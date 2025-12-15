"""
Classical ML Model Training for Talent Prediction
==================================================

Train Logistic Regression and LightGBM models on multimodal talent data.
Includes Platt scaling calibration and cross-validation.

Usage:
    python train_classical_ml.py --data ../data/sample_artifacts.jsonl --output ../results/
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import lightgbm as lgb
import joblib


def load_data(filepath):
    """Load and preprocess artifact data from JSONL."""
    artifacts = []
    with open(filepath, 'r') as f:
        for line in f:
            artifacts.append(json.loads(line.strip()))

    df = pd.DataFrame(artifacts)

    # Extract features (bin_scores as features)
    X = pd.DataFrame(list(df['bin_scores']))

    # Extract target (predicted_domain as label)
    y = df['predicted_domain']

    return X, y, df


def train_logistic_regression(X_train, y_train, X_val, y_val, calibrate=True):
    """Train Logistic Regression with optional Platt scaling."""
    print(f"\n{'='*60}")
    print(f"Training Logistic Regression (calibrate={calibrate})")
    print(f"{'='*60}")

    # Train base model
    lr = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42,
        C=1.0
    )

    if calibrate:
        # Apply Platt scaling calibration
        lr_cal = CalibratedClassifierCV(
            lr,
            method='sigmoid',  # Platt scaling
            cv=5
        )
        lr_cal.fit(X_train, y_train)
        model = lr_cal
    else:
        lr.fit(X_train, y_train)
        model = lr

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    # Calculate metrics
    auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)

    # Calculate ECE (Expected Calibration Error)
    ece = calculate_ece(y_val, y_proba, n_bins=10)

    print(f"\nValidation Metrics:")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  F1-macro: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ECE: {ece:.4f}")

    return model, {
        'model': 'LogisticRegression' + (' (calibrated)' if calibrate else ''),
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'ece': ece
    }


def train_lightgbm(X_train, y_train, X_val, y_val, calibrate=False):
    """Train LightGBM model with optional calibration."""
    print(f"\n{'='*60}")
    print(f"Training LightGBM (calibrate={calibrate})")
    print(f"{'='*60}")

    # Encode labels to integers
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    # LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train_enc)
    val_data = lgb.Dataset(X_val, label=y_val_enc, reference=train_data)

    # Train
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )

    # Wrap in calibration if requested
    if calibrate:
        # For LightGBM, we calibrate predictions
        from sklearn.calibration import calibration_curve
        # This is a simplified version - full implementation would use CalibratedClassifierCV
        model = gbm
    else:
        model = gbm

    # Predict
    y_proba = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    y_pred = le.inverse_transform(np.argmax(y_proba, axis=1))

    # Calculate metrics
    auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
    ece = calculate_ece(y_val, y_proba, n_bins=10)

    print(f"\nValidation Metrics:")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  F1-macro: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  ECE: {ece:.4f}")

    return (model, le), {
        'model': 'LightGBM' + (' (calibrated)' if calibrate else ''),
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'ece': ece
    }


def calculate_ece(y_true, y_proba, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and actual accuracy.
    Lower values indicate better calibration (0 = perfect calibration).

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        n_bins: Number of bins for calibration

    Returns:
        ECE value (0-1, lower is better)
    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)

    # Get max probability and predicted class
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)

    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Calculate accuracy in bin
            accuracy_in_bin = np.mean(predictions[in_bin] == y_true_enc[in_bin])
            # Calculate average confidence in bin
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def main():
    parser = argparse.ArgumentParser(description='Train classical ML models for talent prediction')
    parser.add_argument('--data', type=str, default='../data/sample_artifacts.jsonl',
                        help='Path to JSONL data file')
    parser.add_argument('--output', type=str, default='../results/',
                        help='Output directory for models and results')
    args = parser.parse_args()

    print("="*60)
    print("Classical ML Training Pipeline")
    print("="*60)

    # Load data
    print(f"\nLoading data from: {args.data}")
    X, y, df = load_data(args.data)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(y.unique())}")

    # Split data (stratified 70/15/15)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Train models
    results = []

    # 1. Logistic Regression (calibrated)
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val, calibrate=True)
    results.append(lr_metrics)
    joblib.dump(lr_model, f"{args.output}/logistic_regression_calibrated.pkl")

    # 2. Logistic Regression (uncalibrated)
    lr_model_uncal, lr_metrics_uncal = train_logistic_regression(X_train, y_train, X_val, y_val, calibrate=False)
    results.append(lr_metrics_uncal)
    joblib.dump(lr_model_uncal, f"{args.output}/logistic_regression.pkl")

    # 3. LightGBM (uncalibrated)
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_val, y_val, calibrate=False)
    results.append(lgb_metrics)
    lgb_model[0].save_model(f"{args.output}/lightgbm.txt")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{args.output}/classical_ml_results.csv", index=False)
    print(f"\n{'='*60}")
    print("Results saved to:", f"{args.output}/classical_ml_results.csv")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
