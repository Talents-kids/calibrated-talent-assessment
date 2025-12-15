"""
Bootstrap Confidence Intervals
==============================

Calculate 95% bootstrap confidence intervals for model performance metrics.

Usage:
    python bootstrap_ci.py --predictions predictions.csv --n_iterations 10000
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.utils import resample
from tqdm import tqdm


def bootstrap_metric(y_true, y_pred, y_proba, metric_func, n_iterations=10000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for a metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC)
        metric_func: Metric function to apply
        n_iterations: Number of bootstrap iterations
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Dictionary with mean, lower_bound, upper_bound
    """
    scores = []

    for _ in tqdm(range(n_iterations), desc=f"Bootstrap {metric_func.__name__}"):
        # Resample with replacement
        indices = resample(np.arange(len(y_true)), random_state=None)

        y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_boot = y_pred.iloc[indices] if hasattr(y_pred, 'iloc') else y_pred[indices]

        # Calculate metric
        if metric_func.__name__ == 'roc_auc_score' and y_proba is not None:
            y_proba_boot = y_proba[indices]
            score = metric_func(y_true_boot, y_proba_boot, multi_class='ovr', average='macro')
        else:
            score = metric_func(y_true_boot, y_pred_boot, average='macro')

        scores.append(score)

    scores = np.array(scores)

    # Calculate confidence interval
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(scores, alpha * 100)
    upper_bound = np.percentile(scores, (1 - alpha) * 100)
    mean_score = np.mean(scores)

    return {
        'mean': mean_score,
        'lower': lower_bound,
        'upper': upper_bound,
        'std': np.std(scores)
    }


def calculate_all_metrics_with_ci(y_true, y_pred, y_proba=None, n_iterations=10000):
    """
    Calculate all metrics with bootstrap CIs.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)
        n_iterations: Number of bootstrap iterations

    Returns:
        Dictionary of metrics with CIs
    """
    metrics = {}

    # F1-score
    print("Calculating F1-score CI...")
    metrics['f1'] = bootstrap_metric(y_true, y_pred, y_proba, f1_score, n_iterations)

    # Precision
    print("Calculating Precision CI...")
    metrics['precision'] = bootstrap_metric(y_true, y_pred, y_proba, precision_score, n_iterations)

    # Recall
    print("Calculating Recall CI...")
    metrics['recall'] = bootstrap_metric(y_true, y_pred, y_proba, recall_score, n_iterations)

    # ROC-AUC (if probabilities provided)
    if y_proba is not None:
        print("Calculating ROC-AUC CI...")
        metrics['roc_auc'] = bootstrap_metric(y_true, y_pred, y_proba, roc_auc_score, n_iterations)

    return metrics


def format_ci(metric_dict):
    """Format CI as string."""
    return f"{metric_dict['mean']:.4f} ({metric_dict['lower']:.4f}-{metric_dict['upper']:.4f})"


def main():
    parser = argparse.ArgumentParser(description='Calculate bootstrap confidence intervals')
    parser.add_argument('--predictions', type=str, required=True,
                        help='CSV file with y_true, y_pred, y_proba columns')
    parser.add_argument('--n_iterations', type=int, default=10000,
                        help='Number of bootstrap iterations (default: 10000)')
    parser.add_argument('--output', type=str, default='bootstrap_results.csv',
                        help='Output CSV file for results')
    args = parser.parse_args()

    print("="*60)
    print("Bootstrap Confidence Intervals")
    print("="*60)
    print(f"Iterations: {args.n_iterations}")
    print(f"Confidence: 95%")

    # Load predictions
    print(f"\nLoading predictions from: {args.predictions}")
    df = pd.read_csv(args.predictions)

    # Extract columns
    y_true = df['y_true']
    y_pred = df['y_pred']
    y_proba = df.filter(regex='proba_').values if 'proba_' in df.columns[0] else None

    print(f"Samples: {len(y_true)}")

    # Calculate metrics with CIs
    metrics = calculate_all_metrics_with_ci(y_true, y_pred, y_proba, args.n_iterations)

    # Print results
    print(f"\n{'='*60}")
    print("Results (Mean and 95% CI)")
    print(f"{'='*60}")
    for metric_name, metric_dict in metrics.items():
        print(f"{metric_name.upper()}: {format_ci(metric_dict)}")

    # Save to CSV
    results_df = pd.DataFrame([
        {
            'metric': metric_name,
            'mean': metric_dict['mean'],
            'lower_95': metric_dict['lower'],
            'upper_95': metric_dict['upper'],
            'std': metric_dict['std']
        }
        for metric_name, metric_dict in metrics.items()
    ])

    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
