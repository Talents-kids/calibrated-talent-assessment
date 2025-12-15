# Multimodal Talent Discovery in Children Using Calibrated Baselines

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.17941256-blue)](https://doi.org/10.5281/zenodo.17941256)
[![License](https://img.shields.io/badge/License-Research-blue)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)

This repository contains the complete research materials for the manuscript:

**"Multimodal Talent Discovery in Children Using Calibrated Baselines"**

Dmitriy Sergeev
Talents.Kids, TEMNIKOVA LDA, Portugal

## 📄 Paper Summary

Early identification of children's talents remains a critical challenge in personalized education. We present a multimodal AI system analyzing authentic creative artifacts from 5,237 analyses across 479 children to predict talent profiles across 7 domains.

**Key Results:**
- **Classical ML Performance**: LightGBM (ROC-AUC 0.9999, F1-macro 0.9972, ECE 0.0018)
- **Multi-Agent LLM System**: 34 models from 9 providers, Gemini r>0.98 correlation
- **Temporal Prediction**: F1-macro 0.833 at 5.7 months, 0.742 at 11.4 months
- **Cost Efficiency**: $0.002-$0.091 per prediction

## 📖 Citation

If you use this code or data in your research, please cite:

**Published Paper:**
```bibtex
@article{sergeev2025multimodal,
  title={Multimodal Talent Discovery in Children Using Calibrated Baselines},
  author={Sergeev, Dmitriy},
  year={2025},
  doi={10.5281/zenodo.17941256}
}
```

**Code Repository:**
```bibtex
@software{sergeev2025talent_code,
  author = {Sergeev, Dmitriy},
  title = {Multimodal Talent Discovery in Children: Code and Data},
  year = {2025},
  url = {https://github.com/talents-kids/calibrated-talent-assessment},
  doi = {10.5281/zenodo.17941256},
  version = {1.0.0}
}
```

**Related Work:**
```bibtex
@software{sergeev2025talent_llm,
  author = {Sergeev, Dmitriy},
  title = {TALENT LLM: Multi-Label Talent Prediction in Children Using Fine-Tuned Large Language Models with Calibrated Baselines},
  year = {2025},
  doi = {10.5281/zenodo.17743456},
  note = {Complementary work focusing on LLM fine-tuning approaches for the same talent discovery system}
}
```

## 📊 Dataset

### Overview
- **5,173 artifact assessments** from **479 children** (ages 6-18)
- **8 modality types**: Text (50.8%), Image (30.2%), Musical (17.6%), Audio (0.9%), Video (0.1%), PDF (0.1%), JSON (0.2%), DOCX (<0.1%)
- **7 talent domains**: Academic (41.2%), Artistic (32.3%), Athletic (10.1%), Leadership (8.1%), Service (5.2%), Technology (2.2%), Other (0.9%)
- **306 fine-grained categories** mapped to top-level domains

### Files

```
data/
├── README.md                   # Dataset documentation
├── sample_artifacts.jsonl      # Anonymized artifact samples (n=100)
├── metadata_summary.json       # Dataset statistics
├── llm_metadata.csv            # Multi-agent system usage data
└── llm_accuracy.csv            # Individual model validation metrics
```

**Note**: Full dataset cannot be shared publicly due to GDPR/COPPA compliance for minor participants. Anonymized samples provided for research purposes.

### Data Format

Each artifact assessment contains:

| Field | Description |
|-------|-------------|
| `child_hash` | Anonymized child identifier (SHA256) |
| `analysis_type` | Artifact modality (text, image, musical, audio, video, pdf, json, docx) |
| `age` | Child's age in years |
| `category_scores` | Dict of 306 fine-grained talent scores (0-10) |
| `bin_scores` | Dict of 7 aggregated domain scores |
| `key_talents` | List of top talent categories |

## 📈 Results

### Performance Metrics

**Table: Classical ML Performance (Test Set n=682)**

| Model | ROC-AUC | F1-Macro | ECE | Inference Time |
|-------|---------|----------|-----|----------------|
| Logistic Regression (calibrated) | 0.9956 | 0.9734 | 0.0039 | 0.12ms |
| LightGBM (uncalibrated) | 0.9999 | 0.9972 | 0.0018 | 2.3ms |
| LightGBM (calibrated) | 0.9996 | 0.9920 | 0.0031 | 2.3ms |

**Multi-Agent LLM System (Production Deployment)**

| Model | Invocations | Cost/Pred | Correlation | MAE |
|-------|------------|-----------|-------------|-----|
| Qwen3-235B | 2,379 | $0.012 | - | - |
| DeepSeek-V3 | 2,369 | $0.018 | - | - |
| Kimi-K2 | 2,340 | $0.047 | - | - |
| Llama-4-Scout | 1,608 | $0.002 | - | - |
| gemini-2.5-flash | 7,187 | $0.006 | **0.9997** | **0.0023** |
| gemini-2.5-flash-preview | 4,048 | - | **0.9999** | **0.0012** |

**Total**: 34 models, 9 providers, 12,041 invocations, $213.34 total cost ($0.041/analysis)

**Temporal Prediction (Longitudinal Validation)**

| Prediction Task | F1-Macro | 95% CI | n (children) |
|----------------|----------|--------|--------------|
| S1→S2 (5.7 months) | 0.833 | [0.808, 0.857] | 70 |
| S1→S3 (11.4 months) | 0.742 | [0.715, 0.768] | 127 |

## 🔬 Code

### Notebooks

- `notebooks/analysis.ipynb` - Complete experimental pipeline (Colab-ready)
- `notebooks/figure_generation.ipynb` - All manuscript figures generation

### Scripts

```
code/
├── train_classical_ml.py       # LightGBM, Logistic Regression training
├── calibration_analysis.py     # Platt scaling, isotonic regression
├── temporal_prediction.py      # S1→S2→S3 longitudinal evaluation
├── shap_interpretability.py    # Feature importance analysis
├── bootstrap_ci.py             # Confidence interval computation
└── utils/                      # Helper functions
```

### Running the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Train classical models
python code/train_classical_ml.py --data data/sample_artifacts.jsonl

# Generate figures
python code/generate_figures.py --output figures/

# Run full analysis pipeline
jupyter notebook notebooks/analysis.ipynb
```

## 📊 Figures

### Main Figures

1. **Figure 1**: Multimodal Feature Engineering Pipeline
   - Panel A: Artifact processing workflow
   - Panel B: Classical model architectures
   - Panel C: Calibration methods (Platt, isotonic, temperature)
   - Panel D: 306 categories → 7 domain taxonomy

2. **Figure 2**: Temporal Prediction & Multi-Agent LLM Performance
   - Panel A: Multi-agent cost-efficiency (34 models, cost vs usage)
   - Panel B: Individual Gemini accuracy validation (r>0.98)
   - Panel C: Temporal F1 decay (S1→S2: 0.833, S1→S3: 0.742)
   - Panel D: Per-domain temporal stability

3. **Figure 3**: Model Interpretability via SHAP Analysis
   - Panel A: Academic domain top features
   - Panel B: Sport/Athletic domain top features
   - Panel C: Artistic domain top features
   - Panel D: Cross-domain feature importance comparison

### Supplemental Figures

- **Figure S1**: Complete domain-level confusion matrices (7 domains)
- **Figure S2**: Calibration reliability diagrams (all models)
- **Figure S3**: Dataset composition analysis
- **Figure S4**: Model architecture comparison
- **Figure S5**: Extended temporal analysis (confidence-stratified)

## 🔧 System Requirements

**Software:**
- Python 3.11+
- scikit-learn 1.5.0
- LightGBM 4.3.0
- PyTorch 2.2.1
- SHAP 0.45.0
- Transformers 4.39.0

**Hardware (Training):**
- NVIDIA A100 GPU (40GB) recommended
- 256GB RAM for full dataset
- CPU-only inference supported

**Hardware (Inference):**
- CPU-only (Intel Xeon, 32 cores)
- 16GB RAM minimum

## 📖 Citation

```bibtex
@article{sergeev2025talent,
  title={Multimodal Talent Discovery in Children Using Calibrated Baselines},
  author={Sergeev, Dmitriy},
  year={2025},
  doi={10.5281/zenodo.17941256},
  note={Preprint available on Zenodo}
}
```

## 📜 License

This dataset and code are released for research purposes only.

**Dataset**: Anonymized samples only. Full dataset protected by GDPR/COPPA.
**Code**: MIT License

## 🔗 Links

- **Platform**: [talents.kids](https://talents.kids)
- **Paper**: [10.5281/zenodo.17941256](https://doi.org/10.5281/zenodo.17941256)

- **Code Repository**: [github.com/talents-kids/calibrated-talent-assessment](https://github.com/talents-kids/calibrated-talent-assessment)

## 📧 Contact

- **Author**: Dmitriy Sergeev
- **Organization**: Talents.Kids, TEMNIKOVA LDA
- **ORCID**: 0009-0008-2958-4595
- **Contact**: https://www.talents.kids/contact

## 🙏 Acknowledgments

We thank the children, parents, and educators who participated in this study through the Talents.kids platform. We acknowledge:

- Educational psychologists who validated interpretability patterns
- Open-source ML libraries (scikit-learn, LightGBM, SHAP)
- Computational resources from talents.kids infrastructure

## 📋 Reproducibility

All experiments use:
- Random seed = 42
- 70/15/15 train/val/test split (stratified)
- 10,000 bootstrap iterations for CIs
- McNemar's test for model comparison (χ²=33.47, p<0.001)

Complete hyperparameters and statistical methods detailed in STAR Methods section of manuscript.
