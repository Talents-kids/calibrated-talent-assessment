# Dataset Documentation

This directory contains **anonymized samples** and **metadata summaries** from the talent discovery research study.

## ⚠️ Privacy Notice

**Full dataset is NOT publicly available** due to:
- GDPR compliance (children's data protection)
- COPPA compliance (under-13 participants)
- IRB requirements (human subjects research)

Only **anonymized samples** (n=10) are provided here for reproducibility.

## Files

### 1. `sample_artifacts.jsonl`

**10 anonymized artifact assessments** representing different modalities and age groups.

**Format**: JSONL (one JSON object per line)

**Fields**:
- `child_hash`: SHA256 hash of child identifier (anonymized)
- `age`: Child's age in years
- `analysis_type`: Artifact modality (text, image, musical, video, audio)
- `artifact_content`: "[REDACTED]" - actual content removed for privacy
- `category_scores`: Dict of fine-grained talent scores (0-10)
- `bin_scores`: Dict of 7 aggregated domain scores
- `key_talents`: List of top talent categories
- `predicted_domain`: Primary talent domain
- `confidence`: Model confidence (0-1)

**Example**:
```json
{
  "child_hash": "a1b2c3d4e5f6...",
  "age": 8,
  "analysis_type": "text",
  "artifact_content": "[REDACTED - Creative writing sample]",
  "category_scores": {"writing_creativity": 8.5, "scientific_curiosity": 9.2},
  "bin_scores": {"Academic": 8.4, "Artistic": 6.2, ...},
  "key_talents": ["scientific_curiosity", "writing_creativity"],
  "predicted_domain": "Academic",
  "confidence": 0.92
}
```

### 2. `metadata_summary.json`

**Complete dataset statistics** including:
- Dataset overview (5,173 analyses, 479 children)
- Modality distribution (Text 50.8%, Image 30.2%, etc.)
- Domain distribution (Academic 41.2%, Artistic 32.3%, etc.)
- Temporal data (S1→S2, S1→S3 cohorts)
- Demographics (age distribution, 96% missing gender data)
- Train/val/test split (70/15/15 stratified)
- Anonymization methods

### 3. `llm_metadata.csv`

**Multi-agent LLM system usage data** from production deployment:
- 34 models from 9 providers
- 12,041 model invocations
- Cost per prediction ($0.002 - $0.250)
- Total cost: $213.34 ($0.041/analysis)

**Columns**:
- `model`: Full model identifier
- `provider`: Model provider (Baseten, Google, Groq, OpenAI, xAI, Anthropic, etc.)
- `invocations`: Number of times model was used
- `cost_per_prediction`: Cost in USD per single prediction
- `total_cost`: Total cost for all invocations
- `percentage_usage`: % of total invocations

### 4. `llm_accuracy.csv`

**Individual model validation metrics** (4 Gemini models):
- 11,346 matched predictions with ground truth
- Correlation: 0.986 - 1.000
- MAE: 0.000 - 0.024
- RMSE: 0.000 - 0.148

**Columns**:
- `model`: Gemini model variant
- `predictions`: Number of predictions evaluated
- `correlation`: Pearson correlation with ground truth (finalTalentProfile)
- `mae`: Mean Absolute Error (0-10 scale)
- `rmse`: Root Mean Square Error
- `r2_score`: R² coefficient of determination
- `unique_analyses`: Number of unique artifact analyses
- `unique_children`: Number of unique children
- `unique_categories`: Number of unique talent categories evaluated

## Requesting Full Dataset

For research purposes, the **full dataset** can be requested if you meet the following criteria:

1. **IRB Approval**: Active IRB approval for human subjects research
2. **Data Use Agreement**: Signed DUA with Talents.Kids
3. **GDPR/COPPA Compliance**: Demonstrated compliance with data protection regulations
4. **Research Purpose**: Clear research objective aligned with talent discovery / child development

**Contact**: ds@talents.kids with:
- IRB approval letter
- Research proposal (1-2 pages)
- Institutional affiliation
- Proposed data handling procedures

## Dataset Citation

If using this data, please cite:

```bibtex
@article{sergeev2025talent,
  title={Multimodal Talent Discovery in Children Using Calibrated Baselines},
  author={Sergeev, Dmitriy},
  
  year={2025},
  publisher={Cell Press},
  note={Manuscript under review}
}
```

## Data Format Details

### Category Scores

**306 fine-grained categories** across 7 domains:

**Academic** (125 categories):
- writing_creativity, scientific_curiosity, analytical_thinking, ...

**Artistic** (95 categories):
- visual_arts, music_performance, creative_writing, theater, ...

**Athletic** (32 categories):
- coordination, athletic_skill, physical_fitness, teamwork, ...

**Leadership** (21 categories):
- leadership, public_speaking, strategic_thinking, persuasion, ...

**Service** (14 categories):
- empathy, social_awareness, community_engagement, ...

**Technology** (12 categories):
- programming, problem_solving, technical_documentation, ...

**Other** (7 categories):
- culinary_arts, entrepreneurship, gaming_strategy, ...

### Bin Scores

**7 aggregated domains** (weighted average of category scores):
- `Academic`: STEM, humanities, research
- `Artistic`: Visual arts, music, theater, creative writing
- `Athletic`: Sports, physical fitness, coordination
- `Leadership`: Leadership skills, public speaking, organization
- `Service`: Community service, empathy, social awareness
- `Technology`: Programming, digital media, technical skills
- `Other`: Miscellaneous talents

## Reproducibility Notes

All experiments use:
- **Random seed**: 42
- **Split**: 70/15/15 train/val/test (stratified by domain)
- **Bootstrap CIs**: 10,000 iterations
- **Statistical tests**: McNemar's test (χ²=33.47, p<0.001)

See `../code/` for complete analysis scripts.

---

**Last Updated**: 2025-12-15
**Version**: 1.0
