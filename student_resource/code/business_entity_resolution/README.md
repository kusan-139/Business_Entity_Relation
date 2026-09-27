# Business Entity Resolution — Solution

## Overview

This solution implements a **multi-stage entity resolution pipeline** to match
business records across three independent data sources. It uses multi-key
blocking for candidate generation and a LightGBM gradient-boosted classifier
for match prediction, optimised for the F₀.₅ metric.

## Architecture

```
Pipeline Stages:
  1. Load & Preprocess    — Normalise names, addresses, countries
  2. Multi-Key Blocking   — Reduce comparison space via inverted index
  3. Feature Extraction    — 40+ string similarity features per pair
  4. Model Training        — LightGBM binary classifier
  5. Threshold Optimisation — Maximise F₀.₅ on validation split
  6. Test Prediction       — Score all test candidate pairs
  7. Output Generation     — Write matching_results.tsv & candidate_pairs.tsv
```

## Files

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py           # Package init
│   ├── preprocessing.py      # Text normalisation & blocking keys
│   ├── features.py           # Pairwise similarity feature engineering
│   └── pipeline.py           # Main end-to-end pipeline
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

From the `student_resource/` directory:

```bash
python code/business_entity_resolution/src/pipeline.py
```

This will:
1. Read all training and test TSVs from `dataset/`
2. Train a LightGBM model (saved to `code/business_entity_resolution/models/`)
3. Generate predictions on the test set
4. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`

## Validate Output

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Key Design Decisions

1. **Multi-key blocking** — Generates multiple blocking keys per record (country+prefix, acronyms, ZIP/PIN codes, character n-grams). Records sharing ≥1 key become candidates. Candidate set is capped at 50 per S1 entity to control scale.

2. **Rich feature engineering** — 40+ features spanning name similarity (Levenshtein, Jaccard, token sort/set ratios, n-gram Jaccard), address similarity (token overlap, numeric matching, ZIP code matching), and cross-field signals.

3. **LightGBM classifier** — Fast gradient-boosted trees with scale_pos_weight for class imbalance. Trained on a balanced sample with controlled negative sampling (3:1 neg:pos ratio).

4. **F₀.₅ threshold optimisation** — Threshold tuned on a held-out validation set to maximise the precision-heavy F₀.₅ metric, since false merges are penalised more than missed matches.

5. **Country-agnostic design** — No hard-coded country lists. Blocking and features work with any country string, including the unseen France test entities.
