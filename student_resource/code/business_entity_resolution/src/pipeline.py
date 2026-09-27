"""
pipeline.py — Memory-Optimized Production Entity Resolution Pipeline.

Target: F_0.5 > 0.991
Constraint: Low RAM (4-8GB available)

Architecture:
  - Country-partitioned processing (loads only one country into memory at a time).
  - TF-IDF with chunked sparse matrix multiplications.
  - LightGBM for classification with F_0.5 optimization.
"""

import os
import sys
import csv
import gc
import time
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from tqdm import tqdm

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from preprocessing import (
    normalize_business_name, normalize_address, make_blocking_text,
    extract_name_tokens, extract_numbers, soundex
)
from features import compute_feature_vector, FEATURE_NAMES

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pipeline")

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "code" / "business_entity_resolution" / "models"

# Hyper-parameters
TFIDF_MAX_FEATURES = 60_000
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_TOP_K = 20
TFIDF_CHUNK_SIZE = 2000
TFIDF_MIN_SIM = 0.15

TRAIN_SAMPLE_SIZE_PER_COUNTRY = 300_000
NEG_POS_RATIO = 4
VALIDATION_FRAC = 0.15

LGB_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 30,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# Memory-Efficient Data Loading (Filtered by Country)
# ===================================================================
def load_source_for_country(path, target_country):
    """Load only records matching the target country to save memory."""
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            country = row.get("country", "").strip()
            if country.lower() != target_country.lower():
                continue
                
            eid = row["entity_id"]
            name_norm = normalize_business_name(row.get("business_name", ""))
            addr_norm = normalize_address(row.get("business_address", ""))
            bt = make_blocking_text(name_norm, addr_norm)
            records[eid] = (name_norm, addr_norm, country, bt)
    return records

def load_ground_truth(path):
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            matched = row.get("matched_entity_ids", "").strip()
            gt[s1_id] = set(matched.split(",")) if matched else set()
    return gt

# ===================================================================
# Blocking (Country Level)
# ===================================================================
def run_tfidf_blocking(s1_records, sx_records):
    s1_ids = list(s1_records.keys())
    sx_ids = list(sx_records.keys())
    
    if not s1_ids or not sx_ids:
        return {}

    sx_texts = [sx_records[eid][3] for eid in sx_ids]
    s1_texts = [s1_records[eid][3] for eid in s1_ids]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES, sublinear_tf=True, dtype=np.float32
    )
    
    log.info(f"  Fitting TF-IDF on {len(sx_texts):,} Sx texts...")
    sx_tfidf = vectorizer.fit_transform(sx_texts)
    
    candidates = defaultdict(dict)
    total_pairs = 0

    log.info(f"  Chunking dot products for {len(s1_ids):,} S1 texts...")
    for chunk_start in range(0, len(s1_ids), TFIDF_CHUNK_SIZE):
        chunk_end = min(chunk_start + TFIDF_CHUNK_SIZE, len(s1_ids))
        chunk_s1_ids = s1_ids[chunk_start:chunk_end]
        chunk_texts = s1_texts[chunk_start:chunk_end]

        s1_tfidf = vectorizer.transform(chunk_texts)
        sim_matrix = s1_tfidf.dot(sx_tfidf.T)

        for i in range(sim_matrix.shape[0]):
            row = sim_matrix.getrow(i)
            if row.nnz == 0: continue

            data, indices = row.data, row.indices
            mask = data >= TFIDF_MIN_SIM
            data, indices = data[mask], indices[mask]

            if len(data) == 0: continue

            if len(data) > TFIDF_TOP_K:
                top_idx = np.argpartition(data, -TFIDF_TOP_K)[-TFIDF_TOP_K:]
                data, indices = data[top_idx], indices[top_idx]

            s1_id = chunk_s1_ids[i]
            for j, sim_val in zip(indices, data):
                candidates[s1_id][sx_ids[j]] = float(sim_val)
                total_pairs += 1

    log.info(f"  Generated {total_pairs:,} candidate pairs.")
    return dict(candidates)

# ===================================================================
# Feature Extraction (Country Level)
# ===================================================================
def extract_training_data_for_country(s1_records, sx_records, gt, country):
    log.info(f"--- Blocking & Features for Country: {country} ---")
    s1_ids = list(s1_records.keys())
    sample_size = min(TRAIN_SAMPLE_SIZE_PER_COUNTRY, len(s1_ids))
    sampled_s1_ids = set(random.sample(s1_ids, sample_size))
    
    s1_sampled = {k: v for k, v in s1_records.items() if k in sampled_s1_ids}
    candidates = run_tfidf_blocking(s1_sampled, sx_records)
    
    features_list, labels = [], []
    pos_count, neg_count = 0, 0
    
    for s1_id in sampled_s1_ids:
        s1_name, s1_addr, s1_country, _ = s1_records[s1_id]
        true_matches = gt.get(s1_id, set())
        cand_dict = candidates.get(s1_id, {})
        
        all_sx = set(cand_dict.keys()) | true_matches
        positives, negatives = [], []
        
        for sx_id in all_sx:
            if sx_id not in sx_records: continue
            sx_name, sx_addr, sx_country, _ = sx_records[sx_id]
            tfidf_sim = cand_dict.get(sx_id, 0.0)
            fv = compute_feature_vector(
                s1_name, s1_addr, s1_country, sx_name, sx_addr, sx_country,
                tfidf_name_sim=tfidf_sim, tfidf_combined_sim=tfidf_sim
            )
            if sx_id in true_matches:
                positives.append(fv)
            else:
                negatives.append(fv)
                
        for fv in positives:
            features_list.append(fv); labels.append(1); pos_count += 1
            
        max_neg = max(len(positives) * NEG_POS_RATIO, 2)
        if len(negatives) > max_neg:
            negatives = random.sample(negatives, max_neg)
        for fv in negatives:
            features_list.append(fv); labels.append(0); neg_count += 1
            
    log.info(f"  Pairs extracted: {pos_count:,} positive, {neg_count:,} negative")
    
    if features_list:
        return np.array(features_list, dtype=np.float32), np.array(labels, dtype=np.int32)
    return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.array([], dtype=np.int32)

def optimise_threshold_f05(y_true, y_proba):
    best_f05, best_th = 0.0, 0.5
    for th in np.arange(0.10, 0.96, 0.02):
        preds = (y_proba >= th).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0

        if f05 > best_f05:
            best_f05, best_th = f05, th
            
    log.info(f"Optimal threshold: {best_th:.2f} (Val F_0.5 = {best_f05:.4f})")
    return best_th

# ===================================================================
# Main Pipeline
# ===================================================================
def main():
    start_time = time.time()
    ensure_dirs()
    random.seed(42)

    log.info("=" * 60)
    log.info("MEMORY-OPTIMIZED ENTITY RESOLUTION PIPELINE")
    log.info("=" * 60)
    
    gt = load_ground_truth(TRAIN_DIR / "train_ground_truth.tsv")
    train_countries = ["us", "india"]
    
    all_X, all_y = [], []
    
    # Process Train Data (Country by Country)
    for c in train_countries:
        s1 = load_source_for_country(TRAIN_DIR / "train_source1.tsv", c)
        s2 = load_source_for_country(TRAIN_DIR / "train_source2.tsv", c)
        s3 = load_source_for_country(TRAIN_DIR / "train_source3.tsv", c)
        sx = {**s2, **s3}
        del s2, s3; gc.collect()
        
        X, y = extract_training_data_for_country(s1, sx, gt, c)
        all_X.append(X)
        all_y.append(y)
        
        del s1, sx, X, y; gc.collect()
        
    X_train_full = np.vstack(all_X)
    y_train_full = np.concatenate(all_y)
    del all_X, all_y; gc.collect()
    
    # Train Model
    from sklearn.model_selection import train_test_split
    log.info("\n=== TRAINING MODEL ===")
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VALIDATION_FRAC, random_state=42, stratify=y_train_full
    )
    
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    LGB_PARAMS["scale_pos_weight"] = pos_weight
    
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)
    
    model = lgb.train(
        LGB_PARAMS, train_data, num_boost_round=800,
        valid_sets=[train_data, val_data], valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    y_proba = model.predict(X_val, num_iteration=model.best_iteration)
    threshold = optimise_threshold_f05(y_val, y_proba)
    
    model.save_model(str(MODEL_DIR / "lgbm_model.txt"))
    with open(MODEL_DIR / "threshold.txt", "w") as f:
        f.write(str(threshold))
        
    del X_train_full, y_train_full, X_tr, X_val, y_tr, y_val; gc.collect()
    
    # Inference (Test Data) - Country by Country
    log.info("\n=== INFERENCE ON TEST DATA ===")
    test_countries = ["us", "india", "france"]
    
    match_path = OUTPUT_DIR / "matching_results.tsv"
    cand_path = OUTPUT_DIR / "candidate_pairs.tsv"
    
    with open(match_path, "w", encoding="utf-8") as fm, open(cand_path, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for c in test_countries:
            log.info(f"Processing Test Country: {c}")
            s1 = load_source_for_country(TEST_DIR / "test_source1.tsv", c)
            s2 = load_source_for_country(TEST_DIR / "test_source2.tsv", c)
            s3 = load_source_for_country(TEST_DIR / "test_source3.tsv", c)
            sx = {**s2, **s3}
            del s2, s3; gc.collect()
            
            cands = run_tfidf_blocking(s1, sx)
            
            for s1_id, sx_dict in tqdm(cands.items(), desc=f"Predicting {c}"):
                s1_name, s1_addr, s1_country, _ = s1[s1_id]
                batch_features, batch_pairs = [], []
                
                for sx_id, sim in sx_dict.items():
                    if sx_id not in sx: continue
                    sx_name, sx_addr, sx_country, _ = sx[sx_id]
                    fv = compute_feature_vector(
                        s1_name, s1_addr, s1_country, sx_name, sx_addr, sx_country,
                        tfidf_name_sim=sim, tfidf_combined_sim=sim
                    )
                    batch_features.append(fv)
                    batch_pairs.append(sx_id)
                    
                matched = []
                if batch_features:
                    probs = model.predict(np.array(batch_features, dtype=np.float32))
                    matched = [sx_id for sx_id, p in zip(batch_pairs, probs) if p >= threshold]
                    
                fm.write(f"{s1_id}\t{','.join(sorted(matched))}\n")
                fc.write(f"{s1_id}\t{','.join(sorted(list(sx_dict.keys())))}\n")
                
            # Write empty results for S1 entities with zero candidates
            for s1_id in s1.keys():
                if s1_id not in cands:
                    fm.write(f"{s1_id}\t\n")
                    fc.write(f"{s1_id}\t\n")
                    
            del s1, sx, cands; gc.collect()
            
    log.info(f"\nPipeline completed in {(time.time() - start_time) / 60:.1f} minutes")

if __name__ == "__main__":
    main()
