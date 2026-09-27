"""
Quick sanity check — runs the pipeline on a tiny sample to verify
all stages work end-to-end before the full run.
"""
import sys
import csv
import time
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import normalize_business_name, normalize_address, get_blocking_keys
from features import compute_feature_vector, FEATURE_NAMES

BASE = Path(__file__).resolve().parents[3]
TRAIN_DIR = BASE / "dataset" / "train"

print(f"Feature count: {len(FEATURE_NAMES)}")
print("Loading a small sample of data ...")

# Load first 1000 rows from each source
def load_sample(path, n=1000):
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= n:
                break
            eid = row["entity_id"]
            name_norm = normalize_business_name(row.get("business_name", ""))
            addr_norm = normalize_address(row.get("business_address", ""))
            country = row.get("country", "").strip()
            records[eid] = (name_norm, addr_norm, country)
    return records

s1 = load_sample(TRAIN_DIR / "train_source1.tsv", 1000)
s2 = load_sample(TRAIN_DIR / "train_source2.tsv", 2000)
s3 = load_sample(TRAIN_DIR / "train_source3.tsv", 2000)
print(f"Loaded S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")

# Load ground truth for these S1 IDs
gt = {}
with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        s1_id = row["source1_entity_id"]
        if s1_id in s1:
            matched = row.get("matched_entity_ids", "").strip()
            gt[s1_id] = set(matched.split(",")) if matched else set()
print(f"Ground truth entries: {len(gt)}")

# Test blocking
from collections import defaultdict
sx = {**s2, **s3}
index = defaultdict(set)
for eid, (name, addr, country) in sx.items():
    for k in get_blocking_keys(name, addr, country):
        index[k].add(eid)
print(f"Blocking index keys: {len(index)}")

# Generate candidates for first 10 S1 entities
candidates = {}
for s1_id in list(s1.keys())[:10]:
    name, addr, country = s1[s1_id]
    keys = get_blocking_keys(name, addr, country)
    cands = set()
    for k in keys:
        if k in index:
            cands |= index[k]
    candidates[s1_id] = cands

total_cands = sum(len(v) for v in candidates.values())
print(f"Candidates for 10 S1 entities: {total_cands}")

# Test feature extraction on a few pairs
t0 = time.time()
count = 0
for s1_id, cand_set in candidates.items():
    s1_name, s1_addr, s1_country = s1[s1_id]
    for sx_id in list(cand_set)[:5]:
        if sx_id in sx:
            sx_name, sx_addr, sx_country = sx[sx_id]
            fv = compute_feature_vector(s1_name, s1_addr, s1_country,
                                         sx_name, sx_addr, sx_country)
            count += 1
t1 = time.time()
print(f"Computed {count} feature vectors in {t1-t0:.2f}s")
print(f"Rate: {count/(t1-t0):.0f} pairs/sec")

print("\n✓ Sanity check passed!")
