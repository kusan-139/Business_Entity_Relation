"""
features.py — Advanced pairwise feature engineering for entity resolution.

~45 similarity features covering:
  - Name: fuzzy ratios, Jaro-Winkler, Levenshtein, token overlap, n-grams
  - Address: fuzzy ratios, token overlap, numeric matching, ZIP/PIN
  - Cross-field: combined similarity, metadata
  - External: TF-IDF cosine (injected from blocking stage)

All features are normalised to [0, 1] where possible for stable LightGBM training.
"""

import re

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

try:
    from .preprocessing import (
        extract_name_tokens, extract_address_tokens, extract_numbers,
        get_name_ngrams,
    )
except ImportError:
    from preprocessing import (
        extract_name_tokens, extract_address_tokens, extract_numbers,
        get_name_ngrams,
    )


# ---------------------------------------------------------------------------
# Set similarity helpers
# ---------------------------------------------------------------------------
def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def overlap_coeff(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def dice(a, b):
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def containment(a, b):
    """Fraction of a contained in b."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------
def compute_features(s1_name, s1_addr, s1_country,
                     sx_name, sx_addr, sx_country,
                     tfidf_name_sim=None, tfidf_addr_sim=None,
                     tfidf_combined_sim=None):
    """
    Compute pairwise features. All name/addr inputs are NORMALISED strings.
    Optional TF-IDF cosine similarities can be injected from the blocking stage.

    Returns dict of feature_name -> float.
    """
    f = {}

    # ===================================================================
    # 1. COUNTRY
    # ===================================================================
    f["country_match"] = 1.0 if s1_country == sx_country else 0.0

    # ===================================================================
    # 2. NAME FEATURES
    # ===================================================================
    has_name = bool(s1_name and sx_name)

    # 2a. RapidFuzz string ratios
    f["name_ratio"]        = fuzz.ratio(s1_name, sx_name) / 100.0 if has_name else 0.0
    f["name_partial"]      = fuzz.partial_ratio(s1_name, sx_name) / 100.0 if has_name else 0.0
    f["name_tok_sort"]     = fuzz.token_sort_ratio(s1_name, sx_name) / 100.0 if has_name else 0.0
    f["name_tok_set"]      = fuzz.token_set_ratio(s1_name, sx_name) / 100.0 if has_name else 0.0
    f["name_wratio"]       = fuzz.WRatio(s1_name, sx_name) / 100.0 if has_name else 0.0
    f["name_partial_tok"]  = fuzz.partial_token_sort_ratio(s1_name, sx_name) / 100.0 if has_name else 0.0
    f["name_partial_tset"] = fuzz.partial_token_set_ratio(s1_name, sx_name) / 100.0 if has_name else 0.0

    # 2b. Jaro-Winkler (excellent for name typos)
    f["name_jaro_winkler"] = JaroWinkler.similarity(s1_name, sx_name) if has_name else 0.0

    # 2c. Normalised Levenshtein
    if has_name:
        max_len = max(len(s1_name), len(sx_name))
        f["name_lev_norm"] = 1.0 - Levenshtein.distance(s1_name, sx_name) / max_len if max_len else 0.0
    else:
        f["name_lev_norm"] = 0.0

    # 2d. Token-level set similarities
    s1_ntok = extract_name_tokens(s1_name)
    sx_ntok = extract_name_tokens(sx_name)
    f["name_jaccard"]       = jaccard(s1_ntok, sx_ntok)
    f["name_overlap"]       = overlap_coeff(s1_ntok, sx_ntok)
    f["name_dice"]          = dice(s1_ntok, sx_ntok)
    f["name_contain_s1"]    = containment(s1_ntok, sx_ntok)
    f["name_contain_sx"]    = containment(sx_ntok, s1_ntok)

    # 2e. Character n-gram similarities
    f["name_ng3_jaccard"] = jaccard(get_name_ngrams(s1_name, 3), get_name_ngrams(sx_name, 3))
    f["name_ng4_jaccard"] = jaccard(get_name_ngrams(s1_name, 4), get_name_ngrams(sx_name, 4))
    f["name_ng5_jaccard"] = jaccard(get_name_ngrams(s1_name, 5), get_name_ngrams(sx_name, 5))

    # 2f. Exact / prefix match
    f["name_exact"]      = 1.0 if has_name and s1_name == sx_name else 0.0
    s1c = s1_name.replace(" ", "") if s1_name else ""
    sxc = sx_name.replace(" ", "") if sx_name else ""
    f["name_prefix4"]    = 1.0 if len(s1c) >= 4 and len(sxc) >= 4 and s1c[:4] == sxc[:4] else 0.0
    f["name_prefix6"]    = 1.0 if len(s1c) >= 6 and len(sxc) >= 6 and s1c[:6] == sxc[:6] else 0.0

    # 2g. Length features
    f["name_len_ratio"] = (min(len(s1_name), len(sx_name)) /
                           max(len(s1_name), len(sx_name))
                           if has_name and max(len(s1_name), len(sx_name)) > 0
                           else 0.0)
    f["name_len_diff"]  = abs(len(s1_name) - len(sx_name)) if has_name else 0.0

    # 2h. First meaningful word match
    m_s1 = [w for w in s1_name.split() if len(w) > 2] if s1_name else []
    m_sx = [w for w in sx_name.split() if len(w) > 2] if sx_name else []
    f["name_first_word"] = 1.0 if m_s1 and m_sx and m_s1[0] == m_sx[0] else 0.0

    # 2i. Token count features
    s1_wc = len(s1_name.split()) if s1_name else 0
    sx_wc = len(sx_name.split()) if sx_name else 0
    f["name_tok_cnt_ratio"] = min(s1_wc, sx_wc) / max(s1_wc, sx_wc) if max(s1_wc, sx_wc) > 0 else 0.0

    # ===================================================================
    # 3. ADDRESS FEATURES
    # ===================================================================
    has_addr = bool(s1_addr and sx_addr)

    # 3a. RapidFuzz string ratios
    f["addr_ratio"]    = fuzz.ratio(s1_addr, sx_addr) / 100.0 if has_addr else 0.0
    f["addr_partial"]  = fuzz.partial_ratio(s1_addr, sx_addr) / 100.0 if has_addr else 0.0
    f["addr_tok_sort"] = fuzz.token_sort_ratio(s1_addr, sx_addr) / 100.0 if has_addr else 0.0
    f["addr_tok_set"]  = fuzz.token_set_ratio(s1_addr, sx_addr) / 100.0 if has_addr else 0.0

    # 3b. Jaro-Winkler on address
    f["addr_jaro_winkler"] = JaroWinkler.similarity(s1_addr, sx_addr) if has_addr else 0.0

    # 3c. Token-level set similarities on address
    s1_atok = extract_address_tokens(s1_addr)
    sx_atok = extract_address_tokens(sx_addr)
    f["addr_jaccard"]  = jaccard(s1_atok, sx_atok)
    f["addr_overlap"]  = overlap_coeff(s1_atok, sx_atok)
    f["addr_dice"]     = dice(s1_atok, sx_atok)

    # 3d. Numeric matching (street numbers, ZIP/PIN)
    s1_nums = extract_numbers(s1_addr)
    sx_nums = extract_numbers(sx_addr)
    f["addr_num_jaccard"] = jaccard(s1_nums, sx_nums)
    f["addr_num_contain"] = containment(s1_nums, sx_nums)

    # 3e. ZIP / PIN code match (5+ digit numbers)
    s1_zips = {n for n in s1_nums if len(n) >= 5}
    sx_zips = {n for n in sx_nums if len(n) >= 5}
    f["addr_zip_match"] = 1.0 if s1_zips and sx_zips and s1_zips & sx_zips else 0.0
    f["addr_has_zip"]   = 1.0 if s1_zips or sx_zips else 0.0

    # 3f. Address length and emptiness
    f["addr_len_ratio"]  = (min(len(s1_addr), len(sx_addr)) /
                            max(len(s1_addr), len(sx_addr))
                            if has_addr and max(len(s1_addr), len(sx_addr)) > 0
                            else 0.0)
    f["s1_addr_empty"]   = 1.0 if not s1_addr else 0.0
    f["sx_addr_empty"]   = 1.0 if not sx_addr else 0.0
    f["both_addr_empty"] = 1.0 if not s1_addr and not sx_addr else 0.0

    # ===================================================================
    # 4. CROSS-FIELD / COMBINED FEATURES
    # ===================================================================
    s1_comb = (s1_name + " " + s1_addr).strip()
    sx_comb = (sx_name + " " + sx_addr).strip()
    f["combined_tok_set"] = fuzz.token_set_ratio(s1_comb, sx_comb) / 100.0 if s1_comb and sx_comb else 0.0
    f["combined_tok_sort"] = fuzz.token_sort_ratio(s1_comb, sx_comb) / 100.0 if s1_comb and sx_comb else 0.0

    # Weighted aggregate
    f["agg_score"] = (
        0.45 * f["name_tok_set"] +
        0.25 * f["addr_tok_set"] +
        0.15 * f["name_jaro_winkler"] +
        0.10 * f["country_match"] +
        0.05 * f["addr_num_jaccard"]
    )

    # ===================================================================
    # 5. EXTERNAL TF-IDF FEATURES (injected from blocking)
    # ===================================================================
    f["tfidf_name_sim"]     = tfidf_name_sim if tfidf_name_sim is not None else 0.0
    f["tfidf_addr_sim"]     = tfidf_addr_sim if tfidf_addr_sim is not None else 0.0
    f["tfidf_combined_sim"] = tfidf_combined_sim if tfidf_combined_sim is not None else 0.0

    return f


# Deterministic feature order
FEATURE_NAMES = list(compute_features("", "", "", "", "", "").keys())
NUM_FEATURES = len(FEATURE_NAMES)


def compute_feature_vector(s1_name, s1_addr, s1_country,
                           sx_name, sx_addr, sx_country,
                           tfidf_name_sim=None, tfidf_addr_sim=None,
                           tfidf_combined_sim=None):
    """Return feature values as a list (same order as FEATURE_NAMES)."""
    feats = compute_features(s1_name, s1_addr, s1_country,
                             sx_name, sx_addr, sx_country,
                             tfidf_name_sim, tfidf_addr_sim,
                             tfidf_combined_sim)
    return [feats[fn] for fn in FEATURE_NAMES]
