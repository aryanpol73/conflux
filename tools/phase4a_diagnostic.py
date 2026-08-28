from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(".")
PROC = ROOT / "data/processed/scoring"

FEATURES = [
    "burst_rate_per_minute",
    "link_density",
    "max_transactions_per_shared_card",
    "multi_entity_link_fraction",
    "distinct_merchants_per_transaction",
    "distinct_bins_per_transaction",
]

CAND = "candidate_id"
LABEL = "is_attack_containing"
TS = "last_ts_ns"

df = pd.read_csv(PROC / "candidate_scoring_features.csv")

df[LABEL] = df[LABEL].astype(bool)
df = df.sort_values([TS, CAND]).reset_index(drop=True)

assert len(df) == 4372
assert int(df[LABEL].sum()) == 81

from conflux.scoring.splits import chronological_candidate_split

tr, va, te, meta = chronological_candidate_split(df)

test = df.iloc[te].copy()

print("=" * 90)
print("PHASE 4A — INDIVIDUAL FEATURE DIAGNOSTIC")
print("=" * 90)
print(f"Test candidates: {len(test)}")
print(f"Test positives:  {int(test[LABEL].sum())}")
print(f"Test base rate:  {test[LABEL].mean():.4%}")
print()

def evaluate(name, scores):
    y = test[LABEL].to_numpy()
    s = np.asarray(scores, dtype=float)

    ap = average_precision_score(y, s)
    roc = roc_auc_score(y, s)
    base = y.mean()

    order = np.argsort(-s)

    print(f"{name}")
    print(f"  PR-AUC: {ap:.4f}")
    print(f"  Lift:   {ap / base:.2f}x")
    print(f"  ROC-AUC:{roc:.4f}")

    for k in (10, 20, 50):
        kk = min(k, len(y))
        top = y[order[:kk]]
        precision = top.mean()
        recall = top.sum() / y.sum() if y.sum() else float("nan")
        print(
            f"  @top-{kk:<2d}: precision={precision:.2%} "
            f"recall={recall:.2%}"
        )

    pos = s[y]
    neg = s[~y]

    print(
        f"  POS: median={np.median(pos):.6g} "
        f"p90={np.percentile(pos,90):.6g} "
        f"p95={np.percentile(pos,95):.6g} "
        f"max={np.max(pos):.6g}"
    )

    print(
        f"  NEG: median={np.median(neg):.6g} "
        f"p90={np.percentile(neg,90):.6g} "
        f"p95={np.percentile(neg,95):.6g} "
        f"max={np.max(neg):.6g}"
    )
    print()

print("INDIVIDUAL FEATURES")
print("-" * 90)

for f in FEATURES:
    if f not in test.columns:
        print(f"{f}: MISSING")
        continue

    evaluate(f, test[f].to_numpy())

print("=" * 90)
print("SIMPLE COMBINATIONS")
print("=" * 90)

matrix = test[FEATURES].copy()

# Rank-percentile each feature independently.
# This is diagnostic only; nothing is fitted or selected.
ranks = matrix.rank(method="average", pct=True)

evaluate(
    "Uniform percentile-rank combination",
    ranks.mean(axis=1).to_numpy()
)

# Equal-weight raw standardized features.
z = matrix.copy()

for f in FEATURES:
    mean = df[f].iloc[tr].mean()
    std = df[f].iloc[tr].std()

    if std == 0 or not np.isfinite(std):
        z[f] = 0.0
    else:
        z[f] = (z[f] - mean) / std

evaluate(
    "Uniform train-standardized combination",
    z.mean(axis=1).to_numpy()
)

print("=" * 90)
print("INTERPRETATION")
print("=" * 90)

print("""
This is DIAGNOSTIC ONLY.

Do not tune weights or thresholds from the chronological test set.

Interpretation:

1. If one feature has strong test PR-AUC while the existing combined
   scorer is weak, the scoring formulation is likely destroying useful
   signal.

2. If all individual features are weak, the Phase 4A feature set is
   insufficient for chronological generalization.

3. If individual features are useful but the uniform combinations are
   weak, investigate feature direction/normalization/combination logic.

4. If rank_by_size remains substantially stronger than the individual
   behavioral features, candidate size may be carrying most of the
   available robust signal.

5. Do NOT use these test results to select new weights. They are for
   diagnosis only.
""")