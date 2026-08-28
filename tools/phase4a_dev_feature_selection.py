from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(".")
PROC = ROOT / "data/processed/scoring"

CANDIDATE_ID = "candidate_id"
LABEL = "is_attack_containing"
TS = "last_ts_ns"

FEATURE_SETS = {
    "current_6": [
        "burst_rate_per_minute",
        "link_density",
        "max_transactions_per_shared_card",
        "multi_entity_link_fraction",
        "distinct_merchants_per_transaction",
        "distinct_bins_per_transaction",
    ],
    "direction_corrected_6": [
    "burst_rate_per_minute",
    "link_density",
    "max_transactions_per_shared_card",
    "multi_entity_link_fraction",
    "distinct_merchants_per_transaction",
    "distinct_bins_per_transaction",
    ],

    "direction_corrected_5": [
        "burst_rate_per_minute",
        "link_density",
        "max_transactions_per_shared_card",
        "multi_entity_link_fraction",
        "distinct_bins_per_transaction",
    ],

    "direction_corrected_3": [
        "burst_rate_per_minute",
        "max_transactions_per_shared_card",
        "multi_entity_link_fraction",
    ],
    "burst_only": [
        "burst_rate_per_minute",
    ],
    "burst_card": [
        "burst_rate_per_minute",
        "max_transactions_per_shared_card",
    ],
    "burst_multi_entity": [
        "burst_rate_per_minute",
        "multi_entity_link_fraction",
    ],
    "burst_card_multi_entity": [
        "burst_rate_per_minute",
        "max_transactions_per_shared_card",
        "multi_entity_link_fraction",
    ],
}


def rank_percentile_train_reference(train, frame, features, directions):
    """
    Convert each feature to a percentile rank using TRAIN ONLY
    as the reference population.

    directions:
        +1 = larger value is more suspicious
        -1 = smaller value is more suspicious

    This is diagnostic only. No test data is used.
    """
    result = pd.DataFrame(index=frame.index)

    for feature in features:
        train_values = pd.to_numeric(
            train[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        values = pd.to_numeric(
            frame[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        # Missing values receive the least suspicious rank.
        train_values = train_values.dropna()

        if train_values.empty:
            result[feature] = 0.0
            continue

        sorted_values = np.sort(train_values.to_numpy())

        ranks = np.searchsorted(
            sorted_values,
            values.fillna(sorted_values[0]).to_numpy(),
            side="right",
        ) / len(sorted_values)

        ranks = np.asarray(ranks, dtype=float)

        if directions[feature] < 0:
            ranks = 1.0 - ranks

        result[feature] = ranks

    return result


def evaluate(name, y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if y.sum() == 0:
        print(f"{name}: no positives")
        return

    ap = average_precision_score(y, scores)
    base = y.mean()

    order = np.argsort(-scores)

    print(f"\n{name}")
    print(f"  PR-AUC: {ap:.4f}")
    print(f"  Lift:   {ap / base:.2f}x")

    for k in (10, 20, 50):
        kk = min(k, len(y))
        top = y[order[:kk]]

        precision = top.mean()
        recall = top.sum() / y.sum()

        print(
            f"  top-{kk}: "
            f"precision={precision:.2%} "
            f"recall={recall:.2%}"
        )


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

df = pd.read_csv(PROC / "candidate_scoring_features.csv")

df[LABEL] = df[LABEL].astype(bool)

df = (
    df.sort_values([TS, CANDIDATE_ID])
    .reset_index(drop=True)
)

assert len(df) == 4372
assert int(df[LABEL].sum()) == 81

# ---------------------------------------------------------------------
# EXACT PROTOCOL B SPLIT
# ---------------------------------------------------------------------

from conflux.scoring.splits import chronological_candidate_split

train_idx, val_idx, test_idx, meta = chronological_candidate_split(df)

train = df.iloc[train_idx].copy()
val = df.iloc[val_idx].copy()

# IMPORTANT:
# Test is deliberately NOT used anywhere in feature selection.
test = df.iloc[test_idx].copy()

print("=" * 90)
print("PHASE 4A — DEVELOPMENT FEATURE SELECTION")
print("=" * 90)

print(
    f"Development only: train={len(train)}, "
    f"validation={len(val)}"
)

print(
    f"Train positives={int(train[LABEL].sum())}, "
    f"Validation positives={int(val[LABEL].sum())}"
)

print(
    "\nThe chronological TEST set is intentionally excluded "
    "from all selection decisions."
)

# ---------------------------------------------------------------------
# DETERMINE FEATURE DIRECTIONS FROM TRAIN ONLY
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("TRAIN-ONLY FEATURE DIRECTION")
print("=" * 90)

directions = {}

for feature in sorted(
    {
        f
        for features in FEATURE_SETS.values()
        for f in features
    }
):
    pos = train.loc[train[LABEL], feature].astype(float)
    neg = train.loc[~train[LABEL], feature].astype(float)

    pos_median = pos.median()
    neg_median = neg.median()

    direction = 1 if pos_median >= neg_median else -1

    directions[feature] = direction

    direction_text = (
        "HIGHER = MORE SUSPICIOUS"
        if direction > 0
        else "LOWER = MORE SUSPICIOUS"
    )

    print(
        f"{feature:40s} "
        f"positive median={pos_median:.6g} "
        f"negative median={neg_median:.6g} "
        f"-> {direction_text}"
    )

    # Explicit direction variants for development diagnostics only.
# These do NOT modify production configuration.
DIRECTION_OVERRIDES = {
    "direction_corrected_6": {
        "burst_rate_per_minute": 1,
        "link_density": -1,
        "max_transactions_per_shared_card": 1,
        "multi_entity_link_fraction": 1,
        "distinct_merchants_per_transaction": 1,
        "distinct_bins_per_transaction": -1,
    },
    "direction_corrected_5": {
        "burst_rate_per_minute": 1,
        "link_density": -1,
        "max_transactions_per_shared_card": 1,
        "multi_entity_link_fraction": 1,
        "distinct_bins_per_transaction": -1,
    },
    "direction_corrected_3": {
        "burst_rate_per_minute": 1,
        "max_transactions_per_shared_card": 1,
        "multi_entity_link_fraction": 1,
    },
}

# ---------------------------------------------------------------------
# EVALUATE FEATURE SETS
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("FEATURE-SET COMPARISON")
print("=" * 90)

results = []

for name, features in FEATURE_SETS.items():

    score_directions = DIRECTION_OVERRIDES.get(
        name,
        {feature: directions[feature] for feature in features},
    )

    train_ranks = rank_percentile_train_reference(
        train,
        train,
        features,
        score_directions,
    )

    val_ranks = rank_percentile_train_reference(
        train,
        val,
        features,
        score_directions,
    )

    train_score = train_ranks.mean(axis=1)
    val_score = val_ranks.mean(axis=1)

    train_y = train[LABEL].astype(int).to_numpy()
    val_y = val[LABEL].astype(int).to_numpy()

    train_ap = average_precision_score(
        train_y,
        train_score,
    )

    val_ap = (
        average_precision_score(val_y, val_score)
        if val_y.sum()
        else np.nan
    )

    train_base = train_y.mean()
    val_base = val_y.mean()

    results.append(
        {
            "feature_set": name,
            "train_pr_auc": train_ap,
            "train_lift": train_ap / train_base,
            "validation_pr_auc": val_ap,
            "validation_lift": (
                val_ap / val_base
                if np.isfinite(val_ap)
                else np.nan
            ),
        }
    )

    print(f"\n{name}")
    print(f"  features: {features}")
    print(
        f"  TRAIN: "
        f"PR-AUC={train_ap:.4f} "
        f"lift={train_ap / train_base:.2f}x"
    )
    print(
        f"  VALIDATION: "
        f"PR-AUC={val_ap:.4f} "
        f"lift={val_ap / val_base:.2f}x"
    )

    if val_y.sum():
        order = np.argsort(-val_score)

        for k in (10, 20, 50):
            kk = min(k, len(val_y))
            top = val_y[order[:kk]]

            precision = top.mean()
            recall = top.sum() / val_y.sum()

            print(
                f"  validation top-{kk}: "
                f"precision={precision:.2%} "
                f"recall={recall:.2%}"
            )

# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

result_df = pd.DataFrame(results)

print(
    result_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 90)
print("INTERPRETATION RULE")
print("=" * 90)

print(
    """
This is a DEVELOPMENT experiment only.

Do not use the chronological test set to choose weights,
thresholds, or feature sets.

Look for a feature set that:
1. performs consistently on TRAIN and VALIDATION,
2. does not depend on a single lucky validation result,
3. does not require harmful features to remain in the score,
4. is simpler and more interpretable than the current six-feature score.

The chronological TEST set remains untouched and will be evaluated
only after the development decision is frozen.
"""
)