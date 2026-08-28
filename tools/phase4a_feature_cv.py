from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(".")
PROC = ROOT / "data/processed/scoring"

ID = "candidate_id"
LABEL = "is_attack_containing"
CAMPAIGN = "dominant_campaign_id"
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


def percentile_scores(train, frame, features, directions):
    out = pd.DataFrame(index=frame.index)

    for f in features:
        tr = pd.to_numeric(train[f], errors="coerce")
        x = pd.to_numeric(frame[f], errors="coerce")

        tr = tr.replace([np.inf, -np.inf], np.nan).dropna()

        if len(tr) == 0:
            out[f] = 0.0
            continue

        ref = np.sort(tr.to_numpy())
        x = x.fillna(ref[0]).to_numpy()

        score = np.searchsorted(ref, x, side="right") / len(ref)

        if directions[f] < 0:
            score = 1.0 - score

        out[f] = score

    return out


def evaluate(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    if y.sum() == 0:
        return np.nan

    return average_precision_score(y, score)


df = pd.read_csv(PROC / "candidate_scoring_features.csv")
df[LABEL] = df[LABEL].astype(bool)

df = df.sort_values([TS, ID]).reset_index(drop=True)

assert len(df) == 4372
assert int(df[LABEL].sum()) == 81

# ---------------------------------------------------------------------
# CAMPAIGN GROUPS
# ---------------------------------------------------------------------

campaigns = (
    df.loc[df[LABEL], CAMPAIGN]
    .dropna()
    .astype(str)
    .unique()
)

campaigns = np.sort(campaigns)

print("=" * 90)
print("PHASE 4A — CAMPAIGN-GROUPED FEATURE COMPARISON")
print("=" * 90)

print(f"Attack campaigns: {len(campaigns)}")
print(f"Candidates: {len(df)}")
print(f"Attack-containing candidates: {int(df[LABEL].sum())}")

# Five deterministic campaign folds.
n_folds = 5
folds = [campaigns[i::n_folds] for i in range(n_folds)]

all_results = []

for fold_no in range(n_folds):

    heldout_campaigns = set(folds[fold_no].tolist())

        # Positive candidates are grouped by attack campaign.
    # Negative candidates have no attack campaign, so they are
    # distributed deterministically across folds.
    positive_mask = (
        df[LABEL]
        & df[CAMPAIGN].astype(str).isin(heldout_campaigns)
    )

    negative_mask = ~df[LABEL]

    negative_indices = df.index[negative_mask].to_numpy()

    # Deterministic candidate-level assignment for negatives.
    # This keeps negatives representative in every fold without
    # splitting any attack campaign.
    negative_fold = (
        np.arange(len(negative_indices)) % n_folds
    )

    test_negative_indices = negative_indices[
        negative_fold == fold_no
    ]

    test_mask = positive_mask.copy()
    test_mask.loc[test_negative_indices] = True

    # Remaining candidates form the development set.
    train_mask = ~test_mask

    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()
    assert int(test[LABEL].sum()) > 0
    assert int((~test[LABEL]).sum()) > 0
    assert len(set(train.index) & set(test.index)) == 0

    print("\n" + "-" * 90)
    print(f"FOLD {fold_no + 1}")
    print("-" * 90)

    print(
        f"held-out campaigns={len(heldout_campaigns)} "
        f"train={len(train)} test={len(test)} "
        f"test positives={int(test[LABEL].sum())} "
        f"test negatives={int((~test[LABEL]).sum())}"
    )

    # -------------------------------------------------------------
    # TRAIN-ONLY DIRECTIONS
    # -------------------------------------------------------------

    directions = {}

    for feature in {
        f for fs in FEATURE_SETS.values() for f in fs
    }:
        pos = pd.to_numeric(
            train.loc[train[LABEL], feature],
            errors="coerce",
        )

        neg = pd.to_numeric(
            train.loc[~train[LABEL], feature],
            errors="coerce",
        )

        directions[feature] = (
            1 if pos.median() >= neg.median() else -1
        )

    # -------------------------------------------------------------
    # FEATURE SETS
    # -------------------------------------------------------------

    for name, features in FEATURE_SETS.items():

        train_rank = percentile_scores(
            train,
            train,
            features,
            directions,
        )

        test_rank = percentile_scores(
            train,
            test,
            features,
            directions,
        )

        train_score = train_rank.mean(axis=1)
        test_score = test_rank.mean(axis=1)

        train_y = train[LABEL].astype(int).to_numpy()
        test_y = test[LABEL].astype(int).to_numpy()

        train_ap = evaluate(train_y, train_score)
        test_ap = evaluate(test_y, test_score)

        print(
            f"{name:28s} "
            f"train={train_ap:.4f} "
            f"heldout={test_ap:.4f}"
        )

        all_results.append(
            {
                "fold": fold_no + 1,
                "feature_set": name,
                "train_pr_auc": train_ap,
                "heldout_pr_auc": test_ap,
            }
        )

# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

results = pd.DataFrame(all_results)

print("\n" + "=" * 90)
print("CAMPAIGN-GROUPED SUMMARY")
print("=" * 90)

summary = (
    results
    .groupby("feature_set")
    .agg(
        mean_pr_auc=("heldout_pr_auc", "mean"),
        median_pr_auc=("heldout_pr_auc", "median"),
        std_pr_auc=("heldout_pr_auc", "std"),
        min_pr_auc=("heldout_pr_auc", "min"),
        max_pr_auc=("heldout_pr_auc", "max"),
    )
    .sort_values("mean_pr_auc", ascending=False)
)

print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 90)
print("PER-FOLD RESULTS")
print("=" * 90)

print(
    results.pivot(
        index="fold",
        columns="feature_set",
        values="heldout_pr_auc",
    ).to_string(float_format=lambda x: f"{x:.4f}")
)

print("\n" + "=" * 90)
print("IMPORTANT")
print("=" * 90)

print(
    """
This experiment is diagnostic/development only.

The chronological Protocol-B test window is NOT used here.

No weights or thresholds are tuned.

A feature set should only be considered promising if its advantage
is reasonably consistent across campaign-held-out folds rather than
coming from one unusually favorable fold.

Do not modify the production scorer based on one fold or one mean.
"""
)