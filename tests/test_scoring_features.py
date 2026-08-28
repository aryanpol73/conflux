import numpy as np
import pytest

from conflux.evaluation.campaign_evaluation import GROUND_TRUTH_COLUMNS
from conflux.scoring.config import CORE_FEATURE_NAMES, EXCLUDED_FROM_SCORING
from conflux.scoring.candidate_features import prune_correlated


def test_no_ground_truth_in_feature_frame(scored):
    sf, _ = scored
    assert not [c for c in GROUND_TRUTH_COLUMNS if c in sf.frame.columns]


def test_core_features_are_finite(scored):
    sf, _ = scored
    for c in CORE_FEATURE_NAMES:
        v = sf.frame[c].to_numpy(dtype=float)
        assert np.isfinite(v).all(), f"{c} has NaN/Inf"


def test_raw_size_excluded_from_core(scored):
    for name in EXCLUDED_FROM_SCORING:
        assert name not in CORE_FEATURE_NAMES


def test_shared_card_feature_matches_hand_count(scored):
    """Attack candidates reuse CARD-A{c} exactly ceil(2/3 * size) times."""
    sf, feats = scored
    row = feats.loc[feats["is_attack_containing"]].iloc[0]
    size = int(row["size"])
    expected = sum(1 for i in range(size) if i % 3 in (0, 1))
    assert row["max_transactions_per_shared_card"] == pytest.approx(expected)


def test_negative_candidates_have_no_card_reuse(scored):
    _, feats = scored
    neg = feats.loc[~feats["is_attack_containing"]]
    assert (neg["max_transactions_per_shared_card"] == 1.0).all()


def test_identical_amount_share_bounds(scored):
    sf, _ = scored
    v = sf.frame["max_identical_amount_share"].to_numpy(dtype=float)
    assert (v > 0).all() and (v <= 1.0).all()


def test_burst_rate_uses_span_floor_not_nan(scored):
    sf, _ = scored
    assert np.isfinite(sf.frame["burst_rate_per_minute"].to_numpy(dtype=float)).all()


def test_auth_features_are_ablation_only(scored):
    sf, _ = scored
    assert not any(a in CORE_FEATURE_NAMES for a in sf.auth_features)


def test_decorrelation_respects_cap(scored):
    _, feats = scored
    kept, _ = prune_correlated(feats[list(CORE_FEATURE_NAMES)],
                               list(CORE_FEATURE_NAMES))
    corr = feats[list(kept)].corr(method="spearman").to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    assert np.nanmax(np.abs(corr)) <= 0.70 + 1e-9
