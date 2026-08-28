import numpy as np

from conflux.scoring.config import CORE_FEATURE_NAMES, FEATURE_SIGNS
from conflux.scoring.deterministic_scorer import DeterministicScorer
from conflux.scoring.evaluation import (
    apply_preregistered_rule, average_precision, confusion, precision_at_k,
    recall_exchange_table, select_threshold)
from conflux.scoring.splits import campaign_grouped_folds


def test_threshold_uses_training_rows_only(scored):
    _, feats = scored
    names = list(CORE_FEATURE_NAMES)
    y = feats["is_attack_containing"].to_numpy(dtype=int)
    fold = campaign_grouped_folds(feats, n_folds=3, seeds=(3,))[0]
    tr, te = fold.train_idx, fold.test_idx

    ref = DeterministicScorer.fit(feats.iloc[tr][names], names, signs=FEATURE_SIGNS)
    s_tr, _ = DeterministicScorer.transform(ref, feats.iloc[tr][names])
    s_te, _ = DeterministicScorer.transform(ref, feats.iloc[te][names])

    sel = select_threshold(y[tr], s_tr)
    assert sel["selected_on"] == "training rows only"
    best_test = max(confusion(y[te], s_te, t)["f1"] for t in np.unique(s_te))
    assert confusion(y[te], s_te, sel["threshold"])["f1"] <= best_test + 1e-9


def test_exchange_table_is_monotone_in_candidates_kept(scored):
    _, feats = scored
    names = list(CORE_FEATURE_NAMES)
    ref = DeterministicScorer.fit(feats[names], names, signs=FEATURE_SIGNS)
    s, _ = DeterministicScorer.transform(ref, feats[names])
    t = recall_exchange_table(feats, s)
    assert t["candidates_kept"].is_monotonic_decreasing


def test_average_precision_edge_cases():
    assert average_precision(np.array([1, 1]), np.array([0.9, 0.8])) == 1.0
    assert np.isnan(average_precision(np.array([0, 0]), np.array([0.1, 0.2])))


def test_precision_at_k_bounds(scored):
    _, feats = scored
    y = feats["is_attack_containing"].to_numpy(dtype=int)
    s = np.linspace(0, 1, len(y))
    r = precision_at_k(y, s, 10**6)
    assert r["k"] == len(y)


def test_preregistered_rule_keeps_uniform_on_small_gain():
    d = apply_preregistered_rule([0.50] * 5, [0.51] * 5)
    assert d["decision"] == "keep_uniform_weights"
