import numpy as np
import pandas as pd

from conflux.scoring.config import CORE_FEATURE_NAMES, FEATURE_SIGNS
from conflux.scoring.deterministic_scorer import DeterministicScorer, tune_weights
from conflux.scoring.evaluation import average_precision


def _ref(feats):
    names = list(CORE_FEATURE_NAMES)
    return names, DeterministicScorer.fit(feats[names], names, signs=FEATURE_SIGNS)


def test_scores_are_bounded(scored):
    _, feats = scored
    names, ref = _ref(feats)
    s, _ = DeterministicScorer.transform(ref, feats[names])
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_reruns_are_bit_identical(scored):
    _, feats = scored
    names, r1 = _ref(feats)
    _, r2 = _ref(feats)
    s1, c1 = DeterministicScorer.transform(r1, feats[names])
    s2, c2 = DeterministicScorer.transform(r2, feats[names])
    np.testing.assert_array_equal(s1, s2)
    pd.testing.assert_frame_equal(c1, c2)


def test_monotone_in_each_feature(scored):
    _, feats = scored
    names, ref = _ref(feats)
    base = feats[names].iloc[[0]].copy()
    s0, _ = DeterministicScorer.transform(ref, base)
    for n in names:
        up = base.copy()
        up[n] = feats[n].max() * 10
        s1, _ = DeterministicScorer.transform(ref, up)
        assert s1[0] >= s0[0] - 1e-12, f"{n} not monotone"


def test_contributions_sum_to_score(scored):
    _, feats = scored
    names, ref = _ref(feats)
    s, c = DeterministicScorer.transform(ref, feats[names])
    np.testing.assert_allclose(c.sum(axis=1).to_numpy(), s, atol=1e-12)


def test_tuner_is_deterministic(scored):
    _, feats = scored
    names = list(CORE_FEATURE_NAMES)
    y = feats["is_attack_containing"].to_numpy(dtype=int)
    w1 = tune_weights(feats[names], y, names, objective=average_precision,
                      signs=FEATURE_SIGNS)
    w2 = tune_weights(feats[names], y, names, objective=average_precision,
                      signs=FEATURE_SIGNS)
    assert w1 == w2
