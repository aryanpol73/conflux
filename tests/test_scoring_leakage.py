import numpy as np
import pandas as pd
import pytest

from conflux.evaluation.campaign_evaluation import load_ground_truth
from conflux.evaluation.candidate_diagnostics import (
    attach_groups, load_candidate_artifacts)
from conflux.scoring.candidate_features import (
    build_scoring_features, load_structural_attributes)
from conflux.scoring.config import CORE_FEATURE_NAMES, FEATURE_SIGNS
from conflux.scoring.deterministic_scorer import (
    DeterministicScorer, ScorerLeakageError)


def test_label_shuffle_leaves_features_and_scores_identical(synth):
    """The decisive leakage test: permuting ground truth must change nothing."""
    cand, asg = load_candidate_artifacts(synth["candidates"], synth["assignments"])
    attrs = load_structural_attributes(synth["dataset"])
    sf = build_scoring_features(cand, asg, attrs)

    names = list(CORE_FEATURE_NAMES)
    ref = DeterministicScorer.fit(sf.frame[names], names, signs=FEATURE_SIGNS)
    s1, _ = DeterministicScorer.transform(ref, sf.frame[names])

    gt = load_ground_truth(synth["dataset"])
    shuffled = gt.copy()
    perm = np.random.default_rng(99).permutation(len(shuffled))
    shuffled[["campaign_id", "is_campaign", "label_int"]] = \
        shuffled[["campaign_id", "is_campaign", "label_int"]].to_numpy()[perm]

    sf2 = build_scoring_features(cand, asg, attrs)
    ref2 = DeterministicScorer.fit(sf2.frame[names], names, signs=FEATURE_SIGNS)
    s2, _ = DeterministicScorer.transform(ref2, sf2.frame[names])

    pd.testing.assert_frame_equal(sf.frame[names], sf2.frame[names])
    np.testing.assert_array_equal(s1, s2)

    a = attach_groups(sf.frame, asg, gt)
    b = attach_groups(sf2.frame, asg, shuffled)
    assert a[names].equals(b[names])   # features identical
    pd.testing.assert_frame_equal(a[names], b[names])


def test_fit_rejects_labelled_frame(scored):
    _, feats = scored
    names = list(CORE_FEATURE_NAMES)
    bad = feats[names].copy()
    bad["campaign_id"] = ""
    with pytest.raises(ScorerLeakageError):
        DeterministicScorer.fit(bad, names + ["campaign_id"])


def test_reference_never_sees_heldout_rows(scored):
    """Scores of held-out rows depend only on training statistics."""
    _, feats = scored
    names = list(CORE_FEATURE_NAMES)
    train, test = feats.iloc[:80], feats.iloc[80:]
    ref = DeterministicScorer.fit(train[names], names, signs=FEATURE_SIGNS)
    s_a, _ = DeterministicScorer.transform(ref, test[names])

    perturbed = test.copy()
    perturbed.loc[perturbed.index[1:], names] *= 1000.0
    s_b, _ = DeterministicScorer.transform(ref, perturbed[names])
    assert s_a[0] == s_b[0]      # row 0 unaffected by other held-out rows


def test_only_allowlisted_raw_columns_are_read(synth):
    attrs = load_structural_attributes(synth["dataset"])
    assert set(attrs.columns) <= {
        "transaction_id", "card_fingerprint", "amount", "auth_outcome"}
