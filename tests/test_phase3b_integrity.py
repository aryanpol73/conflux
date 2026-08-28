import hashlib

import pandas as pd

from conflux.evaluation.campaign_evaluation import (
    evaluate_grouping, load_ground_truth)
from conflux.evaluation.candidate_diagnostics import (
    attach_groups, load_candidate_artifacts)
from conflux.scoring.candidate_features import (
    build_scoring_features, load_structural_attributes)


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_phase4_does_not_mutate_phase3b_artifacts(synth):
    before = {k: _sha(synth[k]) for k in ("candidates", "assignments", "dataset")}
    cand, asg = load_candidate_artifacts(synth["candidates"], synth["assignments"])
    attrs = load_structural_attributes(synth["dataset"])
    sf = build_scoring_features(cand, asg, attrs)
    gt = load_ground_truth(synth["dataset"])
    attach_groups(sf.frame, asg, gt)
    assert {k: _sha(synth[k]) for k in before} == before


def test_candidate_membership_is_unchanged(synth, scored):
    _, asg = load_candidate_artifacts(synth["candidates"], synth["assignments"])
    sf, feats = scored
    multi = asg.groupby("candidate_id").size()
    multi = multi[multi >= 2]
    assert set(feats["candidate_id"]) == set(multi.index)
    pd.testing.assert_series_equal(
        feats.set_index("candidate_id")["size"].astype(int).sort_index(),
        multi.astype(int).sort_index(), check_names=False)


def test_no_cross_campaign_merging_introduced(synth):
    _, asg = load_candidate_artifacts(synth["candidates"], synth["assignments"])
    gt = load_ground_truth(synth["dataset"])
    ev = evaluate_grouping(asg, gt, group_col="candidate_id", strict=True)
    assert ev.metrics["D_merging"]["merged_groups"] == 0
