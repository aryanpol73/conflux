"""Phase 3C evaluation tests.

Module under test: conflux.evaluation.campaign_evaluation

Small synthetic data only; the frozen dataset is never read here and the 13
existing candidate tests in tests/test_campaign_candidates.py are untouched.
"""
from __future__ import annotations

import pandas as pd
import pytest

from conflux.evaluation.campaign_evaluation import (
    AlignmentError,
    bin_informativeness,
    bin_only_baseline_grouping,
    evaluate_candidates,
    evaluate_grouping,
    evidence_view,
    normalize_ground_truth,
)


# ---------------------------------------------------------------- helpers
def assignments(pairs):
    df = pd.DataFrame(pairs, columns=["transaction_id", "candidate_id"])
    size = df.groupby("candidate_id")["transaction_id"].transform("size")
    df["candidate_size"] = size
    df["is_isolated"] = size == 1
    return df


def gt(rows):
    df = pd.DataFrame(rows, columns=["transaction_id", "campaign_id"])
    df["label"] = (df["campaign_id"] != "").astype(int).astype(str)
    return normalize_ground_truth(df)


# ------------------------------------------------- 1. perfect detection
def test_perfect_one_candidate_per_campaign():
    a = assignments([("t1", "C1"), ("t2", "C1"), ("t3", "C1"),
                     ("t4", "C2"), ("t5", "C2"), ("t6", "C2")])
    g = gt([("t1", "CMP_A"), ("t2", "CMP_A"), ("t3", "CMP_A"),
            ("t4", "CMP_B"), ("t5", "CMP_B"), ("t6", "CMP_B")])
    m = evaluate_grouping(a, g).metrics

    assert m["A_campaign_coverage"]["ground_truth_campaigns_total"] == 2
    assert m["A_campaign_coverage"]["campaign_coverage_recall"] == 1.0
    assert m["A_campaign_coverage"]["campaigns_majority_captured_by_one_group"] == 2
    assert m["B_purity"]["pure_candidate_count_strict"] == 2
    assert m["B_purity"]["mixed_candidate_count"] == 0
    assert m["B_purity"]["campaign_purity_transaction_weighted"] == 1.0
    assert m["C_fragmentation"]["max_groups_for_one_campaign"] == 1
    assert m["C_fragmentation"]["campaigns_split_across_multiple_groups"] == 0
    assert m["D_merging"]["merged_groups"] == 0
    assert m["E_transaction_coverage"]["campaign_transaction_recall"] == 1.0
    assert m["E_transaction_coverage"]["non_campaign_inclusion_rate"] == 0.0


# ------------------------------------------------ 2. campaign fragmented
def test_one_campaign_split_across_candidates():
    a = assignments([("t1", "C1"), ("t2", "C1"), ("t3", "C2"), ("t4", "C2")])
    g = gt([("t1", "CMP_A"), ("t2", "CMP_A"), ("t3", "CMP_A"), ("t4", "CMP_A")])
    ev = evaluate_grouping(a, g)
    m = ev.metrics

    assert m["A_campaign_coverage"]["campaign_coverage_recall"] == 1.0
    assert m["C_fragmentation"]["campaigns_split_across_multiple_groups"] == 1
    assert m["C_fragmentation"]["max_groups_for_one_campaign"] == 2
    assert m["C_fragmentation"]["mean_groups_per_represented_campaign"] == 2.0
    assert m["D_merging"]["merged_groups"] == 0
    # each fragment is still internally pure
    assert m["B_purity"]["pure_candidate_count_strict"] == 2

    row = ev.campaign_table.iloc[0]
    assert row["n_multi_groups"] == 2
    assert row["best_group_share"] == 0.5
    # a 50/50 split is majority-captured at the >= 0.5 threshold
    assert bool(row["is_majority_captured"]) is True


# --------------------------------------------------- 3. campaigns merged
def test_multiple_campaigns_merged_into_one_candidate():
    a = assignments([("t1", "C1"), ("t2", "C1"), ("t3", "C1"), ("t4", "C1")])
    g = gt([("t1", "CMP_A"), ("t2", "CMP_A"), ("t3", "CMP_B"), ("t4", "CMP_B")])
    ev = evaluate_grouping(a, g)
    m = ev.metrics

    assert m["D_merging"]["merged_groups"] == 1
    assert m["D_merging"]["campaigns_involved_in_a_merge"] == 2
    assert m["D_merging"]["largest_merge_distinct_campaigns"] == 2
    assert m["D_merging"]["merge_rate_over_multi_groups"] == 1.0
    assert m["B_purity"]["mixed_candidate_count"] == 1
    assert m["B_purity"]["pure_candidate_count_strict"] == 0
    # tie on counts -> lexicographically smallest campaign_id wins, deterministically
    assert ev.group_table.iloc[0]["dominant_campaign_id"] == "CMP_A"
    assert m["B_purity"]["campaign_purity_transaction_weighted"] == 0.5


# ------------------------------------------ 4. candidate with normal noise
def test_candidate_containing_normal_traffic():
    a = assignments([("t1", "C1"), ("t2", "C1"), ("t3", "C1"), ("t4", "C1")])
    g = gt([("t1", "CMP_A"), ("t2", "CMP_A"), ("t3", ""), ("t4", "")])
    ev = evaluate_grouping(a, g)
    m = ev.metrics

    assert ev.group_table.iloc[0]["purity_class"] == "pure_campaign_with_normal"
    assert m["B_purity"]["pure_candidate_count_strict"] == 0
    assert m["B_purity"]["pure_candidate_count_allowing_normal_noise"] == 1
    assert m["B_purity"]["campaign_purity_transaction_weighted"] == 1.0
    assert m["E_transaction_coverage"][
        "non_campaign_transactions_in_multi_transaction_groups"] == 2
    assert m["E_transaction_coverage"]["non_campaign_inclusion_rate"] == 1.0
    assert m["E_transaction_coverage"][
        "campaign_share_of_transactions_in_multi_groups"] == 0.5


# ------------------------------------------------- 5. isolated singletons
def test_isolated_transactions_are_not_detections():
    a = assignments([("t1", "C1"), ("t2", "C2"), ("t3", "C3"),
                     ("t4", "C4"), ("t5", "C4")])
    g = gt([("t1", "CMP_A"), ("t2", "CMP_A"), ("t3", ""),
            ("t4", "CMP_B"), ("t5", "CMP_B")])
    m = evaluate_grouping(a, g).metrics

    assert m["grouping"]["groups_singleton"] == 3
    assert m["grouping"]["groups_multi_transaction"] == 1
    # CMP_A exists only as singletons -> not represented
    assert m["A_campaign_coverage"][
        "campaigns_represented_by_a_multi_transaction_group"] == 1
    assert m["A_campaign_coverage"]["ground_truth_campaigns_total"] == 2
    assert m["A_campaign_coverage"]["campaign_coverage_recall"] == 0.5
    assert m["A_campaign_coverage"]["campaigns_entirely_isolated"] == 1
    assert m["C_fragmentation"]["campaign_transactions_left_isolated"] == 2
    assert m["E_transaction_coverage"]["campaign_transaction_recall"] == 0.5


# ------------------------------------- 6. missing / duplicate transaction ids
def test_missing_and_duplicate_ids_are_never_silently_dropped():
    dup = assignments([("t1", "C1"), ("t2", "C1")])
    dup = pd.concat([dup, dup.iloc[[0]]], ignore_index=True)
    g = gt([("t1", "CMP_A"), ("t2", "CMP_A")])
    with pytest.raises(AlignmentError):
        evaluate_grouping(dup, g)

    a = assignments([("t1", "C1"), ("t2", "C1"), ("t9", "C1")])
    with pytest.raises(AlignmentError):
        evaluate_grouping(a, g)

    rep = evaluate_grouping(a, g, strict=False).alignment
    assert rep.aligned is False
    assert rep.assigned_ids_missing_from_ground_truth == ("t9",)

    a2 = assignments([("t1", "C1"), ("t2", "C1")])
    g2 = gt([("t1", "CMP_A"), ("t2", "CMP_A"), ("t3", "CMP_A")])
    rep2 = evaluate_grouping(a2, g2, strict=False).alignment
    assert rep2.ground_truth_ids_missing_from_assignments == ("t3",)


# ---------------------------- 7. ground truth only reaches the evaluator
def test_ground_truth_cannot_enter_candidate_construction():
    from conflux.graph.campaign_detection import CandidateConfig, form_campaign_candidates
    from conflux.graph.temporal_graph import GraphIntegrityError, TemporalEntityGraph

    base = pd.DataFrame({
        "transaction_id": ["a", "b", "c"],
        "timestamp": ["2026-08-26 00:00:01.000000",
                      "2026-08-26 00:00:02.000000",
                      "2026-08-26 05:00:00.000000"],
        "card_fingerprint": ["card1", "card1", "card9"],
        "bin": ["400000", "400000", "400000"],
        "device_fingerprint": ["d1", "d2", "d9"],
        "ip_signature": ["i1", "i2", "i9"],
        "merchant_id": ["M1", "M2", "M3"],
    })
    poisoned = base.assign(label=["1", "1", "0"], campaign_id=["CMP_A", "CMP_A", ""])
    with pytest.raises(GraphIntegrityError):
        TemporalEntityGraph.from_frame(poisoned)

    cs = form_campaign_candidates(TemporalEntityGraph.from_frame(base),
                                  CandidateConfig())
    assert not any(c in cs.assignments.columns for c in ("label", "campaign_id"))
    assert not any(c in cs.links.columns for c in ("label", "campaign_id"))

    # ground truth only now, and only as an evaluation input
    g = gt([("a", "CMP_A"), ("b", "CMP_A"), ("c", "")])
    before = cs.assignments["candidate_id"].tolist()
    evaluate_candidates(cs.assignments, g, evidence=cs.candidate_frame())
    assert cs.assignments["candidate_id"].tolist() == before


# ------------------------------------------------------- 8. determinism
def test_evaluation_is_deterministic_and_order_independent():
    a = assignments([("t1", "C1"), ("t2", "C1"), ("t3", "C2"),
                     ("t4", "C2"), ("t5", "C3")])
    g = gt([("t1", "CMP_A"), ("t2", "CMP_B"), ("t3", "CMP_B"),
            ("t4", "CMP_B"), ("t5", "")])

    m1 = evaluate_grouping(a, g).metrics
    m2 = evaluate_grouping(a, g).metrics
    m3 = evaluate_grouping(
        a.sample(frac=1.0, random_state=7).reset_index(drop=True),
        g.sample(frac=1.0, random_state=3).reset_index(drop=True),
    ).metrics
    assert m1 == m2 == m3

    t1 = evaluate_grouping(a, g).group_table
    t2 = evaluate_grouping(a, g).group_table
    assert t1.equals(t2)


# --------------------------- 9. BIN context cannot change membership
def test_bin_context_does_not_alter_candidate_membership():
    a = assignments([("t1", "C1"), ("t2", "C1"), ("t3", "C2"), ("t4", "C2")])
    g = gt([("t1", "CMP_A"), ("t2", "CMP_A"), ("t3", "CMP_B"), ("t4", "CMP_B")])
    evidence = pd.DataFrame({
        "candidate_id": ["C1", "C2"],
        "distinct_cards": [1, 2],
        "distinct_devices": [2, 1],
        "distinct_ips": [1, 1],
        "distinct_merchants": [2, 2],
        "distinct_bins": [1, 2],
        "bin_ids_context": ["400000", "400000|411111"],
        "time_span_seconds": [10.0, 20.0],
        "link_edge_count": [1, 1],
    })
    with_bin = evaluate_candidates(a, g, evidence=evidence, include_bin_context=True)
    without = evaluate_candidates(a, g, evidence=evidence, include_bin_context=False)

    assert with_bin.grouping.metrics == without.grouping.metrics
    assert with_bin.candidate_table[["candidate_id", "size"]].equals(
        without.candidate_table[["candidate_id", "size"]])
    assert "distinct_bins" in with_bin.candidate_table.columns
    assert "distinct_bins" not in without.candidate_table.columns
    assert "bin_ids_context" not in evidence_view(evidence, include_bin=False).columns


# ------------------------------ 10. BIN baseline respects the window
def test_bin_baseline_respects_the_window_and_never_returns_candidates():
    txns = pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4"],
        "timestamp": ["2026-08-26 00:00:00.000000", "2026-08-26 00:10:00.000000",
                      "2026-08-26 09:00:00.000000", "2026-08-26 00:05:00.000000"],
        "bin": ["400000", "400000", "400000", "411111"],
    })
    base = bin_only_baseline_grouping(txns, window_seconds=3600)
    gid = dict(zip(base["transaction_id"], base["baseline_group_id"]))

    assert gid["t1"] == gid["t2"]           # same BIN, 10 minutes apart
    assert gid["t3"] != gid["t1"]           # same BIN, 9 hours later
    assert gid["t4"] != gid["t1"]           # different BIN
    assert "candidate_id" not in base.columns   # never masquerades as candidates
    assert set(base.columns) == {"transaction_id", "baseline_group_id",
                                 "baseline_group_size"}


# ------------------ 11. baseline goes through the same metric code path
def test_bin_baseline_is_evaluated_with_the_same_metric_code():
    txns = pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3"],
        "timestamp": ["2026-08-26 00:00:00.000000", "2026-08-26 00:01:00.000000",
                      "2026-08-26 00:02:00.000000"],
        "bin": ["400000", "400000", "400000"],
    })
    g = gt([("t1", "CMP_A"), ("t2", "CMP_B"), ("t3", "")])
    base = bin_only_baseline_grouping(txns, window_seconds=3600)
    m = evaluate_grouping(base, g, group_col="baseline_group_id",
                          name="bin_only_baseline").metrics

    assert m["grouping"]["groups_multi_transaction"] == 1
    assert m["D_merging"]["merged_groups"] == 1          # BIN alone merges campaigns
    assert m["D_merging"]["largest_merge_distinct_campaigns"] == 2


# --------------------------------- 12. BIN informativeness both directions
def test_bin_informativeness_reports_spread_both_ways():
    txns = pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4"],
        "bin": ["400000", "400000", "411111", "422222"],
        "timestamp": ["2026-08-26 00:00:00.000000"] * 4,
    })
    g = gt([("t1", "CMP_A"), ("t2", "CMP_B"), ("t3", "CMP_A"), ("t4", "")])
    info = bin_informativeness(txns, g)

    assert info["distinct_bins"] == 3
    assert info["bins_hosting_multiple_campaigns"] == 1     # 400000 hosts A and B
    assert info["campaigns_spanning_multiple_bins"] == 1    # CMP_A spans 2 bins


# ------------------------------------------ 13. non-campaign-only groups
def test_non_campaign_only_groups_are_classified_and_not_counted_as_detections():
    a = assignments([("t1", "C1"), ("t2", "C1")])
    g = gt([("t1", ""), ("t2", "")])
    ev = evaluate_grouping(a, g)

    assert ev.group_table.iloc[0]["purity_class"] == "non_campaign"
    assert ev.metrics["B_purity"]["non_campaign_group_count"] == 1
    assert ev.metrics["A_campaign_coverage"]["ground_truth_campaigns_total"] == 0
    assert ev.metrics["E_transaction_coverage"]["campaign_transaction_recall"] == 0.0


# ------------------------------------- 14. empty campaign_id normalization
def test_empty_campaign_id_variants_are_normalized_consistently():
    df = pd.DataFrame({
        "transaction_id": ["a", "b", "c", "d"],
        "campaign_id": ["", "  ", None, "CMP_A"],
        "label": ["0", "0", "0", "1"],
    })
    n = normalize_ground_truth(df)

    assert n["is_campaign"].tolist() == [False, False, False, True]
    assert n["campaign_id"].tolist() == ["", "", "", "CMP_A"]
