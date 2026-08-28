"""CONFLUX Phase 3B runner -- build and verify campaign candidates.

Builds the temporal entity graph via the existing loader, forms candidates, runs
the eight required checks against the real dataset, prints real numbers.
Writes only to the processed output directory. Never touches data/raw/.

STOP POINT: no campaign scoring, no ML integration, no campaign-level metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from conflux.config import PROCESSED_DIR, RAW_DATASET_PATH
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from conflux.config import PROCESSED_DIR, RAW_DATASET_PATH  # type: ignore

from conflux.graph.config import (  # noqa: E402
    ENTITY_COLUMNS, FORBIDDEN_GRAPH_INPUTS, ID_COL, TS_COL, GraphConfig,
)
from conflux.graph.temporal_graph import (  # noqa: E402
    BinConnectivityError, TemporalEntityGraph,
)
from conflux.graph.campaign_detection import (  # noqa: E402
    BLOCKED_CANDIDATE_LINK_TYPES, CandidateConfig, CandidateSet,
    ContextEntityConnectivityError, form_campaign_candidates,
)

log = logging.getLogger("conflux.graph.build_candidates")
DEFAULT_OUT_DIR = PROCESSED_DIR / "graph"
FROZEN_PATHS = {Path(RAW_DATASET_PATH).resolve()}


@dataclass
class CheckResult:
    name: str
    verdict: str
    evidence: dict[str, Any] = field(default_factory=dict)


# 1 -------------------------------------------------------------------
def check_no_future(cs: CandidateSet, graph: TemporalEntityGraph) -> CheckResult:
    lk = cs.links
    has = len(lk) > 0

    # real member-vs-decision-time comparison (v1 compared a field to itself)
    dec = pd.DataFrame({"candidate_id": [c.candidate_id for c in cs.candidates],
                        "decision_ts_ns": [c.decision_ts_ns for c in cs.candidates]})
    merged = cs.assignments.merge(dec, on="candidate_id", how="left")

    ev = {
        "links_checked": int(len(lk)),
        "links_with_prior_after_anchor_wallclock":
            int((lk["prior_ts_ns"] > lk["anchor_ts_ns"]).sum()) if has else 0,
        "links_with_prior_at_or_after_anchor_position":
            int((lk["prior_pos"] >= lk["anchor_pos"]).sum()) if has else 0,
        "links_with_positive_delta":
            int((lk["delta_seconds"] > 0).sum()) if has else 0,
        "members_after_candidate_decision_time":
            int((merged["ts_ns"] > merged["decision_ts_ns"]).sum()),
        "members_missing_decision_time": int(merged["decision_ts_ns"].isna().sum()),
    }

    step = max(1, graph.n_transactions // 500)
    probe = graph.transactions[ID_COL].to_numpy()[::step]
    bad = 0
    for tid in probe:
        cc = cs.causal_candidate(tid)
        for l in cc["links"]:
            if l["prior_ts_ns"] > cc["anchor_ts_ns"] or l["prior_pos"] >= l["anchor_pos"]:
                bad += 1
    ev["anchor_probe_size"] = int(len(probe))
    ev["anchor_probe_future_violations"] = bad

    ok = (ev["links_with_prior_after_anchor_wallclock"] == 0
          and ev["links_with_prior_at_or_after_anchor_position"] == 0
          and ev["links_with_positive_delta"] == 0
          and ev["members_after_candidate_decision_time"] == 0
          and ev["members_missing_decision_time"] == 0
          and bad == 0)
    return CheckResult("1_no_future_transaction_in_candidate", "PASS" if ok else "FAIL", ev)


# 2 / 3 ---------------------------------------------------------------
def check_context_entities_cannot_connect(cs: CandidateSet,
                                          graph: TemporalEntityGraph) -> CheckResult:
    present = sorted(et for et in cs.config.connectivity_entity_types
                     if len(cs.links) and bool(cs.links[f"shares_{et}"].any()))
    ev: dict[str, Any] = {
        "blocked_link_entity_types": list(BLOCKED_CANDIDATE_LINK_TYPES),
        "link_types_actually_present": present,
        "blocked_types_found_in_link_table":
            sorted(set(present) & set(BLOCKED_CANDIDATE_LINK_TYPES)),
        "blocked_evidence_columns_present": [
            c for b in BLOCKED_CANDIDATE_LINK_TYPES
            for c in (f"shares_{b}", f"{b}_entity_id") if c in cs.links.columns],
    }

    for et, exc in (("bin", BinConnectivityError),
                    ("merchant", ContextEntityConnectivityError)):
        try:
            form_campaign_candidates(graph, CandidateConfig(
                connectivity_entity_types=(et,)))
            ev[f"config_rejects_{et}"] = False
        except exc as e:
            ev[f"config_rejects_{et}"] = True
            ev[f"{et}_rejection_message"] = str(e)

    try:
        graph.temporal_neighbors(graph.transactions[ID_COL].iloc[0], entity_types=["bin"])
        ev["graph_query_rejects_bin"] = False
    except BinConnectivityError:
        ev["graph_query_rejects_bin"] = True

    # every multi-transaction candidate must be spanned purely by card/device/ip pairs
    disconnected = []
    for c in cs.candidates:
        if c.size < 2:
            continue
        idx = {t: i for i, t in enumerate(c.transaction_ids)}
        adj: dict[int, set[int]] = {i: set() for i in range(c.size)}
        for l in c.links:
            a, b = idx[l["anchor_transaction_id"]], idx[l["prior_transaction_id"]]
            adj[a].add(b)
            adj[b].add(a)
        seen = {0}
        stack = [0]
        while stack:
            for nb in adj[stack.pop()]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(seen) != c.size:
            disconnected.append(c.candidate_id)
    ev["multi_candidates_not_spanned_by_card_device_ip_links"] = disconnected[:20]
    ev["n_multi_candidates_not_spanned"] = len(disconnected)

    ok = (not ev["blocked_types_found_in_link_table"]
          and not ev["blocked_evidence_columns_present"]
          and ev["config_rejects_bin"] and ev["config_rejects_merchant"]
          and ev["graph_query_rejects_bin"] and not disconnected)
    return CheckResult("2_3_bin_and_merchant_cannot_connect",
                       "PASS" if ok else "FAIL", ev)


# 4 -------------------------------------------------------------------
def check_connectivity_works(cs: CandidateSet) -> CheckResult:
    has = len(cs.links) > 0
    counts = {et: int(cs.links[f"shares_{et}"].sum()) if has else 0
              for et in cs.config.connectivity_entity_types}
    per_type_groups = {
        et: int(sum(1 for c in cs.candidates if et in c.link_counts))
        for et in cs.config.connectivity_entity_types
    }
    ev = {
        "connected_pairs_by_entity_type": counts,
        "connected_pairs_total": int(len(cs.links)),
        "multi_entity_pairs":
            int((cs.links["n_link_entity_types"] >= 2).sum()) if has else 0,
        "multi_candidates_using_each_type": per_type_groups,
        "expected_types": list(cs.config.connectivity_entity_types),
    }
    ok = all(v > 0 for v in counts.values())
    return CheckResult("4_card_device_ip_connectivity_works",
                       "PASS" if ok else "FAIL", ev)


# 5 -------------------------------------------------------------------
def check_window_respected(cs: CandidateSet, graph: TemporalEntityGraph) -> CheckResult:
    lk = cs.links
    has = len(lk) > 0
    w = cs.config.window_seconds
    tight = form_campaign_candidates(
        graph, CandidateConfig(window_seconds=max(1.0, w / 60.0)))
    tight_pairs = set(zip(tight.links["anchor_transaction_id"],
                          tight.links["prior_transaction_id"]))
    wide_pairs = set(zip(lk["anchor_transaction_id"], lk["prior_transaction_id"]))
    ev = {
        "window_seconds": w,
        "max_age_seconds": float(lk["age_seconds"].max()) if has else 0.0,
        "min_age_seconds": float(lk["age_seconds"].min()) if has else 0.0,
        "links_outside_window": int((lk["age_seconds"] > w).sum()) if has else 0,
        "links_with_negative_age": int((lk["age_seconds"] < 0).sum()) if has else 0,
        "tight_window_seconds": max(1.0, w / 60.0),
        "tight_window_links": int(len(tight.links)),
        "tight_links_are_subset_of_default": tight_pairs.issubset(wide_pairs),
    }
    ok = (ev["links_outside_window"] == 0 and ev["links_with_negative_age"] == 0
          and ev["tight_links_are_subset_of_default"]
          and ev["tight_window_links"] <= len(lk))
    return CheckResult("5_temporal_window_respected", "PASS" if ok else "FAIL", ev)


# 6 -------------------------------------------------------------------
def check_determinism(cs: CandidateSet, graph: TemporalEntityGraph) -> CheckResult:
    again = form_campaign_candidates(graph, CandidateConfig())
    a1 = cs.assignments[[ID_COL, "candidate_id"]].reset_index(drop=True)
    a2 = again.assignments[[ID_COL, "candidate_id"]].reset_index(drop=True)
    ev = {
        "assignments_identical": bool(a1.equals(a2)),
        "candidate_frames_identical": bool(cs.candidate_frame().equals(again.candidate_frame())),
        "link_tables_identical": bool(cs.links.equals(again.links)),
        "exploded_link_tables_identical":
            bool(cs.explode_links().equals(again.explode_links())),
        "candidate_count_run1": len(cs.candidates),
        "candidate_count_run2": len(again.candidates),
    }
    ok = all([ev["assignments_identical"], ev["candidate_frames_identical"],
              ev["link_tables_identical"], ev["exploded_link_tables_identical"]])
    return CheckResult("6_deterministic_construction", "PASS" if ok else "FAIL", ev)


# 7 -------------------------------------------------------------------
def check_no_ground_truth(cs: CandidateSet, graph: TemporalEntityGraph,
                          dataset_path: Path) -> CheckResult:
    header = pd.read_csv(dataset_path, nrows=0).columns.tolist()
    frames = {"graph_transactions": graph.transactions, "links": cs.links,
              "exploded_links": cs.explode_links(), "assignments": cs.assignments,
              "candidate_frame": cs.candidate_frame()}
    found = {name: [c for c in FORBIDDEN_GRAPH_INPUTS if c in df.columns]
             for name, df in frames.items()}
    blob = json.dumps([c.as_row() for c in cs.candidates[:50]], default=str)
    ev = {
        "forbidden_columns": list(FORBIDDEN_GRAPH_INPUTS),
        "present_in_csv_header": [c for c in FORBIDDEN_GRAPH_INPUTS if c in header],
        "found_in_outputs": found,
        "forbidden_token_in_serialized_candidates":
            any(tok in blob for tok in FORBIDDEN_GRAPH_INPUTS),
        "candidate_construction_inputs":
            [ENTITY_COLUMNS[et] for et in cs.config.connectivity_entity_types]
            + [TS_COL, ID_COL],
    }
    ok = (not any(found.values()) and not ev["forbidden_token_in_serialized_candidates"])
    return CheckResult("7_label_and_campaign_id_absent_from_inputs",
                       "PASS" if ok else "FAIL", ev)


# 8 -------------------------------------------------------------------
def check_accounting(cs: CandidateSet, graph: TemporalEntityGraph) -> CheckResult:
    assigned = cs.assignments[ID_COL]
    all_ids = set(graph.transactions[ID_COL])
    member_ids = [t for c in cs.candidates for t in c.transaction_ids]
    ev = {
        "graph_transactions": graph.n_transactions,
        "assigned_rows": int(len(assigned)),
        "assigned_unique": int(assigned.nunique()),
        "duplicate_assignments": int(len(assigned) - assigned.nunique()),
        "unassigned_transactions": len(all_ids - set(assigned)),
        "sum_of_candidate_sizes": len(member_ids),
        "member_ids_unique": len(set(member_ids)) == len(member_ids),
        "isolated_candidates": int(sum(1 for c in cs.candidates if c.is_isolated)),
        "multi_candidates": int(sum(1 for c in cs.candidates if not c.is_isolated)),
    }
    ok = (ev["assigned_unique"] == graph.n_transactions
          and ev["duplicate_assignments"] == 0
          and ev["unassigned_transactions"] == 0
          and ev["sum_of_candidate_sizes"] == graph.n_transactions
          and ev["member_ids_unique"])
    return CheckResult("8_all_transactions_accounted_for", "PASS" if ok else "FAIL", ev)


# optional post-hoc audit (ground truth read ONLY after formation) ------
def ground_truth_audit(cs: CandidateSet, dataset_path: Path) -> dict[str, Any]:
    gt = pd.read_csv(dataset_path, usecols=[ID_COL, "label", "campaign_id"], dtype=str)
    merged = cs.assignments[[ID_COL, "candidate_id", "candidate_size"]].merge(
        gt, on=ID_COL, how="left")
    attack = merged[merged["label"] == "1"]
    return {
        "note": ("Descriptive accounting only, computed strictly AFTER candidate "
                 "formation. Not a metric, not a score, not used anywhere upstream."),
        "labelled_attack_transactions": int(len(attack)),
        "attack_transactions_in_multi_transaction_candidates":
            int((attack["candidate_size"] > 1).sum()),
        "attack_transactions_isolated": int((attack["candidate_size"] == 1).sum()),
        "distinct_ground_truth_campaigns": int(gt["campaign_id"].nunique()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CONFLUX Phase 3B: campaign candidates.")
    ap.add_argument("--dataset", default=str(RAW_DATASET_PATH))
    ap.add_argument("--window-seconds", type=float, default=3600.0)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--write-exploded-links", action="store_true",
                    help="also write the long one-row-per-entity-type link view")
    ap.add_argument("--ground-truth-audit", action="store_true",
                    help="post-hoc descriptive audit; never influences formation")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s %(message)s")

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    if out_dir.resolve() in FROZEN_PATHS:
        raise SystemExit(f"refusing to write to a frozen path: {out_dir}")

    graph = TemporalEntityGraph.from_csv(
        dataset_path, config=GraphConfig(campaign_window_seconds=args.window_seconds))
    cs = form_campaign_candidates(graph, CandidateConfig())

    results = [
        check_no_future(cs, graph),
        check_context_entities_cannot_connect(cs, graph),
        check_connectivity_works(cs),
        check_window_respected(cs, graph),
        check_determinism(cs, graph),
        check_no_ground_truth(cs, graph, dataset_path),
        check_accounting(cs, graph),
    ]
    report = {
        "status": "OK" if all(r.verdict == "PASS" for r in results) else "FAILED",
        "candidates": cs.summary(),
        "checks": [asdict(r) for r in results],
        "check_summary": {"pass": sum(r.verdict == "PASS" for r in results),
                          "fail": sum(r.verdict == "FAIL" for r in results)},
    }
    if args.ground_truth_audit:
        report["post_hoc_ground_truth_audit"] = ground_truth_audit(cs, dataset_path)

    print("\n=== CANDIDATES ===")
    print(json.dumps(report["candidates"], indent=2, default=str))
    print("\n=== CHECKS ===")
    print(json.dumps(report["checks"], indent=2, default=str))
    print("\n=== RESULT ===")
    print(json.dumps(report["check_summary"] | {"status": report["status"]}, indent=2))

    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        cs.candidate_frame().to_csv(out_dir / "campaign_candidates.csv", index=False)
        cs.links.to_csv(out_dir / "campaign_candidate_links.csv", index=False)
        cs.assignments.to_csv(out_dir / "campaign_candidate_assignments.csv", index=False)
        if args.write_exploded_links:
            cs.explode_links().to_csv(
                out_dir / "campaign_candidate_links_by_entity.csv", index=False)
        (out_dir / "campaign_candidate_report.json").write_text(
            json.dumps(report, indent=2, default=str))
        print(f"\nwrote candidate artifacts to {out_dir}")

    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
