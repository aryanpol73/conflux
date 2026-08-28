"""CONFLUX Phase 3C runner -- evaluate existing campaign candidates.

Module: conflux.evaluation.run_campaign_evaluation
Invoke: py -3.14 -m conflux.evaluation.run_campaign_evaluation

Default behaviour reads the candidate assignments ALREADY WRITTEN by
conflux.graph.build_candidates, so the candidates under evaluation are exactly
the ones that were generated and verified in Phase 3B. If no artifact is found
it re-forms them in-process using the unmodified Phase 3B layer and says so.

Writes only to data/processed/evaluation/. Never touches data/raw/.

STOP POINT: no score, no weights, no thresholds, no ML, no frontend.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from conflux.config import PROCESSED_DIR, RAW_DATASET_PATH
except ImportError:  # allow `python src/conflux/evaluation/run_campaign_evaluation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from conflux.config import PROCESSED_DIR, RAW_DATASET_PATH  # type: ignore

from conflux.graph.campaign_detection import (  # noqa: E402
    CandidateConfig,
    form_campaign_candidates,
)
from conflux.graph.config import (  # noqa: E402
    FORBIDDEN_GRAPH_INPUTS,
    GraphConfig,
)
from conflux.graph.temporal_graph import TemporalEntityGraph  # noqa: E402

from conflux.evaluation.campaign_evaluation import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    bin_informativeness,
    bin_only_baseline_grouping,
    candidate_frames_from_candidate_set,
    compare_groupings,
    evaluate_candidates,
    evaluate_grouping,
    load_ground_truth,
    render_campaign_table,
    render_candidate_table,
    render_summary,
)

log = logging.getLogger("conflux.evaluation.run_campaign_evaluation")

CANDIDATE_ARTIFACT_DIR = PROCESSED_DIR / "graph"
DEFAULT_OUT_DIR = PROCESSED_DIR / "evaluation"
FROZEN_PATHS = {Path(RAW_DATASET_PATH).resolve()}


# ----------------------------------------------------------------------
# candidate acquisition -- read what exists, do not invent a second generator
# ----------------------------------------------------------------------
def load_candidates(dataset_path: Path,
                    args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    a_path = (Path(args.assignments) if args.assignments
              else CANDIDATE_ARTIFACT_DIR / "campaign_candidate_assignments.csv")
    e_path = (Path(args.candidates) if args.candidates
              else CANDIDATE_ARTIFACT_DIR / "campaign_candidates.csv")

    if not args.rebuild and a_path.exists():
        assignments = pd.read_csv(
            a_path, dtype={"transaction_id": str, "candidate_id": str})
        evidence = (pd.read_csv(e_path, dtype={"candidate_id": str})
                    if e_path.exists() else pd.DataFrame())
        return assignments, evidence, {
            "candidate_source": "phase_3b_artifacts",
            "assignments_path": str(a_path),
            "candidates_path": str(e_path) if e_path.exists() else None,
            "note": "candidates were NOT re-derived; the stored Phase 3B output is used",
        }

    log.warning("no candidate artifact at %s (or --rebuild given); re-forming "
                "candidates in-process with the unmodified Phase 3B layer", a_path)
    graph = TemporalEntityGraph.from_csv(
        dataset_path, config=GraphConfig(campaign_window_seconds=args.window_seconds))
    candidate_set = form_campaign_candidates(graph, CandidateConfig())
    assignments, evidence = candidate_frames_from_candidate_set(candidate_set)
    return assignments, evidence, {
        "candidate_source": "reformed_in_process",
        "window_seconds": args.window_seconds,
        "note": ("artifact absent; Phase 3B layer was invoked unchanged. Run "
                 "conflux.graph.build_candidates first if you need the stored run."),
    }


# ----------------------------------------------------------------------
# leakage safety
# ----------------------------------------------------------------------
def leakage_checks(assignments: pd.DataFrame, evidence: pd.DataFrame,
                   dataset_path: Path, *, run_invariance: bool,
                   window_seconds: float) -> dict[str, Any]:
    found = {
        "assignments": [c for c in FORBIDDEN_GRAPH_INPUTS if c in assignments.columns],
        "candidate_evidence": ([c for c in FORBIDDEN_GRAPH_INPUTS
                                if c in evidence.columns] if len(evidence) else []),
    }
    out: dict[str, Any] = {
        "forbidden_columns": list(FORBIDDEN_GRAPH_INPUTS),
        "forbidden_columns_found_in_candidate_outputs": found,
        "candidate_outputs_free_of_ground_truth": not any(found.values()),
        "ordering_note": ("ground truth is read only after the assignments above "
                          "already exist; evaluation has no write path back into "
                          "graph construction or candidate formation"),
    }

    try:
        TemporalEntityGraph.from_frame(pd.DataFrame({
            "transaction_id": ["a"],
            "timestamp": ["2026-08-26 00:00:01.000000"],
            "card_fingerprint": ["c"],
            "bin": ["400000"],
            "device_fingerprint": ["d"],
            "ip_signature": ["i"],
            "merchant_id": ["M0001"],
            "label": ["1"],
            "campaign_id": ["X"],
        }))
        graph_refuses = False
    except Exception as exc:  # GraphIntegrityError
        graph_refuses = True
        out["graph_rejection_message"] = str(exc)
    out["graph_construction_rejects_ground_truth_columns"] = graph_refuses

    out["assignment_invariance_when_ground_truth_removed"] = (
        _invariance_check(assignments, dataset_path, window_seconds)
        if run_invariance else "SKIPPED")
    return out


def _invariance_check(assignments: pd.DataFrame, dataset_path: Path,
                      window_seconds: float) -> dict[str, Any]:
    """Physically delete label/campaign_id from a copy of the dataset, rebuild
    candidates, and require identical assignments."""
    raw = pd.read_csv(dataset_path, dtype=str, keep_default_na=False,
                      na_values=[], low_memory=False)
    removed = [c for c in FORBIDDEN_GRAPH_INPUTS if c in raw.columns]
    stripped = raw.drop(columns=removed)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "dataset_no_ground_truth.csv"
        stripped.to_csv(p, index=False)
        graph = TemporalEntityGraph.from_csv(
            p, config=GraphConfig(campaign_window_seconds=window_seconds))
        candidate_set = form_campaign_candidates(graph, CandidateConfig())

    a = (assignments[["transaction_id", "candidate_id"]].astype(str)
         .sort_values("transaction_id", kind="mergesort").reset_index(drop=True))
    b = (candidate_set.assignments[["transaction_id", "candidate_id"]].astype(str)
         .sort_values("transaction_id", kind="mergesort").reset_index(drop=True))

    identical = bool(a.equals(b))
    diff = int((a["candidate_id"] != b["candidate_id"]).sum()) if len(a) == len(b) else -1
    return {
        "ground_truth_columns_removed": removed,
        "rows_compared": int(len(a)),
        "assignments_identical": identical,
        "transactions_with_changed_candidate": diff,
        "verdict": "PASS" if identical else "FAIL",
    }


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="conflux.evaluation.run_campaign_evaluation",
        description="CONFLUX Phase 3C: evaluate campaign candidates vs ground truth.")
    ap.add_argument("--dataset", default=str(RAW_DATASET_PATH))
    ap.add_argument("--assignments", default=None,
                    help="candidate assignments CSV (default: Phase 3B artifact)")
    ap.add_argument("--candidates", default=None,
                    help="candidate evidence CSV (default: Phase 3B artifact)")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-form candidates in-process instead of reading artifacts")
    ap.add_argument("--window-seconds", type=float, default=3600.0)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-bin-baseline", action="store_true")
    ap.add_argument("--skip-invariance", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s %(message)s")

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    if out_dir.resolve() in FROZEN_PATHS:
        raise SystemExit(f"refusing to write to a frozen path: {out_dir}")

    # ---- 1. candidates FIRST, ground truth strictly afterwards -------
    assignments, evidence, provenance = load_candidates(dataset_path, args)
    leak = leakage_checks(assignments, evidence, dataset_path,
                          run_invariance=not args.skip_invariance,
                          window_seconds=args.window_seconds)

    ground_truth = load_ground_truth(dataset_path)          # <-- first GT read

    # ---- 2. evaluation, full evidence and BIN-suppressed evidence ----
    full = evaluate_candidates(assignments, ground_truth, evidence=evidence,
                               include_bin_context=True,
                               name="graph_candidates_full")
    no_bin = evaluate_candidates(assignments, ground_truth, evidence=evidence,
                                 include_bin_context=False,
                                 name="graph_candidates_without_bin_context")
    bin_view_equivalence = {
        "metrics_identical": full.grouping.metrics == no_bin.grouping.metrics,
        "membership_identical": bool(
            full.candidate_table[["candidate_id", "size"]].equals(
                no_bin.candidate_table[["candidate_id", "size"]])),
        "bin_columns_present_full": [c for c in ("distinct_bins", "bin_ids_context")
                                     if c in full.candidate_table.columns],
        "bin_columns_present_no_bin": [c for c in ("distinct_bins", "bin_ids_context")
                                       if c in no_bin.candidate_table.columns],
        "meaning": ("BIN is reporting context. Suppressing it changes the report "
                    "and nothing else; candidate membership is identical."),
    }

    # ---- 3. BIN-only baseline (separate, never a candidate generator) ----
    bin_block: dict[str, Any] = {"enabled": not args.no_bin_baseline}
    if not args.no_bin_baseline:
        txns = pd.read_csv(dataset_path,
                           usecols=["transaction_id", "timestamp", "bin"],
                           dtype=str, keep_default_na=False, na_values=[],
                           low_memory=False)
        base = bin_only_baseline_grouping(txns, window_seconds=args.window_seconds)
        base_eval = evaluate_grouping(base, ground_truth,
                                      group_col="baseline_group_id",
                                      name="bin_only_baseline")
        bin_block.update({
            "baseline_definition": (
                "shared BIN + same 3600s causal chain. Evaluation-only; component-"
                "equivalent to Phase 3B with BIN as the sole connectivity type, "
                "which the graph layer refuses to run. Never written as candidates."),
            "baseline_metrics": base_eval.metrics,
            "comparison": compare_groupings(full.grouping, base_eval),
            "bin_informativeness": bin_informativeness(txns, ground_truth),
        })

    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "phase": "3C_evaluation_only",
        "candidate_provenance": provenance,
        "leakage_safety": leak,
        "ground_truth_consistency": full.ground_truth_consistency,
        "graph_candidates": full.grouping.as_dict(),
        "bin_context_view_equivalence": bin_view_equivalence,
        "bin_comparison": bin_block,
        "scope_note": ("Evaluation only. No campaign risk score, no weights, no "
                       "thresholds, no ML integration, no decisions."),
    }

    print("\n=== CANDIDATE PROVENANCE ===")
    print(json.dumps(provenance, indent=2))
    print("\n=== LEAKAGE SAFETY ===")
    print(json.dumps(leak, indent=2, default=str))
    print("\n=== GRAPH CANDIDATE EVALUATION ===")
    print(render_summary(full.grouping.metrics))
    print(f"=== TOP {args.top_n} MULTI-TRANSACTION CANDIDATES ===")
    print(render_candidate_table(full.candidate_table, top_n=args.top_n))
    print("\n=== PER-CAMPAIGN COVERAGE / FRAGMENTATION ===")
    print(render_campaign_table(full.grouping.campaign_table))
    print("\n=== BIN CONTEXT VIEW EQUIVALENCE ===")
    print(json.dumps(bin_view_equivalence, indent=2, default=str))
    if not args.no_bin_baseline:
        print("\n=== BIN-ONLY BASELINE (separate; not a candidate generator) ===")
        print(json.dumps(bin_block["comparison"], indent=2, default=str))
        print("\n=== BIN INFORMATIVENESS ===")
        print(json.dumps(bin_block["bin_informativeness"], indent=2, default=str))

    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        full.candidate_table.to_csv(out_dir / "candidate_evaluation.csv", index=False)
        full.grouping.campaign_table.to_csv(out_dir / "campaign_evaluation.csv",
                                            index=False)
        (out_dir / "campaign_evaluation_report.json").write_text(
            json.dumps(report, indent=2, default=str))
        print(f"\nwrote evaluation artifacts to {out_dir}")

    inv = leak["assignment_invariance_when_ground_truth_removed"]
    inv_ok = (inv == "SKIPPED") or (isinstance(inv, dict) and inv["verdict"] == "PASS")
    ok = (leak["candidate_outputs_free_of_ground_truth"]
          and leak["graph_construction_rejects_ground_truth_columns"]
          and full.grouping.alignment.aligned
          and bin_view_equivalence["metrics_identical"]
          and inv_ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
