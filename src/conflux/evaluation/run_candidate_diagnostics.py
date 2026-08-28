"""CONFLUX Phase 3C runner -- diagnostic report on candidate groups.

Module: conflux.evaluation.run_candidate_diagnostics
Invoke: py -3.14 -m conflux.evaluation.run_candidate_diagnostics

Compares the attack-containing candidates against the non-campaign candidates
produced by Phase 3B. Reads the stored Phase 3B artifacts; never re-forms them.

STOP POINT: no model, no training, no scoring, no weights, no Phase 3B change.
Writes only to data/processed/evaluation/phase3c_diagnostics/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from conflux.config import PROCESSED_DIR, RAW_DATASET_PATH
except ImportError:  # allow direct file execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from conflux.config import PROCESSED_DIR, RAW_DATASET_PATH  # type: ignore

from conflux.evaluation.campaign_evaluation import (  # noqa: E402
    GROUND_TRUTH_COLUMNS,
    label_campaign_consistency,
    load_ground_truth,
)
from conflux.evaluation.candidate_diagnostics import (  # noqa: E402
    DIAGNOSTIC_SCHEMA_VERSION,
    GROUP_BY_CHOICES,
    attach_groups,
    attack_rate_crosstab,
    build_candidate_features,
    compare_boolean,
    compare_numeric,
    group_summary,
    load_candidate_artifacts,
    load_transaction_attributes,
    redundancy_matrix,
    render_boolean_table,
    render_crosstab,
    render_numeric_table,
    strongest_separations,
)

log = logging.getLogger("conflux.evaluation.run_candidate_diagnostics")

CANDIDATE_ARTIFACT_DIR = PROCESSED_DIR / "graph"
DEFAULT_OUT_DIR = PROCESSED_DIR / "evaluation" / "phase3c_diagnostics"
FROZEN_PATHS = {Path(RAW_DATASET_PATH).resolve()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="conflux.evaluation.run_candidate_diagnostics",
        description=("CONFLUX Phase 3C: structural/behavioural diagnostic comparison "
                     "of attack-containing vs non-campaign candidates. No training."))
    ap.add_argument("--dataset", default=str(RAW_DATASET_PATH))
    ap.add_argument("--candidates",
                    default=str(CANDIDATE_ARTIFACT_DIR / "campaign_candidates.csv"))
    ap.add_argument("--assignments",
                    default=str(CANDIDATE_ARTIFACT_DIR / "campaign_candidate_assignments.csv"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--min-size", type=int, default=2,
                    help="minimum candidate size to include (singletons excluded)")
    ap.add_argument("--group-by", choices=list(GROUP_BY_CHOICES), default="campaign_id",
                    help="how a candidate is called attack-containing")
    ap.add_argument("--expect-attack", type=int, default=None,
                    help="expected attack-containing candidate count, e.g. 81")
    ap.add_argument("--expect-other", type=int, default=None,
                    help="expected non-campaign candidate count, e.g. 4291")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--no-behaviour", action="store_true",
                    help="skip amount/auth_outcome features (structural only)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s %(message)s")

    out_dir = Path(args.out_dir)
    if out_dir.resolve() in FROZEN_PATHS:
        raise SystemExit(f"refusing to write to a frozen path: {out_dir}")

    # ---- 1. Phase 3B artifacts + non-ground-truth attributes ----------
    candidates, assignments = load_candidate_artifacts(args.candidates, args.assignments)
    attributes = None if args.no_behaviour else load_transaction_attributes(args.dataset)

    fs = build_candidate_features(candidates, assignments, attributes,
                                  min_size=args.min_size)

    leakage = {
        "forbidden_columns": list(GROUND_TRUTH_COLUMNS),
        "found_in_candidate_artifacts": [
            c for c in GROUND_TRUTH_COLUMNS
            if c in candidates.columns or c in assignments.columns],
        "found_in_feature_table": [
            c for c in GROUND_TRUTH_COLUMNS if c in fs.frame.columns],
        "feature_construction_inputs": (
            ["campaign_candidates.csv", "campaign_candidate_assignments.csv"]
            + ([] if args.no_behaviour else ["amount", "auth_outcome"])),
        "ordering_note": ("the feature table above is complete before ground truth "
                          "is read; the group flag is added afterwards and is "
                          "excluded from every comparison input"),
    }
    leakage["clean"] = not (leakage["found_in_candidate_artifacts"]
                            or leakage["found_in_feature_table"])

    # ---- 2. ground truth: FIRST read, groups only ---------------------
    ground_truth = load_ground_truth(args.dataset)
    features = attach_groups(fs.frame, assignments, ground_truth,
                             group_by=args.group_by)

    groups = group_summary(features, group_by=args.group_by,
                           expected_attack=args.expect_attack,
                           expected_other=args.expect_other)

    mask_a = features["is_attack_containing"]
    mask_b = ~features["is_attack_containing"]

    # ---- 3. comparisons ------------------------------------------------
    numeric = compare_numeric(features, fs.numeric_features,
                              mask_a=mask_a, mask_b=mask_b)
    boolean = compare_boolean(features, fs.boolean_features,
                              mask_a=mask_a, mask_b=mask_b)

    crosstabs = pd.concat(
        [attack_rate_crosstab(features, "size_bucket"),
         attack_rate_crosstab(features, "span_bucket"),
         attack_rate_crosstab(features, "dominant_link_type"),
         attack_rate_crosstab(features, "purity_class")],
        ignore_index=True)

    top = strongest_separations(numeric, boolean, top_n=args.top_n)
    top_names = [r["feature"] for r in top["top_numeric"]]
    redundancy = redundancy_matrix(features, top_names)

    # sensitivity: campaign-majority candidates only, same comparison
    majority = features["campaign_share"] >= 0.5
    sensitivity = compare_numeric(features, fs.numeric_features,
                                  mask_a=majority, mask_b=mask_b)

    report: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "phase": "3C_diagnostic_only",
        "scope_note": ("Descriptive diagnostics. No model trained, no candidate "
                       "created or altered, no threshold adopted, Phase 3B untouched."),
        "inputs": {
            "candidates": str(Path(args.candidates)),
            "assignments": str(Path(args.assignments)),
            "dataset_read_for": (["label", "campaign_id"] if args.no_behaviour
                                 else ["amount", "auth_outcome", "label", "campaign_id"]),
            "behavioural_features_enabled": not args.no_behaviour,
            "min_candidate_size": args.min_size,
        },
        "leakage_safety": leakage,
        "ground_truth_consistency": label_campaign_consistency(ground_truth),
        "groups": groups,
        "feature_inventory": {
            "numeric_features": list(fs.numeric_features),
            "boolean_features": list(fs.boolean_features),
            "notes": fs.notes,
        },
        "strongest_separations": top,
        "numeric_comparison": numeric.to_dict("records"),
        "boolean_comparison": boolean.to_dict("records"),
        "attack_rate_crosstabs": crosstabs.to_dict("records"),
        "top_feature_redundancy_spearman": (redundancy.to_dict()
                                            if len(redundancy) else {}),
        "sensitivity_campaign_majority_vs_non_campaign": {
            "n_campaign_majority_candidates": int(majority.sum()),
            "note": ("same comparison restricted to candidates whose members are "
                     ">= 50% campaign transactions; guards against conclusions "
                     "driven by single-attack-transaction candidates"),
            "top_numeric": sensitivity.head(args.top_n).to_dict("records"),
        },
    }

    # ---- 4. print --------------------------------------------------------
    print("\n=== LEAKAGE SAFETY ===")
    print(json.dumps(leakage, indent=2, default=str))
    print("\n=== GROUPS ===")
    print(json.dumps(groups, indent=2, default=str))
    print(f"\n=== TOP {args.top_n} NUMERIC SEPARATIONS (by |Cliff's delta|) ===")
    print(render_numeric_table(numeric, top_n=args.top_n))
    print(f"\n=== TOP {args.top_n} BOOLEAN / STRUCTURAL FLAGS ===")
    print(render_boolean_table(boolean, top_n=args.top_n))
    print("\n=== ATTACK RATE BY BUCKET ===")
    print(render_crosstab(crosstabs))
    if len(redundancy):
        print("\n=== REDUNDANCY AMONG TOP FEATURES (Spearman) ===")
        print(redundancy.to_string())
    print("\n=== INTERPRETATION POLICY ===")
    print(top["interpretation_policy"])

    # ---- 5. write --------------------------------------------------------
    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_dir / "candidate_diagnostic_features.csv", index=False)
        numeric.to_csv(out_dir / "numeric_comparison.csv", index=False)
        boolean.to_csv(out_dir / "boolean_comparison.csv", index=False)
        crosstabs.to_csv(out_dir / "attack_rate_crosstabs.csv", index=False)
        sensitivity.to_csv(out_dir / "sensitivity_campaign_majority.csv", index=False)
        if len(redundancy):
            redundancy.to_csv(out_dir / "top_feature_redundancy_spearman.csv")
        (out_dir / "phase3c_diagnostic_report.json").write_text(
            json.dumps(report, indent=2, default=str))
        print(f"\nwrote Phase 3C diagnostic artifacts to {out_dir}")

    check = groups.get("expected_counts_check")
    ok = leakage["clean"] and (check is None or check["matches"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
