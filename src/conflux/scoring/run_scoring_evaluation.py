"""CONFLUX Phase 4A runner -- deterministic candidate scorer evaluation.

Invoke: py -3.14 -m conflux.scoring.run_scoring_evaluation

Reads the stored Phase 3B artifacts and never re-forms them. Writes only to
data/processed/scoring/. Refuses to write anywhere near data/raw/ or
data/processed/graph/.

STOP POINT: no ML model. Phase 4B builds one only after these numbers exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from conflux.config import RAW_DATASET_PATH
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from conflux.config import RAW_DATASET_PATH  # type: ignore

from conflux.evaluation.campaign_evaluation import load_ground_truth  # noqa: E402
from conflux.evaluation.candidate_diagnostics import (  # noqa: E402
    attach_groups, load_candidate_artifacts,
)
from conflux.scoring.candidate_features import (  # noqa: E402
    build_scoring_features, load_structural_attributes, prune_correlated,
    spearman_matrix,
)
from conflux.scoring.config import (  # noqa: E402
    ASSIGNMENTS_PATH, BIN_FEATURES, CANDIDATES_PATH, CORE_FEATURES,
    CORE_FEATURE_NAMES, FEATURE_SIGNS, FROZEN_PATHS, N_FOLDS,
    SCORING_OUT_DIR, SCORING_SCHEMA_VERSION, SPLIT_SEEDS,
)
from conflux.scoring.deterministic_scorer import (  # noqa: E402
    DeterministicScorer, tune_weights,
)
from conflux.scoring.evaluation import (  # noqa: E402
    apply_preregistered_rule, average_precision, bootstrap_ci, campaign_metrics,
    comparator_scores, confusion, normalize_purity, precision_at_k,
    recall_exchange_table, render_exchange_table, roc_auc, select_threshold,
)
from conflux.scoring.splits import (  # noqa: E402
    campaign_grouped_folds, chronological_candidate_split, cluster_ids,
    split_feasibility_probe,
)

log = logging.getLogger("conflux.scoring.run_scoring_evaluation")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _oof_protocol_a(features: pd.DataFrame, names: list[str], *,
                    n_folds: int, seeds: tuple[int, ...],
                    tuned: bool) -> tuple[np.ndarray, list[dict[str, Any]], list[float]]:
    """Out-of-fold scores + per-fold held-out metrics. Nested thresholds."""
    y = features["is_attack_containing"].to_numpy(dtype=int)
    folds = campaign_grouped_folds(features, n_folds=n_folds, seeds=seeds)

    oof_sum = np.zeros(len(features)); oof_n = np.zeros(len(features))
    per_fold: list[dict[str, Any]] = []
    praucs: list[float] = []

    for fold in folds:
        tr, te = fold.train_idx, fold.test_idx
        train_frame = features.iloc[tr]
        test_frame = features.iloc[te]

        weights = None
        if tuned:
            weights = tune_weights(train_frame[names], y[tr], names,
                                   objective=average_precision,
                                   signs=FEATURE_SIGNS)

        ref = DeterministicScorer.fit(train_frame[names], names,
                                      signs=FEATURE_SIGNS, weights=weights,
                                      fit_scope=f"A:r{fold.repeat}f{fold.fold} train")
        s_tr, _ = DeterministicScorer.transform(ref, train_frame[names])
        s_te, _ = DeterministicScorer.transform(ref, test_frame[names])

        oof_sum[te] += s_te; oof_n[te] += 1

        sel = select_threshold(y[tr], s_tr, rule="max_f1")
        ap = average_precision(y[te], s_te)
        praucs.append(ap)
        per_fold.append({
            **fold.as_dict(),
            "weights": ref.as_dict()["weights"],
            "threshold_selected_on_train": sel,
            "held_out_pr_auc": round(ap, 6) if np.isfinite(ap) else None,
            "held_out_roc_auc": round(roc_auc(y[te], s_te), 6),
            "held_out_at_train_threshold": confusion(y[te], s_te, sel["threshold"]),
            "held_out_campaign_metrics":
                campaign_metrics(test_frame, s_te, sel["threshold"]),
        })

    oof = np.divide(oof_sum, np.maximum(oof_n, 1))
    return oof, per_fold, praucs


def _evaluate_pooled(features: pd.DataFrame, s: np.ndarray, label: str
                     ) -> dict[str, Any]:
    y = features["is_attack_containing"].to_numpy(dtype=int)
    clusters = cluster_ids(features)
    return {
        "name": label,
        "pr_auc": bootstrap_ci(y, s, clusters, average_precision),
        "roc_auc": bootstrap_ci(y, s, clusters, roc_auc),
        "precision_at_k": [precision_at_k(y, s, k) for k in (50, 100, 200, 500)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="conflux.scoring.run_scoring_evaluation",
        description="CONFLUX Phase 4A: deterministic candidate triage scorer.")
    ap.add_argument("--dataset", default=str(RAW_DATASET_PATH))
    ap.add_argument("--candidates", default=str(CANDIDATES_PATH))
    ap.add_argument("--assignments", default=str(ASSIGNMENTS_PATH))
    ap.add_argument("--out-dir", default=str(SCORING_OUT_DIR))
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--expect-attack", type=int, default=None)
    ap.add_argument("--expect-other", type=int, default=None)
    ap.add_argument("--expect-campaigns", type=int, default=None)
    ap.add_argument("--skip-tuned", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s %(message)s")

    out_dir = Path(args.out_dir)
    if out_dir.resolve() in FROZEN_PATHS or any(
            p in out_dir.resolve().parents for p in FROZEN_PATHS):
        raise SystemExit(f"refusing to write to a frozen path: {out_dir}")

    cand_path, asg_path = Path(args.candidates), Path(args.assignments)
    artifact_hashes_before = {p.name: _sha256(p) for p in (cand_path, asg_path)}

    # ---- 1. inputs (read-only) ----------------------------------------
    candidates, assignments = load_candidate_artifacts(cand_path, asg_path)
    attributes = load_structural_attributes(args.dataset)
    sf = build_scoring_features(candidates, assignments, attributes, min_size=2)

    # ---- 2. ground truth: fold groups and metrics ONLY -----------------
    ground_truth = load_ground_truth(args.dataset)
    features = attach_groups(sf.frame, assignments, ground_truth,
                             group_by="campaign_id")
    features["purity_class"] = normalize_purity(features["purity_class"])
    features["is_campaign_majority"] = features["campaign_share"] >= 0.5

    n_attack = int(features["is_attack_containing"].sum())
    n_other = int(len(features) - n_attack)
    n_campaigns = int(features.loc[features["is_attack_containing"],
                                   "dominant_campaign_id"].nunique())

    population = {
        "multi_transaction_candidates": int(len(features)),
        "attack_containing": n_attack,
        "non_campaign": n_other,
        "base_rate": round(n_attack / len(features), 6),
        "campaigns_represented": n_campaigns,
        "expected_counts_check": {
            "expected_attack": args.expect_attack,
            "expected_other": args.expect_other,
            "expected_campaigns": args.expect_campaigns,
            "matches": bool(
                (args.expect_attack in (None, n_attack))
                and (args.expect_other in (None, n_other))
                and (args.expect_campaigns in (None, n_campaigns))),
        },
    }

    # ---- 3. decorrelation on the UNLABELLED training population --------
    tr_idx, _, _, chrono_probe_meta = chronological_candidate_split(features)
    unlabelled_train = features.iloc[tr_idx][list(CORE_FEATURE_NAMES)]
    kept, dropped = prune_correlated(unlabelled_train, list(CORE_FEATURE_NAMES))
    core = list(kept)
    corr = spearman_matrix(unlabelled_train, list(CORE_FEATURE_NAMES))

    feature_sets: dict[str, list[str]] = {
        "core": core,
        "core_no_bin": [c for c in core if c not in BIN_FEATURES],
    }
    if sf.auth_features:
        feature_sets["core_plus_auth_ablation"] = core + list(sf.auth_features)

    # ---- 4. Protocol A ---------------------------------------------------
    protocol_a: dict[str, Any] = {}
    uniform_praucs: list[float] = []
    tuned_praucs: list[float] = []
    oof_core: np.ndarray | None = None

    for label, names in feature_sets.items():
        oof, per_fold, praucs = _oof_protocol_a(
            features, names, n_folds=args.folds, seeds=SPLIT_SEEDS, tuned=False)
        protocol_a[label] = {
            "feature_names": names,
            "per_fold": per_fold,
            "mean_held_out_pr_auc": round(float(np.nanmean(praucs)), 6),
            "pooled_out_of_fold": _evaluate_pooled(features, oof, f"{label}_uniform"),
        }
        if label == "core":
            oof_core, uniform_praucs = oof, praucs

    if not args.skip_tuned:
        oof_t, per_fold_t, tuned_praucs = _oof_protocol_a(
            features, core, n_folds=args.folds, seeds=SPLIT_SEEDS, tuned=True)
        protocol_a["core_tuned_weights"] = {
            "feature_names": core, "per_fold": per_fold_t,
            "mean_held_out_pr_auc": round(float(np.nanmean(tuned_praucs)), 6),
            "pooled_out_of_fold": _evaluate_pooled(features, oof_t, "core_tuned"),
        }

    weight_decision = (apply_preregistered_rule(uniform_praucs, tuned_praucs)
                       if tuned_praucs else
                       {"decision": "keep_uniform_weights",
                        "note": "tuned comparison skipped via --skip-tuned"})

    # ---- 5. Protocol B ---------------------------------------------------
    tr, va, te, chrono_probe_meta = chronological_candidate_split(features)
    y = features["is_attack_containing"].to_numpy(dtype=int)
    ref_b = DeterministicScorer.fit(features.iloc[tr][core], core,
                                    signs=FEATURE_SIGNS,
                                    fit_scope="B: chronological train only")
    s_va, _ = DeterministicScorer.transform(ref_b, features.iloc[va][core])
    s_te, _ = DeterministicScorer.transform(ref_b, features.iloc[te][core])
    sel_b = select_threshold(y[va], s_va, rule="max_f1")   # validation, not test
    protocol_b = {
        "split": chrono_probe_meta,
        "threshold_selected_on_validation": sel_b,
        "test_pr_auc": round(average_precision(y[te], s_te), 6),
        "test_roc_auc": round(roc_auc(y[te], s_te), 6),
        "test_at_validation_threshold": confusion(y[te], s_te, sel_b["threshold"]),
        "test_campaign_metrics": campaign_metrics(features.iloc[te], s_te,
                                                  sel_b["threshold"]),
        "caveat": ("test metrics use a threshold chosen on validation; no metric "
                   "here is reported at a threshold chosen on the same rows."),
    }

    # ---- 6. comparators + exchange table (pooled out-of-fold) -----------
    comparators = {name: _evaluate_pooled(features, s, name)
                   for name, s in comparator_scores(features).items()}
    exchange = recall_exchange_table(features, oof_core)

    # secondary positive definition
    feat2 = features.copy()
    feat2["is_attack_containing"] = feat2["is_campaign_majority"]
    secondary = _evaluate_pooled(feat2, oof_core, "core_campaign_majority")

    report: dict[str, Any] = {
        "schema_version": SCORING_SCHEMA_VERSION,
        "phase": "4A_deterministic_candidate_triage",
        "scope_note": ("Retrospective candidate triage over FINALIZED Phase 3B "
                       "components. Not a decision-time transaction score. No "
                       "candidate created, merged, split or re-ordered."),
        "phase3b_artifact_sha256": artifact_hashes_before,
        "population": population,
        "feature_design": {
            "core_features": [
                {"name": f.name, "family": f.family, "sign": f.sign,
                 "source": f.source, "rationale": f.rationale}
                for f in CORE_FEATURES],
            "retained_after_decorrelation": core,
            "dropped_by_correlation_cap": dropped,
            "spearman_on_unlabelled_training_population": corr.to_dict(),
            "notes": sf.notes,
        },
        "split_feasibility_probe": split_feasibility_probe(features),
        "protocol_a_campaign_grouped": protocol_a,
        "protocol_b_chronological": protocol_b,
        "weight_decision_preregistered": weight_decision,
        "comparator_rankers": comparators,
        "secondary_positive_definition_campaign_majority": secondary,
        "recall_exchange_table": exchange.to_dict("records"),
    }

    # ---- 7. print ---------------------------------------------------------
    print("\n=== POPULATION ===")
    print(json.dumps(population, indent=2, default=str))
    print("\n=== SPLIT FEASIBILITY ===")
    print(json.dumps(report["split_feasibility_probe"], indent=2, default=str))
    print("\n=== PROTOCOL A: HELD-OUT PR-AUC BY FEATURE SET ===")
    for k, v in protocol_a.items():
        print(f"  {k:32s} mean held-out PR-AUC = {v['mean_held_out_pr_auc']}")
    print("\n=== COMPARATORS (pooled out-of-fold PR-AUC) ===")
    for k, v in comparators.items():
        print(f"  {k:32s} {v['pr_auc']['point']} "
              f"[{v['pr_auc']['ci_low']}, {v['pr_auc']['ci_high']}]")
    print("\n=== PRE-REGISTERED WEIGHT DECISION ===")
    print(json.dumps(weight_decision, indent=2, default=str))
    print("\n=== RECALL EXCHANGE TABLE (pooled out-of-fold) ===")
    print(render_exchange_table(exchange))
    print("\n=== PROTOCOL B (chronological) ===")
    print(json.dumps(protocol_b, indent=2, default=str))

    # ---- 8. write + integrity re-check -----------------------------------
    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_dir / "candidate_scoring_features.csv", index=False)
        pd.DataFrame({"candidate_id": features["candidate_id"],
                      "out_of_fold_score": oof_core}).to_csv(
            out_dir / "out_of_fold_scores.csv", index=False)
        exchange.to_csv(out_dir / "recall_exchange_table.csv", index=False)
        if len(corr):
            corr.to_csv(out_dir / "core_feature_spearman.csv")
        (out_dir / "phase4a_scoring_report.json").write_text(
            json.dumps(report, indent=2, default=str))
        print(f"\nwrote Phase 4A artifacts to {out_dir}")

    after = {p.name: _sha256(p) for p in (cand_path, asg_path)}
    intact = after == artifact_hashes_before
    print(f"\nPhase 3B artifacts unchanged: {intact}")

    ok = intact and population["expected_counts_check"]["matches"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
