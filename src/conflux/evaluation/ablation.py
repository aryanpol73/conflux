"""CONFLUX evaluation layer -- behavioural feature-group ablation.

Measures the contribution of the six locked signal groups (FEATURE_SPEC.md) using
the EXACT methodology of the validated baseline. This module imports the baseline's
own split / preprocessing / model / convergence / threshold / metric functions from
conflux.models.train_baseline rather than reimplementing any of them, so it cannot
silently diverge from the baseline it is supposed to be ablating.

WHAT THIS MODULE DOES NOT DO
----------------------------
* It never writes to, or reloads, src/conflux/models/artifacts/baseline_model.pkl.
  Reduced-feature models are fitted in memory and discarded.
* It never touches data/raw/dataset_v4_final.csv or data/processed/features_v4.csv.
* It writes exactly one file: data/processed/ablation_report.json.
* TEST is never used for preprocessing fitting, feature selection, threshold
  selection, convergence decisions, or any other tuning. The threshold is chosen on
  VALIDATION only and frozen before test is scored, once, per experiment.

Group membership is read from data/processed/feature_dictionary.csv (columns `name`
and `group`). No individual feature name is hardcoded anywhere in this file.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

try:
    from conflux.config import (
        FEATURE_DICTIONARY_PATH,
        FEATURES_TABLE_PATH,
        FORBIDDEN_MODEL_INPUTS,
        PROCESSED_DIR,
        PROJECT_ROOT,
        RAW_DATASET_PATH,
    )
except ImportError:  # allow `python src/conflux/evaluation/ablation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from conflux.config import (  # type: ignore
        FEATURE_DICTIONARY_PATH,
        FEATURES_TABLE_PATH,
        FORBIDDEN_MODEL_INPUTS,
        PROCESSED_DIR,
        PROJECT_ROOT,
        RAW_DATASET_PATH,
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conflux.models.train_baseline import (  # noqa: E402  (reuse, never reimplement)
    EXPECTED_FEATURE_COUNT,
    ID_COL,
    LABEL_COL,
    TS_COL,
    best_f1_threshold,
    build_pipeline,
    chronological_split,
    fit_with_convergence_check,
    load_aligned,
    score_block,
)

log = logging.getLogger("conflux.evaluation.ablation")

REPORT_SCHEMA_VERSION = "conflux.ablation.logreg.v1"
DEFAULT_OUT_PATH = PROCESSED_DIR / "ablation_report.json"
DEFAULT_BASELINE_REPORT = (
    PROJECT_ROOT / "src" / "conflux" / "models" / "artifacts" / "baseline_logreg_v4_report.json"
)

# The six locked signal groups of FEATURE_SPEC.md. This is a spec expectation used to
# VALIDATE the dictionary -- group membership itself is read from the dictionary.
EXPECTED_GROUPS: tuple[str, ...] = ("amount", "bin", "decline", "device", "merchant", "velocity")

# Paths this module must never write to.
FROZEN_PATHS = {RAW_DATASET_PATH.resolve(), FEATURES_TABLE_PATH.resolve(),
                FEATURE_DICTIONARY_PATH.resolve()}


class DictionaryError(ValueError):
    """Raised when feature_dictionary.csv does not describe exactly the model features."""


# ---------------------------------------------------------------------------
# Feature groups -- from the ACTUAL dictionary, never hardcoded
# ---------------------------------------------------------------------------
def load_feature_groups(dictionary_path: Path, feature_names: list[str]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Return {group: [features in feature-table order]} plus integrity evidence."""
    fd = pd.read_csv(dictionary_path)
    for col in ("name", "group"):
        if col not in fd.columns:
            raise DictionaryError(
                f"{dictionary_path} has no '{col}' column; found {list(fd.columns)}. "
                "Refusing to guess the schema."
            )

    dict_names = fd["name"].astype(str).tolist()
    dup_dict = sorted({n for n in dict_names if dict_names.count(n) > 1})
    if dup_dict:
        raise DictionaryError(f"duplicate feature name(s) in the dictionary: {dup_dict}")
    dup_table = sorted({n for n in feature_names if feature_names.count(n) > 1})
    if dup_table:
        raise DictionaryError(f"duplicate feature column(s) in the feature table: {dup_table}")

    set_dict, set_table = set(dict_names), set(feature_names)
    if set_dict != set_table:
        raise DictionaryError(
            f"dictionary/table mismatch: {len(set_dict - set_table)} dictionary-only "
            f"(e.g. {sorted(set_dict - set_table)[:5]}), {len(set_table - set_dict)} table-only "
            f"(e.g. {sorted(set_table - set_dict)[:5]})"
        )

    forbidden_in_dict = [n for n in dict_names if n in set(FORBIDDEN_MODEL_INPUTS) | {ID_COL, TS_COL}]
    if forbidden_in_dict:
        raise DictionaryError(f"dictionary declares forbidden column(s) as features: {forbidden_in_dict}")

    group_of = dict(zip(fd["name"].astype(str), fd["group"].astype(str)))
    observed_groups = tuple(sorted(set(group_of.values())))
    if observed_groups != tuple(sorted(EXPECTED_GROUPS)):
        raise DictionaryError(
            f"dictionary groups {observed_groups} are not the six locked signal groups "
            f"{tuple(sorted(EXPECTED_GROUPS))}"
        )

    # Preserve feature-table column order inside each group: deterministic matrices.
    groups: dict[str, list[str]] = {g: [] for g in sorted(EXPECTED_GROUPS)}
    for name in feature_names:
        groups[group_of[name]].append(name)

    counts = {g: len(v) for g, v in groups.items()}
    total = sum(counts.values())
    if total != len(feature_names) or total != EXPECTED_FEATURE_COUNT:
        raise DictionaryError(
            f"group counts sum to {total}; expected {len(feature_names)} table columns "
            f"and {EXPECTED_FEATURE_COUNT} contracted features"
        )
    if any(v == 0 for v in counts.values()):
        raise DictionaryError(f"empty feature group(s): {[g for g, v in counts.items() if v == 0]}")

    evidence = {
        "dictionary_path": str(dictionary_path),
        "dictionary_rows": int(len(fd)),
        "dictionary_matches_feature_table_exactly": True,
        "duplicate_names_in_dictionary": 0,
        "duplicate_columns_in_feature_table": 0,
        "forbidden_columns_in_dictionary": [],
        "groups_observed": list(observed_groups),
        "group_counts": counts,
        "group_counts_sum": total,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
    }
    return groups, evidence


def build_experiments(groups: dict[str, list[str]], feature_names: list[str]) -> list[dict[str, Any]]:
    """The 13 locked experiments: full, only_<g> x6, without_<g> x6."""
    ordered = sorted(groups)
    plans: list[dict[str, Any]] = [{
        "experiment": "full",
        "kind": "full",
        "groups_included": ordered,
        "groups_excluded": [],
        "features": list(feature_names),
    }]
    for g in ordered:
        plans.append({
            "experiment": f"only_{g}",
            "kind": "only",
            "groups_included": [g],
            "groups_excluded": [x for x in ordered if x != g],
            "features": [f for f in feature_names if f in set(groups[g])],
        })
    for g in ordered:
        keep = [x for x in ordered if x != g]
        keep_set = {f for x in keep for f in groups[x]}
        plans.append({
            "experiment": f"without_{g}",
            "kind": "without",
            "groups_included": keep,
            "groups_excluded": [g],
            "features": [f for f in feature_names if f in keep_set],
        })
    if len(plans) != 13:
        raise AssertionError(f"expected 13 experiments, built {len(plans)}")
    return plans


# ---------------------------------------------------------------------------
# One experiment == one fresh pipeline, fitted on the train slice only
# ---------------------------------------------------------------------------
def run_experiment(plan: dict[str, Any], X: pd.DataFrame, y: np.ndarray,
                   tr: np.ndarray, va: np.ndarray, te: np.ndarray,
                   class_weight: str | None, C: float, max_iter: int, solver: str,
                   max_iter_ceiling: int) -> dict[str, Any]:
    feats = plan["features"]
    if not feats:
        raise AssertionError(f"experiment '{plan['experiment']}' selected zero features")

    Xe = X.loc[:, feats]
    X_tr, X_va, X_te = Xe.iloc[tr], Xe.iloc[va], Xe.iloc[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    # Fresh, unfitted pipeline. Never reused between experiments, never the saved artifact.
    pipe = build_pipeline(class_weight, C, max_iter, solver)
    conv = fit_with_convergence_check(pipe, X_tr, y_tr, max_iter_ceiling)

    imp = pipe.named_steps["impute"]
    scaler = pipe.named_steps["scale"]
    clf = pipe.named_steps["clf"]
    n_ind = int(len(imp.indicator_.features_)) if imp.indicator_ is not None else 0
    fit_evidence = {
        "train_rows": int(len(tr)),
        "scaler_n_samples_seen": int(np.atleast_1d(scaler.n_samples_seen_).max()),
        "scaler_saw_only_train": bool(int(np.atleast_1d(scaler.n_samples_seen_).max()) == len(tr)),
        "imputer_n_features_in": int(imp.n_features_in_),
        "imputer_matches_selected_features": bool(int(imp.n_features_in_) == len(feats)),
        "indicator_columns": n_ind,
        "matrix_width_after_impute": int(imp.n_features_in_) + n_ind,
        "clf_n_features_in": int(clf.n_features_in_),
        "clf_width_matches": bool(int(clf.n_features_in_) == int(imp.n_features_in_) + n_ind),
        "validation_rows_transform_only": True,
        "test_rows_transform_only": True,
    }
    assert fit_evidence["scaler_saw_only_train"], "preprocessing saw rows outside the train slice"
    assert fit_evidence["imputer_matches_selected_features"]
    assert fit_evidence["clf_width_matches"]

    pos = int(list(clf.classes_).index(1))
    p_tr = pipe.predict_proba(X_tr)[:, pos]
    p_va = pipe.predict_proba(X_va)[:, pos]
    thr = best_f1_threshold(y_va, p_va)      # VALIDATION ONLY
    p_te = pipe.predict_proba(X_te)[:, pos]  # test scored once, threshold already frozen

    coef = clf.coef_.ravel()
    ind_src = [feats[i] for i in imp.indicator_.features_] if imp.indicator_ is not None else []
    names_out = list(feats) + [f"__missing__{n}" for n in ind_src]
    top = sorted(zip(names_out, coef.tolist()), key=lambda kv: abs(kv[1]), reverse=True)[:15]

    return {
        "experiment": plan["experiment"],
        "kind": plan["kind"],
        "groups_included": plan["groups_included"],
        "groups_excluded": plan["groups_excluded"],
        "feature_count": len(feats),
        "features": list(feats),
        "convergence": conv,
        "preprocessing_fit_evidence": fit_evidence,
        "operating_threshold": thr,
        "threshold_selection": "max F1 on VALIDATION only; test never used for tuning",
        "train": score_block(y_tr, p_tr, thr),
        "validation": score_block(y_va, p_va, thr),
        "test": score_block(y_te, p_te, thr),
        "top_abs_coefficients": top,
    }


def compare_full_to_baseline(full_result: dict[str, Any], baseline_report_path: Path,
                             atol: float = 1e-9) -> dict[str, Any]:
    """Reproducibility control. Reads the baseline report; never modifies it."""
    if not baseline_report_path.exists():
        return {"status": "NOT_EXECUTED",
                "reason": f"baseline report not found at {baseline_report_path}"}
    br = json.loads(baseline_report_path.read_text())
    bm, fm = br.get("metrics", {}), full_result
    out: dict[str, Any] = {"status": "EXECUTED", "baseline_report": str(baseline_report_path),
                           "comparisons": {}}
    thr_b = bm.get("operating_threshold")
    out["comparisons"]["operating_threshold"] = {
        "baseline": thr_b, "ablation_full": fm["operating_threshold"],
        "abs_diff": abs(float(thr_b) - fm["operating_threshold"]) if thr_b is not None else None,
    }
    for split in ("train", "validation", "test"):
        for metric in ("roc_auc", "pr_auc", "brier", "precision", "recall", "f1",
                       "tp", "fp", "fn", "tn", "alert_rate",
                       "recall_at_top_1pct", "recall_at_top_5pct"):
            b = bm.get(split, {}).get(metric)
            f = fm.get(split, {}).get(metric)
            if b is None or f is None:
                continue
            out["comparisons"][f"{split}.{metric}"] = {
                "baseline": b, "ablation_full": f, "abs_diff": abs(float(b) - float(f)),
            }
    diffs = [v["abs_diff"] for v in out["comparisons"].values() if v.get("abs_diff") is not None]
    out["max_abs_diff"] = float(max(diffs)) if diffs else None
    out["reproduces_baseline_within_atol"] = bool(diffs and max(diffs) <= atol)
    out["atol"] = atol
    out["note"] = ("The full experiment refits the baseline configuration in memory. It does not "
                   "load, modify, or overwrite baseline_model.pkl.")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CONFLUX behavioural feature-group ablation (13 experiments).")
    ap.add_argument("--features", default=str(FEATURES_TABLE_PATH))
    ap.add_argument("--dataset", default=str(RAW_DATASET_PATH))
    ap.add_argument("--dictionary", default=str(FEATURE_DICTIONARY_PATH))
    ap.add_argument("--baseline-report", default=str(DEFAULT_BASELINE_REPORT))
    ap.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--solver", default="lbfgs")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--max-iter-ceiling", type=int, default=20000)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s %(message)s")

    out_path = Path(args.out)
    if out_path.resolve() in FROZEN_PATHS:
        raise SystemExit(f"refusing to write to a frozen path: {out_path}")

    # Load + align exactly as the baseline does (156-feature contract enforced there).
    df, feature_names, diag = load_aligned(Path(args.features), Path(args.dataset))
    groups, dict_evidence = load_feature_groups(Path(args.dictionary), feature_names)

    # ONE split, shared by all 13 experiments.
    tr, va, te, split_meta = chronological_split(df, args.train_frac, args.val_frac)

    X = df[feature_names]
    y = df[LABEL_COL].to_numpy()
    n_inf = int(np.isinf(X.to_numpy(dtype=np.float64)).sum())
    if n_inf:
        log.warning("%s Inf cell(s) found; converting to NaN so the imputer handles them", n_inf)
        X = X.replace([np.inf, -np.inf], np.nan)

    class_weight = None if args.class_weight == "none" else "balanced"
    plans = build_experiments(groups, feature_names)

    results: list[dict[str, Any]] = []
    for plan in plans:
        log.info("running experiment %s (%s features)", plan["experiment"], len(plan["features"]))
        results.append(run_experiment(plan, X, y, tr, va, te, class_weight, args.C,
                                      args.max_iter, args.solver, args.max_iter_ceiling))

    full_result = next(r for r in results if r["experiment"] == "full")
    baseline_check = compare_full_to_baseline(full_result, Path(args.baseline_report))

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "OK",
        "methodology": {
            "purpose": "Measure the contribution of the six locked behavioural signal groups.",
            "model": "sklearn.linear_model.LogisticRegression, identical configuration to the baseline",
            "code_reuse": ("split, preprocessing, model construction, convergence policy, threshold "
                           "rule and metrics are imported from conflux.models.train_baseline; no "
                           "parallel implementation exists in this module"),
            "split": "chronological cutoff on timestamp, computed once and shared by all experiments",
            "preprocessing": ("median imputation with add_indicator=True and keep_empty_features=True, "
                              "then StandardScaler; refitted from scratch on each experiment's TRAIN "
                              "slice only"),
            "threshold_rule": "max F1 on VALIDATION only, frozen before test is scored once",
            "test_usage": ("TEST was NOT used for preprocessing fitting, feature selection, threshold "
                           "selection, convergence decisions, or any other tuning."),
            "artifact_policy": ("no model artifact is written or overwritten; reduced-feature models are "
                                "fitted in memory and discarded"),
            "model_params": {"class_weight": args.class_weight, "C": args.C, "solver": args.solver,
                             "max_iter": args.max_iter, "max_iter_ceiling": args.max_iter_ceiling,
                             "penalty": "l2", "tol": 1e-4, "random_state": 0},
        },
        "test_not_used_for_tuning": True,
        "dataset": diag,
        "split": split_meta,
        "feature_groups": {
            "source": str(args.dictionary),
            "name_column": "name",
            "group_column": "group",
            "counts": dict_evidence["group_counts"],
            "members": groups,
            "integrity": dict_evidence,
        },
        "experiment_count": len(results),
        "experiments": results,
        "full_vs_existing_baseline": baseline_check,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print(json.dumps({
        "experiments": [
            {"experiment": r["experiment"], "n_features": r["feature_count"],
             "converged": r["convergence"]["converged"], "n_iter": r["convergence"]["n_iter"],
             "threshold": r["operating_threshold"],
             "val_roc_auc": r["validation"].get("roc_auc"), "val_pr_auc": r["validation"].get("pr_auc"),
             "test_roc_auc": r["test"].get("roc_auc"), "test_pr_auc": r["test"].get("pr_auc"),
             "test_f1": r["test"].get("f1"), "test_recall": r["test"].get("recall"),
             "test_precision": r["test"].get("precision")}
            for r in results
        ],
        "full_vs_baseline_max_abs_diff": baseline_check.get("max_abs_diff"),
    }, indent=2))
    print(f"\nreport -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
