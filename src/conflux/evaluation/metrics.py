"""CONFLUX evaluation layer -- formal metrics for the EXISTING baseline.

Reuses train_baseline.py's own load_aligned / chronological_split / score_block
/ best_f1_threshold so this module cannot silently diverge from how the
baseline itself defines the split, the threshold rule, or a metric -- it is
not a second, parallel implementation.

This module does NOT retrain. It:
  1. loads the frozen feature table + raw dataset the same way train_baseline.py does,
  2. reproduces the identical chronological train/val/test index split,
  3. scores each split with the EXISTING saved artifact's pipeline (transform-only,
     predict_proba only -- see predict.py's `_FIT_FORBIDDEN` guard, mirrored here),
  4. reports the same metric set train_baseline.py reports, using the ARTIFACT's own
     saved operating_threshold (also re-derivable on validation only, for comparison).

Requires the raw dataset (label + timestamp) to do anything beyond load the artifact
and score the feature table label-free. Without it, `main()` still runs predict-only
inference (real, executed) and reports why labeled metrics were not computed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from conflux.config import FEATURES_TABLE_PATH, PROJECT_ROOT, RAW_DATASET_PATH
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from conflux.config import FEATURES_TABLE_PATH, PROJECT_ROOT, RAW_DATASET_PATH  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conflux.models.train_baseline import (  # noqa: E402  (reuse, do not reimplement)
    best_f1_threshold,
    chronological_split,
    load_aligned,
    score_block,
)

log = logging.getLogger("conflux.evaluation.metrics")
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "src" / "conflux" / "models" / "artifacts" / "baseline_model.pkl"


def evaluate_existing_baseline(features_path: Path, dataset_path: Path, artifact_path: Path,
                                train_frac: float = 0.70, val_frac: float = 0.15) -> dict[str, Any]:
    """Re-derive train/val/test metrics for the EXISTING artifact. Transform-only."""
    art = joblib.load(artifact_path)
    pipe = art["pipeline"]
    feature_names = art["feature_names"]

    df, feature_names_loaded, diag = load_aligned(features_path, dataset_path)
    if feature_names_loaded != feature_names:
        raise AssertionError(
            "feature table column order does not match the order the artifact was trained on; "
            "refusing to score with a possibly-misaligned matrix."
        )

    tr, va, te, split_meta = chronological_split(df, train_frac, val_frac)
    X = df[feature_names]
    y = df["label"].to_numpy()
    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    for name, step in pipe.steps:  # belt-and-braces, same guard as predict.py
        if hasattr(step, "fit") and not hasattr(step, "predict_proba") and not hasattr(step, "transform"):
            raise AssertionError(f"pipeline step '{name}' looks unfitted")

    p_tr = pipe.predict_proba(X_tr)[:, art["positive_class_index"]]
    p_va = pipe.predict_proba(X_va)[:, art["positive_class_index"]]
    p_te = pipe.predict_proba(X_te)[:, art["positive_class_index"]]

    thr_recomputed = best_f1_threshold(y_va, p_va)          # independently re-derived, validation only
    thr_artifact = float(art["operating_threshold"])         # as saved at training time

    result = {
        "artifact_path": str(artifact_path),
        "artifact_schema_version": art["schema_version"],
        "split": split_meta,
        "threshold_selection": "max F1 on VALIDATION only; test never used for tuning",
        "threshold_recomputed_on_validation": thr_recomputed,
        "threshold_saved_in_artifact": thr_artifact,
        "threshold_reproduces_artifact": bool(np.isclose(thr_recomputed, thr_artifact)),
        "metrics_using_artifact_threshold": {
            "train": score_block(y_tr, p_tr, thr_artifact),
            "validation": score_block(y_va, p_va, thr_artifact),
            "test": score_block(y_te, p_te, thr_artifact),
        },
    }
    if not result["threshold_reproduces_artifact"]:
        result["metrics_using_recomputed_threshold"] = {
            "train": score_block(y_tr, p_tr, thr_recomputed),
            "validation": score_block(y_va, p_va, thr_recomputed),
            "test": score_block(y_te, p_te, thr_recomputed),
        }
    return result


def predict_only_summary(features_path: Path, artifact_path: Path) -> dict[str, Any]:
    """Label-free execution: real inference over the full feature table with the real artifact."""
    art = joblib.load(artifact_path)
    df = pd.read_csv(features_path, dtype={"transaction_id": str}, low_memory=False)
    X = df[art["feature_names"]]
    p = art["pipeline"].predict_proba(X)[:, art["positive_class_index"]]
    thr = float(art["operating_threshold"])
    return {
        "rows_scored": int(len(df)),
        "prob_min": float(p.min()), "prob_max": float(p.max()), "prob_mean": float(p.mean()),
        "all_finite": bool(np.all(np.isfinite(p))),
        "threshold": thr,
        "flagged": int((p >= thr).sum()),
        "note": "Label-free scoring only -- no ROC-AUC/PR-AUC/F1 possible without ground-truth "
                "labels, which live in the raw dataset (label column), not in the feature table.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the EXISTING CONFLUX baseline (no retraining).")
    ap.add_argument("--features", default=str(FEATURES_TABLE_PATH))
    ap.add_argument("--dataset", default=str(RAW_DATASET_PATH))
    ap.add_argument("--artifact", default=str(DEFAULT_ARTIFACT_PATH))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    dataset_path = Path(args.dataset)
    report: dict[str, Any] = {
        "features_path": str(args.features),
        "dataset_path": str(dataset_path),
        "dataset_path_exists": dataset_path.exists(),
    }
    if dataset_path.exists():
        report["labeled_metrics"] = evaluate_existing_baseline(Path(args.features), dataset_path, Path(args.artifact))
        report["labeled_metrics_status"] = "EXECUTED"
    else:
        report["labeled_metrics_status"] = "NOT_EXECUTED"
        report["labeled_metrics_blocked_reason"] = (
            f"raw dataset (label + timestamp) not found at {dataset_path}; ROC-AUC / PR-AUC / "
            "F1 / confusion matrix / recall@k require the ground-truth label column, which is "
            "not present in the feature table by design (Rule 5, ground-truth leakage)."
        )
        report["predict_only_summary"] = predict_only_summary(Path(args.features), Path(args.artifact))

    print(json.dumps(report, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())