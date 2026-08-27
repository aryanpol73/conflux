"""CONFLUX baseline ML layer -- Logistic Regression on the frozen v4 feature table.

SCOPE (locked): Logistic Regression only. No XGBoost, no Random Forest, no graph, no
scoring, no API. This module reads the frozen dataset and the validated feature table
and writes exactly one artifact plus one training report. It never writes to, or
modifies, anything under src/conflux/features/.

CAUSALITY / LEAKAGE CONTRACT
----------------------------
1. Predictors are exactly the 156 behavioural feature columns of features_v4.csv.
   transaction_id, timestamp, label, campaign_id, _source_type are never predictors.
2. The split is CHRONOLOGICAL, by timestamp cutoff, never random. Cutoffs are derived
   from the observed timestamp distribution at run time and printed.
3. Every fitted object (imputer statistics, missingness-indicator column set, scaler
   mean/scale, model coefficients) is fitted on the TRAIN slice only. Validation and
   test are transform-only.
4. The operating threshold is selected on VALIDATION only. Test is scored once.

Run `--inspect-only` first to see the temporal label distribution before committing
to cutoffs.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from conflux.config import (
        FEATURES_TABLE_PATH,
        FORBIDDEN_MODEL_INPUTS,
        PROJECT_ROOT,
        RAW_DATASET_PATH,
    )
except ImportError:  # allow `python src/conflux/models/train_baseline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from conflux.config import (  # type: ignore
        FEATURES_TABLE_PATH,
        FORBIDDEN_MODEL_INPUTS,
        PROJECT_ROOT,
        RAW_DATASET_PATH,
    )

log = logging.getLogger("conflux.models.train_baseline")

ARTIFACT_SCHEMA_VERSION = "conflux.baseline.logreg.v1"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "src" / "conflux" / "models" / "artifacts"
DEFAULT_ARTIFACT_NAME = "baseline_model.pkl"
EXPECTED_FEATURE_COUNT = 156
EXPECTED_ROWS = 31_873
ID_COL = "transaction_id"
TS_COL = "timestamp"
LABEL_COL = "label"


# ---------------------------------------------------------------------------
# Load + align
# ---------------------------------------------------------------------------
def load_aligned(features_path: Path, dataset_path: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load the feature table, attach the target by transaction_id, verify alignment."""
    feats = pd.read_csv(features_path, dtype={ID_COL: str}, low_memory=False)
    if ID_COL not in feats.columns:
        raise KeyError(f"{features_path} has no {ID_COL} column")

    feature_names = [c for c in feats.columns if c != ID_COL]
    forbidden_in_features = [c for c in FORBIDDEN_MODEL_INPUTS if c in feature_names]
    if forbidden_in_features:
        raise AssertionError(f"forbidden column(s) present in the feature table: {forbidden_in_features}")
    if TS_COL in feature_names:
        raise AssertionError("timestamp must not be a predictor; it is present in the feature table")
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"expected {EXPECTED_FEATURE_COUNT} behavioural features, found {len(feature_names)}. "
            "The feature layer is locked; refusing to train on a changed feature set."
        )

    raw = pd.read_csv(
        dataset_path,
        dtype=str,
        usecols=lambda c: c in {ID_COL, LABEL_COL, TS_COL},
        low_memory=False,
    )
    for col in (ID_COL, LABEL_COL, TS_COL):
        if col not in raw.columns:
            raise KeyError(f"{dataset_path} has no {col} column")

    # --- alignment checks, all fatal -------------------------------------
    if feats[ID_COL].duplicated().any():
        raise AssertionError(f"{int(feats[ID_COL].duplicated().sum())} duplicate transaction_id in the feature table")
    if raw[ID_COL].duplicated().any():
        raise AssertionError(f"{int(raw[ID_COL].duplicated().sum())} duplicate transaction_id in the dataset")
    f_ids, r_ids = set(feats[ID_COL]), set(raw[ID_COL])
    if f_ids != r_ids:
        raise AssertionError(
            f"transaction_id sets differ: {len(f_ids - r_ids)} feature-only, {len(r_ids - f_ids)} dataset-only"
        )

    df = feats.merge(raw, on=ID_COL, how="inner", validate="one_to_one")
    if len(df) != len(feats):
        raise AssertionError(f"merge changed row count: {len(feats)} -> {len(df)}")

    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    if y.isna().any():
        raise AssertionError(f"{int(y.isna().sum())} missing/unparseable label(s); refusing to drop rows")
    bad = sorted(set(y.unique()) - {0.0, 1.0})
    if bad:
        raise AssertionError(f"label must be binary 0/1, found {bad}")
    df[LABEL_COL] = y.astype(np.int8)

    ts = pd.to_datetime(df[TS_COL], errors="coerce")
    if ts.isna().any():
        raise AssertionError(f"{int(ts.isna().sum())} unparseable timestamp(s)")
    df[TS_COL] = ts

    # Deterministic causal order, identical tie-break rule to the feature layer.
    df = df.sort_values([TS_COL, ID_COL], kind="mergesort").reset_index(drop=True)

    diagnostics = {
        "features_path": str(features_path),
        "dataset_path": str(dataset_path),
        "rows": int(len(df)),
        "rows_expected": EXPECTED_ROWS,
        "rows_match_expected": bool(len(df) == EXPECTED_ROWS),
        "feature_count": len(feature_names),
        "duplicate_transaction_ids": 0,
        "missing_targets": 0,
        "id_sets_identical": True,
        "positives_total": int(df[LABEL_COL].sum()),
        "negatives_total": int((df[LABEL_COL] == 0).sum()),
        "prevalence_total": float(df[LABEL_COL].mean()),
        "timestamp_min": str(df[TS_COL].min()),
        "timestamp_max": str(df[TS_COL].max()),
        "distinct_timestamps": int(df[TS_COL].nunique()),
    }
    return df, feature_names, diagnostics


def temporal_label_profile(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    g = df.set_index(TS_COL)[LABEL_COL].resample(freq)
    out = pd.DataFrame({"n": g.size(), "positives": g.sum()})
    out["prevalence"] = np.where(out["n"] > 0, out["positives"] / out["n"].where(out["n"] > 0), np.nan)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Split by TIMESTAMP CUTOFF (never randomly). Identical timestamps stay together."""
    n = len(df)
    ts = df[TS_COL].to_numpy()

    def boundary(frac: float) -> int:
        i = int(round(frac * n))
        i = min(max(i, 1), n - 1)
        while i < n and ts[i] == ts[i - 1]:  # never split a shared timestamp
            i += 1
        return i

    i_tr = boundary(train_frac)
    i_va = boundary(train_frac + val_frac)
    if not (0 < i_tr < i_va < n):
        raise ValueError(f"degenerate split boundaries: train_end={i_tr} val_end={i_va} n={n}")

    idx = np.arange(n)
    tr, va, te = idx[:i_tr], idx[i_tr:i_va], idx[i_va:]

    # Hard temporal ordering assertions.
    assert df[TS_COL].iloc[tr].max() < df[TS_COL].iloc[va].min(), "train/val overlap in time"
    assert df[TS_COL].iloc[va].max() < df[TS_COL].iloc[te].min(), "val/test overlap in time"
    assert set(df[ID_COL].iloc[tr]).isdisjoint(df[ID_COL].iloc[va]), "train/val share rows"
    assert set(df[ID_COL].iloc[va]).isdisjoint(df[ID_COL].iloc[te]), "val/test share rows"
    assert set(df[ID_COL].iloc[tr]).isdisjoint(df[ID_COL].iloc[te]), "train/test share rows"
    assert len(tr) + len(va) + len(te) == n

    def block(name: str, ix: np.ndarray) -> dict[str, Any]:
        y = df[LABEL_COL].to_numpy()[ix]
        return {
            "split": name,
            "rows": int(len(ix)),
            "positives": int(y.sum()),
            "negatives": int((y == 0).sum()),
            "prevalence": float(y.mean()) if len(ix) else float("nan"),
            "ts_start": str(df[TS_COL].iloc[ix].min()),
            "ts_end": str(df[TS_COL].iloc[ix].max()),
        }

    meta = {
        "method": "chronological cutoff on timestamp (no shuffling, no random_state)",
        "requested_fractions": {"train": train_frac, "val": val_frac, "test": round(1 - train_frac - val_frac, 6)},
        "cutoff_train_end_exclusive": str(df[TS_COL].iloc[i_tr]),
        "cutoff_val_end_exclusive": str(df[TS_COL].iloc[i_va]),
        "splits": [block("train", tr), block("val", va), block("test", te)],
    }
    for b in meta["splits"]:
        if b["positives"] == 0:
            raise ValueError(
                f"split '{b['split']}' contains ZERO positives ({b['ts_start']} .. {b['ts_end']}). "
                "A chronological split at these fractions is not usable. Run --inspect-only, "
                "look at the temporal label profile, and choose cutoffs that place attack "
                "activity on both sides. Refusing to report meaningless metrics."
            )
    return tr, va, te, meta


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(class_weight: str | None, C: float, max_iter: int, solver: str) -> Pipeline:
    try:
        imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    except TypeError as exc:  # sklearn < 1.2
        raise RuntimeError(
            f"scikit-learn {sklearn.__version__} does not support keep_empty_features. "
            "Upgrade to >=1.2: without it, columns that are entirely NaN in train are "
            "silently dropped and the 156-feature contract breaks."
        ) from exc
    return Pipeline(
        [
            ("impute", imputer),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                    class_weight=class_weight,
                    tol=1e-4,
                    n_jobs=None,
                    random_state=0,  # affects nothing for lbfgs; pinned for reproducibility
                ),
            ),
        ]
    )


def fit_with_convergence_check(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, max_iter_ceiling: int) -> dict[str, Any]:
    """Fit, escalating max_iter until lbfgs converges or the ceiling is hit."""
    history = []
    while True:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipe.fit(X, y)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        clf: LogisticRegression = pipe.named_steps["clf"]
        n_iter = int(np.max(clf.n_iter_))
        history.append({"max_iter": int(clf.max_iter), "n_iter": n_iter, "converged": bool(converged)})
        if converged or clf.max_iter >= max_iter_ceiling:
            break
        clf.set_params(max_iter=min(clf.max_iter * 4, max_iter_ceiling))
        log.warning("LogisticRegression did not converge; retrying with max_iter=%s", clf.max_iter)
    return {
        "converged": history[-1]["converged"],
        "n_iter": history[-1]["n_iter"],
        "final_max_iter": history[-1]["max_iter"],
        "attempts": history,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def score_block(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    if not np.all(np.isfinite(p)):
        raise AssertionError("non-finite probability produced")
    if p.min() < 0.0 or p.max() > 1.0:
        raise AssertionError(f"probability outside [0,1]: min={p.min()} max={p.max()}")
    out: dict[str, Any] = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "prob_min": float(p.min()),
        "prob_max": float(p.max()),
        "prob_mean": float(p.mean()),
    }
    if len(np.unique(y)) < 2:
        out["note"] = "single-class block; ranking metrics undefined"
        return out
    out["roc_auc"] = float(roc_auc_score(y, p))
    out["pr_auc"] = float(average_precision_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    out.update(
        {
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
            "alert_rate": float(yhat.mean()),
        }
    )
    # Realistic fraud-ops view: fixed review budget.
    for pct in (0.01, 0.05):
        k = max(1, int(round(pct * len(p))))
        top = np.argsort(-p, kind="stable")[:k]
        out[f"recall_at_top_{int(pct * 100)}pct"] = float(y[top].sum() / y.sum())
    return out


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold selected on VALIDATION ONLY."""
    prec, rec, thr = precision_recall_curve(y, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the CONFLUX baseline Logistic Regression.")
    ap.add_argument("--features", default=str(FEATURES_TABLE_PATH))
    ap.add_argument("--dataset", default=str(RAW_DATASET_PATH))
    ap.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    ap.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--solver", default="lbfgs")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--max-iter-ceiling", type=int, default=20000)
    ap.add_argument("--inspect-only", action="store_true", help="print the temporal label profile and exit")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s %(message)s")
    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("train_frac + val_frac must leave a non-empty test slice")

    df, feature_names, diag = load_aligned(Path(args.features), Path(args.dataset))
    print("\n=== DATASET / ALIGNMENT ===")
    print(json.dumps(diag, indent=2))

    profile = temporal_label_profile(df)
    print("\n=== TEMPORAL LABEL PROFILE (hourly) ===")
    print(profile.to_string(index=False))
    if args.inspect_only:
        return 0
    if diag["positives_total"] == 0:
        raise SystemExit("the dataset contains no positive labels; nothing to train")

    tr, va, te, split_meta = chronological_split(df, args.train_frac, args.val_frac)
    print("\n=== CHRONOLOGICAL SPLIT ===")
    print(json.dumps(split_meta, indent=2))

    X = df[feature_names]
    y = df[LABEL_COL].to_numpy()
    # Defensive: features_v4 declares 0 Inf cells, but never let one reach the scaler.
    n_inf = int(np.isinf(X.to_numpy(dtype=np.float64)).sum())
    if n_inf:
        log.warning("%s Inf cell(s) found; converting to NaN so the imputer handles them", n_inf)
        X = X.replace([np.inf, -np.inf], np.nan)

    X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    class_weight = None if args.class_weight == "none" else "balanced"
    pipe = build_pipeline(class_weight, args.C, args.max_iter, args.solver)
    conv = fit_with_convergence_check(pipe, X_tr, y_tr, args.max_iter_ceiling)
    print("\n=== CONVERGENCE ===")
    print(json.dumps(conv, indent=2))
    if not conv["converged"]:
        raise SystemExit(f"LogisticRegression failed to converge at max_iter={conv['final_max_iter']}")

    imp: SimpleImputer = pipe.named_steps["impute"]
    scaler: StandardScaler = pipe.named_steps["scale"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    # PREPROCESSING-LEAKAGE PROOF: fitted objects saw exactly len(train) samples.
    fit_evidence = {
        "train_rows": int(len(tr)),
        "imputer_n_features_in": int(imp.n_features_in_),
        "scaler_n_samples_seen": int(np.atleast_1d(scaler.n_samples_seen_).max()),
        "scaler_saw_only_train": bool(int(np.atleast_1d(scaler.n_samples_seen_).max()) == len(tr)),
        "indicator_columns": int(len(imp.indicator_.features_)) if imp.indicator_ is not None else 0,
        "matrix_width_after_impute": int(imp.n_features_in_ + (len(imp.indicator_.features_) if imp.indicator_ is not None else 0)),
    }
    assert fit_evidence["scaler_saw_only_train"], "scaler was fitted on more rows than the train slice"
    assert fit_evidence["imputer_n_features_in"] == EXPECTED_FEATURE_COUNT
    print("\n=== PREPROCESSING FIT EVIDENCE ===")
    print(json.dumps(fit_evidence, indent=2))

    p_tr = pipe.predict_proba(X_tr)[:, 1]
    p_va = pipe.predict_proba(X_va)[:, 1]
    thr = best_f1_threshold(y_va, p_va)          # chosen on VALIDATION only
    p_te = pipe.predict_proba(X_te)[:, 1]        # test scored once, after the threshold is frozen

    metrics = {
        "operating_threshold": thr,
        "threshold_selection": "max F1 on VALIDATION only; test never used for tuning",
        "train": score_block(y_tr, p_tr, thr),
        "validation": score_block(y_va, p_va, thr),
        "test": score_block(y_te, p_te, thr),
    }
    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))

    coef = pipe.named_steps["clf"].coef_.ravel()
    ind_src = [feature_names[i] for i in imp.indicator_.features_] if imp.indicator_ is not None else []
    names_out = feature_names + [f"__missing__{n}" for n in ind_src]
    top = sorted(zip(names_out, coef.tolist()), key=lambda kv: abs(kv[1]), reverse=True)[:25]

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "sklearn.linear_model.LogisticRegression",
        "pipeline": pipe,                       # imputer + indicators + scaler + model
        "feature_names": feature_names,         # exact order predict.py must supply
        "feature_count": len(feature_names),
        "forbidden_inputs": list(FORBIDDEN_MODEL_INPUTS) + [ID_COL, TS_COL],
        "positive_class_index": int(list(pipe.named_steps["clf"].classes_).index(1)),
        "operating_threshold": thr,
        "model_params": pipe.named_steps["clf"].get_params(),
        "class_weight": args.class_weight,
        "convergence": conv,
        "split": split_meta,
        "dataset_diagnostics": diag,
        "preprocessing_fit_evidence": fit_evidence,
        "metrics": metrics,
        "top_abs_coefficients": top,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "notes": [
            "Probabilities from a class_weight='balanced' fit are NOT calibrated to the "
            "true prevalence; they are inflated. Use ranking metrics (ROC-AUC, PR-AUC, "
            "recall@top-k) as primary. Brier is reported but is not a calibration verdict.",
            "Preprocessing is fitted on the chronological TRAIN slice only.",
        ],
    }

    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / args.artifact_name
    joblib.dump(artifact, artifact_path)

    # --- reload + byte-identical inference check, in the same run ---------
    reloaded = joblib.load(artifact_path)
    p_re = reloaded["pipeline"].predict_proba(X_te[reloaded["feature_names"]])[:, 1]
    reload_check = {
        "artifact_path": str(artifact_path),
        "artifact_bytes": int(artifact_path.stat().st_size),
        "schema_version_match": reloaded["schema_version"] == ARTIFACT_SCHEMA_VERSION,
        "feature_names_match": reloaded["feature_names"] == feature_names,
        "probabilities_identical": bool(np.array_equal(p_re, p_te)),
        "max_abs_diff": float(np.max(np.abs(p_re - p_te))),
        "all_finite": bool(np.all(np.isfinite(p_re))),
        "within_unit_interval": bool(p_re.min() >= 0.0 and p_re.max() <= 1.0),
    }
    assert reload_check["probabilities_identical"], "reloaded artifact does not reproduce predictions"
    print("\n=== RELOAD CHECK ===")
    print(json.dumps(reload_check, indent=2))

    report = {
        "status": "OK",
        "artifact": str(artifact_path),
        "dataset": diag,
        "split": split_meta,
        "preprocessing_fit_evidence": fit_evidence,
        "convergence": conv,
        "metrics": metrics,
        "reload_check": reload_check,
        "top_abs_coefficients": top,
        "environment": artifact["environment"],
    }
    report_path = out_dir / "baseline_logreg_v4_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nartifact -> {artifact_path}\nreport   -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
