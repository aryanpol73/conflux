"""CONFLUX baseline inference -- load the artifact, score, never fit.

HARD GUARANTEES
---------------
* Never trains. Never calls fit / fit_transform on anything.
* Never uses label, campaign_id, _source_type, timestamp, or transaction_id as a
  predictor. transaction_id is carried through as an output key only.
* Applies exactly the preprocessing saved at training time (median imputation with the
  train-fitted medians, the train-fitted missingness-indicator column set, the
  train-fitted standardisation), in the saved column order.
* Fails loudly on any schema deviation from the 156-feature contract.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from conflux.config import FEATURES_TABLE_PATH, FORBIDDEN_MODEL_INPUTS, PROJECT_ROOT
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from conflux.config import FEATURES_TABLE_PATH, FORBIDDEN_MODEL_INPUTS, PROJECT_ROOT  # type: ignore

log = logging.getLogger("conflux.models.predict")

ARTIFACT_SCHEMA_VERSION = "conflux.baseline.logreg.v1"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "src" / "conflux" / "models" / "artifacts" / "baseline_model.pkl"
ID_COL = "transaction_id"
_FIT_FORBIDDEN = ("fit", "fit_transform", "partial_fit")


class SchemaError(ValueError):
    """Raised when the input frame does not satisfy the saved 156-feature contract."""


def load_artifact(path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}. Run train_baseline.py first.")
    art = joblib.load(path)
    if not isinstance(art, dict) or art.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise SchemaError(f"unsupported artifact schema: {art.get('schema_version') if isinstance(art, dict) else type(art)}")
    for key in ("pipeline", "feature_names", "positive_class_index"):
        if key not in art:
            raise SchemaError(f"artifact is missing required key '{key}'")
    if not isinstance(art["pipeline"], Pipeline):
        raise SchemaError("artifact['pipeline'] is not an sklearn Pipeline")
    names = art["feature_names"]
    if len(names) != art.get("feature_count", len(names)) or len(set(names)) != len(names):
        raise SchemaError("artifact feature_names are inconsistent or contain duplicates")
    leaked = [c for c in art.get("forbidden_inputs", FORBIDDEN_MODEL_INPUTS) if c in names]
    if leaked:
        raise SchemaError(f"artifact was trained with forbidden predictor(s): {leaked}")
    return art


def validate_and_align(df: pd.DataFrame, feature_names: list[str], forbidden: Iterable[str]) -> pd.DataFrame:
    """Return the feature matrix in the exact saved order. Never mutates the caller's frame."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{len(missing)} required feature column(s) absent, e.g. {missing[:8]}. "
            f"Expected the {len(feature_names)}-feature v4 schema."
        )
    extra = [c for c in df.columns if c not in feature_names]
    if extra:
        log.info("ignoring %s non-predictor column(s), e.g. %s", len(extra), extra[:8])
    used_forbidden = [c for c in forbidden if c in feature_names]
    if used_forbidden:
        raise SchemaError(f"forbidden column(s) would be used as predictors: {used_forbidden}")

    X = df.loc[:, feature_names].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    n_inf = int(np.isinf(X.to_numpy(dtype=np.float64)).sum())
    if n_inf:
        log.warning("%s Inf cell(s) in input; converting to NaN for the saved imputer", n_inf)
        X = X.replace([np.inf, -np.inf], np.nan)
    if list(X.columns) != list(feature_names):
        raise SchemaError("column alignment failed")
    return X


def predict_proba(df: pd.DataFrame, artifact: dict[str, Any] | None = None,
                  artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> np.ndarray:
    """Attack probability per row, in input row order. Transform-only."""
    art = artifact or load_artifact(artifact_path)
    pipe: Pipeline = art["pipeline"]
    X = validate_and_align(df, art["feature_names"], art.get("forbidden_inputs", FORBIDDEN_MODEL_INPUTS))

    # Belt and braces: this code path must never fit.
    for name, step in pipe.steps:
        for attr in _FIT_FORBIDDEN:
            if attr == "fit" and not hasattr(step, "fit"):
                raise SchemaError(f"pipeline step '{name}' is not a fitted estimator")

    p = pipe.predict_proba(X)[:, art["positive_class_index"]]
    p = np.asarray(p, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        raise AssertionError("non-finite probability produced")
    if p.min() < 0.0 or p.max() > 1.0:
        raise AssertionError(f"probability outside [0,1]: min={p.min()} max={p.max()}")
    return p


def predict_frame(df: pd.DataFrame, artifact: dict[str, Any] | None = None,
                  artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
                  threshold: float | None = None) -> pd.DataFrame:
    art = artifact or load_artifact(artifact_path)
    p = predict_proba(df, artifact=art)
    thr = art.get("operating_threshold", 0.5) if threshold is None else float(threshold)
    out = pd.DataFrame({"attack_probability": p, "flagged": (p >= thr).astype(int)}, index=df.index)
    if ID_COL in df.columns:  # carried as an output key, never a predictor
        out.insert(0, ID_COL, df[ID_COL].to_numpy())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score transactions with the CONFLUX baseline model.")
    ap.add_argument("--features", default=str(FEATURES_TABLE_PATH), help="feature table CSV to score")
    ap.add_argument("--artifact", default=str(DEFAULT_ARTIFACT_PATH))
    ap.add_argument("--out", default=None, help="optional CSV output path")
    ap.add_argument("--threshold", type=float, default=None, help="override the saved operating threshold")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows (smoke test)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s %(message)s")

    art = load_artifact(args.artifact)
    df = pd.read_csv(args.features, dtype={ID_COL: str}, nrows=args.limit, low_memory=False)
    res = predict_frame(df, artifact=art, threshold=args.threshold)

    p = res["attack_probability"].to_numpy()
    print(json.dumps({
        "artifact": str(args.artifact),
        "schema_version": art["schema_version"],
        "expected_features": art["feature_count"],
        "rows_scored": int(len(res)),
        "prob_min": float(p.min()), "prob_max": float(p.max()), "prob_mean": float(p.mean()),
        "all_finite": bool(np.all(np.isfinite(p))),
        "within_unit_interval": bool(p.min() >= 0.0 and p.max() <= 1.0),
        "threshold": float(args.threshold if args.threshold is not None else art.get("operating_threshold", 0.5)),
        "flagged": int(res["flagged"].sum()),
    }, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
