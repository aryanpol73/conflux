"""CONFLUX Phase 4B -- rebuild seam (IMPORT / API-VERIFICATION REPAIR PASS).

SCOPE OF THIS FILE IN ITS CURRENT STATE
---------------------------------------
This module must import cleanly and verify_pipeline_api(strict=False) must
return a JSON-serializable diagnostic report. Nothing else is guaranteed yet.

WHY DYNAMIC RESOLUTION
----------------------
Several frozen APIs (conflux.graph.*, conflux.scoring.candidate_features,
conflux.evaluation.*) have NOT been signature-verified against this file. A
static `from X import Y` on an unverified symbol turns a diagnosable mismatch
into an ImportError at module load, which is exactly the failure mode Phase 4B
has been stuck in. They are therefore resolved lazily through _resolve() and
reported by verify_pipeline_api(). Call sites below are PROVISIONAL until that
report pins the real signatures.

NOTHING HERE MODIFIES PHASE 3B OR PHASE 4A.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import tempfile
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---- verified static imports only -------------------------------------
from conflux.config import PROCESSED_DIR, RAW_DATASET_PATH
from conflux.scoring.config import FROZEN_PATHS, SCORING_OUT_DIR
from conflux.scoring.deterministic_scorer import (
    DeterministicScorer, ScorerReference,
)

log = logging.getLogger("conflux.robustness.rebuild")

# Local derivation from a verified constant. NOT a new config entry.
ROBUSTNESS_OUT_DIR: Path = PROCESSED_DIR / "robustness"
WORLDS_DIR: Path = ROBUSTNESS_OUT_DIR / "worlds"

# The frozen Phase 4A candidate table is the ONLY source of baseline counts.
BASELINE_FEATURES_PATH: Path = SCORING_OUT_DIR / "candidate_scoring_features.csv"

# Column names as they appear in candidate_scoring_features.csv (verified from
# the artifact schema, not from memory of any module constant).
FEAT_ID_COL = "candidate_id"
FEAT_LABEL_COL = "is_attack_containing"
FEAT_CAMPAIGN_COL = "dominant_campaign_id"

TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class WorldRebuildError(RuntimeError):
    """A frozen pipeline API is unavailable/mismatched, or a world frame is
    not a valid dataset."""


# ----------------------------------------------------------------------
# dynamic resolution of UNVERIFIED frozen APIs
# ----------------------------------------------------------------------
_DYNAMIC_TARGETS: dict[str, tuple[str, str]] = {
    "TemporalEntityGraph":        ("conflux.graph.temporal_graph", "TemporalEntityGraph"),
    "CandidateConfig":            ("conflux.graph.campaign_detection", "CandidateConfig"),
    "CandidateSet":               ("conflux.graph.campaign_detection", "CandidateSet"),
    "form_campaign_candidates":   ("conflux.graph.campaign_detection", "form_campaign_candidates"),
    "GraphConfig":                ("conflux.graph.config", "GraphConfig"),
    "ID_COL":                     ("conflux.graph.config", "ID_COL"),
    "TS_COL":                     ("conflux.graph.config", "TS_COL"),
    "STRUCTURAL_COLUMNS":         ("conflux.graph.config", "STRUCTURAL_COLUMNS"),
    "ATTRIBUTE_COLUMNS":          ("conflux.graph.config", "ATTRIBUTE_COLUMNS"),
    "FORBIDDEN_GRAPH_INPUTS":     ("conflux.graph.config", "FORBIDDEN_GRAPH_INPUTS"),
    "ScoringFeatures":            ("conflux.scoring.candidate_features", "ScoringFeatures"),
    "build_scoring_features":     ("conflux.scoring.candidate_features", "build_scoring_features"),
    "load_structural_attributes": ("conflux.scoring.candidate_features", "load_structural_attributes"),
    "attach_groups":              ("conflux.evaluation.candidate_diagnostics", "attach_groups"),
    "GROUP_COLUMNS":              ("conflux.evaluation.candidate_diagnostics", "GROUP_COLUMNS"),
    "load_ground_truth":          ("conflux.evaluation.campaign_evaluation", "load_ground_truth"),
    "normalize_ground_truth":     ("conflux.evaluation.campaign_evaluation", "normalize_ground_truth"),
    "evaluate_grouping":          ("conflux.evaluation.campaign_evaluation", "evaluate_grouping"),
    "CAMPAIGN_COL":               ("conflux.evaluation.campaign_evaluation", "CAMPAIGN_COL"),
    "GT_ID_COL":                  ("conflux.evaluation.campaign_evaluation", "ID_COL"),
}

_cache: dict[str, Any] = {}


def _try_resolve(key: str) -> tuple[Any, str | None]:
    """Return (object, error_string). Never raises."""
    if key in _cache:
        return _cache[key], None
    try:
        mod_name, attr = _DYNAMIC_TARGETS[key]
    except KeyError:
        return None, f"KeyError: '{key}' is not a registered dynamic target"
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:                      # noqa: BLE001 - diagnostic
        return None, f"{type(exc).__name__}: {exc}"
    if not hasattr(mod, attr):
        return None, f"AttributeError: '{mod_name}' has no attribute '{attr}'"
    obj = getattr(mod, attr)
    _cache[key] = obj
    return obj, None


def _resolve(key: str) -> Any:
    """Strict resolution for call sites. Raises WorldRebuildError, not ImportError."""
    obj, err = _try_resolve(key)
    if err is not None:
        mod_name, attr = _DYNAMIC_TARGETS.get(key, ("<unregistered>", key))
        raise WorldRebuildError(
            f"required frozen API '{attr}' could not be resolved from "
            f"'{mod_name}': {err}. Run verify_pipeline_api(strict=False) and "
            "supply the real module source before calling this path.")
    return obj


# ----------------------------------------------------------------------
# API verification -- every probe is individually guarded
# ----------------------------------------------------------------------
def _params(fn: Any) -> tuple[str, ...]:
    sig = inspect.signature(fn)
    return tuple(p for p, spec in sig.parameters.items()
                 if spec.kind not in (spec.VAR_POSITIONAL, spec.VAR_KEYWORD))


def _probe_callable(key: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"target": ".".join(_DYNAMIC_TARGETS[key]),
                             "status": "unknown", "signature": None,
                             "parameters": None, "error": None}
    obj, err = _try_resolve(key)
    if err is not None:
        entry["status"] = "not_found"
        entry["error"] = err
        return entry
    try:
        entry["signature"] = str(inspect.signature(obj))
        entry["parameters"] = list(_params(obj))
        entry["status"] = "found"
    except Exception as exc:                      # noqa: BLE001 - diagnostic
        entry["status"] = "probe_error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _probe_method(key: str, method: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"target": f"{_DYNAMIC_TARGETS[key][1]}.{method}",
                             "status": "unknown", "signature": None,
                             "parameters": None, "error": None}
    obj, err = _try_resolve(key)
    if err is not None:
        entry["status"] = "not_found"
        entry["error"] = err
        return entry
    if not hasattr(obj, method):
        entry["status"] = "not_found"
        entry["error"] = f"AttributeError: no method '{method}'"
        return entry
    try:
        fn = getattr(obj, method)
        entry["signature"] = str(inspect.signature(fn))
        entry["parameters"] = list(_params(fn))
        entry["status"] = "found"
    except Exception as exc:                      # noqa: BLE001 - diagnostic
        entry["status"] = "probe_error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _probe_value(key: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"target": ".".join(_DYNAMIC_TARGETS[key]),
                             "status": "unknown", "value": None, "error": None}
    obj, err = _try_resolve(key)
    if err is not None:
        entry["status"] = "not_found"
        entry["error"] = err
        return entry
    try:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            entry["value"] = obj
        elif isinstance(obj, (list, tuple, set, frozenset)):
            entry["value"] = [str(v) for v in obj]
        else:
            entry["value"] = str(obj)
        entry["status"] = "found"
    except Exception as exc:                      # noqa: BLE001 - diagnostic
        entry["status"] = "probe_error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _probe_dataclass_fields(key: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"target": ".".join(_DYNAMIC_TARGETS[key]),
                             "status": "unknown", "is_dataclass": None,
                             "fields": None, "error": None}
    obj, err = _try_resolve(key)
    if err is not None:
        entry["status"] = "not_found"
        entry["error"] = err
        return entry
    try:
        entry["is_dataclass"] = bool(is_dataclass(obj))
        if entry["is_dataclass"]:
            entry["fields"] = list(obj.__dataclass_fields__)
        else:
            entry["fields"] = list(inspect.signature(obj).parameters)
        entry["status"] = "found"
    except Exception as exc:                      # noqa: BLE001 - diagnostic
        entry["status"] = "probe_error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _probe_default_construction(key: str, attrs: tuple[str, ...]) -> dict[str, Any]:
    """Instantiating with no args can raise TypeError. Fully guarded."""
    entry: dict[str, Any] = {"target": f"{_DYNAMIC_TARGETS[key][1]}()",
                             "status": "unknown", "attributes": {}, "error": None}
    obj, err = _try_resolve(key)
    if err is not None:
        entry["status"] = "not_found"
        entry["error"] = err
        return entry
    try:
        inst = obj()
    except Exception as exc:                      # noqa: BLE001 - diagnostic
        entry["status"] = "construction_error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        return entry
    entry["status"] = "found"
    for a in attrs:
        try:
            v = getattr(inst, a)
            entry["attributes"][a] = (
                v if isinstance(v, (str, int, float, bool)) or v is None
                else [str(x) for x in v] if isinstance(v, (list, tuple, set, frozenset))
                else str(v))
        except Exception as exc:                  # noqa: BLE001 - diagnostic
            entry["attributes"][a] = f"ERROR {type(exc).__name__}: {exc}"
    return entry


def verify_pipeline_api(*, strict: bool = True) -> dict[str, Any]:
    """Introspect every frozen API this module intends to call.

    Returns a JSON-serializable report. No expected-signature table is asserted:
    the previous table was written from memory and is deliberately not carried
    forward. Signatures are REPORTED so they can be pinned from real evidence.

    strict=False : never raises. strict=True : raises only on CONFIRMED
    failures (not_found / probe_error / construction_error).
    """
    report: dict[str, Any] = {"schema": "conflux.robustness.rebuild.verify.v2"}

    report["static_imports"] = {
        "conflux.config.PROCESSED_DIR": str(PROCESSED_DIR),
        "conflux.config.RAW_DATASET_PATH": str(RAW_DATASET_PATH),
        "conflux.scoring.config.SCORING_OUT_DIR": str(SCORING_OUT_DIR),
        "conflux.scoring.config.FROZEN_PATHS": sorted(str(p) for p in FROZEN_PATHS),
        "DeterministicScorer.fit": str(inspect.signature(DeterministicScorer.fit)),
        "DeterministicScorer.transform": str(inspect.signature(DeterministicScorer.transform)),
        "ScorerReference.fields": list(ScorerReference.__dataclass_fields__),
    }

    callables: dict[str, Any] = {}
    for key in ("form_campaign_candidates", "build_scoring_features",
                "load_structural_attributes", "attach_groups",
                "load_ground_truth", "normalize_ground_truth",
                "evaluate_grouping"):
        callables[key] = _probe_callable(key)
    callables["TemporalEntityGraph.from_frame"] = _probe_method("TemporalEntityGraph", "from_frame")
    callables["TemporalEntityGraph.from_csv"] = _probe_method("TemporalEntityGraph", "from_csv")
    callables["TemporalEntityGraph.summary"] = _probe_method("TemporalEntityGraph", "summary")
    callables["CandidateSet.candidate_frame"] = _probe_method("CandidateSet", "candidate_frame")
    report["callables"] = callables

    types_: dict[str, Any] = {
        "CandidateConfig": _probe_dataclass_fields("CandidateConfig"),
        "CandidateConfig_defaults": _probe_default_construction(
            "CandidateConfig", ("include_singletons", "window_seconds")),
        "ScoringFeatures": _probe_dataclass_fields("ScoringFeatures"),
        "GraphConfig": _probe_dataclass_fields("GraphConfig"),
    }
    report["types"] = types_

    constants: dict[str, Any] = {}
    for key in ("ID_COL", "TS_COL", "STRUCTURAL_COLUMNS", "ATTRIBUTE_COLUMNS",
                "FORBIDDEN_GRAPH_INPUTS", "GROUP_COLUMNS", "CAMPAIGN_COL",
                "GT_ID_COL"):
        constants[key] = _probe_value(key)
    report["constants"] = constants

    art: dict[str, Any] = {"path": str(BASELINE_FEATURES_PATH),
                           "exists": BASELINE_FEATURES_PATH.is_file()}
    if art["exists"]:
        try:
            head = pd.read_csv(BASELINE_FEATURES_PATH, nrows=1)
            art["columns"] = list(head.columns)
            art["has_expected_columns"] = all(
                c in head.columns
                for c in (FEAT_ID_COL, FEAT_LABEL_COL, FEAT_CAMPAIGN_COL))
        except Exception as exc:                  # noqa: BLE001 - diagnostic
            art["error"] = f"{type(exc).__name__}: {exc}"
    report["baseline_artifact"] = art

    failures: list[str] = []
    for section in ("callables", "types", "constants"):
        for name, entry in report[section].items():
            if entry.get("status") in ("not_found", "probe_error",
                                       "construction_error"):
                failures.append(f"{section}.{name}: {entry.get('status')} "
                                f"-- {entry.get('error')}")

    report["failures"] = failures
    report["all_ok"] = not failures

    if failures and strict:
        raise WorldRebuildError(
            "frozen pipeline API verification failed; refusing to call it:\n  "
            + "\n  ".join(failures))
    return report


# ----------------------------------------------------------------------
# baseline population, derived -- never hard-coded
# ----------------------------------------------------------------------
def _as_bool(series: pd.Series) -> pd.Series:
    """CSV round-trip can make a boolean column strings. Normalize safely."""
    if series.dtype == bool:
        return series
    return (series.astype(str).str.strip().str.lower()
            .isin({"true", "1", "yes", "t"}))


def frozen_baseline_population(path: Path | None = None) -> dict[str, Any]:
    """Read the frozen Phase 4A candidate table and derive the reference counts.

    This file is the single source of truth for parity. No literal counts appear
    anywhere in this module.
    """
    p = Path(path) if path is not None else BASELINE_FEATURES_PATH
    if not p.is_file():
        raise WorldRebuildError(
            f"frozen Phase 4A candidate table not found at {p}; baseline parity "
            "cannot be derived and must not be hard-coded.")
    df = pd.read_csv(p)
    for c in (FEAT_ID_COL, FEAT_LABEL_COL, FEAT_CAMPAIGN_COL):
        if c not in df.columns:
            raise WorldRebuildError(
                f"{p} is missing column '{c}'; columns are {list(df.columns)}")
    lab = _as_bool(df[FEAT_LABEL_COL])
    pos = int(lab.sum())
    camps = int(df.loc[lab, FEAT_CAMPAIGN_COL]
                .replace("", np.nan).dropna().astype(str).nunique())
    n = int(len(df))
    return {"source": str(p),
            "multi_transaction_candidates": n,
            "attack_containing_candidates": pos,
            "base_rate": round(pos / n, 6) if n else None,
            "distinct_campaigns_represented": camps}


# ----------------------------------------------------------------------
# world frame -> world file
# ----------------------------------------------------------------------
def world_columns() -> tuple[str, ...]:
    """Resolved lazily: depends on graph.config + campaign_evaluation."""
    structural = tuple(_resolve("STRUCTURAL_COLUMNS"))
    attribute = tuple(_resolve("ATTRIBUTE_COLUMNS"))
    campaign = _resolve("CAMPAIGN_COL")
    return (*structural, *attribute, "label", campaign)


def render_timestamps(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.strftime(TS_FORMAT)
    return values.astype(str)


def ns_to_timestamp_strings(ts_ns: np.ndarray) -> pd.Series:
    return (pd.to_datetime(pd.Series(np.asarray(ts_ns, dtype="int64")), unit="ns")
              .dt.strftime(TS_FORMAT))


def validate_world_frame(frame: pd.DataFrame) -> None:
    id_col = _resolve("ID_COL")
    cols = world_columns()
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise WorldRebuildError(f"world frame missing column(s): {missing}")
    for col in _resolve("STRUCTURAL_COLUMNS"):
        s = frame[col]
        bad = int(s.isna().sum() + (s.astype(str).str.strip() == "").sum())
        if bad:
            raise WorldRebuildError(
                f"{bad} blank/NaN value(s) in structural column '{col}'")
    dup = int(frame[id_col].duplicated().sum())
    if dup:
        raise WorldRebuildError(f"{dup} duplicate {id_col} in the world frame")
    if "amount" in frame.columns and frame["amount"].isna().any():
        raise WorldRebuildError("world frame has NaN amount")


def write_world_csv(frame: pd.DataFrame, path: Path) -> Path:
    resolved = Path(path).resolve()
    frozen = {Path(p).resolve() for p in FROZEN_PATHS}
    if resolved in frozen or any(p in resolved.parents for p in frozen):
        raise WorldRebuildError(f"refusing to write into a frozen path: {path}")
    validate_world_frame(frame)
    ts_col = _resolve("TS_COL")
    out = frame.loc[:, list(world_columns())].copy()
    out[ts_col] = render_timestamps(out[ts_col])
    resolved.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(resolved, index=False)
    log.info("wrote world: %s rows -> %s", len(out), resolved)
    return resolved


# ----------------------------------------------------------------------
# rebuild  (call sites PROVISIONAL until verify_pipeline_api pins signatures)
# ----------------------------------------------------------------------
@dataclass
class RebuiltWorld:
    name: str
    world_path: Path
    n_transactions: int
    graph_summary: dict[str, Any]
    candidate_set: Any
    candidate_frame: pd.DataFrame
    assignments: pd.DataFrame
    scoring_features: Any
    labelled_features: pd.DataFrame
    ground_truth: pd.DataFrame
    grouping_metrics: dict[str, Any]

    @property
    def frame(self) -> pd.DataFrame:
        return self.labelled_features

    def population(self) -> dict[str, Any]:
        f = self.labelled_features
        lab = _as_bool(f[FEAT_LABEL_COL])
        pos = int(lab.sum())
        camps = int(f.loc[lab, FEAT_CAMPAIGN_COL]
                    .replace("", np.nan).dropna().astype(str).nunique())
        return {"transactions": int(self.n_transactions),
                "multi_transaction_candidates": int(len(f)),
                "attack_containing_candidates": pos,
                "base_rate": round(pos / len(f), 6) if len(f) else None,
                "distinct_campaigns_represented": camps}


def rebuild_world(frame: pd.DataFrame, *, name: str,
                  graph_config: Any = None,
                  candidate_config: Any = None,
                  min_size: int = 2,
                  world_dir: Path | None = None,
                  keep_world_file: bool = False,
                  strict_alignment: bool = True) -> RebuiltWorld:
    verify_pipeline_api(strict=True)

    TemporalEntityGraph = _resolve("TemporalEntityGraph")
    form_campaign_candidates = _resolve("form_campaign_candidates")
    CandidateConfig = _resolve("CandidateConfig")
    GraphConfig = _resolve("GraphConfig")
    build_scoring_features = _resolve("build_scoring_features")
    load_structural_attributes = _resolve("load_structural_attributes")
    load_ground_truth = _resolve("load_ground_truth")
    attach_groups = _resolve("attach_groups")
    evaluate_grouping = _resolve("evaluate_grouping")

    gcfg = graph_config if graph_config is not None else GraphConfig()
    ccfg = candidate_config if candidate_config is not None else CandidateConfig()

    tmp: tempfile.TemporaryDirectory | None = None
    if world_dir is None and not keep_world_file:
        tmp = tempfile.TemporaryDirectory(prefix="conflux_4b_world_")
        target_dir = Path(tmp.name)
    else:
        target_dir = Path(world_dir) if world_dir is not None else WORLDS_DIR
    world_path = target_dir / f"world_{name}.csv"

    try:
        write_world_csv(frame, world_path)

        graph = TemporalEntityGraph.from_csv(world_path, config=gcfg)
        cset = form_campaign_candidates(graph, ccfg)
        candidates = cset.candidate_frame()
        assignments = cset.assignments
        attributes = load_structural_attributes(world_path)
        features = build_scoring_features(candidates, assignments, attributes,
                                          min_size=min_size)
        gt = load_ground_truth(world_path)
        labelled = attach_groups(features.frame, assignments, gt,
                                 group_by="campaign_id")
        grouping = evaluate_grouping(assignments, gt, group_col="candidate_id",
                                     name=f"4B_world_{name}",
                                     strict=strict_alignment)

        world = RebuiltWorld(
            name=name, world_path=world_path, n_transactions=int(len(frame)),
            graph_summary=graph.summary(), candidate_set=cset,
            candidate_frame=candidates, assignments=assignments,
            scoring_features=features, labelled_features=labelled,
            ground_truth=gt, grouping_metrics=grouping.metrics)
        log.info("rebuilt world '%s': %s", name, world.population())
        return world
    finally:
        if tmp is not None:
            tmp.cleanup()


def rebuild_baseline(raw_path: Path | None = None, **kwargs: Any) -> RebuiltWorld:
    p = Path(raw_path) if raw_path is not None else Path(RAW_DATASET_PATH)
    frame = pd.read_csv(p, dtype=str, keep_default_na=False,
                        na_values=[], low_memory=False)
    if "amount" in frame.columns:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="raise")
    return rebuild_world(frame, name=kwargs.pop("name", "baseline"), **kwargs)


def assert_baseline_parity(world: RebuiltWorld, *, strict: bool = True,
                           reference_path: Path | None = None) -> dict[str, Any]:
    expected = frozen_baseline_population(reference_path)
    observed = world.population()
    keys = ("multi_transaction_candidates", "attack_containing_candidates",
            "distinct_campaigns_represented")
    failures = {k: {"observed": observed[k], "expected": expected[k]}
                for k in keys if observed[k] != expected[k]}
    result = {"parity": not failures, "observed": observed,
              "expected": expected, "failures": failures}
    if failures and strict:
        raise WorldRebuildError(
            "baseline rebuild does not reproduce the frozen Phase 4A "
            "population; scenario deltas would be uninterpretable. "
            + repr(failures))
    return result


# ----------------------------------------------------------------------
# scoring with the FROZEN reference -- transform() only
# ----------------------------------------------------------------------
def score_world(ref: ScorerReference, world: RebuiltWorld) -> np.ndarray:
    names = list(ref.feature_names)
    missing = [n for n in names if n not in world.labelled_features.columns]
    if missing:
        raise WorldRebuildError(
            f"world '{world.name}' is missing frozen feature(s) {missing}")
    scores, _ = DeterministicScorer.transform(ref, world.labelled_features[names])
    return scores


def labels_of(world: RebuiltWorld) -> np.ndarray:
    return _as_bool(world.labelled_features[FEAT_LABEL_COL]).to_numpy(dtype=int)


if __name__ == "__main__":
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Phase 4B rebuild seam")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    rep = verify_pipeline_api(strict=args.strict)
    print(_json.dumps(rep, indent=2))
    if not args.verify_only:
        rep2 = frozen_baseline_population()
        print(_json.dumps(rep2, indent=2))

