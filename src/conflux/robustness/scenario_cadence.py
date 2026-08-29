"""CONFLUX Phase 4B -- S1: CADENCE ROBUSTNESS.

QUESTION
--------
How does the frozen pipeline behave when coordinated groups execute FASTER or
SLOWER, while every non-temporal property of every transaction is unchanged?

PIPELINE
--------
    baseline frame
        -> scale_group_cadence(factor)        (perturbations.py, unmodified)
        -> build_world(...)                   (world.py, invariants + provenance)
        -> assert_only_timestamps_changed()   (scenario-level guard, this file)
        -> rebuild_world(...)                 (rebuild.py, frozen pipeline)
        -> population / grouping metrics      (read off RebuiltWorld)
        -> JSON-serializable comparison

NO PARALLEL PIPELINE. This module contains no cadence arithmetic of its own,
no graph code, no metric definitions and NO SCORER. Every number it reports is
read from RebuiltWorld.population() / .grouping_metrics / .graph_summary.

SCORING
-------
Deliberately absent. No serialized Phase 4A ScorerReference exists in the
repository, and fitting one here would silently invent a scorer. S1 is
answerable from candidate formation and grouping metrics alone.

IDENTITY ARM
------------
factor == 1.0 is arithmetically a no-op inside scale_group_cadence, but the
timestamp column is still re-rendered through TS_FORMAT. The control arm is
therefore compared on PARSED NANOSECONDS, never on raw strings or fingerprints.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from conflux.robustness.perturbations import (
    NS_PER_SECOND, PerturbationError, SchemaView, attack_mask, resolve_schema,
    scale_group_cadence, timestamps_as_ns,
)
from conflux.robustness.rebuild import RebuiltWorld, rebuild_world
from conflux.robustness.world import (
    PerturbedWorld, WorldError, baseline_world, build_world, to_json_safe,
)

log = logging.getLogger("conflux.robustness.scenario_cadence")

SCENARIO_ID = "S1_cadence"
SCENARIO_NAME = "S1 cadence robustness"
RESULT_SCHEMA = "conflux.robustness.scenario_cadence.v1"

#: 0.5 = twice as fast, 1.0 = control, 2.0 = half as fast.
DEFAULT_CADENCE_FACTORS: tuple[float, ...] = (0.5, 1.0, 2.0)
IDENTITY_FACTOR: float = 1.0

#: Mirrors the anchor contract of perturbations.scale_group_cadence.
VALID_ANCHORS: tuple[str, ...] = ("start", "end", "median")

#: Group-label values scale_group_cadence treats as "no group".
_NULL_GROUPS = frozenset({"", "nan", "None"})

RebuildFn = Callable[..., RebuiltWorld]


class CadenceScenarioError(ValueError):
    """S1 configuration is invalid, or a cadence world violates an invariant."""


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
def _validate_factor(factor: Any) -> float:
    try:
        value = float(factor)
    except (TypeError, ValueError) as exc:
        raise CadenceScenarioError(
            f"cadence factor must be a number, got {factor!r}") from exc
    if not math.isfinite(value):
        raise CadenceScenarioError(
            f"cadence factor must be finite, got {factor!r}")
    if value <= 0.0:
        raise CadenceScenarioError(
            f"cadence factor must be > 0 (see scale_group_cadence), got {value}")
    return value


def _validate_anchor(anchor: Any) -> str:
    if anchor not in VALID_ANCHORS:
        raise CadenceScenarioError(
            f"anchor must be one of {list(VALID_ANCHORS)}, got {anchor!r}")
    return str(anchor)


@dataclass(frozen=True)
class CadenceScenarioConfig:
    """Everything that makes an S1 run reproducible."""

    factors: tuple[float, ...] = DEFAULT_CADENCE_FACTORS
    anchor: str = "start"
    group_col: str | None = None          # None -> schema.campaign_col
    group_values: tuple[Any, ...] | None = None   # None -> all real campaigns
    scenario_id: str = SCENARIO_ID
    min_size: int = 2
    world_dir: Path | None = None
    keep_world_file: bool = False
    strict_alignment: bool = True
    graph_config: Any = None
    candidate_config: Any = None

    def __post_init__(self) -> None:
        factors = tuple(self.factors or ())
        if not factors:
            raise CadenceScenarioError("factors must not be empty")
        object.__setattr__(self, "factors",
                           tuple(_validate_factor(f) for f in factors))
        object.__setattr__(self, "anchor", _validate_anchor(self.anchor))
        if self.group_values is not None:
            values = tuple(self.group_values)
            if not values:
                raise CadenceScenarioError(
                    "group_values must be None or a non-empty sequence")
            object.__setattr__(self, "group_values", values)
        if int(self.min_size) < 2:
            raise CadenceScenarioError("min_size must be >= 2")

    def rebuild_kwargs(self) -> dict[str, Any]:
        return {"graph_config": self.graph_config,
                "candidate_config": self.candidate_config,
                "min_size": int(self.min_size),
                "world_dir": self.world_dir,
                "keep_world_file": bool(self.keep_world_file),
                "strict_alignment": bool(self.strict_alignment)}

    def as_dict(self) -> dict[str, Any]:
        return {"factors": list(self.factors),
                "anchor": self.anchor,
                "group_col": self.group_col,
                "group_values": (None if self.group_values is None
                                 else [str(v) for v in self.group_values]),
                "scenario_id": self.scenario_id,
                "min_size": int(self.min_size),
                "world_dir": None if self.world_dir is None else str(self.world_dir),
                "keep_world_file": bool(self.keep_world_file),
                "strict_alignment": bool(self.strict_alignment),
                "graph_config": None if self.graph_config is None else str(self.graph_config),
                "candidate_config": None if self.candidate_config is None else str(self.candidate_config)}


def factor_tag(factor: float) -> str:
    """Filesystem-safe arm suffix: 0.5 -> '0p5', 2.0 -> '2'."""
    return f"{float(factor):g}".replace(".", "p").replace("-", "neg")


# ----------------------------------------------------------------------
# cadence measurement (reporting only -- no timestamps are written here)
# ----------------------------------------------------------------------
def cadence_profile(frame: pd.DataFrame, *,
                    group_col: str | None = None,
                    group_values: Sequence[Any] | None = None,
                    schema: SchemaView | None = None) -> dict[str, dict[str, Any]]:
    """Per-group temporal footprint: span and inter-arrival spacing.

    Group selection mirrors scale_group_cadence exactly, so the profile
    describes precisely the rows the primitive acts on.
    """
    s = schema or resolve_schema()
    gcol = group_col or s.campaign_col
    if gcol not in frame.columns:
        raise CadenceScenarioError(
            f"group column '{gcol}' not in frame; columns are {list(frame.columns)}")
    ts = timestamps_as_ns(frame, schema=s)
    groups = frame[gcol].astype(str)
    selected = (set(str(v) for v in group_values) if group_values is not None
                else {g for g in groups.unique() if g not in _NULL_GROUPS})

    out: dict[str, dict[str, Any]] = {}
    for g in sorted(selected):
        idx = np.flatnonzero((groups == g).to_numpy())
        if idx.size < 2:
            continue
        vals = np.sort(ts[idx])
        gaps = np.diff(vals)
        out[g] = {
            "transactions": int(idx.size),
            "span_seconds": float(vals[-1] - vals[0]) / NS_PER_SECOND,
            "mean_interarrival_seconds": float(gaps.mean()) / NS_PER_SECOND,
            "median_interarrival_seconds": float(np.median(gaps)) / NS_PER_SECOND,
        }
    return out


def cadence_span_ratios(before: Mapping[str, Mapping[str, Any]],
                        after: Mapping[str, Mapping[str, Any]]
                        ) -> dict[str, float | None]:
    """after/before span ratio per group. ~= factor when the primitive worked."""
    ratios: dict[str, float | None] = {}
    for g, b in before.items():
        a = after.get(g)
        b_span = float(b["span_seconds"])
        if a is None or b_span <= 0.0:
            ratios[g] = None
        else:
            ratios[g] = float(a["span_seconds"]) / b_span
    return ratios


def _median_ratio(ratios: Mapping[str, float | None]) -> float | None:
    vals = [v for v in ratios.values() if v is not None and math.isfinite(v)]
    return float(np.median(vals)) if vals else None


# ----------------------------------------------------------------------
# invariants -- only timestamps may move
# ----------------------------------------------------------------------
def assert_only_timestamps_changed(before: pd.DataFrame, after: pd.DataFrame, *,
                                   schema: SchemaView | None = None
                                   ) -> dict[str, Any]:
    """Prove a cadence perturbation touched the timestamp column and nothing else.

    world.build_world already enforces ID/label/campaign integrity and row
    accounting. This adds the S1-specific claim: EVERY other column -- entities,
    amount, auth outcome -- is byte-identical per transaction.
    """
    s = schema or resolve_schema()
    if list(before.columns) != list(after.columns):
        raise CadenceScenarioError(
            f"column set/order changed: {list(before.columns)} -> "
            f"{list(after.columns)}")
    if len(before) != len(after):
        raise CadenceScenarioError(
            f"row count changed: {len(before)} -> {len(after)}; cadence scaling "
            "must never add or remove a transaction")

    b = before.set_index(before[s.id_col].astype(str)).sort_index(kind="mergesort")
    a = after.set_index(after[s.id_col].astype(str)).sort_index(kind="mergesort")
    if list(b.index) != list(a.index):
        b_ids, a_ids = set(b.index), set(a.index)
        raise CadenceScenarioError(
            f"transaction IDs changed: {len(b_ids - a_ids)} lost, "
            f"{len(a_ids - b_ids)} new")

    changed: dict[str, int] = {}
    for col in before.columns:
        if col == s.ts_col:
            continue
        lhs = b[col].astype(str).to_numpy()
        rhs = a[col].astype(str).to_numpy()
        n = int((lhs != rhs).sum())
        if n:
            changed[str(col)] = n
    if changed:
        raise CadenceScenarioError(
            f"cadence scaling modified non-temporal column(s) {changed}; only "
            f"'{s.ts_col}' may change")

    b_ns = timestamps_as_ns(b, schema=s)
    a_ns = timestamps_as_ns(a, schema=s)
    moved = int((b_ns != a_ns).sum())
    shift = np.abs(a_ns.astype("int64") - b_ns.astype("int64"))
    return {"rows": int(len(before)),
            "timestamps_changed": moved,
            "timestamps_unchanged": int(len(before) - moved),
            "max_shift_seconds": float(shift.max()) / NS_PER_SECOND if len(shift) else 0.0,
            "non_temporal_columns_verified": [str(c) for c in before.columns
                                              if c != s.ts_col]}


# ----------------------------------------------------------------------
# world construction
# ----------------------------------------------------------------------
def cadence_transform(*, factor: float, anchor: str = "start",
                      group_col: str | None = None,
                      group_values: Sequence[Any] | None = None,
                      schema: SchemaView | None = None
                      ) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Bind one cadence factor into a side-effect-free frame transform.

    Thin closure over the verified primitive. No cadence maths lives here.
    """
    f = _validate_factor(factor)
    a = _validate_anchor(anchor)

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        return scale_group_cadence(frame, factor=f, group_col=group_col,
                                   group_values=(None if group_values is None
                                                 else list(group_values)),
                                   anchor=a, schema=schema)
    return _apply


def build_cadence_world(baseline: pd.DataFrame, *, factor: float,
                        config: CadenceScenarioConfig | None = None,
                        schema: SchemaView | None = None) -> PerturbedWorld:
    """One cadence arm as a provenance-carrying PerturbedWorld."""
    cfg = config or CadenceScenarioConfig()
    s = schema or resolve_schema()
    f = _validate_factor(factor)
    arm_id = f"{cfg.scenario_id}_factor_{factor_tag(f)}"

    return build_world(
        baseline,
        scenario_id=arm_id,
        name=f"{SCENARIO_NAME}: factor={f:g}, anchor={cfg.anchor}",
        transform=cadence_transform(factor=f, anchor=cfg.anchor,
                                    group_col=cfg.group_col,
                                    group_values=cfg.group_values, schema=s),
        seed=None,                       # cadence scaling is deterministic
        parameters={"factor": f, "anchor": cfg.anchor,
                    "group_col": cfg.group_col or s.campaign_col,
                    "group_values": (None if cfg.group_values is None
                                     else [str(v) for v in cfg.group_values])},
        allow_removals=False,
        allow_additions=False,
        notes={"scenario": "S1", "primitive": "scale_group_cadence",
               "identity_arm": f == IDENTITY_FACTOR},
        schema=s,
    )


# ----------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------
def _frame_summary(frame: pd.DataFrame, schema: SchemaView) -> dict[str, Any]:
    return {"transactions": int(len(frame)),
            "attack_transactions": int(attack_mask(frame, schema=schema).sum())}


def _rebuild_metrics(rw: RebuiltWorld) -> dict[str, Any]:
    """Read metrics off the frozen rebuild result. Nothing is computed here."""
    return {"name": getattr(rw, "name", None),
            "transactions": int(getattr(rw, "n_transactions", 0)),
            "population": rw.population(),
            "grouping_metrics": getattr(rw, "grouping_metrics", None),
            "graph_summary": getattr(rw, "graph_summary", None)}


def run_cadence_arm(baseline: pd.DataFrame, *, factor: float,
                    config: CadenceScenarioConfig | None = None,
                    schema: SchemaView | None = None,
                    rebuild_fn: RebuildFn | None = None) -> dict[str, Any]:
    """Perturb -> validate -> rebuild -> collect, for a single factor."""
    cfg = config or CadenceScenarioConfig()
    s = schema or resolve_schema()
    do_rebuild = rebuild_fn or rebuild_world
    f = _validate_factor(factor)

    world = build_cadence_world(baseline, factor=f, config=cfg, schema=s)
    invariants = assert_only_timestamps_changed(baseline, world.frame, schema=s)

    before = cadence_profile(baseline, group_col=cfg.group_col,
                             group_values=cfg.group_values, schema=s)
    after = cadence_profile(world.frame, group_col=cfg.group_col,
                            group_values=cfg.group_values, schema=s)
    ratios = cadence_span_ratios(before, after)

    rebuilt = do_rebuild(world.frame, name=world.scenario_id,
                         **cfg.rebuild_kwargs())

    return {"factor": f,
            "anchor": cfg.anchor,
            "arm_id": world.scenario_id,
            "is_identity": f == IDENTITY_FACTOR,
            "world": world.describe(),
            "frame": _frame_summary(world.frame, s),
            "cadence": {"groups_measured": len(before),
                        "span_ratio_by_group": ratios,
                        "median_span_ratio": _median_ratio(ratios),
                        "profile_before": before,
                        "profile_after": after},
            "invariants": invariants,
            "rebuild": _rebuild_metrics(rebuilt)}


def _control_arm(baseline: pd.DataFrame, *, config: CadenceScenarioConfig,
                 schema: SchemaView,
                 rebuild_fn: RebuildFn | None) -> dict[str, Any]:
    """Unperturbed control, used only when 1.0 is absent from the grid."""
    do_rebuild = rebuild_fn or rebuild_world
    world = baseline_world(baseline, scenario_id=f"{config.scenario_id}_control",
                           name=f"{SCENARIO_NAME}: unperturbed control",
                           schema=schema)
    rebuilt = do_rebuild(world.frame, name=world.scenario_id,
                         **config.rebuild_kwargs())
    profile = cadence_profile(baseline, group_col=config.group_col,
                              group_values=config.group_values, schema=schema)
    return {"factor": IDENTITY_FACTOR,
            "anchor": config.anchor,
            "arm_id": world.scenario_id,
            "is_identity": True,
            "world": world.describe(),
            "frame": _frame_summary(world.frame, schema),
            "cadence": {"groups_measured": len(profile),
                        "span_ratio_by_group": {g: 1.0 for g in profile},
                        "median_span_ratio": 1.0 if profile else None,
                        "profile_before": profile,
                        "profile_after": profile},
            "invariants": {"rows": int(len(baseline)), "timestamps_changed": 0,
                           "timestamps_unchanged": int(len(baseline)),
                           "max_shift_seconds": 0.0,
                           "non_temporal_columns_verified": [
                               str(c) for c in baseline.columns
                               if c != schema.ts_col]},
            "rebuild": _rebuild_metrics(rebuilt)}


def _compare(arm: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    """Baseline-relative deltas over whatever the rebuild layer reported."""
    out: dict[str, Any] = {"factor": arm["factor"],
                           "arm_id": arm["arm_id"],
                           "median_span_ratio": arm["cadence"]["median_span_ratio"],
                           "population_delta": {}, "grouping_delta": {}}

    a_pop = arm["rebuild"]["population"] or {}
    c_pop = control["rebuild"]["population"] or {}
    for key, a_val in a_pop.items():
        c_val = c_pop.get(key)
        if isinstance(a_val, (int, float)) and isinstance(c_val, (int, float)):
            out["population_delta"][key] = {
                "control": c_val, "perturbed": a_val,
                "delta": a_val - c_val,
                "ratio": (a_val / c_val) if c_val else None}

    a_grp = arm["rebuild"]["grouping_metrics"] or {}
    c_grp = control["rebuild"]["grouping_metrics"] or {}
    if isinstance(a_grp, Mapping) and isinstance(c_grp, Mapping):
        for key, a_val in a_grp.items():
            c_val = c_grp.get(key)
            if isinstance(a_val, (int, float)) and not isinstance(a_val, bool) \
                    and isinstance(c_val, (int, float)) and not isinstance(c_val, bool):
                out["grouping_delta"][str(key)] = {
                    "control": c_val, "perturbed": a_val,
                    "delta": a_val - c_val,
                    "ratio": (a_val / c_val) if c_val else None}
    return out


# ----------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------
def run_cadence_scenario(baseline: pd.DataFrame, *,
                         config: CadenceScenarioConfig | None = None,
                         schema: SchemaView | None = None,
                         rebuild_fn: RebuildFn | None = None,
                         strict: bool = True) -> dict[str, Any]:
    """Run the full S1 cadence grid and return a JSON-serializable comparison.

    strict=True (default) lets an arm failure propagate. strict=False records
    the failure on the arm and continues -- it never fabricates metrics, and
    all_arms_ok goes false.
    """
    cfg = config or CadenceScenarioConfig()
    s = schema or resolve_schema()

    arms: list[dict[str, Any]] = []
    errors: list[str] = []
    for factor in cfg.factors:
        try:
            arms.append(run_cadence_arm(baseline, factor=factor, config=cfg,
                                        schema=s, rebuild_fn=rebuild_fn))
        except (CadenceScenarioError, PerturbationError, WorldError) as exc:
            if strict:
                raise
            msg = f"factor={factor:g}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            arms.append({"factor": float(factor), "anchor": cfg.anchor,
                         "arm_id": f"{cfg.scenario_id}_factor_{factor_tag(factor)}",
                         "is_identity": float(factor) == IDENTITY_FACTOR,
                         "status": "error", "error": msg})

    ok_arms = [a for a in arms if a.get("status") != "error"]
    control = next((a for a in ok_arms if a["is_identity"]), None)
    control_source = "identity_arm"
    if control is None and ok_arms:
        control = _control_arm(baseline, config=cfg, schema=s,
                               rebuild_fn=rebuild_fn)
        control_source = "unperturbed_control"

    comparisons = ([_compare(a, control) for a in ok_arms]
                   if control is not None else [])

    result = {
        "schema": RESULT_SCHEMA,
        "scenario_id": cfg.scenario_id,
        "name": SCENARIO_NAME,
        "question": ("does detection survive when coordinated groups run "
                     "faster or slower, with everything else held fixed?"),
        "config": cfg.as_dict(),
        "schema_view": s.as_dict(),
        "baseline": _frame_summary(baseline, s),
        "control_source": control_source,
        "control": control,
        "arms": arms,
        "comparisons": comparisons,
        "scoring": {"performed": False,
                    "reason": ("no serialized Phase 4A ScorerReference exists; "
                               "S1 reports candidate-formation and grouping "
                               "metrics only and does not fit a scorer")},
        "errors": errors,
        "all_arms_ok": not errors,
    }
    log.info("S1 cadence: %s arm(s), ok=%s", len(arms), not errors)
    return to_json_safe(result)
