"""CONFLUX Phase 4B -- S2: ENTITY REUSE ROBUSTNESS.

QUESTION
--------
Coordinated campaigns are detectable largely BECAUSE they reuse entities. What
happens when they reuse fewer? This scenario severs a controlled fraction of
one entity column's shared values on attack rows and re-runs the frozen
pipeline. Nothing else about any transaction changes.

PIPELINE
--------
    baseline frame
        -> weaken_entity_reuse(fraction, seed)   (perturbations.py, unmodified)
        -> build_world(...)                      (world.py, invariants)
        -> assert_only_entity_column_changed()   (scenario-level guard, here)
        -> rebuild_world(...)                    (rebuild.py, frozen pipeline)
        -> population / grouping metrics
        -> JSON-serializable comparison

NO PARALLEL PIPELINE. No entity rewriting happens in this file; every token is
produced by weaken_entity_reuse. No metric is defined here that the rebuild
layer already reports.

IDENTITY ARM
------------
fraction == 0.0 makes k == 0 inside weaken_entity_reuse, which returns an
unmodified copy. It is the control arm, exactly as factor == 1.0 is for S1.

TARGET COLUMN
-------------
SchemaView.entity holds LOGICAL entity names ('card', 'ip', ...), not column
names, so find_entity_column() is ambiguous by default. The target is resolved
through world.entity_column(), the verified logical -> column map.

SCORING
-------
Deliberately absent, matching S1: no serialized Phase 4A ScorerReference exists
and fitting one here would invent a scorer.
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
    PerturbationError, SchemaView, attack_mask, resolve_schema,
    weaken_entity_reuse,
)
from conflux.robustness.rebuild import RebuiltWorld, rebuild_world
from conflux.robustness.world import (
    PerturbedWorld, WorldError, baseline_world, build_world, entity_column,
    to_json_safe,
)

log = logging.getLogger("conflux.robustness.scenario_entity_reuse")

SCENARIO_ID = "S2_entity_reuse"
SCENARIO_NAME = "S2 entity reuse robustness"
RESULT_SCHEMA = "conflux.robustness.scenario_entity_reuse.v1"

#: Fraction of TARGETED rows whose shared entity value is severed.
#: 0.0 = control (weaken_entity_reuse returns an unmodified copy).
DEFAULT_REUSE_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
IDENTITY_FRACTION: float = 0.0

#: Logical entity type. Resolved to a real column via world.entity_column().
DEFAULT_ENTITY_TYPE: str = "card"

#: weaken_entity_reuse requires a seed; Phase 4B must be reproducible.
DEFAULT_SEED: int = 20240201

_NULL_GROUPS = frozenset({"", "nan", "None"})

RebuildFn = Callable[..., RebuiltWorld]


class EntityReuseScenarioError(ValueError):
    """S2 configuration is invalid, or a world violates an S2 invariant."""


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
def _validate_fraction(fraction: Any) -> float:
    try:
        value = float(fraction)
    except (TypeError, ValueError) as exc:
        raise EntityReuseScenarioError(
            f"fraction must be a number, got {fraction!r}") from exc
    if not math.isfinite(value):
        raise EntityReuseScenarioError(
            f"fraction must be finite, got {fraction!r}")
    if not 0.0 <= value <= 1.0:
        raise EntityReuseScenarioError(
            f"fraction must be in [0, 1] (see weaken_entity_reuse), got {value}")
    return value


def _validate_seed(seed: Any) -> int:
    if seed is None or isinstance(seed, bool):
        raise EntityReuseScenarioError(
            "seed must be an int; weaken_entity_reuse refuses seed=None")
    try:
        return int(seed)
    except (TypeError, ValueError) as exc:
        raise EntityReuseScenarioError(
            f"seed must be an int, got {seed!r}") from exc


@dataclass(frozen=True)
class EntityReuseScenarioConfig:
    """Everything that makes an S2 run reproducible."""

    fractions: tuple[float, ...] = DEFAULT_REUSE_FRACTIONS
    entity_type: str = DEFAULT_ENTITY_TYPE      # logical: card / device / ip / ...
    entity_column: str | None = None            # explicit override, wins if set
    campaign_values: tuple[Any, ...] | None = None   # None -> all attack rows
    seed: int = DEFAULT_SEED
    scenario_id: str = SCENARIO_ID
    min_size: int = 2
    world_dir: Path | None = None
    keep_world_file: bool = False
    strict_alignment: bool = True
    graph_config: Any = None
    candidate_config: Any = None

    def __post_init__(self) -> None:
        fractions = tuple(self.fractions or ())
        if not fractions:
            raise EntityReuseScenarioError("fractions must not be empty")
        object.__setattr__(self, "fractions",
                           tuple(_validate_fraction(f) for f in fractions))
        object.__setattr__(self, "seed", _validate_seed(self.seed))

        if self.entity_column is None:
            # Fail at construction, not three layers down inside the primitive.
            try:
                entity_column(self.entity_type)
            except WorldError as exc:
                raise EntityReuseScenarioError(str(exc)) from exc
        elif not str(self.entity_column).strip():
            raise EntityReuseScenarioError("entity_column must not be blank")

        if self.campaign_values is not None:
            values = tuple(self.campaign_values)
            if not values:
                raise EntityReuseScenarioError(
                    "campaign_values must be None or a non-empty sequence")
            object.__setattr__(self, "campaign_values", values)
        if int(self.min_size) < 2:
            raise EntityReuseScenarioError("min_size must be >= 2")

    def resolve_entity_column(self) -> str:
        """The real dataframe column this run will weaken."""
        if self.entity_column is not None:
            return str(self.entity_column)
        try:
            return entity_column(self.entity_type)
        except WorldError as exc:
            raise EntityReuseScenarioError(str(exc)) from exc

    def rebuild_kwargs(self) -> dict[str, Any]:
        return {"graph_config": self.graph_config,
                "candidate_config": self.candidate_config,
                "min_size": int(self.min_size),
                "world_dir": self.world_dir,
                "keep_world_file": bool(self.keep_world_file),
                "strict_alignment": bool(self.strict_alignment)}

    def as_dict(self) -> dict[str, Any]:
        return {"fractions": list(self.fractions),
                "entity_type": self.entity_type,
                "entity_column": self.resolve_entity_column(),
                "entity_column_explicit": self.entity_column is not None,
                "campaign_values": (None if self.campaign_values is None
                                    else [str(v) for v in self.campaign_values]),
                "seed": int(self.seed),
                "scenario_id": self.scenario_id,
                "min_size": int(self.min_size),
                "world_dir": None if self.world_dir is None else str(self.world_dir),
                "keep_world_file": bool(self.keep_world_file),
                "strict_alignment": bool(self.strict_alignment),
                "graph_config": None if self.graph_config is None else str(self.graph_config),
                "candidate_config": None if self.candidate_config is None else str(self.candidate_config)}


def fraction_tag(fraction: float) -> str:
    """Filesystem-safe arm suffix: 0.25 -> '0p25', 1.0 -> '1'."""
    return f"{float(fraction):g}".replace(".", "p").replace("-", "neg")


# ----------------------------------------------------------------------
# row targeting
# ----------------------------------------------------------------------
def target_mask(frame: pd.DataFrame, *,
                campaign_values: Sequence[Any] | None = None,
                schema: SchemaView | None = None) -> pd.Series:
    """Rows whose entity reuse will be severed: attack rows, optionally
    narrowed to specific campaigns. Ground truth is READ for selection only."""
    s = schema or resolve_schema()
    mask = attack_mask(frame, schema=s)
    if campaign_values is not None:
        if s.campaign_col not in frame.columns:
            raise EntityReuseScenarioError(
                f"campaign column '{s.campaign_col}' not in frame")
        wanted = {str(v) for v in campaign_values}
        mask = mask & frame[s.campaign_col].astype(str).isin(wanted)
    return mask


# ----------------------------------------------------------------------
# reuse measurement (reporting only -- nothing is rewritten here)
# ----------------------------------------------------------------------
def entity_reuse_profile(frame: pd.DataFrame, *, entity_column: str,
                         mask: pd.Series | None = None,
                         group_col: str | None = None,
                         schema: SchemaView | None = None) -> dict[str, Any]:
    """How concentrated is `entity_column` across the targeted rows?

    reuse_index = 1 - distinct_values / rows. 1.0 means every targeted row
    shares one value; 0.0 means no value is reused at all.
    """
    s = schema or resolve_schema()
    if entity_column not in frame.columns:
        raise EntityReuseScenarioError(
            f"entity column '{entity_column}' not in frame; columns are "
            f"{list(frame.columns)}")
    gcol = group_col or s.campaign_col
    sel = (target_mask(frame, schema=s) if mask is None else mask)
    sel_arr = np.asarray(sel, dtype=bool)
    if sel_arr.shape != (len(frame),):
        raise EntityReuseScenarioError(
            f"mask length {sel_arr.shape} does not match frame length {len(frame)}")

    rows = frame.loc[sel_arr]
    n = int(len(rows))
    if n == 0:
        return {"entity_column": entity_column, "targeted_rows": 0,
                "distinct_values": 0, "shared_value_rows": 0,
                "shared_row_fraction": None, "max_transactions_per_value": 0,
                "mean_transactions_per_value": None, "reuse_index": None,
                "per_group": {}}

    counts = rows[entity_column].astype(str).value_counts()
    distinct = int(counts.size)
    shared_rows = int(counts[counts > 1].sum())

    per_group: dict[str, Any] = {}
    if gcol in rows.columns:
        for g, block in rows.groupby(rows[gcol].astype(str), sort=True):
            if g in _NULL_GROUPS:
                continue
            gc = block[entity_column].astype(str).value_counts()
            per_group[str(g)] = {
                "transactions": int(len(block)),
                "distinct_values": int(gc.size),
                "max_transactions_per_value": int(gc.max()),
                "reuse_index": 1.0 - (int(gc.size) / int(len(block))),
            }

    return {"entity_column": entity_column,
            "targeted_rows": n,
            "distinct_values": distinct,
            "shared_value_rows": shared_rows,
            "shared_row_fraction": shared_rows / n,
            "max_transactions_per_value": int(counts.max()),
            "mean_transactions_per_value": n / distinct,
            "reuse_index": 1.0 - (distinct / n),
            "per_group": per_group}


def reuse_reduction(before: Mapping[str, Any],
                    after: Mapping[str, Any]) -> dict[str, Any]:
    """Before/after view of how much reuse the perturbation actually removed."""
    def _delta(key: str) -> float | None:
        b, a = before.get(key), after.get(key)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            return float(a) - float(b)
        return None

    return {"reuse_index_before": before.get("reuse_index"),
            "reuse_index_after": after.get("reuse_index"),
            "reuse_index_delta": _delta("reuse_index"),
            "distinct_values_before": before.get("distinct_values"),
            "distinct_values_after": after.get("distinct_values"),
            "distinct_values_increase": _delta("distinct_values"),
            "shared_row_fraction_before": before.get("shared_row_fraction"),
            "shared_row_fraction_after": after.get("shared_row_fraction"),
            "shared_row_fraction_delta": _delta("shared_row_fraction"),
            "max_transactions_per_value_before": before.get("max_transactions_per_value"),
            "max_transactions_per_value_after": after.get("max_transactions_per_value")}


# ----------------------------------------------------------------------
# invariants -- only the target entity column may move
# ----------------------------------------------------------------------
def assert_only_entity_column_changed(before: pd.DataFrame, after: pd.DataFrame,
                                      *, entity_column: str,
                                      mask: pd.Series | None = None,
                                      schema: SchemaView | None = None
                                      ) -> dict[str, Any]:
    """Prove the perturbation touched ONE column, on TARGETED rows only.

    world.build_world already guards IDs, labels, campaign IDs and row
    accounting. This adds the S2-specific claim: timestamps, amounts, auth
    outcomes and every non-target entity column are byte-identical, and no
    untargeted row lost its entity value.
    """
    s = schema or resolve_schema()
    if entity_column not in before.columns:
        raise EntityReuseScenarioError(
            f"entity column '{entity_column}' not in frame")
    if list(before.columns) != list(after.columns):
        raise EntityReuseScenarioError(
            f"column set/order changed: {list(before.columns)} -> "
            f"{list(after.columns)}")
    if len(before) != len(after):
        raise EntityReuseScenarioError(
            f"row count changed: {len(before)} -> {len(after)}; weakening "
            "entity reuse must never add or remove a transaction")

    b = before.set_index(before[s.id_col].astype(str)).sort_index(kind="mergesort")
    a = after.set_index(after[s.id_col].astype(str)).sort_index(kind="mergesort")
    if list(b.index) != list(a.index):
        b_ids, a_ids = set(b.index), set(a.index)
        raise EntityReuseScenarioError(
            f"transaction IDs changed: {len(b_ids - a_ids)} lost, "
            f"{len(a_ids - b_ids)} new")

    changed: dict[str, int] = {}
    for col in before.columns:
        if col == entity_column:
            continue
        lhs = b[col].astype(str).to_numpy()
        rhs = a[col].astype(str).to_numpy()
        n = int((lhs != rhs).sum())
        if n:
            changed[str(col)] = n
    if changed:
        raise EntityReuseScenarioError(
            f"weakening entity reuse modified column(s) {changed}; only "
            f"'{entity_column}' may change")

    diff = (b[entity_column].astype(str).to_numpy()
            != a[entity_column].astype(str).to_numpy())
    changed_ids = set(np.asarray(b.index)[diff])

    outside = 0
    if mask is not None:
        sel = np.asarray(mask, dtype=bool)
        if sel.shape != (len(before),):
            raise EntityReuseScenarioError(
                "mask length does not match frame length")
        targeted_ids = set(before.loc[sel, s.id_col].astype(str))
        stray = changed_ids - targeted_ids
        outside = len(stray)
        if stray:
            raise EntityReuseScenarioError(
                f"{outside} untargeted transaction(s) had '{entity_column}' "
                f"rewritten, e.g. {sorted(stray)[:5]}")

    return {"rows": int(len(before)),
            "entity_column": entity_column,
            "entity_values_rewritten": int(len(changed_ids)),
            "entity_values_unchanged": int(len(before) - len(changed_ids)),
            "rewritten_outside_target": outside,
            "columns_verified_unchanged": [str(c) for c in before.columns
                                           if c != entity_column]}


# ----------------------------------------------------------------------
# world construction
# ----------------------------------------------------------------------
def entity_reuse_transform(*, entity_column: str, fraction: float, seed: int,
                           campaign_values: Sequence[Any] | None = None,
                           schema: SchemaView | None = None
                           ) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Bind one severity level into a side-effect-free frame transform.

    Thin closure over the verified primitive. No entity rewriting lives here.
    """
    frac = _validate_fraction(fraction)
    sd = _validate_seed(seed)

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        s = schema or resolve_schema()
        mask = target_mask(frame, campaign_values=campaign_values, schema=s)
        return weaken_entity_reuse(frame, entity_column=entity_column,
                                   fraction=frac, seed=sd, mask=mask, schema=s)
    return _apply


def build_entity_reuse_world(baseline: pd.DataFrame, *, fraction: float,
                             config: EntityReuseScenarioConfig | None = None,
                             schema: SchemaView | None = None) -> PerturbedWorld:
    """One severity arm as a provenance-carrying PerturbedWorld."""
    cfg = config or EntityReuseScenarioConfig()
    s = schema or resolve_schema()
    frac = _validate_fraction(fraction)
    col = cfg.resolve_entity_column()
    if col not in baseline.columns:
        raise EntityReuseScenarioError(
            f"entity column '{col}' not in frame; columns are "
            f"{list(baseline.columns)}")
    arm_id = f"{cfg.scenario_id}_fraction_{fraction_tag(frac)}"

    return build_world(
        baseline,
        scenario_id=arm_id,
        name=f"{SCENARIO_NAME}: {col} fraction={frac:g}",
        transform=entity_reuse_transform(entity_column=col, fraction=frac,
                                         seed=cfg.seed,
                                         campaign_values=cfg.campaign_values,
                                         schema=s),
        seed=cfg.seed,
        parameters={"fraction": frac, "entity_column": col,
                    "entity_type": cfg.entity_type,
                    "campaign_values": (None if cfg.campaign_values is None
                                        else [str(v) for v in cfg.campaign_values])},
        allow_removals=False,
        allow_additions=False,
        notes={"scenario": "S2", "primitive": "weaken_entity_reuse",
               "identity_arm": frac == IDENTITY_FRACTION},
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


def run_entity_reuse_arm(baseline: pd.DataFrame, *, fraction: float,
                         config: EntityReuseScenarioConfig | None = None,
                         schema: SchemaView | None = None,
                         rebuild_fn: RebuildFn | None = None) -> dict[str, Any]:
    """Perturb -> validate -> rebuild -> collect, for a single severity level."""
    cfg = config or EntityReuseScenarioConfig()
    s = schema or resolve_schema()
    do_rebuild = rebuild_fn or rebuild_world
    frac = _validate_fraction(fraction)
    col = cfg.resolve_entity_column()

    mask = target_mask(baseline, campaign_values=cfg.campaign_values, schema=s)
    if int(mask.sum()) == 0:
        raise EntityReuseScenarioError(
            "no rows selected for weakening; check campaign_values and the "
            f"label column '{s.label_col}'")

    world = build_entity_reuse_world(baseline, fraction=frac, config=cfg,
                                     schema=s)
    invariants = assert_only_entity_column_changed(
        baseline, world.frame, entity_column=col, mask=mask, schema=s)

    before = entity_reuse_profile(baseline, entity_column=col, mask=mask,
                                  schema=s)
    after = entity_reuse_profile(world.frame, entity_column=col, mask=mask,
                                 schema=s)

    rebuilt = do_rebuild(world.frame, name=world.scenario_id,
                         **cfg.rebuild_kwargs())

    return {"fraction": frac,
            "entity_column": col,
            "seed": int(cfg.seed),
            "arm_id": world.scenario_id,
            "is_identity": frac == IDENTITY_FRACTION,
            "world": world.describe(),
            "frame": _frame_summary(world.frame, s),
            "reuse": {"targeted_rows": before["targeted_rows"],
                      "profile_before": before,
                      "profile_after": after,
                      "reduction": reuse_reduction(before, after)},
            "invariants": invariants,
            "rebuild": _rebuild_metrics(rebuilt)}


def _control_arm(baseline: pd.DataFrame, *,
                 config: EntityReuseScenarioConfig, schema: SchemaView,
                 rebuild_fn: RebuildFn | None) -> dict[str, Any]:
    """Unperturbed control, used only when 0.0 is absent from the grid."""
    do_rebuild = rebuild_fn or rebuild_world
    col = config.resolve_entity_column()
    world = baseline_world(baseline,
                           scenario_id=f"{config.scenario_id}_control",
                           name=f"{SCENARIO_NAME}: unperturbed control",
                           schema=schema)
    rebuilt = do_rebuild(world.frame, name=world.scenario_id,
                         **config.rebuild_kwargs())
    mask = target_mask(baseline, campaign_values=config.campaign_values,
                       schema=schema)
    profile = entity_reuse_profile(baseline, entity_column=col, mask=mask,
                                   schema=schema)
    return {"fraction": IDENTITY_FRACTION,
            "entity_column": col,
            "seed": int(config.seed),
            "arm_id": world.scenario_id,
            "is_identity": True,
            "world": world.describe(),
            "frame": _frame_summary(world.frame, schema),
            "reuse": {"targeted_rows": profile["targeted_rows"],
                      "profile_before": profile,
                      "profile_after": profile,
                      "reduction": reuse_reduction(profile, profile)},
            "invariants": {"rows": int(len(baseline)),
                           "entity_column": col,
                           "entity_values_rewritten": 0,
                           "entity_values_unchanged": int(len(baseline)),
                           "rewritten_outside_target": 0,
                           "columns_verified_unchanged": [
                               str(c) for c in baseline.columns if c != col]},
            "rebuild": _rebuild_metrics(rebuilt)}


def _compare(arm: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    """Control-relative deltas over whatever the rebuild layer reported."""
    out: dict[str, Any] = {
        "fraction": arm["fraction"],
        "arm_id": arm["arm_id"],
        "entity_column": arm["entity_column"],
        "reuse_index_delta": arm["reuse"]["reduction"]["reuse_index_delta"],
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
def run_entity_reuse_scenario(baseline: pd.DataFrame, *,
                              config: EntityReuseScenarioConfig | None = None,
                              schema: SchemaView | None = None,
                              rebuild_fn: RebuildFn | None = None,
                              strict: bool = True) -> dict[str, Any]:
    """Run the full S2 severity grid and return a JSON-serializable comparison.

    strict=True (default) lets an arm failure propagate. strict=False records
    the failure on the arm and continues -- it never fabricates metrics, and
    all_arms_ok goes false.
    """
    cfg = config or EntityReuseScenarioConfig()
    s = schema or resolve_schema()

    arms: list[dict[str, Any]] = []
    errors: list[str] = []
    for fraction in cfg.fractions:
        try:
            arms.append(run_entity_reuse_arm(baseline, fraction=fraction,
                                             config=cfg, schema=s,
                                             rebuild_fn=rebuild_fn))
        except (EntityReuseScenarioError, PerturbationError, WorldError) as exc:
            if strict:
                raise
            msg = f"fraction={fraction:g}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            arms.append({"fraction": float(fraction),
                         "entity_column": cfg.resolve_entity_column(),
                         "seed": int(cfg.seed),
                         "arm_id": f"{cfg.scenario_id}_fraction_{fraction_tag(fraction)}",
                         "is_identity": float(fraction) == IDENTITY_FRACTION,
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
        "question": ("does detection survive when coordinated transactions "
                     "share fewer entities, with everything else held fixed?"),
        "config": cfg.as_dict(),
        "schema_view": s.as_dict(),
        "baseline": _frame_summary(baseline, s),
        "control_source": control_source,
        "control": control,
        "arms": arms,
        "comparisons": comparisons,
        "scoring": {"performed": False,
                    "reason": ("no serialized Phase 4A ScorerReference exists; "
                               "S2 reports candidate-formation and grouping "
                               "metrics only and does not fit a scorer")},
        "errors": errors,
        "all_arms_ok": not errors,
    }
    log.info("S2 entity reuse: %s arm(s) on '%s', ok=%s",
             len(arms), cfg.resolve_entity_column(), not errors)
    return to_json_safe(result)
