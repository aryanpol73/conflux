"""CONFLUX Phase 4B -- perturbed world construction (TRANSACTION LAYER ONLY).

RESPONSIBILITY BOUNDARY
-----------------------
world.py     : baseline load, perturbation application, provenance, invariants.
rebuild.py   : frame -> graph -> candidates -> features -> labels (FROZEN).
scorer       : frozen Phase 4A ScorerReference, transform() only.

This module never builds a graph, never scores, never computes a metric and
never calls fit(). It owns exactly one question: "is this perturbed frame a
legitimate world, and can I prove how it differs from the baseline?"

DETERMINISM / PROVENANCE
------------------------
Every world carries its scenario id, parameters, seed, the fingerprint of the
baseline it came from and its own fingerprint. Two runs with identical inputs
produce identical fingerprints, which is what the scenario tests assert on.

NO MUTATION
-----------
build_world() fingerprints the baseline before and after the transform and
raises if the transform touched it. A perturbation that mutates its input is a
bug, not a scenario.

GROUND TRUTH
------------
label / campaign_id are carried, never computed from and never used to build a
feature. assert_labels_preserved() is the guard scenarios use to prove it.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from conflux.config import RAW_DATASET_PATH
from conflux.graph.config import (
    ATTRIBUTE_COLUMNS, ENTITY_COLUMNS, ID_COL, STRUCTURAL_COLUMNS, TS_COL,
)
from conflux.robustness.perturbations import (
    PerturbationError, SchemaView, attack_mask, resolve_schema,
    summarize_perturbation,
)
from conflux.robustness.rebuild import RebuiltWorld, rebuild_world

log = logging.getLogger("conflux.robustness.world")

WORLD_SCHEMA_VERSION = "conflux.robustness.world.v1"

# Logical entity type -> real dataframe column. ENTITY_COLUMNS is a MAPPING;
# SchemaView.entity holds the logical keys, which are NOT column names. Every
# scenario must go through here instead of guessing.
ENTITY_COLUMN_MAP: dict[str, str] = dict(ENTITY_COLUMNS)
LINKING_ENTITY_TYPES: tuple[str, ...] = ("card", "device", "ip")
CONTEXT_ENTITY_TYPES: tuple[str, ...] = ("bin", "merchant")


class WorldError(RuntimeError):
    """A perturbed world violates an invariant the frozen pipeline relies on."""


# ----------------------------------------------------------------------
# JSON safety -- used by every downstream scenario report
# ----------------------------------------------------------------------
def to_json_safe(obj: Any) -> Any:
    """Recursively convert to something json.dumps() accepts without default=."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, BaseException):
        return f"{type(obj).__name__}: {obj}"
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": {"rows": int(len(obj)),
                                  "columns": [str(c) for c in obj.columns]}}
    if isinstance(obj, pd.Series):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_json_safe(v) for v in obj]
    return str(obj)


# ----------------------------------------------------------------------
# schema helpers -- resolved, never hard-coded
# ----------------------------------------------------------------------
def entity_column(entity_type: str) -> str:
    try:
        return ENTITY_COLUMN_MAP[entity_type]
    except KeyError as exc:
        raise WorldError(
            f"unknown entity type '{entity_type}'; known: "
            f"{sorted(ENTITY_COLUMN_MAP)}") from exc


def entity_columns(entity_types: Sequence[str]) -> tuple[str, ...]:
    return tuple(entity_column(t) for t in entity_types)


def linking_entity_columns() -> tuple[str, ...]:
    """Columns that can actually join two transactions (card / device / ip)."""
    return entity_columns(LINKING_ENTITY_TYPES)


def context_entity_columns() -> tuple[str, ...]:
    """bin / merchant. Context only -- never a connectivity mechanism."""
    return entity_columns(CONTEXT_ENTITY_TYPES)


def world_frame_columns(schema: SchemaView | None = None) -> tuple[str, ...]:
    """The columns rebuild.write_world_csv() will demand."""
    s = schema or resolve_schema()
    return (*STRUCTURAL_COLUMNS, *ATTRIBUTE_COLUMNS, s.label_col, s.campaign_col)


# ----------------------------------------------------------------------
# baseline
# ----------------------------------------------------------------------
def load_baseline_frame(path: str | Path | None = None) -> pd.DataFrame:
    """Read the frozen raw dataset with EXACTLY rebuild_baseline()'s semantics.

    dtype=str + keep_default_na=False so that an empty campaign_id stays "" and
    never becomes NaN, and so timestamps survive byte-identically. amount is the
    single numeric column, matching rebuild_baseline().
    """
    p = Path(path) if path is not None else Path(RAW_DATASET_PATH)
    if not p.is_file():
        raise WorldError(f"raw dataset not found: {p}")
    frame = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[],
                        low_memory=False)
    if "amount" in frame.columns:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="raise")
    log.info("baseline frame: %s rows x %s columns from %s",
             len(frame), frame.shape[1], p)
    return frame


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Order-independent content hash. Two frames differing only in row order
    hash the same, so a perturbation that merely re-sorts is not a change."""
    cols = sorted(str(c) for c in frame.columns)
    canon = frame.loc[:, cols]
    if ID_COL in canon.columns:
        canon = canon.sort_values(ID_COL, kind="mergesort")
    payload = canon.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ----------------------------------------------------------------------
# invariants
# ----------------------------------------------------------------------
def validate_world_frame(frame: pd.DataFrame, *,
                         schema: SchemaView | None = None,
                         context: str = "world") -> None:
    """Everything rebuild.validate_world_frame() checks, checked EARLY.

    Failing here gives a scenario-level error message instead of an opaque
    failure three layers down inside the frozen pipeline.
    """
    s = schema or resolve_schema()
    required = world_frame_columns(s)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise WorldError(f"{context}: missing required column(s) {missing}")
    if frame.empty:
        raise WorldError(f"{context}: frame is empty")
    for col in STRUCTURAL_COLUMNS:
        col_s = frame[col]
        bad = int(col_s.isna().sum() + (col_s.astype(str).str.strip() == "").sum())
        if bad:
            raise WorldError(
                f"{context}: {bad} blank/NaN value(s) in structural column '{col}'")
    dup = int(frame[ID_COL].duplicated().sum())
    if dup:
        raise WorldError(f"{context}: {dup} duplicate {ID_COL}")
    if "amount" in frame.columns and pd.to_numeric(
            frame["amount"], errors="coerce").isna().any():
        raise WorldError(f"{context}: non-numeric or NaN amount")


def assert_labels_preserved(before: pd.DataFrame, after: pd.DataFrame, *,
                            allow_removals: bool = False,
                            allow_additions: bool = True,
                            additions_must_be_negative: bool = True,
                            schema: SchemaView | None = None) -> dict[str, Any]:
    """Ground-truth integrity guard. Returns a JSON-safe delta description.

    Surviving transactions must keep their exact label and campaign_id. Injected
    transactions must be negative unless a scenario explicitly says otherwise.
    """
    s = schema or resolve_schema()
    b = before.set_index(before[ID_COL].astype(str))
    a = after.set_index(after[ID_COL].astype(str))

    removed = sorted(set(b.index) - set(a.index))
    added = sorted(set(a.index) - set(b.index))
    common = sorted(set(b.index) & set(a.index))

    if removed and not allow_removals:
        raise WorldError(
            f"{len(removed)} transaction(s) removed but allow_removals=False; "
            f"e.g. {removed[:5]}")
    if added and not allow_additions:
        raise WorldError(
            f"{len(added)} transaction(s) added but allow_additions=False; "
            f"e.g. {added[:5]}")

    for col in (s.label_col, s.campaign_col):
        lhs = b.loc[common, col].astype(str).to_numpy()
        rhs = a.loc[common, col].astype(str).to_numpy()
        diff = int((lhs != rhs).sum())
        if diff:
            raise WorldError(
                f"'{col}' changed on {diff} surviving transaction(s); a "
                "perturbation may never rewrite ground truth")

    added_positive = 0
    if added:
        added_rows = after.loc[after[ID_COL].astype(str).isin(set(added))]
        added_positive = int(attack_mask(added_rows, schema=s).sum())
        if additions_must_be_negative and added_positive:
            raise WorldError(
                f"{added_positive} injected transaction(s) carry a positive "
                "label; injected traffic must be benign")

    return {"transactions_removed": len(removed),
            "transactions_added": len(added),
            "transactions_retained": len(common),
            "injected_positive_rows": added_positive}


# ----------------------------------------------------------------------
# the world object
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PerturbedWorld:
    """A transaction world plus the provenance needed to defend it."""

    scenario_id: str
    name: str
    seed: int | None
    parameters: dict[str, Any]
    frame: pd.DataFrame = field(repr=False, compare=False)
    baseline_rows: int
    baseline_fingerprint: str
    fingerprint: str
    perturbation_summary: dict[str, Any]
    label_delta: dict[str, Any]
    notes: dict[str, Any] = field(default_factory=dict)
    schema_version: str = WORLD_SCHEMA_VERSION

    @property
    def n_transactions(self) -> int:
        return int(len(self.frame))

    @property
    def is_baseline(self) -> bool:
        return self.fingerprint == self.baseline_fingerprint

    def describe(self) -> dict[str, Any]:
        """JSON-serializable. Never contains a DataFrame."""
        return to_json_safe({
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "name": self.name,
            "seed": self.seed,
            "parameters": self.parameters,
            "baseline_rows": self.baseline_rows,
            "baseline_fingerprint": self.baseline_fingerprint,
            "world_rows": self.n_transactions,
            "world_fingerprint": self.fingerprint,
            "is_baseline": self.is_baseline,
            "perturbation_summary": self.perturbation_summary,
            "label_delta": self.label_delta,
            "notes": self.notes,
        })

    def rebuild(self, **kwargs: Any) -> RebuiltWorld:
        """Delegate to the FROZEN Phase 4B rebuild seam. No logic duplicated."""
        kwargs.setdefault("name", self.scenario_id)
        return rebuild_world(self.frame, **kwargs)


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------
def build_world(baseline: pd.DataFrame, *,
                scenario_id: str,
                name: str,
                transform: Callable[[pd.DataFrame], pd.DataFrame],
                seed: int | None = None,
                parameters: Mapping[str, Any] | None = None,
                allow_removals: bool = False,
                allow_additions: bool = True,
                additions_must_be_negative: bool = True,
                notes: Mapping[str, Any] | None = None,
                schema: SchemaView | None = None) -> PerturbedWorld:
    """Apply one perturbation pipeline and wrap it with proof of what changed.

    `transform` receives the baseline frame and must RETURN A NEW FRAME. If it
    mutates its argument, the fingerprint check below fails loudly.
    """
    s = schema or resolve_schema()
    validate_world_frame(baseline, schema=s, context=f"{scenario_id}: baseline")

    before_fp = frame_fingerprint(baseline)
    try:
        perturbed = transform(baseline)
    except PerturbationError as exc:
        raise WorldError(
            f"{scenario_id}: perturbation failed -- {type(exc).__name__}: {exc}"
        ) from exc

    if frame_fingerprint(baseline) != before_fp:
        raise WorldError(
            f"{scenario_id}: the transform MUTATED the baseline frame in place; "
            "perturbations must be side-effect free")
    if perturbed is baseline:
        raise WorldError(f"{scenario_id}: the transform returned its input object")
    if not isinstance(perturbed, pd.DataFrame):
        raise WorldError(
            f"{scenario_id}: transform returned {type(perturbed).__name__}, "
            "expected DataFrame")

    if list(perturbed.columns) != list(baseline.columns):
        raise WorldError(
            f"{scenario_id}: column set/order changed; the frozen pipeline "
            f"expects {list(baseline.columns)}")

    validate_world_frame(perturbed, schema=s, context=f"{scenario_id}: world")

    label_delta = assert_labels_preserved(
        baseline, perturbed, allow_removals=allow_removals,
        allow_additions=allow_additions,
        additions_must_be_negative=additions_must_be_negative, schema=s)

    summary = summarize_perturbation(baseline, perturbed, name=scenario_id,
                                     schema=s)

    world = PerturbedWorld(
        scenario_id=scenario_id,
        name=name,
        seed=None if seed is None else int(seed),
        parameters=dict(parameters or {}),
        frame=perturbed,
        baseline_rows=int(len(baseline)),
        baseline_fingerprint=before_fp,
        fingerprint=frame_fingerprint(perturbed),
        perturbation_summary=to_json_safe(summary),
        label_delta=to_json_safe(label_delta),
        notes=dict(notes or {}),
    )
    log.info("world '%s': %s -> %s rows (added=%s removed=%s) fp=%s",
             scenario_id, world.baseline_rows, world.n_transactions,
             label_delta["transactions_added"], label_delta["transactions_removed"],
             world.fingerprint[:12])
    return world


def baseline_world(baseline: pd.DataFrame, *,
                   scenario_id: str = "S0_baseline",
                   name: str = "unperturbed baseline",
                   schema: SchemaView | None = None) -> PerturbedWorld:
    """The control world. Identity transform, so parity must reproduce exactly."""
    return build_world(baseline, scenario_id=scenario_id, name=name,
                       transform=lambda f: f.copy(), seed=None,
                       parameters={}, allow_removals=False,
                       allow_additions=False,
                       notes={"role": "control"}, schema=schema)


def chain(*steps: Callable[[pd.DataFrame], pd.DataFrame]
          ) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Compose perturbation steps left to right into one transform."""
    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame
        for step in steps:
            out = step(out)
        return out
    return _apply
