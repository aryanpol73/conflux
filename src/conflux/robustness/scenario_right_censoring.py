"""Phase 4B Scenario S5 - Right Censoring Robustness.

Truncates the observation window with the real ``right_censor`` primitive and
pushes every arm through the established world/rebuild pipeline.

Contract notes derived from the primitive source:

* ``right_censor`` is deterministic and takes **no seed**; it consumes exactly
  one of ``cutoff_ts_ns`` or ``keep_fraction``.
* Row count intentionally changes.  Rows are only ever *dropped* - labels,
  campaign ids, transaction ids and entity columns of surviving rows are
  untouched.
* The primitive returns ``_sorted_by_id(...)`` with a reset index, so the
  output row *order* differs from the input.  All invariant checks therefore
  join on the schema id column instead of comparing positionally.

Nothing in this module imports or fits the frozen scorer.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from conflux.robustness.perturbations import (
    PerturbationError,
    SchemaView,
    attack_mask,
    describe_schema,
    observation_window_ns,
    resolve_schema,
    right_censor,
    summarize_perturbation,
    timestamps_as_ns,
)
from conflux.robustness.rebuild import rebuild_world

try:  # pragma: no cover - presence depends on the world module revision
    from conflux.robustness.world import build_world
except Exception:  # pragma: no cover
    build_world = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

SCENARIO_ID = "phase4b_s5_right_censoring"
SCENARIO_NAME = "Right Censoring Robustness"
PRIMITIVE_NAME = "right_censor"
CONTROL_ARM_ID = "control"

__all__ = [
    "SCENARIO_ID",
    "SCENARIO_NAME",
    "PRIMITIVE_NAME",
    "CONTROL_ARM_ID",
    "RightCensoringError",
    "RightCensoringConfig",
    "json_safe",
    "primitive_signature",
    "censor_call_kwargs",
    "apply_right_censor",
    "predicted_cutoff_ns",
    "frame_fingerprint",
    "resolve_id_column",
    "resolve_label_column",
    "resolve_campaign_column",
    "resolve_timestamp_column",
    "schema_entity_columns",
    "window_profile",
    "censoring_profile",
    "assert_surviving_rows_unchanged",
    "assert_cutoff_consistency",
    "id_set",
    "check_monotone_nesting",
    "run_right_censoring_arm",
    "run_right_censoring_scenario",
]


class RightCensoringError(RuntimeError):
    """Raised for scenario-level (not primitive-level) contract violations."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _float_tuple(values: Any, *, label: str) -> tuple[float, ...]:
    if values is None:
        raise RightCensoringError(f"{label} must be a sequence of numbers")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RightCensoringError(f"{label} must be a sequence of numbers")
    out: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise RightCensoringError(f"{label} entries must be numbers, got {v!r}")
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            raise RightCensoringError(f"{label} entries must be finite, got {v!r}")
        out.append(f)
    if len(set(out)) != len(out):
        raise RightCensoringError(f"{label} must not contain duplicates: {out!r}")
    return tuple(out)


def _int_tuple(values: Any, *, label: str) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RightCensoringError(f"{label} must be a sequence of integers")
    out: list[int] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise RightCensoringError(f"{label} entries must be integers, got {v!r}")
        out.append(int(v))
    if len(set(out)) != len(out):
        raise RightCensoringError(f"{label} must not contain duplicates: {out!r}")
    return tuple(out)


@dataclass(frozen=True)
class RightCensoringConfig:
    """Severity configuration for the right-censoring scenario.

    Exactly one of ``keep_fractions`` or ``cutoff_ts_ns`` must be populated,
    mirroring the primitive's own ``cutoff_ts_ns`` / ``keep_fraction`` rule.
    """

    keep_fractions: tuple[float, ...] = (0.9, 0.75, 0.5, 0.25)
    cutoff_ts_ns: tuple[int, ...] = ()
    include_control: bool = True
    strict_invariants: bool = True
    require_monotone_nesting: bool = True
    tolerate_empty_attack_side: bool = True
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "keep_fractions", _float_tuple(self.keep_fractions, label="keep_fractions")
        )
        object.__setattr__(
            self, "cutoff_ts_ns", _int_tuple(self.cutoff_ts_ns, label="cutoff_ts_ns")
        )
        if not isinstance(self.extra_kwargs, Mapping):
            raise RightCensoringError("extra_kwargs must be a mapping")

    # -- validation ---------------------------------------------------------

    def validate(self) -> "RightCensoringConfig":
        has_fractions = bool(self.keep_fractions)
        has_cutoffs = bool(self.cutoff_ts_ns)
        if has_fractions == has_cutoffs:
            raise RightCensoringError(
                "supply exactly one of keep_fractions or cutoff_ts_ns "
                f"(got keep_fractions={self.keep_fractions!r}, "
                f"cutoff_ts_ns={self.cutoff_ts_ns!r})"
            )
        for f in self.keep_fractions:
            if not 0.0 < f <= 1.0:
                raise RightCensoringError(
                    f"keep_fractions entries must lie in (0, 1], got {f!r}"
                )
        for name in self.extra_kwargs:
            if not isinstance(name, str):
                raise RightCensoringError("extra_kwargs keys must be strings")
        return self

    # -- derived ------------------------------------------------------------

    @property
    def severity_mode(self) -> str:
        return "keep_fraction" if self.keep_fractions else "cutoff_ts_ns"

    def severity_levels(self) -> tuple[Any, ...]:
        return self.keep_fractions if self.keep_fractions else self.cutoff_ts_ns

    def arm_id(self, level: Any) -> str:
        if self.severity_mode == "keep_fraction":
            return f"keep_fraction={float(level):g}"
        return f"cutoff_ts_ns={int(level)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "keep_fractions": list(self.keep_fractions),
            "cutoff_ts_ns": list(self.cutoff_ts_ns),
            "severity_mode": self.severity_mode,
            "include_control": bool(self.include_control),
            "strict_invariants": bool(self.strict_invariants),
            "require_monotone_nesting": bool(self.require_monotone_nesting),
            "tolerate_empty_attack_side": bool(self.tolerate_empty_attack_side),
            "extra_kwargs": dict(self.extra_kwargs),
        }


# ---------------------------------------------------------------------------
# json safety
# ---------------------------------------------------------------------------


def json_safe(obj: Any) -> Any:
    """Coerce nested structures into json-serializable primitives."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted((json_safe(v) for v in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# schema reflection
# ---------------------------------------------------------------------------

_ID_ATTRS = (
    "id_column",
    "id_col",
    "transaction_id_column",
    "transaction_id_col",
    "txn_id_column",
    "tx_id_column",
    "id",
)
_LABEL_ATTRS = ("label_column", "label_col", "target_column", "label")
_CAMPAIGN_ATTRS = (
    "campaign_column",
    "campaign_col",
    "campaign_id_column",
    "campaign_id_col",
    "campaign",
)
_TS_ATTRS = (
    "timestamp_column",
    "timestamp_col",
    "ts_column",
    "ts_col",
    "time_column",
    "timestamp",
)
_ENTITY_LIST_ATTRS = (
    "entity_columns",
    "entity_cols",
    "shared_entity_columns",
    "entity_column_names",
    "entities",
)
_ENTITY_SINGLE_ATTRS = (
    "card_column",
    "device_column",
    "ip_column",
    "bin_column",
    "entity_column",
)


def _first_string_attr(
    schema: SchemaView, names: Sequence[str], frame: pd.DataFrame | None = None
) -> str | None:
    for name in names:
        value = getattr(schema, name, None)
        if isinstance(value, str) and value:
            if frame is None or value in frame.columns:
                return value
    return None


def resolve_id_column(
    frame: pd.DataFrame | None = None, *, schema: SchemaView | None = None
) -> str:
    s = schema or resolve_schema()
    col = _first_string_attr(s, _ID_ATTRS, frame)
    if col is None:
        col = _first_string_attr(s, _ID_ATTRS, None)
    if col is None:
        raise RightCensoringError(
            "could not resolve the transaction id column from the SchemaView; "
            f"probed attributes: {_ID_ATTRS!r}"
        )
    if frame is not None and col not in frame.columns:
        raise PerturbationError(f"id column {col!r} missing from frame")
    return col


def resolve_label_column(
    frame: pd.DataFrame | None = None, *, schema: SchemaView | None = None
) -> str | None:
    s = schema or resolve_schema()
    return _first_string_attr(s, _LABEL_ATTRS, frame)


def resolve_campaign_column(
    frame: pd.DataFrame | None = None, *, schema: SchemaView | None = None
) -> str | None:
    s = schema or resolve_schema()
    return _first_string_attr(s, _CAMPAIGN_ATTRS, frame)


def resolve_timestamp_column(
    frame: pd.DataFrame | None = None, *, schema: SchemaView | None = None
) -> str | None:
    s = schema or resolve_schema()
    return _first_string_attr(s, _TS_ATTRS, frame)


def schema_entity_columns(
    frame: pd.DataFrame | None = None, *, schema: SchemaView | None = None
) -> tuple[str, ...]:
    """Best-effort entity column discovery, used for invariance reporting."""
    s = schema or resolve_schema()
    found: list[str] = []
    for name in _ENTITY_LIST_ATTRS:
        value = getattr(s, name, None)
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item and item not in found:
                    found.append(item)
    for name in _ENTITY_SINGLE_ATTRS:
        value = getattr(s, name, None)
        if isinstance(value, str) and value and value not in found:
            found.append(value)
    if frame is not None:
        found = [c for c in found if c in frame.columns]
    return tuple(found)


def _schema_description(schema: SchemaView) -> Any:
    try:
        return describe_schema(schema)
    except TypeError:
        try:
            return describe_schema()
        except Exception:  # pragma: no cover
            return None
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# primitive adapter
# ---------------------------------------------------------------------------


def primitive_signature() -> inspect.Signature:
    """Live signature of the ``right_censor`` primitive."""
    return inspect.signature(right_censor)


def censor_call_kwargs(
    *,
    keep_fraction: float | None = None,
    cutoff_ts_ns: int | None = None,
    schema: SchemaView | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact kwargs for ``right_censor``, binding against its signature."""
    sig = primitive_signature()
    params = sig.parameters

    if (keep_fraction is None) == (cutoff_ts_ns is None):
        raise RightCensoringError(
            "supply exactly one of keep_fraction or cutoff_ts_ns"
        )

    kwargs: dict[str, Any] = {}
    if keep_fraction is not None:
        if "keep_fraction" not in params:
            raise RightCensoringError(
                f"{PRIMITIVE_NAME} no longer accepts 'keep_fraction'; "
                f"signature is {sig}"
            )
        kwargs["keep_fraction"] = float(keep_fraction)
    else:
        if "cutoff_ts_ns" not in params:
            raise RightCensoringError(
                f"{PRIMITIVE_NAME} no longer accepts 'cutoff_ts_ns'; "
                f"signature is {sig}"
            )
        kwargs["cutoff_ts_ns"] = int(cutoff_ts_ns)

    if schema is not None and "schema" in params:
        kwargs["schema"] = schema

    for name, value in (extra or {}).items():
        if name not in params:
            raise RightCensoringError(
                f"extra kwarg {name!r} is not accepted by {PRIMITIVE_NAME}; "
                f"signature is {sig}"
            )
        kwargs[name] = value

    required = [
        name
        for name, p in params.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
        and p.default is inspect.Parameter.empty
        and name not in kwargs
    ]
    if required:
        raise RightCensoringError(
            f"{PRIMITIVE_NAME} requires keyword args {required!r} that the "
            f"scenario does not supply; signature is {sig}"
        )

    try:
        sig.bind(pd.DataFrame(), **kwargs)
    except TypeError as exc:  # pragma: no cover - defensive
        raise RightCensoringError(
            f"cannot bind {kwargs!r} to {PRIMITIVE_NAME}{sig}: {exc}"
        ) from exc
    return kwargs


def predicted_cutoff_ns(
    frame: pd.DataFrame, keep_fraction: float, *, schema: SchemaView | None = None
) -> int:
    """Mirror of the primitive's quantile cutoff, for reporting only.

    The scenario never uses this to *decide* anything - it is reported next to
    the observed cutoff so signature/semantics drift shows up loudly.
    """
    s = schema or resolve_schema()
    ts = timestamps_as_ns(frame, schema=s)
    return int(np.quantile(np.asarray(ts), float(keep_fraction)))


def apply_right_censor(
    frame: pd.DataFrame,
    *,
    keep_fraction: float | None = None,
    cutoff_ts_ns: int | None = None,
    schema: SchemaView | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Call the primitive and assert it did not mutate the input frame."""
    s = schema or resolve_schema()
    kwargs = censor_call_kwargs(
        keep_fraction=keep_fraction,
        cutoff_ts_ns=cutoff_ts_ns,
        schema=s,
        extra=extra,
    )
    before = frame_fingerprint(frame, schema=s)
    out = right_censor(frame, **kwargs)
    after = frame_fingerprint(frame, schema=s)
    if before != after:
        raise RightCensoringError(
            f"{PRIMITIVE_NAME} mutated the input frame in place"
        )
    if not isinstance(out, pd.DataFrame):
        raise RightCensoringError(
            f"{PRIMITIVE_NAME} returned {type(out)!r}, expected a DataFrame"
        )
    reported = {k: v for k, v in kwargs.items() if k != "schema"}
    reported["schema_passed"] = "schema" in kwargs
    return out, reported


# ---------------------------------------------------------------------------
# fingerprinting
# ---------------------------------------------------------------------------


def frame_fingerprint(
    frame: pd.DataFrame, *, schema: SchemaView | None = None
) -> str:
    """Order-insensitive sha256 fingerprint of a frame's content."""
    work = frame.copy()
    work = work.reindex(sorted(map(str, work.columns)), axis=1)
    s = schema or resolve_schema()
    id_col = _first_string_attr(s, _ID_ATTRS, frame)
    if id_col is not None:
        work = work.sort_values(id_col, kind="mergesort")
    work = work.reset_index(drop=True)
    payload = work.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# value comparison (tolerates dtype drift, catches value tampering)
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover
        return False


def _scalar_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    lm, rm = _is_missing(left), _is_missing(right)
    if lm or rm:
        return lm and rm
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return bool(left) == bool(right)
    numeric = (int, float, np.integer, np.floating)
    if isinstance(left, numeric) and isinstance(right, numeric):
        return float(left) == float(right)
    try:
        if left == right:
            return True
    except Exception:  # pragma: no cover
        pass
    for a, b in ((left, right),):
        try:
            return pd.Timestamp(a) == pd.Timestamp(b)
        except Exception:
            break
    return str(left) == str(right)


def _series_values_equal(left: pd.Series, right: pd.Series) -> bool:
    a = left.reset_index(drop=True)
    b = right.reset_index(drop=True)
    if len(a) != len(b):
        return False
    if a.equals(b):
        return True
    for x, y in zip(a.tolist(), b.tolist()):
        if not _scalar_equal(x, y):
            return False
    return True


# ---------------------------------------------------------------------------
# invariant guards
# ---------------------------------------------------------------------------


def id_set(frame: pd.DataFrame, *, schema: SchemaView | None = None) -> set[Any]:
    s = schema or resolve_schema()
    id_col = resolve_id_column(frame, schema=s)
    return set(frame[id_col].tolist())


def assert_surviving_rows_unchanged(
    baseline: pd.DataFrame,
    censored: pd.DataFrame,
    *,
    schema: SchemaView | None = None,
    arm_id: str = "",
) -> dict[str, Any]:
    """S5 guard: rows may only *disappear*.

    Every surviving row must be identical to its baseline counterpart, no new
    transaction ids may appear, and the column set must be unchanged.
    """
    s = schema or resolve_schema()
    id_col = resolve_id_column(baseline, schema=s)
    where = f" [{arm_id}]" if arm_id else ""

    if list(map(str, censored.columns)) != list(map(str, baseline.columns)):
        raise RightCensoringError(
            f"column set changed{where}: baseline={list(baseline.columns)!r} "
            f"censored={list(censored.columns)!r}"
        )
    if id_col not in censored.columns:
        raise PerturbationError(f"id column {id_col!r} missing from output{where}")
    if baseline[id_col].duplicated().any():
        raise RightCensoringError(f"baseline contains duplicate ids{where}")
    if censored[id_col].duplicated().any():
        raise RightCensoringError(f"censored frame contains duplicate ids{where}")

    base_ids = baseline[id_col]
    kept_ids = censored[id_col]
    novel = set(kept_ids.tolist()) - set(base_ids.tolist())
    if novel:
        raise RightCensoringError(
            f"right censoring introduced {len(novel)} new transaction id(s)"
            f"{where}: {sorted(map(str, novel))[:5]!r}"
        )
    if len(censored) > len(baseline):
        raise RightCensoringError(
            f"censored frame grew from {len(baseline)} to {len(censored)} rows{where}"
        )

    aligned = baseline.set_index(id_col).loc[kept_ids.tolist()]
    mismatched: list[str] = []
    for column in baseline.columns:
        if column == id_col:
            continue
        if not _series_values_equal(aligned[column], censored[column]):
            mismatched.append(str(column))
    if mismatched:
        raise RightCensoringError(
            f"surviving rows were modified{where} in column(s): {mismatched!r}"
        )

    removed = set(base_ids.tolist()) - set(kept_ids.tolist())
    return {
        "id_column": id_col,
        "columns_stable": True,
        "no_new_ids": True,
        "surviving_rows_identical": True,
        "baseline_rows": int(len(baseline)),
        "surviving_rows": int(len(censored)),
        "removed_rows": int(len(removed)),
    }


def assert_cutoff_consistency(
    baseline: pd.DataFrame,
    censored: pd.DataFrame,
    *,
    schema: SchemaView | None = None,
    arm_id: str = "",
) -> dict[str, Any]:
    """Every kept timestamp must precede every removed timestamp."""
    s = schema or resolve_schema()
    id_col = resolve_id_column(baseline, schema=s)
    where = f" [{arm_id}]" if arm_id else ""

    base_ts = pd.Series(
        np.asarray(timestamps_as_ns(baseline, schema=s)), index=baseline[id_col].tolist()
    )
    kept_ids = censored[id_col].tolist()
    removed_ids = [i for i in base_ts.index if i not in set(kept_ids)]

    kept_ts = base_ts.loc[kept_ids]
    max_kept = int(kept_ts.max()) if len(kept_ts) else None
    min_removed = int(base_ts.loc[removed_ids].min()) if removed_ids else None

    if max_kept is not None and min_removed is not None and max_kept > min_removed:
        raise RightCensoringError(
            f"censoring is not right-sided{where}: max kept ts {max_kept} exceeds "
            f"min removed ts {min_removed}"
        )
    return {
        "max_kept_ts_ns": max_kept,
        "min_removed_ts_ns": min_removed,
        "right_sided": True,
    }


def check_monotone_nesting(
    id_sets: Sequence[tuple[Any, set[Any]]],
) -> dict[str, Any]:
    """Levels ordered by increasing retention must yield nested id sets."""
    ordered = sorted(id_sets, key=lambda pair: len(pair[1]))
    violations: list[dict[str, Any]] = []
    for (small_level, small), (big_level, big) in zip(ordered, ordered[1:]):
        if not small.issubset(big):
            violations.append(
                {
                    "smaller_level": json_safe(small_level),
                    "larger_level": json_safe(big_level),
                    "escaped_ids": int(len(small - big)),
                }
            )
    return {
        "checked_levels": [json_safe(level) for level, _ in ordered],
        "retained_counts": [int(len(ids)) for _, ids in ordered],
        "nested": not violations,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------


def _attack_flags(frame: pd.DataFrame, schema: SchemaView) -> np.ndarray | None:
    for kwargs in ({"schema": schema}, {}):
        try:
            mask = attack_mask(frame, **kwargs)
        except TypeError:
            continue
        except Exception:  # pragma: no cover
            return None
        return np.asarray(pd.Series(mask).fillna(False)).astype(bool)
    return None


def window_profile(frame: pd.DataFrame, *, schema: SchemaView | None = None) -> dict[str, Any]:
    """Observation window statistics via the repository helper."""
    s = schema or resolve_schema()
    raw: Any = None
    for kwargs in ({"schema": s}, {}):
        try:
            raw = observation_window_ns(frame, **kwargs)
        except TypeError:
            continue
        except Exception as exc:  # pragma: no cover
            return {"error": f"{type(exc).__name__}: {exc}"}
        break

    start = end = span = None
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        start, end = int(raw[0]), int(raw[1])
        span = end - start
    elif isinstance(raw, Mapping):
        start = raw.get("start_ns", raw.get("start"))
        end = raw.get("end_ns", raw.get("end"))
        span = raw.get("span_ns", raw.get("span"))
        start = None if start is None else int(start)
        end = None if end is None else int(end)
        span = None if span is None else int(span)
    elif isinstance(raw, (int, np.integer)):
        span = int(raw)
    return {
        "start_ns": start,
        "end_ns": end,
        "span_ns": span,
        "helper": "observation_window_ns",
    }


def _label_counts(frame: pd.DataFrame, schema: SchemaView) -> dict[str, int] | None:
    label_col = resolve_label_column(frame, schema=schema)
    if label_col is None:
        return None
    counts = frame[label_col].value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _primitive_summary(
    baseline: pd.DataFrame, censored: pd.DataFrame, schema: SchemaView
) -> Any:
    for args, kwargs in (
        ((baseline, censored), {"schema": schema}),
        ((baseline, censored), {}),
        ((censored,), {"schema": schema}),
        ((censored,), {}),
    ):
        try:
            return json_safe(summarize_perturbation(*args, **kwargs))
        except TypeError:
            continue
        except Exception:  # pragma: no cover
            return None
    return None


def censoring_profile(
    baseline: pd.DataFrame,
    censored: pd.DataFrame,
    *,
    schema: SchemaView | None = None,
    keep_fraction: float | None = None,
) -> dict[str, Any]:
    """Everything measurable about one censoring arm - no hardcoded stats."""
    s = schema or resolve_schema()
    before_rows = int(len(baseline))
    after_rows = int(len(censored))
    removed = before_rows - after_rows

    before_window = window_profile(baseline, schema=s)
    after_window = window_profile(censored, schema=s)
    span_before = before_window.get("span_ns")
    span_after = after_window.get("span_ns")
    span_ratio = None
    if isinstance(span_before, int) and span_before > 0 and isinstance(span_after, int):
        span_ratio = span_after / span_before

    profile: dict[str, Any] = {
        "rows_before": before_rows,
        "rows_after": after_rows,
        "rows_removed": removed,
        "removed_fraction": (removed / before_rows) if before_rows else None,
        "retained_fraction": (after_rows / before_rows) if before_rows else None,
        "window_before": before_window,
        "window_after": after_window,
        "span_retained_ratio": span_ratio,
        "labels_before": _label_counts(baseline, s),
        "labels_after": _label_counts(censored, s),
        "primitive_summary": _primitive_summary(baseline, censored, s),
    }

    flags_before = _attack_flags(baseline, s)
    flags_after = _attack_flags(censored, s)
    profile["attack_rows_before"] = (
        int(flags_before.sum()) if flags_before is not None else None
    )
    profile["attack_rows_after"] = (
        int(flags_after.sum()) if flags_after is not None else None
    )
    profile["attack_side_empty"] = (
        bool(flags_after is not None and int(flags_after.sum()) == 0)
    )

    campaign_col = resolve_campaign_column(baseline, schema=s)
    if campaign_col is not None:
        profile["campaign_column"] = campaign_col
        profile["campaigns_before"] = int(baseline[campaign_col].nunique(dropna=True))
        profile["campaigns_after"] = int(censored[campaign_col].nunique(dropna=True))

    entity_cols = schema_entity_columns(baseline, schema=s)
    profile["entity_columns"] = list(entity_cols)

    if keep_fraction is not None:
        try:
            predicted = predicted_cutoff_ns(baseline, keep_fraction, schema=s)
        except Exception:  # pragma: no cover
            predicted = None
        profile["predicted_cutoff_ns"] = predicted
    return profile


# ---------------------------------------------------------------------------
# world / rebuild adapter
# ---------------------------------------------------------------------------


def _filtered_kwargs(fn: Any, candidates: Mapping[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return {}
    return {k: v for k, v in candidates.items() if k in params and v is not None}


def _maybe_build_world(
    frame: pd.DataFrame, *, schema: SchemaView | None
) -> tuple[Any, str | None]:
    if build_world is None:
        return None, "build_world unavailable"
    try:
        return build_world(frame, **_filtered_kwargs(build_world, {"schema": schema})), None
    except TypeError:
        try:
            return build_world(frame), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _rebuild_metrics(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        source: Mapping[str, Any] = result
    elif hasattr(result, "__dict__"):
        source = {k: v for k, v in vars(result).items() if not k.startswith("_")}
    else:
        return {"value": json_safe(result)}
    scalars: dict[str, Any] = {}
    for key, value in source.items():
        if isinstance(value, (bool, int, float, str, np.integer, np.floating, np.bool_)):
            scalars[str(key)] = json_safe(value)
        elif value is None:
            scalars[str(key)] = None
    return scalars


def _invoke_rebuild(
    frame: pd.DataFrame, world: Any, *, schema: SchemaView | None
) -> dict[str, Any]:
    """Invoke the real rebuild_world pipeline on the arm frame."""

    del world, schema

    try:
        result = rebuild_world(
            frame,
            name="S5_right_censoring",
        )
    except Exception as exc:
        return {
            "invoked": True,
            "function": getattr(rebuild_world, "__name__", str(rebuild_world)),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": {},
        }

    return {
        "invoked": True,
        "function": getattr(rebuild_world, "__name__", str(rebuild_world)),
        "ok": True,
        "error": None,
        "result_type": type(result).__name__,
        "metrics": _rebuild_metrics(result),
    }


# ---------------------------------------------------------------------------
# arm runner
# ---------------------------------------------------------------------------


def run_right_censoring_arm(
    baseline: pd.DataFrame,
    *,
    arm_id: str,
    keep_fraction: float | None = None,
    cutoff_ts_ns: int | None = None,
    config: RightCensoringConfig | None = None,
    schema: SchemaView | None = None,
    control: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one arm; returns ``(record, arm_frame)``.

    The control arm skips the primitive entirely and rebuilds the untouched
    baseline, so control-vs-perturbed deltas are attributable to censoring.
    """
    if not isinstance(baseline, pd.DataFrame):
        raise RightCensoringError("baseline must be a pandas DataFrame")
    if baseline.empty:
        raise RightCensoringError("baseline frame is empty")

    cfg = (config or RightCensoringConfig()).validate()
    s = schema or resolve_schema()
    baseline_fp = frame_fingerprint(baseline, schema=s)

    record: dict[str, Any] = {
        "arm_id": arm_id,
        "kind": "control" if control else "perturbed",
        "primitive": None if control else PRIMITIVE_NAME,
        "keep_fraction": None if keep_fraction is None else float(keep_fraction),
        "cutoff_ts_ns": None if cutoff_ts_ns is None else int(cutoff_ts_ns),
        "status": "ok",
    }

    if control:
        frame = baseline.copy()
        record["primitive_kwargs"] = None
    else:
        frame, used = apply_right_censor(
            baseline,
            keep_fraction=keep_fraction,
            cutoff_ts_ns=cutoff_ts_ns,
            schema=s,
            extra=cfg.extra_kwargs,
        )
        record["primitive_kwargs"] = json_safe(used)

    invariants = assert_surviving_rows_unchanged(
        baseline, frame, schema=s, arm_id=arm_id
    )
    if not control:
        invariants.update(
            assert_cutoff_consistency(baseline, frame, schema=s, arm_id=arm_id)
        )
    else:
        if frame_fingerprint(frame, schema=s) != baseline_fp:
            raise RightCensoringError("control arm frame diverged from the baseline")
        invariants["control_identical_to_baseline"] = True
    record["invariants"] = json_safe(invariants)

    record["profile"] = json_safe(
        censoring_profile(baseline, frame, schema=s, keep_fraction=keep_fraction)
    )
    record["fingerprint"] = frame_fingerprint(frame, schema=s)
    record["baseline_fingerprint"] = baseline_fp

    world, world_error = _maybe_build_world(frame, schema=s)
    record["world"] = {
        "built": world is not None,
        "type": type(world).__name__ if world is not None else None,
        "error": world_error,
    }

    rebuild = _invoke_rebuild(frame, world, schema=s)
    record["rebuild"] = json_safe(rebuild)

    if not rebuild.get("ok", False):
        empty_attacks = bool(record["profile"].get("attack_side_empty"))
        if empty_attacks and cfg.tolerate_empty_attack_side:
            record["status"] = "rebuild_rejected_empty_attack_side"
            log.info(
                "%s: arm %s rejected by the rebuild layer with no attack rows left",
                SCENARIO_ID,
                arm_id,
            )
        elif cfg.strict_invariants and not empty_attacks:
            record["status"] = "rebuild_failed"
        else:
            record["status"] = "rebuild_rejected"

    if frame_fingerprint(baseline, schema=s) != baseline_fp:
        raise RightCensoringError(
            f"arm {arm_id!r} mutated the baseline frame in place"
        )
    return record, frame


# ---------------------------------------------------------------------------
# scenario runner
# ---------------------------------------------------------------------------


def _compare_to_control(control: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    cp = control.get("profile", {}) or {}
    ap = arm.get("profile", {}) or {}

    def _delta(key: str) -> Any:
        a, b = ap.get(key), cp.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a - b
        return None

    return {
        "arm_id": arm.get("arm_id"),
        "keep_fraction": arm.get("keep_fraction"),
        "cutoff_ts_ns": arm.get("cutoff_ts_ns"),
        "rows_delta": _delta("rows_after"),
        "attack_rows_delta": _delta("attack_rows_after"),
        "campaigns_delta": _delta("campaigns_after"),
        "retained_fraction": ap.get("retained_fraction"),
        "span_retained_ratio": ap.get("span_retained_ratio"),
        "fingerprint_differs": arm.get("fingerprint") != control.get("fingerprint"),
        "rebuild_ok": (arm.get("rebuild") or {}).get("ok"),
    }


def run_right_censoring_scenario(
    baseline: pd.DataFrame,
    config: RightCensoringConfig | None = None,
    *,
    schema: SchemaView | None = None,
) -> dict[str, Any]:
    """Run the full right-censoring scenario and return a json-safe result."""
    if not isinstance(baseline, pd.DataFrame):
        raise RightCensoringError("baseline must be a pandas DataFrame")
    cfg = (config or RightCensoringConfig()).validate()
    s = schema or resolve_schema()

    input_fp = frame_fingerprint(baseline, schema=s)
    sig = primitive_signature()

    control_record: dict[str, Any] | None = None
    if cfg.include_control:
        control_record, _ = run_right_censoring_arm(
            baseline,
            arm_id=CONTROL_ARM_ID,
            config=cfg,
            schema=s,
            control=True,
        )

    arms: list[dict[str, Any]] = []
    retained: list[tuple[Any, set[Any]]] = []
    for level in cfg.severity_levels():
        arm_id = cfg.arm_id(level)
        kwargs: dict[str, Any] = {}
        if cfg.severity_mode == "keep_fraction":
            kwargs["keep_fraction"] = float(level)
        else:
            kwargs["cutoff_ts_ns"] = int(level)
        record, frame = run_right_censoring_arm(
            baseline, arm_id=arm_id, config=cfg, schema=s, **kwargs
        )
        arms.append(record)
        retained.append((level, id_set(frame, schema=s)))

    nesting = check_monotone_nesting(retained)
    if cfg.require_monotone_nesting and not nesting["nested"]:
        raise RightCensoringError(
            f"retained id sets are not nested across severity levels: "
            f"{nesting['violations']!r}"
        )

    comparisons = (
        [_compare_to_control(control_record, arm) for arm in arms]
        if control_record is not None
        else []
    )

    statuses = [arm["status"] for arm in arms]
    summary = {
        "arm_count": len(arms),
        "has_control": control_record is not None,
        "severity_mode": cfg.severity_mode,
        "severity_levels": [json_safe(level) for level in cfg.severity_levels()],
        "all_invariants_held": True,
        "monotone_nesting_ok": bool(nesting["nested"]),
        "statuses": statuses,
        "arms_rebuilt": sum(
            1 for arm in arms if (arm.get("rebuild") or {}).get("ok") is True
        ),
        "min_retained_fraction": min(
            (
                arm["profile"].get("retained_fraction")
                for arm in arms
                if isinstance(arm["profile"].get("retained_fraction"), (int, float))
            ),
            default=None,
        ),
        "max_removed_fraction": max(
            (
                arm["profile"].get("removed_fraction")
                for arm in arms
                if isinstance(arm["profile"].get("removed_fraction"), (int, float))
            ),
            default=None,
        ),
        "baseline_input_unchanged": frame_fingerprint(baseline, schema=s) == input_fp,
    }
    if not summary["baseline_input_unchanged"]:
        raise RightCensoringError("scenario mutated the baseline input frame")

    result = {
        "scenario_id": SCENARIO_ID,
        "scenario_name": SCENARIO_NAME,
        "phase": "4B",
        "primitive": PRIMITIVE_NAME,
        "primitive_signature": str(sig),
        "deterministic": True,
        "seeded": "seed" in sig.parameters,
        "config": cfg.as_dict(),
        "schema": json_safe(_schema_description(s)),
        "baseline": {
            "rows": int(len(baseline)),
            "columns": [str(c) for c in baseline.columns],
            "fingerprint": input_fp,
            "window": window_profile(baseline, schema=s),
        },
        "control": control_record,
        "arms": arms,
        "comparisons": comparisons,
        "nesting": nesting,
        "summary": summary,
    }
    return json_safe(result)
