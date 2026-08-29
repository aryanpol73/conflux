"""Phase 4B - S4: Legitimate Volume.

Dilutes the population with additional *negative* background traffic and reports
whether extra legitimate volume alone perturbs the candidate structures.

Hard rules honoured here:
  * the resampling primitive is NOT reimplemented; it is imported from
    ``conflux.robustness.perturbations`` and bound exactly to its real
    signature,
  * the input frame is never mutated,
  * every original row survives value-for-value (the primitive re-sorts the
    frame by id, so original rows move - that is position, not content),
  * no scorer is constructed, fitted or refitted anywhere in this module.

Bound primitive::

    resample_legitimate_transactions(
        frame, *, n_new=None, multiplier=None, seed, refresh_entities=True,
        entity_columns=None, time_mode="uniform", id_prefix="SYNLEG-",
        schema=None,
    ) -> pd.DataFrame

Note that ``multiplier`` scales the *donor pool* (the negative rows), not the
full population, so ``multiplier=1.0`` roughly doubles the benign traffic while
leaving the attack rows untouched.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from conflux.robustness.perturbations import (
    PerturbationError,
    attack_mask,
    resample_legitimate_transactions,
    resolve_schema,
)
from conflux.robustness.rebuild import rebuild_world

try:  # world abstraction is used when its call shape is compatible
    from conflux.robustness.world import build_world as _build_world
except Exception:  # pragma: no cover
    _build_world = None

try:
    from conflux.robustness.world import to_json_safe as _to_json_safe
except Exception:  # pragma: no cover
    _to_json_safe = None


LOGGER = logging.getLogger(__name__)

SCENARIO_ID = "s4_legit_volume"
SCENARIO_NAME = "Legitimate Volume"
CONTROL_ARM_ID = "control"

MULTIPLIER_MODE = "multiplier"
ABSOLUTE_MODE = "n_new"


class LegitVolumeError(RuntimeError):
    """Raised on invalid configuration, unmappable primitive API, or a broken invariant."""


# --------------------------------------------------------------------------- #
# schema helpers (attribute names probed, never assumed)
# --------------------------------------------------------------------------- #

_ID_ATTRS = ("id_col", "id_column", "transaction_id_col", "transaction_id")
_LABEL_ATTRS = ("label_col", "label_column", "label")
_CAMPAIGN_ATTRS = ("campaign_col", "campaign_column", "campaign_id_col", "campaign_id")
_TIME_ATTRS = ("ts_col", "time_col", "timestamp_col", "timestamp_column", "time_column")
_ENTITY_LIST_ATTRS = ("entity", "entity_columns", "entity_cols", "entities")


def _schema_attr(schema: Any, attrs: Sequence[str], frame: pd.DataFrame | None = None) -> str | None:
    for name in attrs:
        value = getattr(schema, name, None)
        if isinstance(value, str) and value:
            if frame is None or value in frame.columns:
                return value
    return None


def scenario_schema(frame: pd.DataFrame, schema: Any = None) -> Any:
    """Resolve the repository schema view for ``frame`` (or pass one through)."""
    if schema is not None:
        return schema
    try:
        return resolve_schema(frame)
    except TypeError:
        return resolve_schema()


def id_column(frame: pd.DataFrame, schema: Any) -> str:
    col = _schema_attr(schema, _ID_ATTRS, frame)
    if col is None:
        raise LegitVolumeError(
            "Could not resolve the transaction id column from the resolved schema "
            f"({type(schema).__name__}); probed attributes: {_ID_ATTRS}."
        )
    return col


def schema_entity_columns(frame: pd.DataFrame, schema: Any) -> tuple[str, ...]:
    """The entity columns the primitive would refresh by default (``schema.entity``)."""
    for name in _ENTITY_LIST_ATTRS:
        value = getattr(schema, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:  # pragma: no cover
                continue
        if isinstance(value, str):
            value = (value,)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            columns = [c for c in value if isinstance(c, str) and c in frame.columns]
            if columns:
                return tuple(dict.fromkeys(columns))
    return ()


# --------------------------------------------------------------------------- #
# volume levels
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VolumeLevel:
    """One scenario arm: either a donor-pool multiplier or an absolute row count."""

    kind: str
    value: float | int

    @property
    def arm_id(self) -> str:
        if self.kind == MULTIPLIER_MODE:
            return f"{MULTIPLIER_MODE}={self.value:g}"
        return f"{ABSOLUTE_MODE}={int(self.value)}"

    def as_dict(self) -> dict[str, Any]:
        value = float(self.value) if self.kind == MULTIPLIER_MODE else int(self.value)
        return {"kind": self.kind, "value": value, "arm_id": self.arm_id}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LegitVolumeConfig:
    """Configuration for S4.

    Exactly one of ``multipliers`` / ``absolute_counts`` may be set, mirroring
    the primitive's own "supply exactly one of n_new or multiplier" rule. Each
    entry becomes one perturbed arm.

    ``entity_columns=None`` and ``id_prefix=None`` mean "use the primitive's own
    defaults" (``schema.entity`` and ``SYNLEG-`` respectively), so neither is
    duplicated here. ``time_mode`` is passed through unvalidated so the
    repository's ``PerturbationError`` remains the single source of truth for
    which modes exist.
    """

    multipliers: tuple[float, ...] | None = (0.5, 1.0, 2.0)
    absolute_counts: tuple[int, ...] | None = None
    seed: int = 4208
    refresh_entities: bool = True
    entity_columns: tuple[str, ...] | None = None
    time_mode: str = "uniform"
    id_prefix: str | None = None
    include_control: bool = True
    strict_invariants: bool = True
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        multipliers = None if self.multipliers is None else tuple(self.multipliers)
        counts = None if self.absolute_counts is None else tuple(self.absolute_counts)
        object.__setattr__(self, "multipliers", multipliers)
        object.__setattr__(self, "absolute_counts", counts)

        if (multipliers is None) == (counts is None):
            raise LegitVolumeError(
                "Supply exactly one of multipliers or absolute_counts "
                f"(got multipliers={multipliers!r}, absolute_counts={counts!r})."
            )

        if multipliers is not None:
            if not multipliers:
                raise LegitVolumeError("multipliers must contain at least one level.")
            for value in multipliers:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise LegitVolumeError(f"multipliers entries must be numeric, got {value!r}.")
                if float(value) <= 0.0:
                    raise LegitVolumeError(f"multipliers entries must be > 0, got {value!r}.")
            if len({float(v) for v in multipliers}) != len(multipliers):
                raise LegitVolumeError(f"multipliers must be unique, got {multipliers!r}.")

        if counts is not None:
            if not counts:
                raise LegitVolumeError("absolute_counts must contain at least one level.")
            for value in counts:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise LegitVolumeError(f"absolute_counts entries must be ints, got {value!r}.")
                if value <= 0:
                    raise LegitVolumeError(f"absolute_counts entries must be > 0, got {value!r}.")
            if len(set(counts)) != len(counts):
                raise LegitVolumeError(f"absolute_counts must be unique, got {counts!r}.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise LegitVolumeError(f"seed must be a non-negative int, got {self.seed!r}.")

        if not isinstance(self.refresh_entities, bool):
            raise LegitVolumeError(
                f"refresh_entities must be a bool, got {type(self.refresh_entities).__name__}."
            )

        if self.entity_columns is not None:
            if isinstance(self.entity_columns, str):
                raise LegitVolumeError(
                    "entity_columns must be a sequence of column names, not a bare string "
                    f"({self.entity_columns!r})."
                )
            columns = tuple(self.entity_columns)
            if not columns:
                raise LegitVolumeError(
                    "entity_columns must be None (use the schema default) or non-empty."
                )
            for column in columns:
                if not isinstance(column, str) or not column:
                    raise LegitVolumeError(
                        f"entity_columns entries must be non-empty strings, got {column!r}."
                    )
            if len(set(columns)) != len(columns):
                raise LegitVolumeError(f"entity_columns must be unique, got {columns!r}.")
            object.__setattr__(self, "entity_columns", columns)

        if not isinstance(self.time_mode, str) or not self.time_mode:
            raise LegitVolumeError(f"time_mode must be a non-empty string, got {self.time_mode!r}.")

        if self.id_prefix is not None:
            if not isinstance(self.id_prefix, str) or not self.id_prefix:
                raise LegitVolumeError(
                    f"id_prefix must be None or a non-empty string, got {self.id_prefix!r}."
                )

        if not isinstance(self.extra_kwargs, Mapping):
            raise LegitVolumeError("extra_kwargs must be a mapping.")
        object.__setattr__(self, "extra_kwargs", dict(self.extra_kwargs))

    @property
    def mode(self) -> str:
        return MULTIPLIER_MODE if self.multipliers is not None else ABSOLUTE_MODE

    def volume_levels(self) -> tuple[VolumeLevel, ...]:
        if self.multipliers is not None:
            return tuple(VolumeLevel(MULTIPLIER_MODE, float(v)) for v in self.multipliers)
        return tuple(VolumeLevel(ABSOLUTE_MODE, int(v)) for v in (self.absolute_counts or ()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "multipliers": None if self.multipliers is None else [float(v) for v in self.multipliers],
            "absolute_counts": (
                None if self.absolute_counts is None else [int(v) for v in self.absolute_counts]
            ),
            "seed": self.seed,
            "refresh_entities": self.refresh_entities,
            "entity_columns": None if self.entity_columns is None else list(self.entity_columns),
            "time_mode": self.time_mode,
            "id_prefix": self.id_prefix,
            "include_control": self.include_control,
            "strict_invariants": self.strict_invariants,
            "extra_kwargs": dict(self.extra_kwargs),
        }


# --------------------------------------------------------------------------- #
# donor pool
# --------------------------------------------------------------------------- #

def donor_mask(frame: pd.DataFrame, schema: Any) -> pd.Series:
    """The negative rows the primitive resamples from."""
    try:
        mask = attack_mask(frame, schema=schema)
    except TypeError:
        mask = attack_mask(frame)
    return ~pd.Series(mask, index=frame.index).astype(bool)


def donor_frame(frame: pd.DataFrame, schema: Any) -> pd.DataFrame:
    return frame.loc[donor_mask(frame, schema)]


def donor_count(frame: pd.DataFrame, schema: Any) -> int:
    return int(donor_mask(frame, schema).sum())


def expected_injected_rows(
    frame: pd.DataFrame,
    *,
    level: VolumeLevel,
    schema: Any,
) -> int:
    """Predict the injected row count using the primitive's own arithmetic.

    This is a prediction for metrics and assertions only; the primitive remains
    the sole implementation.
    """
    if level.kind == ABSOLUTE_MODE:
        return int(level.value)
    return int(round(donor_count(frame, schema) * float(level.value)))


# --------------------------------------------------------------------------- #
# value / representation comparison
#
# pd.concat inside the primitive can widen a column's dtype without touching a
# value. That is representation drift, not tampering: repair it, but never let
# it mask a real change.
# --------------------------------------------------------------------------- #

def _is_datetimelike(series: pd.Series) -> bool:
    dtype = series.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return True
    return getattr(dtype, "kind", "") == "M"


def _as_utc_ns(series: pd.Series):
    return pd.to_datetime(series, utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]")


def _scalar_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    left_na, right_na = pd.isna(left), pd.isna(right)
    if bool(left_na) or bool(right_na):
        return bool(left_na) == bool(right_na)
    try:
        if bool(left == right):
            return True
    except Exception:  # pragma: no cover
        return False
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def _series_values_equal(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return False
    if left.dtype == right.dtype:
        return left.reset_index(drop=True).equals(right.reset_index(drop=True))
    if _is_datetimelike(left) or _is_datetimelike(right):
        try:
            left_ns, right_ns = _as_utc_ns(left), _as_utc_ns(right)
        except (TypeError, ValueError, OverflowError):
            return False
        both_na = pd.isna(left_ns) & pd.isna(right_ns)
        return bool(((left_ns == right_ns) | both_na).all())
    for a, b in zip(left.to_numpy(dtype=object), right.to_numpy(dtype=object)):
        if not _scalar_equal(a, b):
            return False
    return True


def _difference_sample(
    left: pd.DataFrame, right: pd.DataFrame, column: str, limit: int = 3
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    left_col, right_col = left[column], right[column]
    for key in left_col.index:
        if _scalar_equal(left_col.loc[key], right_col.loc[key]):
            continue
        samples.append(
            {
                "id": str(key),
                "baseline": repr(left_col.loc[key]),
                "perturbed": repr(right_col.loc[key]),
            }
        )
        if len(samples) >= limit:
            break
    return samples


def _coerce_to_dtype(series: pd.Series, dtype: Any) -> pd.Series | None:
    try:
        return series.astype(dtype)
    except (TypeError, ValueError, OverflowError):
        pass
    is_tz = isinstance(dtype, pd.DatetimeTZDtype)
    if is_tz or getattr(dtype, "kind", "") == "M":
        try:
            converted = pd.to_datetime(series, utc=True, errors="raise")
            if is_tz:
                return converted.dt.tz_convert(dtype.tz).astype(dtype)
            return converted.dt.tz_localize(None).astype(dtype)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def restore_baseline_representation(
    baseline: pd.DataFrame, perturbed: pd.DataFrame
) -> pd.DataFrame:
    """Undo concat-driven dtype/column-order drift without changing any value.

    Row order is deliberately left alone: the primitive re-sorts by id and that
    ordering is part of its contract.
    """
    out = perturbed
    copied = False
    for column in baseline.columns:
        if column not in out.columns:
            continue
        target = baseline[column].dtype
        current = out[column]
        if current.dtype == target:
            continue
        recast = _coerce_to_dtype(current, target)
        if recast is None or not _series_values_equal(current, recast):
            LOGGER.warning(
                "Could not restore baseline dtype %s for column %r (found %s); "
                "the invariant guard will compare values instead.",
                target, column, current.dtype,
            )
            continue
        if not copied:
            out = out.copy()
            copied = True
        out[column] = recast

    ordered = [c for c in baseline.columns if c in out.columns]
    extras = [c for c in out.columns if c not in baseline.columns]
    if list(out.columns) != ordered + extras:
        out = out.reindex(columns=ordered + extras)
    return out


def dtype_drift(baseline: pd.DataFrame, perturbed: pd.DataFrame) -> dict[str, dict[str, str]]:
    drift: dict[str, dict[str, str]] = {}
    for column in baseline.columns:
        if column not in perturbed.columns:
            continue
        if perturbed[column].dtype != baseline[column].dtype:
            drift[column] = {
                "baseline": str(baseline[column].dtype),
                "perturbed": str(perturbed[column].dtype),
            }
    return drift


# --------------------------------------------------------------------------- #
# primitive adapter - exact binding
# --------------------------------------------------------------------------- #

def primitive_signature() -> inspect.Signature:
    return inspect.signature(resample_legitimate_transactions)


def _primitive_default(name: str) -> Any:
    parameter = primitive_signature().parameters.get(name)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


def effective_id_prefix(config: LegitVolumeConfig) -> str | None:
    """Configured prefix, else the primitive's own declared default."""
    if config.id_prefix is not None:
        return config.id_prefix
    value = _primitive_default("id_prefix")
    return value if isinstance(value, str) else None


def resolve_entity_columns(
    frame: pd.DataFrame, *, config: LegitVolumeConfig, schema: Any
) -> tuple[str, ...]:
    """The columns the primitive will refresh: configured, else ``schema.entity``.

    Missing columns are not pre-checked here - the primitive raises
    ``PerturbationError`` for those and stays the single source of truth.
    """
    if config.entity_columns is not None:
        return tuple(config.entity_columns)
    return schema_entity_columns(frame, schema)


def volume_call_kwargs(
    *,
    config: LegitVolumeConfig,
    level: VolumeLevel,
    schema: Any = None,
) -> dict[str, Any]:
    """Build the keyword arguments for the real primitive."""
    signature = primitive_signature()
    params = signature.parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    if level.kind not in (MULTIPLIER_MODE, ABSOLUTE_MODE):
        raise LegitVolumeError(
            f"Unknown volume level kind {level.kind!r}; "
            f"expected {MULTIPLIER_MODE!r} or {ABSOLUTE_MODE!r}."
        )

    kwargs: dict[str, Any] = {
        "seed": int(config.seed),
        "refresh_entities": bool(config.refresh_entities),
        "time_mode": config.time_mode,
    }
    if level.kind == MULTIPLIER_MODE:
        kwargs["multiplier"] = float(level.value)
    else:
        kwargs["n_new"] = int(level.value)

    if config.entity_columns is not None:
        kwargs["entity_columns"] = tuple(config.entity_columns)
    if config.id_prefix is not None:
        kwargs["id_prefix"] = config.id_prefix
    if schema is not None and "schema" in params:
        kwargs["schema"] = schema

    for key, value in config.extra_kwargs.items():
        if key in params or accepts_var_kw:
            kwargs[key] = value
        else:
            raise LegitVolumeError(
                f"extra_kwargs key {key!r} is not accepted by "
                f"resample_legitimate_transactions{signature}."
            )

    unexpected = [key for key in kwargs if key not in params and not accepts_var_kw]
    if unexpected:
        raise LegitVolumeError(
            f"resample_legitimate_transactions{signature} does not accept {unexpected!r}; "
            "the primitive signature changed - update the adapter in "
            "scenario_legit_volume.py (do not edit perturbations.py)."
        )

    positional = [
        name for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    frame_param = positional[0] if positional else None
    missing = [
        name for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                       inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       inspect.Parameter.KEYWORD_ONLY)
        and name != frame_param
        and name not in kwargs
    ]
    if missing:
        raise LegitVolumeError(
            "Cannot map LegitVolumeConfig onto the real primitive: "
            f"resample_legitimate_transactions{signature} requires {missing!r}. "
            "Supply them via LegitVolumeConfig.extra_kwargs."
        )
    return kwargs


def apply_legit_volume(
    frame: pd.DataFrame,
    *,
    config: LegitVolumeConfig,
    level: VolumeLevel,
    schema: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Call the real primitive; return ``(perturbed_frame, reported_kwargs)``."""
    schema = scenario_schema(frame, schema)
    kwargs = volume_call_kwargs(config=config, level=level, schema=schema)
    try:
        result = resample_legitimate_transactions(frame, **kwargs)
    except PerturbationError:
        raise
    except TypeError as exc:
        raise LegitVolumeError(
            f"resample_legitimate_transactions rejected the adapted call "
            f"{sorted(kwargs)}: {exc}"
        ) from exc

    if isinstance(result, tuple) and result and isinstance(result[0], pd.DataFrame):
        result = result[0]
    if not isinstance(result, pd.DataFrame):
        raise LegitVolumeError(
            f"resample_legitimate_transactions returned {type(result).__name__}, "
            "expected a DataFrame."
        )

    result = restore_baseline_representation(frame, result)

    reported = {k: v for k, v in kwargs.items() if k != "schema"}
    reported["level"] = level.as_dict()
    reported["entity_columns"] = list(resolve_entity_columns(frame, config=config, schema=schema))
    reported["id_prefix"] = effective_id_prefix(config)
    return result, reported


# --------------------------------------------------------------------------- #
# invariants and profiling
# --------------------------------------------------------------------------- #

def injected_ids(baseline: pd.DataFrame, perturbed: pd.DataFrame, id_col: str) -> list[Any]:
    original = set(baseline[id_col].tolist())
    return [value for value in perturbed[id_col].tolist() if value not in original]


def injected_rows(baseline: pd.DataFrame, perturbed: pd.DataFrame, id_col: str) -> pd.DataFrame:
    original = set(baseline[id_col].tolist())
    return perturbed.loc[~perturbed[id_col].isin(original)]


def negative_label_value(frame: pd.DataFrame, schema: Any) -> Any:
    """Infer the negative label literal from the baseline itself."""
    label_col = _schema_attr(schema, _LABEL_ATTRS, frame)
    if label_col is None:
        return None
    negatives = frame.loc[donor_mask(frame, schema), label_col]
    values = pd.unique(negatives.dropna())
    if len(values) == 1:
        return values[0]
    counts = frame[label_col].value_counts(dropna=True)
    return counts.index[0] if len(counts) else None


def assert_original_rows_preserved(
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
    *,
    schema: Any,
    context: str = "",
) -> dict[str, Any]:
    """Verify every baseline row survives untouched; injected rows may be added.

    Rows are matched by id, so the primitive's re-sort is not mistaken for a
    change. Comparison is by value, so dtype widening is not either - but any
    changed, dropped or duplicated original row is.
    """
    id_col = id_column(baseline, schema)
    where = f" [{context}]" if context else ""

    if perturbed[id_col].duplicated().any():
        dupes = perturbed.loc[perturbed[id_col].duplicated(), id_col].unique().tolist()[:5]
        raise LegitVolumeError(f"Duplicate transaction ids after resampling{where}: {dupes}.")

    missing = set(baseline[id_col].tolist()) - set(perturbed[id_col].tolist())
    if missing:
        raise LegitVolumeError(
            f"{len(missing)} original transaction id(s) disappeared after resampling{where}."
        )

    lost_columns = [c for c in baseline.columns if c not in perturbed.columns]
    if lost_columns:
        raise LegitVolumeError(f"Columns dropped by resampling{where}: {lost_columns}.")

    left = baseline.set_index(id_col).sort_index()
    right = perturbed.set_index(id_col).loc[left.index].sort_index()
    changed = [
        column for column in left.columns
        if not _series_values_equal(left[column], right[column])
    ]
    if changed:
        detail = {column: _difference_sample(left, right, column) for column in changed}
        raise LegitVolumeError(
            f"Original rows were modified by legitimate-volume resampling{where}; "
            f"columns: {changed}; examples: {detail}."
        )

    drift = dtype_drift(baseline, perturbed)
    if drift:
        LOGGER.warning(
            "Column dtypes drifted during resampling%s (values intact): %s", where, drift
        )

    return {
        "original_ids_preserved": True,
        "original_rows_unchanged": True,
        "unique_ids": True,
        "original_dtypes_preserved": not drift,
        "dtype_drift": drift,
        "injected_row_count": int(len(perturbed) - len(baseline)),
    }


def legit_volume_profile(
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
    *,
    schema: Any,
    level: VolumeLevel | None = None,
    entity_columns: Sequence[str] = (),
    id_prefix: str | None = None,
    refresh_entities: bool | None = None,
) -> dict[str, Any]:
    id_col = id_column(baseline, schema)
    label_col = _schema_attr(schema, _LABEL_ATTRS, baseline)
    campaign_col = _schema_attr(schema, _CAMPAIGN_ATTRS, baseline)
    time_col = _schema_attr(schema, _TIME_ATTRS, baseline)

    new_rows = injected_rows(baseline, perturbed, id_col)
    donors = donor_count(baseline, schema)
    negative_value = negative_label_value(baseline, schema)

    profile: dict[str, Any] = {
        "baseline_rows": int(len(baseline)),
        "perturbed_rows": int(len(perturbed)),
        "injected_rows": int(len(new_rows)),
        "donor_rows": donors,
        "attack_rows": int(len(baseline)) - donors,
        "row_growth_ratio": (float(len(perturbed)) / float(len(baseline))) if len(baseline) else None,
        "donor_growth_ratio": (float(len(new_rows)) / float(donors)) if donors else None,
        "label_column": label_col,
        "campaign_column": campaign_col,
        "negative_label_value": negative_value,
    }

    if level is not None:
        expected = expected_injected_rows(baseline, level=level, schema=schema)
        profile["expected_injected_rows"] = expected
        profile["injected_rows_match_expected"] = bool(len(new_rows) == expected)

    if len(new_rows):
        ids = new_rows[id_col].astype(str)
        profile["injected_ids_unique"] = bool(not new_rows[id_col].duplicated().any())
        if id_prefix:
            profile["injected_ids_use_prefix"] = bool(ids.str.startswith(id_prefix).all())
        if label_col is not None:
            labels = new_rows[label_col]
            profile["injected_label_values"] = sorted({str(v) for v in labels.tolist()})
            if negative_value is not None:
                profile["injected_all_negative"] = bool((labels == negative_value).all())
        if campaign_col is not None:
            values = sorted({str(v) for v in new_rows[campaign_col].tolist()})
            profile["injected_campaign_values"] = values
            profile["injected_campaigns_empty"] = bool(len(values) == 1 and values[0] in ("", "nan"))
        if time_col is not None:
            new_stamps = pd.to_datetime(new_rows[time_col], errors="coerce", utc=True).dropna()
            base_stamps = pd.to_datetime(baseline[time_col], errors="coerce", utc=True).dropna()
            if len(new_stamps) and len(base_stamps):
                profile["injected_time_span_seconds"] = float(
                    (new_stamps.max() - new_stamps.min()).total_seconds()
                )
                profile["injected_within_observation_window"] = bool(
                    (new_stamps >= base_stamps.min()).all() and (new_stamps <= base_stamps.max()).all()
                )
                profile["injected_timestamps_reused"] = bool(
                    new_stamps.isin(set(base_stamps)).all()
                )

        present = [c for c in entity_columns if c in new_rows.columns]
        if present:
            tokens = new_rows[present].astype(str).agg("|".join, axis=1)
            baseline_tokens = set(baseline[present].astype(str).agg("|".join, axis=1))
            profile["entity_columns"] = list(present)
            profile["distinct_injected_entity_tuples"] = int(tokens.nunique())
            profile["injected_entity_tuples_reused_from_baseline"] = int(
                sum(1 for t in tokens if t in baseline_tokens)
            )
            profile["injected_entities_all_fresh"] = bool(
                tokens.nunique() == len(tokens)
                and not any(t in baseline_tokens for t in tokens)
            )
    if refresh_entities is not None:
        profile["refresh_entities"] = bool(refresh_entities)
    return profile


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Order-independent-by-column, row-order-sensitive content fingerprint."""
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# world / rebuild plumbing
# --------------------------------------------------------------------------- #

def build_legit_volume_world(frame: pd.DataFrame, *, label: str) -> Any:
    if _build_world is None:
        return None
    attempts: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (
        ((frame,), {"label": label}),
        ((frame,), {"name": label}),
        ((frame,), {}),
    )
    for args, kwargs in attempts:
        try:
            return _build_world(*args, **kwargs)
        except TypeError:
            continue
        except Exception as exc:
            LOGGER.warning("build_world rejected arm %s: %s", label, exc)
            return None
    return None


def _world_fingerprint(world: Any) -> str | None:
    for attr in ("fingerprint", "world_fingerprint", "digest", "hash"):
        value = getattr(world, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:  # pragma: no cover
                continue
        if isinstance(value, str):
            return value
    return None


def _invoke_rebuild(rebuild_fn: Callable[..., Any], frame: pd.DataFrame, arm_id: str) -> Any:
    try:
        accepted = inspect.signature(rebuild_fn).parameters
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())
    except (TypeError, ValueError):  # pragma: no cover
        accepted, accepts_var_kw = {}, False
    for name in ("arm_id", "label", "name"):
        if name in accepted or accepts_var_kw:
            return rebuild_fn(frame, **{name: arm_id})
    return rebuild_fn(frame)


def _numeric(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def rebuild_metrics(rebuilt: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if rebuilt is None:
        return metrics
    if isinstance(rebuilt, Mapping):
        items = rebuilt.items()
    else:
        items = [(name, getattr(rebuilt, name)) for name in dir(rebuilt)
                 if not name.startswith("_") and not callable(getattr(rebuilt, name, None))]
    for name, value in items:
        if isinstance(value, pd.DataFrame):
            metrics[f"{name}_rows"] = int(len(value))
            metrics[f"{name}_columns"] = int(value.shape[1])
        elif isinstance(value, pd.Series):
            metrics[f"{name}_length"] = int(len(value))
        elif isinstance(value, (list, tuple, set, frozenset)):
            metrics[f"{name}_count"] = int(len(value))
        elif isinstance(value, Mapping):
            metrics[f"{name}_keys"] = int(len(value))
        else:
            number = _numeric(value)
            if number is not None:
                metrics[name] = number
    return metrics


_STRUCTURE_HINTS = ("candidate", "group", "component", "cluster", "campaign", "community")


def _structure_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if any(h in k.lower() for h in _STRUCTURE_HINTS)}


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #

def run_legit_volume_arm(
    baseline: pd.DataFrame,
    *,
    level: VolumeLevel | None,
    config: LegitVolumeConfig,
    rebuild_fn: Callable[..., Any] | None = None,
    schema: Any = None,
) -> dict[str, Any]:
    """Run one arm. ``level=None`` means the unperturbed control arm."""
    if not isinstance(baseline, pd.DataFrame):
        raise LegitVolumeError(f"baseline must be a DataFrame, got {type(baseline).__name__}.")
    if baseline.empty:
        raise LegitVolumeError("baseline frame is empty; S4 needs at least one transaction.")

    schema = scenario_schema(baseline, schema)
    rebuild_fn = rebuild_fn or rebuild_world
    source = baseline.copy(deep=True)

    if level is None:
        arm_id = CONTROL_ARM_ID
        kind = "control"
        frame = source
        used_kwargs: dict[str, Any] = {}
    else:
        arm_id = level.arm_id
        kind = "perturbed"
        frame, used_kwargs = apply_legit_volume(
            source, config=config, level=level, schema=schema
        )

    invariants = assert_original_rows_preserved(baseline, frame, schema=schema, context=arm_id)
    profile = legit_volume_profile(
        baseline, frame, schema=schema, level=level,
        entity_columns=used_kwargs.get("entity_columns", ()),
        id_prefix=effective_id_prefix(config) if level is not None else None,
        refresh_entities=config.refresh_entities if level is not None else None,
    )

    if config.strict_invariants and kind == "perturbed":
        if profile["injected_rows"] <= 0:
            raise LegitVolumeError(
                f"Arm {arm_id} injected no rows; resample_legitimate_transactions was "
                f"called with {sorted(used_kwargs)} but the population did not grow."
            )
        if profile.get("injected_rows_match_expected") is False:
            raise LegitVolumeError(
                f"Arm {arm_id} injected {profile['injected_rows']} rows but "
                f"{profile['expected_injected_rows']} were expected from a donor pool of "
                f"{profile['donor_rows']}."
            )
        if profile.get("injected_all_negative") is False:
            raise LegitVolumeError(
                f"Arm {arm_id} injected non-negative rows: "
                f"{profile.get('injected_label_values')}."
            )

    world = build_legit_volume_world(frame, label=f"{SCENARIO_ID}:{arm_id}")
    rebuilt = _invoke_rebuild(rebuild_fn, frame, arm_id)
    metrics = rebuild_metrics(rebuilt)

    return {
        "arm_id": arm_id,
        "kind": kind,
        "scenario": SCENARIO_ID,
        "parameters": {"level": None if level is None else level.as_dict(), **used_kwargs},
        "population": profile,
        "invariants": invariants,
        "frame_fingerprint": frame_fingerprint(frame),
        "world_fingerprint": _world_fingerprint(world),
        "rebuild": {"metrics": metrics, "structure_metrics": _structure_metrics(metrics)},
    }


def _compare(control: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    control_metrics = control["rebuild"]["metrics"]
    arm_metrics = arm["rebuild"]["metrics"]
    deltas: dict[str, Any] = {}
    for key, value in arm_metrics.items():
        base = control_metrics.get(key)
        if _numeric(base) is not None and _numeric(value) is not None:
            deltas[key] = value - base
    structure_deltas = _structure_metrics(deltas)
    positive = {k: v for k, v in structure_deltas.items() if v > 0}
    return {
        "arm_id": arm["arm_id"],
        "level": arm["parameters"].get("level"),
        "injected_rows": arm["population"]["injected_rows"],
        "row_growth_ratio": arm["population"]["row_growth_ratio"],
        "metric_deltas": deltas,
        "structure_deltas": structure_deltas,
        "additional_structures": (bool(positive) if structure_deltas else None),
        "additional_structure_metrics": positive,
        "frames_differ": arm["frame_fingerprint"] != control["frame_fingerprint"],
    }


# --------------------------------------------------------------------------- #
# JSON safety
# --------------------------------------------------------------------------- #

def _fallback_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _fallback_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_fallback_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _fallback_json_safe(value.item())
        except Exception:  # pragma: no cover
            pass
    return str(value)


def json_safe(payload: Any) -> Any:
    if _to_json_safe is not None:
        try:
            candidate = _to_json_safe(payload)
            json.dumps(candidate)
            return candidate
        except Exception:  # pragma: no cover
            LOGGER.debug("world.to_json_safe could not coerce the S4 result; using fallback.")
    return _fallback_json_safe(payload)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def run_legit_volume_scenario(
    frame: pd.DataFrame,
    *,
    config: LegitVolumeConfig | None = None,
    rebuild_fn: Callable[..., Any] | None = None,
    schema: Any = None,
) -> dict[str, Any]:
    """Run S4 end-to-end and return a deterministic, JSON-safe result."""
    config = config or LegitVolumeConfig()
    if not isinstance(config, LegitVolumeConfig):
        raise LegitVolumeError(
            f"config must be a LegitVolumeConfig, got {type(config).__name__}."
        )
    if not isinstance(frame, pd.DataFrame):
        raise LegitVolumeError(f"frame must be a DataFrame, got {type(frame).__name__}.")

    guard = frame_fingerprint(frame)
    baseline = frame.copy(deep=True)
    schema = scenario_schema(baseline, schema)
    rebuild_fn = rebuild_fn or rebuild_world

    arms: list[dict[str, Any]] = []
    control: dict[str, Any] | None = None
    if config.include_control:
        control = run_legit_volume_arm(
            baseline, level=None, config=config, rebuild_fn=rebuild_fn, schema=schema
        )
        arms.append(control)

    for level in config.volume_levels():
        arms.append(
            run_legit_volume_arm(
                baseline, level=level, config=config,
                rebuild_fn=rebuild_fn, schema=schema,
            )
        )

    comparisons = (
        [_compare(control, arm) for arm in arms if arm["kind"] == "perturbed"]
        if control is not None else []
    )

    flagged = [c["arm_id"] for c in comparisons if c["additional_structures"]]
    summary = {
        "arm_count": len(arms),
        "perturbed_arm_count": sum(1 for a in arms if a["kind"] == "perturbed"),
        "mode": config.mode,
        "arms_with_additional_structures": flagged,
        "structures_added_by_volume": (bool(flagged) if comparisons else None),
        "baseline_rows": int(len(baseline)),
        "donor_rows": donor_count(baseline, schema),
        "max_injected_rows": max((a["population"]["injected_rows"] for a in arms), default=0),
    }

    if frame_fingerprint(frame) != guard:
        raise LegitVolumeError("The input frame was mutated during the S4 scenario.")

    result = {
        "scenario": SCENARIO_ID,
        "scenario_name": SCENARIO_NAME,
        "config": config.as_dict(),
        "primitive": {
            "name": "resample_legitimate_transactions",
            "signature": str(primitive_signature()),
        },
        "control": control,
        "arms": arms,
        "comparisons": comparisons,
        "summary": summary,
    }
    return json_safe(result)


__all__ = [
    "SCENARIO_ID",
    "SCENARIO_NAME",
    "CONTROL_ARM_ID",
    "MULTIPLIER_MODE",
    "ABSOLUTE_MODE",
    "LegitVolumeConfig",
    "LegitVolumeError",
    "VolumeLevel",
    "apply_legit_volume",
    "assert_original_rows_preserved",
    "build_legit_volume_world",
    "donor_count",
    "donor_frame",
    "donor_mask",
    "dtype_drift",
    "effective_id_prefix",
    "expected_injected_rows",
    "frame_fingerprint",
    "id_column",
    "injected_ids",
    "injected_rows",
    "json_safe",
    "legit_volume_profile",
    "negative_label_value",
    "primitive_signature",
    "rebuild_metrics",
    "resolve_entity_columns",
    "restore_baseline_representation",
    "run_legit_volume_arm",
    "run_legit_volume_scenario",
    "scenario_schema",
    "schema_entity_columns",
    "volume_call_kwargs",
]
