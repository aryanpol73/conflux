"""Phase 4B - S3: False Positive Stress.

Stresses the rebuild pipeline with synthetic *benign* traffic bursts and reports
whether that alone manufactures additional suspicious candidate structures.

Hard rules honoured here:
  * the benign-burst primitive is NOT reimplemented; it is imported from
    ``conflux.robustness.perturbations`` and bound exactly to its real
    signature,
  * the input frame is never mutated,
  * original transaction ids / labels / campaign ids / payload columns are
    verified value-for-value after injection, and their baseline dtypes are
    restored so the frame handed downstream is byte-identical in the original
    rows,
  * no scorer is constructed, fitted or refitted anywhere in this module.

Bound primitive::

    inject_benign_bursts(
        frame, *, n_bursts, burst_size, span_seconds, seed,
        shared_entity_columns, vary_columns=(), id_prefix="SYNFP-", schema=None,
    ) -> pd.DataFrame
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from conflux.robustness.perturbations import (  # verified in earlier phases
    PerturbationError,
    inject_benign_bursts,
    resolve_schema,
)
from conflux.robustness.rebuild import rebuild_world

try:  # optional helpers, present in perturbations.py in earlier phases
    from conflux.robustness.perturbations import attack_mask as _attack_mask
except Exception:  # pragma: no cover - defensive only
    _attack_mask = None

try:
    from conflux.robustness.perturbations import find_entity_column as _find_entity_column
except Exception:  # pragma: no cover - defensive only
    _find_entity_column = None

try:  # world abstraction is used when its call shape is compatible
    from conflux.robustness.world import build_world as _build_world
except Exception:  # pragma: no cover
    _build_world = None

try:
    from conflux.robustness.world import to_json_safe as _to_json_safe
except Exception:  # pragma: no cover
    _to_json_safe = None


LOGGER = logging.getLogger(__name__)

SCENARIO_ID = "s3_false_positive_stress"
SCENARIO_NAME = "False Positive Stress"
CONTROL_ARM_ID = "control"


class FalsePositiveStressError(RuntimeError):
    """Raised on invalid configuration, unmappable primitive API, or a broken invariant."""


# --------------------------------------------------------------------------- #
# schema helpers (attribute names probed, never assumed)
# --------------------------------------------------------------------------- #

_ID_ATTRS = ("id_col", "id_column", "transaction_id_col", "transaction_id")
_LABEL_ATTRS = ("label_col", "label_column", "label")
_CAMPAIGN_ATTRS = ("campaign_col", "campaign_column", "campaign_id_col", "campaign_id")
_TIME_ATTRS = ("ts_col", "time_col", "timestamp_col", "timestamp_column", "time_column")


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
        return resolve_schema()  # signature variant without a frame argument


def id_column(frame: pd.DataFrame, schema: Any) -> str:
    col = _schema_attr(schema, _ID_ATTRS, frame)
    if col is None:
        raise FalsePositiveStressError(
            "Could not resolve the transaction id column from the resolved schema "
            f"({type(schema).__name__}); probed attributes: {_ID_ATTRS}."
        )
    return col


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise FalsePositiveStressError(
            f"{field_name} must be a sequence of column names, not a bare string ({value!r})."
        )
    if not isinstance(value, Sequence):
        raise FalsePositiveStressError(
            f"{field_name} must be a sequence of column names, got {type(value).__name__}."
        )
    columns = tuple(value)
    for column in columns:
        if not isinstance(column, str) or not column:
            raise FalsePositiveStressError(
                f"{field_name} entries must be non-empty strings, got {column!r}."
            )
    if len(set(columns)) != len(columns):
        raise FalsePositiveStressError(f"{field_name} must not contain duplicates: {columns!r}.")
    return columns


@dataclass(frozen=True)
class FalsePositiveStressConfig:
    """Configuration for S3.

    ``burst_counts`` defines one scenario arm per stress level; each level is
    forwarded to ``inject_benign_bursts`` as ``n_bursts``.

    ``shared_entity_columns`` is the entity linkage the synthetic benign rows
    share within a burst. ``None`` means "resolve from the repository schema at
    run time"; an explicit tuple is passed through unchanged and validated
    against the frame before the primitive is called.

    ``id_prefix=None`` means "use the primitive's own default", so that constant
    is never duplicated here.
    """

    burst_counts: tuple[int, ...] = (1, 3, 5)
    burst_size: int = 8
    span_seconds: float = 300.0
    seed: int = 4207
    shared_entity_columns: tuple[str, ...] | None = None
    vary_columns: tuple[str, ...] = ()
    id_prefix: str | None = None
    include_control: bool = True
    strict_invariants: bool = True
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counts = tuple(self.burst_counts or ())
        object.__setattr__(self, "burst_counts", counts)
        if not counts:
            raise FalsePositiveStressError("burst_counts must contain at least one stress level.")
        for value in counts:
            if isinstance(value, bool) or not isinstance(value, int):
                raise FalsePositiveStressError(f"burst_counts entries must be ints, got {value!r}.")
            if value <= 0:
                raise FalsePositiveStressError(f"burst_counts entries must be > 0, got {value!r}.")
        if len(set(counts)) != len(counts):
            raise FalsePositiveStressError(f"burst_counts must be unique, got {counts!r}.")

        if isinstance(self.burst_size, bool) or not isinstance(self.burst_size, int) or self.burst_size < 1:
            raise FalsePositiveStressError(f"burst_size must be an int >= 1, got {self.burst_size!r}.")

        if not isinstance(self.span_seconds, (int, float)) or isinstance(self.span_seconds, bool):
            raise FalsePositiveStressError(f"span_seconds must be numeric, got {self.span_seconds!r}.")
        if float(self.span_seconds) <= 0.0:
            raise FalsePositiveStressError(f"span_seconds must be > 0, got {self.span_seconds!r}.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise FalsePositiveStressError(f"seed must be a non-negative int, got {self.seed!r}.")

        if self.shared_entity_columns is not None:
            shared = _string_tuple(self.shared_entity_columns, "shared_entity_columns")
            if not shared:
                raise FalsePositiveStressError(
                    "shared_entity_columns must be None (resolve from schema) or a non-empty sequence."
                )
            object.__setattr__(self, "shared_entity_columns", shared)
        else:
            shared = ()

        vary = _string_tuple(self.vary_columns, "vary_columns")
        object.__setattr__(self, "vary_columns", vary)
        overlap = sorted(set(shared) & set(vary))
        if overlap:
            raise FalsePositiveStressError(
                f"Columns cannot be both shared and varied within a burst: {overlap}."
            )

        if self.id_prefix is not None:
            if not isinstance(self.id_prefix, str) or not self.id_prefix:
                raise FalsePositiveStressError(
                    f"id_prefix must be None or a non-empty string, got {self.id_prefix!r}."
                )

        if not isinstance(self.extra_kwargs, Mapping):
            raise FalsePositiveStressError("extra_kwargs must be a mapping.")
        object.__setattr__(self, "extra_kwargs", dict(self.extra_kwargs))

    def as_dict(self) -> dict[str, Any]:
        return {
            "burst_counts": list(self.burst_counts),
            "burst_size": self.burst_size,
            "span_seconds": float(self.span_seconds),
            "seed": self.seed,
            "shared_entity_columns": (
                None if self.shared_entity_columns is None else list(self.shared_entity_columns)
            ),
            "vary_columns": list(self.vary_columns),
            "id_prefix": self.id_prefix,
            "include_control": self.include_control,
            "strict_invariants": self.strict_invariants,
            "extra_kwargs": dict(self.extra_kwargs),
        }


# --------------------------------------------------------------------------- #
# entity-column resolution (schema-driven, never hardcoded)
# --------------------------------------------------------------------------- #

_ENTITY_LIST_ATTRS = (
    "entity_columns", "entity_cols", "entities",
    "shared_entity_columns", "entity_column_names", "entity",
)
_ENTITY_SINGLE_ATTRS = ("entity_column", "entity_col")


def schema_entity_columns(frame: pd.DataFrame, schema: Any) -> tuple[str, ...]:
    """Best-effort entity columns for ``frame``, sourced only from repository mechanisms."""
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

    for name in _ENTITY_SINGLE_ATTRS:
        value = getattr(schema, name, None)
        if isinstance(value, str) and value in frame.columns:
            return (value,)

    if _find_entity_column is not None:
        try:
            column = _find_entity_column(frame, schema=schema)
        except TypeError:
            try:
                column = _find_entity_column(frame)
            except Exception:  # pragma: no cover
                column = None
        except Exception:  # pragma: no cover
            column = None
        if isinstance(column, str) and column in frame.columns:
            return (column,)

    return ()


def resolve_shared_entity_columns(
    frame: pd.DataFrame,
    *,
    config: FalsePositiveStressConfig,
    schema: Any,
) -> tuple[str, ...]:
    """Return the ``shared_entity_columns`` to hand to the primitive.

    Explicitly configured columns are validated against ``frame`` and a missing
    column raises the repository's ``PerturbationError`` rather than continuing.
    """
    if config.shared_entity_columns is not None:
        missing = [c for c in config.shared_entity_columns if c not in frame.columns]
        if missing:
            raise PerturbationError(
                f"shared_entity_columns not present in the frame: {missing}; "
                f"available columns: {sorted(frame.columns)}."
            )
        return tuple(config.shared_entity_columns)

    resolved = schema_entity_columns(frame, schema)
    if not resolved:
        raise FalsePositiveStressError(
            "shared_entity_columns is required by inject_benign_bursts but could not be "
            f"resolved from the schema ({type(schema).__name__}); probed "
            f"{_ENTITY_LIST_ATTRS + _ENTITY_SINGLE_ATTRS} and find_entity_column. "
            "Set FalsePositiveStressConfig.shared_entity_columns explicitly."
        )
    return resolved


def resolve_vary_columns(
    frame: pd.DataFrame,
    *,
    config: FalsePositiveStressConfig,
) -> tuple[str, ...]:
    missing = [c for c in config.vary_columns if c not in frame.columns]
    if missing:
        raise PerturbationError(
            f"vary_columns not present in the frame: {missing}; "
            f"available columns: {sorted(frame.columns)}."
        )
    return tuple(config.vary_columns)


# --------------------------------------------------------------------------- #
# value / representation comparison
#
# pd.concat inside the primitive can widen a column's dtype (datetime
# resolution, or object holding Timestamps) without touching any value. That is
# a representation change, not a data change: it must be repaired, not treated
# as tampering. Real value changes must still be caught.
# --------------------------------------------------------------------------- #

def _is_datetimelike(series: pd.Series) -> bool:
    dtype = series.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return True
    return getattr(dtype, "kind", "") == "M"


def _as_utc_ns(series: pd.Series):
    """Instant-preserving normalisation used only for datetime-like columns."""
    converted = pd.to_datetime(series, utc=True, errors="coerce")
    return converted.to_numpy(dtype="datetime64[ns]")


def _scalar_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    left_na, right_na = pd.isna(left), pd.isna(right)
    if bool(left_na) or bool(right_na):
        return bool(left_na) == bool(right_na)
    try:
        if bool(left == right):
            return True
    except Exception:  # pragma: no cover - exotic __eq__
        return False
    try:  # int 1 vs float 1.0 after a concat-driven widening
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def _series_values_equal(left: pd.Series, right: pd.Series) -> bool:
    """Value equality that ignores dtype/representation but nothing else."""
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
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
    limit: int = 3,
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
    """Cast back to ``dtype``; ``None`` when no instant/value-preserving cast exists."""
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
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
) -> pd.DataFrame:
    """Undo concat-driven dtype/column-order drift without changing any value.

    A cast is kept only when it round-trips every value in the column, so this
    can never paper over an actual modification.
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
# primitive adapter - exact binding to inject_benign_bursts
# --------------------------------------------------------------------------- #

_REQUIRED_PRIMITIVE_ARGS = (
    "n_bursts", "burst_size", "span_seconds", "seed", "shared_entity_columns",
)
_OPTIONAL_PRIMITIVE_ARGS = ("vary_columns", "id_prefix", "schema")


def primitive_signature() -> inspect.Signature:
    return inspect.signature(inject_benign_bursts)


def effective_id_prefix(config: FalsePositiveStressConfig) -> str | None:
    """Configured prefix, else the primitive's own declared default."""
    if config.id_prefix is not None:
        return config.id_prefix
    parameter = primitive_signature().parameters.get("id_prefix")
    if parameter is not None and parameter.default is not inspect.Parameter.empty:
        value = parameter.default
        return value if isinstance(value, str) else None
    return None


def burst_call_kwargs(
    *,
    config: FalsePositiveStressConfig,
    burst_count: int,
    shared_entity_columns: Sequence[str],
    vary_columns: Sequence[str] = (),
    schema: Any = None,
) -> dict[str, Any]:
    """Build the keyword arguments for the real ``inject_benign_bursts``."""
    signature = primitive_signature()
    params = signature.parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs: dict[str, Any] = {
        "n_bursts": int(burst_count),
        "burst_size": int(config.burst_size),
        "span_seconds": float(config.span_seconds),
        "seed": int(config.seed),
        "shared_entity_columns": tuple(shared_entity_columns),
    }
    if vary_columns:
        kwargs["vary_columns"] = tuple(vary_columns)
    if config.id_prefix is not None:
        kwargs["id_prefix"] = config.id_prefix
    if schema is not None and "schema" in params:
        kwargs["schema"] = schema

    for key, value in config.extra_kwargs.items():
        if key in params or accepts_var_kw:
            kwargs[key] = value
        else:
            raise FalsePositiveStressError(
                f"extra_kwargs key {key!r} is not accepted by inject_benign_bursts{signature}."
            )

    unexpected = [key for key in kwargs if key not in params and not accepts_var_kw]
    if unexpected:
        raise FalsePositiveStressError(
            f"inject_benign_bursts{signature} does not accept {unexpected!r}; "
            "the primitive signature changed - update the adapter in "
            "scenario_false_positive_stress.py (do not edit perturbations.py)."
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
        raise FalsePositiveStressError(
            "Cannot map FalsePositiveStressConfig onto the real primitive: "
            f"inject_benign_bursts{signature} requires {missing!r}. "
            "Supply them via FalsePositiveStressConfig.extra_kwargs."
        )
    return kwargs


def apply_benign_bursts(
    frame: pd.DataFrame,
    *,
    config: FalsePositiveStressConfig,
    burst_count: int,
    schema: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Call the real primitive; return ``(perturbed_frame, reported_kwargs)``.

    The returned frame carries the baseline's column order and dtypes so the
    original rows are byte-identical to the input.
    """
    schema = scenario_schema(frame, schema)
    shared = resolve_shared_entity_columns(frame, config=config, schema=schema)
    vary = resolve_vary_columns(frame, config=config)
    kwargs = burst_call_kwargs(
        config=config,
        burst_count=burst_count,
        shared_entity_columns=shared,
        vary_columns=vary,
        schema=schema,
    )
    try:
        result = inject_benign_bursts(frame, **kwargs)
    except PerturbationError:
        raise
    except TypeError as exc:  # signature mismatch surfaced explicitly
        raise FalsePositiveStressError(
            f"inject_benign_bursts rejected the adapted call {sorted(kwargs)}: {exc}"
        ) from exc

    if isinstance(result, tuple) and result and isinstance(result[0], pd.DataFrame):
        result = result[0]
    if not isinstance(result, pd.DataFrame):
        raise FalsePositiveStressError(
            f"inject_benign_bursts returned {type(result).__name__}, expected a DataFrame."
        )

    result = restore_baseline_representation(frame, result)

    reported = {k: v for k, v in kwargs.items() if k != "schema"}
    reported["shared_entity_columns"] = list(shared)
    reported["vary_columns"] = list(vary)
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


def benign_label_value(frame: pd.DataFrame, schema: Any) -> Any:
    """Infer the benign label value from the baseline itself (no hardcoded semantics)."""
    label_col = _schema_attr(schema, _LABEL_ATTRS, frame)
    if label_col is None:
        return None
    if _attack_mask is not None:
        try:
            mask = _attack_mask(frame, schema=schema)
        except TypeError:
            mask = _attack_mask(frame)
        except Exception:  # pragma: no cover - primitive refused this frame
            mask = None
        if mask is not None:
            benign = frame.loc[~pd.Series(mask, index=frame.index).astype(bool), label_col]
            values = pd.unique(benign.dropna())
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

    Comparison is by value, so a concat-driven dtype widening is not mistaken
    for tampering - but any changed, dropped or duplicated original row is.
    """
    id_col = id_column(baseline, schema)
    where = f" [{context}]" if context else ""

    if perturbed[id_col].duplicated().any():
        dupes = perturbed.loc[perturbed[id_col].duplicated(), id_col].unique().tolist()[:5]
        raise FalsePositiveStressError(f"Duplicate transaction ids after injection{where}: {dupes}.")

    missing = set(baseline[id_col].tolist()) - set(perturbed[id_col].tolist())
    if missing:
        raise FalsePositiveStressError(
            f"{len(missing)} original transaction id(s) disappeared after injection{where}."
        )

    lost_columns = [c for c in baseline.columns if c not in perturbed.columns]
    if lost_columns:
        raise FalsePositiveStressError(f"Columns dropped by injection{where}: {lost_columns}.")

    left = baseline.set_index(id_col).sort_index()
    right = perturbed.set_index(id_col).loc[left.index].sort_index()
    changed = [
        column for column in left.columns
        if not _series_values_equal(left[column], right[column])
    ]
    if changed:
        detail = {column: _difference_sample(left, right, column) for column in changed}
        raise FalsePositiveStressError(
            f"Original rows were modified by benign burst injection{where}; "
            f"columns: {changed}; examples: {detail}."
        )

    drift = dtype_drift(baseline, perturbed)
    if drift:
        LOGGER.warning(
            "Column dtypes drifted during injection%s (values intact): %s", where, drift
        )

    return {
        "original_ids_preserved": True,
        "original_rows_unchanged": True,
        "unique_ids": True,
        "original_dtypes_preserved": not drift,
        "dtype_drift": drift,
        "injected_row_count": int(len(perturbed) - len(baseline)),
    }


def benign_injection_profile(
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
    *,
    schema: Any,
    shared_entity_columns: Sequence[str] = (),
    id_prefix: str | None = None,
) -> dict[str, Any]:
    id_col = id_column(baseline, schema)
    label_col = _schema_attr(schema, _LABEL_ATTRS, baseline)
    campaign_col = _schema_attr(schema, _CAMPAIGN_ATTRS, baseline)
    time_col = _schema_attr(schema, _TIME_ATTRS, baseline)

    new_rows = injected_rows(baseline, perturbed, id_col)
    benign_value = benign_label_value(baseline, schema)

    profile: dict[str, Any] = {
        "baseline_rows": int(len(baseline)),
        "perturbed_rows": int(len(perturbed)),
        "injected_rows": int(len(new_rows)),
        "row_growth_ratio": (float(len(perturbed)) / float(len(baseline))) if len(baseline) else None,
        "label_column": label_col,
        "campaign_column": campaign_col,
        "benign_label_value": benign_value,
    }
    if label_col is not None and len(new_rows):
        labels = new_rows[label_col]
        profile["injected_label_values"] = sorted({str(v) for v in labels.tolist()})
        if benign_value is not None:
            profile["injected_all_benign"] = bool((labels == benign_value).all())
    if campaign_col is not None and len(new_rows):
        profile["injected_campaign_values"] = sorted({str(v) for v in new_rows[campaign_col].tolist()})
    if time_col is not None and len(new_rows):
        stamps = pd.to_datetime(new_rows[time_col], errors="coerce", utc=True).dropna()
        if len(stamps):
            profile["injected_time_span_seconds"] = float((stamps.max() - stamps.min()).total_seconds())
    if id_prefix and len(new_rows):
        ids = new_rows[id_col].astype(str)
        profile["injected_ids_use_prefix"] = bool(ids.str.startswith(id_prefix).all())
    shared = [c for c in shared_entity_columns if c in new_rows.columns]
    if shared and len(new_rows):
        groups = new_rows[shared].astype(str).agg("|".join, axis=1)
        profile["shared_entity_columns"] = list(shared)
        profile["distinct_shared_entity_tuples"] = int(groups.nunique())
    return profile


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Order-independent-by-column, row-order-sensitive content fingerprint."""
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# world / rebuild plumbing
# --------------------------------------------------------------------------- #

def build_false_positive_world(frame: pd.DataFrame, *, label: str) -> Any:
    """Use the repository world abstraction when its call shape is compatible."""
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
        except Exception as exc:  # world guards rejected the frame - report, do not hide
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
    """Call the rebuild layer, passing an arm identifier only if it is accepted."""
    try:
        signature = inspect.signature(rebuild_fn)
        accepted = signature.parameters
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())
    except (TypeError, ValueError):  # pragma: no cover - builtins / C callables
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
    """Extract scalar metrics from whatever the rebuild layer returns, without
    hardcoding candidate field names."""
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

def run_false_positive_arm(
    baseline: pd.DataFrame,
    *,
    burst_count: int | None,
    config: FalsePositiveStressConfig,
    rebuild_fn: Callable[..., Any] | None = None,
    schema: Any = None,
) -> dict[str, Any]:
    """Run one arm. ``burst_count=None`` means the unperturbed control arm."""
    if not isinstance(baseline, pd.DataFrame):
        raise FalsePositiveStressError(f"baseline must be a DataFrame, got {type(baseline).__name__}.")
    if baseline.empty:
        raise FalsePositiveStressError("baseline frame is empty; S3 needs at least one transaction.")

    schema = scenario_schema(baseline, schema)
    rebuild_fn = rebuild_fn or rebuild_world
    source = baseline.copy(deep=True)

    if burst_count is None:
        arm_id = CONTROL_ARM_ID
        kind = "control"
        frame = source
        used_kwargs: dict[str, Any] = {}
    else:
        arm_id = f"bursts={burst_count}"
        kind = "perturbed"
        frame, used_kwargs = apply_benign_bursts(
            source, config=config, burst_count=burst_count, schema=schema
        )

    prefix = effective_id_prefix(config) if burst_count is not None else None
    shared = list(used_kwargs.get("shared_entity_columns", ()))

    invariants = assert_original_rows_preserved(baseline, frame, schema=schema, context=arm_id)
    profile = benign_injection_profile(
        baseline, frame, schema=schema,
        shared_entity_columns=shared, id_prefix=prefix,
    )

    if config.strict_invariants and kind == "perturbed" and profile["injected_rows"] <= 0:
        raise FalsePositiveStressError(
            f"Arm {arm_id} injected no rows; inject_benign_bursts was called with "
            f"{sorted(used_kwargs)} but the population did not grow."
        )

    world = build_false_positive_world(frame, label=f"{SCENARIO_ID}:{arm_id}")
    rebuilt = _invoke_rebuild(rebuild_fn, frame, arm_id)
    metrics = rebuild_metrics(rebuilt)

    return {
        "arm_id": arm_id,
        "kind": kind,
        "scenario": SCENARIO_ID,
        "parameters": {"burst_count": burst_count, **used_kwargs},
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
        "burst_count": arm["parameters"].get("burst_count"),
        "injected_rows": arm["population"]["injected_rows"],
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
        except Exception:  # pragma: no cover - fall back to the local coercion
            LOGGER.debug("world.to_json_safe could not coerce the S3 result; using fallback.")
    return _fallback_json_safe(payload)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def run_false_positive_stress_scenario(
    frame: pd.DataFrame,
    *,
    config: FalsePositiveStressConfig | None = None,
    rebuild_fn: Callable[..., Any] | None = None,
    schema: Any = None,
) -> dict[str, Any]:
    """Run S3 end-to-end and return a deterministic, JSON-safe result."""
    config = config or FalsePositiveStressConfig()
    if not isinstance(config, FalsePositiveStressConfig):
        raise FalsePositiveStressError(
            f"config must be a FalsePositiveStressConfig, got {type(config).__name__}."
        )
    if not isinstance(frame, pd.DataFrame):
        raise FalsePositiveStressError(f"frame must be a DataFrame, got {type(frame).__name__}.")

    guard = frame_fingerprint(frame)
    baseline = frame.copy(deep=True)
    schema = scenario_schema(baseline, schema)
    rebuild_fn = rebuild_fn or rebuild_world

    arms: list[dict[str, Any]] = []
    control: dict[str, Any] | None = None
    if config.include_control:
        control = run_false_positive_arm(
            baseline, burst_count=None, config=config, rebuild_fn=rebuild_fn, schema=schema
        )
        arms.append(control)

    for burst_count in config.burst_counts:
        arms.append(
            run_false_positive_arm(
                baseline, burst_count=burst_count, config=config,
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
        "arms_with_additional_structures": flagged,
        "false_positive_pressure_detected": (bool(flagged) if comparisons else None),
        "baseline_rows": int(len(baseline)),
        "max_injected_rows": max((a["population"]["injected_rows"] for a in arms), default=0),
    }

    if frame_fingerprint(frame) != guard:
        raise FalsePositiveStressError("The input frame was mutated during the S3 scenario.")

    result = {
        "scenario": SCENARIO_ID,
        "scenario_name": SCENARIO_NAME,
        "config": config.as_dict(),
        "primitive": {
            "name": "inject_benign_bursts",
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
    "FalsePositiveStressConfig",
    "FalsePositiveStressError",
    "apply_benign_bursts",
    "assert_original_rows_preserved",
    "benign_injection_profile",
    "benign_label_value",
    "build_false_positive_world",
    "burst_call_kwargs",
    "dtype_drift",
    "effective_id_prefix",
    "frame_fingerprint",
    "id_column",
    "injected_ids",
    "injected_rows",
    "json_safe",
    "primitive_signature",
    "rebuild_metrics",
    "restore_baseline_representation",
    "resolve_shared_entity_columns",
    "resolve_vary_columns",
    "run_false_positive_arm",
    "run_false_positive_stress_scenario",
    "scenario_schema",
    "schema_entity_columns",
]
