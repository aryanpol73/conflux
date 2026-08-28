"""CONFLUX Phase 4B -- deterministic world perturbation primitives.

WHAT THIS FILE IS
-----------------
Composable, seeded, side-effect-free DataFrame transformations that build a
PERTURBED WORLD. A world produced here is fed to
conflux.robustness.rebuild.rebuild_world(), which runs it through the FROZEN
Phase 3B / Phase 4A pipeline.

WHAT THIS FILE IS NOT
---------------------
It contains no scenarios, no metrics, no scorer, no thresholds and no model.
It never calls rebuild_world(); scenario modules do that.

SCHEMA DISCIPLINE
-----------------
No raw column name is hard-coded except those verified in
conflux.scoring.config.ALLOWED_RAW_COLUMNS. Everything else is resolved at
runtime from conflux.graph.config. If a name cannot be resolved, the function
raises PerturbationError naming the exact missing symbol.

GROUND TRUTH
------------
label / campaign_id are read ONLY to select which rows a perturbation acts on.
They are always carried through unmodified and are never used to compute a
feature. Injected rows are always labelled negative.

DETERMINISM
-----------
Every stochastic function takes `seed` and uses numpy.random.Generator. No
global RNG state is touched. Identical (frame, seed, kwargs) gives identical
output, including row order.

TIME UNIT CONTRACT
------------------
timestamps_as_ns() returns TRUE NANOSECONDS since the Unix epoch, always.
pandas >= 2.0 parses timestamp strings to the lowest sufficient resolution, so
a '%f' format yields datetime64[us]; a bare .astype("int64") on that silently
returns microseconds. _to_ns_int64() normalizes the unit before the integer
cast so the contract holds regardless of the parsed resolution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# TS_FORMAT / ns rendering are shared with the rebuild seam so a perturbed
# timestamp is byte-identical in form to an unperturbed one. rebuild.py is NOT
# modified by this import.
from conflux.robustness.rebuild import TS_FORMAT, ns_to_timestamp_strings

log = logging.getLogger("conflux.robustness.perturbations")

NS_PER_SECOND: int = 1_000_000_000


class PerturbationError(ValueError):
    """A perturbation cannot be performed on the supplied frame."""


# ----------------------------------------------------------------------
# schema resolution -- runtime, never hard-coded
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchemaView:
    """The real raw schema, read from conflux.graph.config at runtime."""

    id_col: str
    ts_col: str
    structural: tuple[str, ...]
    attribute: tuple[str, ...]
    entity: tuple[str, ...]
    forbidden: tuple[str, ...]
    label_col: str
    campaign_col: str
    amount_col: str = "amount"
    auth_col: str = "auth_outcome"
    unresolved: tuple[str, ...] = field(default=())

    def all_columns(self) -> tuple[str, ...]:
        return (*self.structural, *self.attribute, self.label_col,
                self.campaign_col)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id_col": self.id_col,
            "ts_col": self.ts_col,
            "structural_columns": list(self.structural),
            "attribute_columns": list(self.attribute),
            "entity_columns": list(self.entity),
            "forbidden_graph_inputs": list(self.forbidden),
            "label_col": self.label_col,
            "campaign_col": self.campaign_col,
            "amount_col": self.amount_col,
            "auth_col": self.auth_col,
            "unresolved_symbols": list(self.unresolved),
        }


def _import_attr(module: str, attr: str) -> tuple[Any, str | None]:
    import importlib

    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - diagnostic
        return None, f"{module}: {type(exc).__name__}: {exc}"
    if not hasattr(mod, attr):
        return None, f"{module}.{attr}: not found"
    return getattr(mod, attr), None


def resolve_schema() -> SchemaView:
    """Read the actual raw schema from the frozen project configuration.

    Raises PerturbationError naming the missing symbol rather than guessing a
    column name.
    """
    missing: list[str] = []

    id_col, e = _import_attr("conflux.graph.config", "ID_COL")
    if e:
        missing.append(e)
    ts_col, e = _import_attr("conflux.graph.config", "TS_COL")
    if e:
        missing.append(e)
    structural, e = _import_attr("conflux.graph.config", "STRUCTURAL_COLUMNS")
    if e:
        missing.append(e)
    attribute, e = _import_attr("conflux.graph.config", "ATTRIBUTE_COLUMNS")
    if e:
        missing.append(e)
    forbidden, e = _import_attr("conflux.graph.config", "FORBIDDEN_GRAPH_INPUTS")
    if e:
        missing.append(e)

    if missing:
        raise PerturbationError(
            "cannot resolve the raw schema from conflux.graph.config; refusing "
            "to guess column names. Missing: " + "; ".join(missing))

    # ENTITY_COLUMNS is optional: fall back to structural minus id/timestamp.
    unresolved: list[str] = []
    entity, e = _import_attr("conflux.graph.config", "ENTITY_COLUMNS")
    if e:
        unresolved.append(e)
        entity = tuple(c for c in structural if c not in (id_col, ts_col))

    campaign_col, e = _import_attr("conflux.evaluation.campaign_evaluation",
                                   "CAMPAIGN_COL")
    if e:
        raise PerturbationError(
            "cannot resolve CAMPAIGN_COL from conflux.evaluation."
            "campaign_evaluation: " + e)

    label_col, e = _import_attr("conflux.evaluation.campaign_evaluation",
                                "LABEL_COL")
    if e:
        unresolved.append(e)
        label_col = "label"

    return SchemaView(
        id_col=str(id_col), ts_col=str(ts_col),
        structural=tuple(structural), attribute=tuple(attribute),
        entity=tuple(entity), forbidden=tuple(forbidden),
        label_col=str(label_col), campaign_col=str(campaign_col),
        unresolved=tuple(unresolved))


def describe_schema(frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Inspection helper. Print this BEFORE writing any scenario."""
    schema = resolve_schema()
    out = schema.as_dict()
    if frame is not None:
        out["frame_columns"] = list(frame.columns)
        out["missing_from_frame"] = [c for c in schema.all_columns()
                                     if c not in frame.columns]
        out["extra_in_frame"] = [c for c in frame.columns
                                 if c not in schema.all_columns()]
    return out


def find_entity_column(substring: str, *, schema: SchemaView | None = None,
                       candidates: Sequence[str] | None = None) -> str:
    """Locate a real column by substring against the RESOLVED schema.

    This is inspection, not guessing: it fails loudly on zero or multiple
    matches rather than returning an assumed name.
    """
    s = schema or resolve_schema()
    pool = list(candidates) if candidates is not None else list(
        dict.fromkeys((*s.entity, *s.structural, *s.attribute)))
    hits = [c for c in pool if substring.lower() in c.lower()]
    if not hits:
        raise PerturbationError(
            f"no column matching '{substring}' in resolved schema {pool}")
    if len(hits) > 1:
        raise PerturbationError(
            f"'{substring}' is ambiguous: matches {hits}. Pass the column "
            "explicitly.")
    return hits[0]


# ----------------------------------------------------------------------
# small shared utilities
# ----------------------------------------------------------------------
def make_rng(seed: int) -> np.random.Generator:
    """The ONLY randomness source in Phase 4B perturbations."""
    if seed is None:
        raise PerturbationError("a seed is required; Phase 4B must be "
                                "reproducible")
    return np.random.default_rng(int(seed))


def _require_columns(frame: pd.DataFrame, cols: Iterable[str], op: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise PerturbationError(
            f"{op}: frame is missing required column(s) {missing}; present "
            f"columns are {list(frame.columns)}")


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (series.astype(str).str.strip().str.lower()
            .isin({"true", "1", "yes", "t"}))


def _to_ns_int64(parsed: pd.Series) -> np.ndarray:
    """datetime64 -> TRUE nanoseconds since epoch.

    pandas >= 2.0 parses strings to the LOWEST sufficient resolution, so a
    '%f' format yields datetime64[us]. A bare .astype("int64") on that returns
    MICROSECONDS, silently violating this module's nanosecond contract and
    producing a 1000x scale error on round-trip. The unit is therefore
    normalized explicitly before the integer cast.
    """
    ser = pd.Series(parsed)
    if isinstance(ser.dtype, pd.DatetimeTZDtype):
        ser = ser.dt.tz_convert("UTC").dt.tz_localize(None)
    as_unit = getattr(ser.dt, "as_unit", None)
    ser = as_unit("ns") if as_unit is not None else ser.astype("datetime64[ns]")
    return ser.astype("int64").to_numpy()


def attack_mask(frame: pd.DataFrame, *, schema: SchemaView | None = None
                ) -> pd.Series:
    """Boolean mask of attack rows. Used ONLY for row selection."""
    s = schema or resolve_schema()
    _require_columns(frame, [s.label_col], "attack_mask")
    return _as_bool_series(frame[s.label_col])


def _negative_label_literal(frame: pd.DataFrame, schema: SchemaView) -> Any:
    """Reproduce the exact on-disk representation of a negative label."""
    mask = attack_mask(frame, schema=schema)
    neg = frame.loc[~mask, schema.label_col]
    if len(neg) == 0:
        raise PerturbationError("frame contains no negative rows; cannot infer "
                                "the negative label representation")
    return neg.iloc[0]


def _empty_campaign_literal(frame: pd.DataFrame, schema: SchemaView) -> Any:
    mask = attack_mask(frame, schema=schema)
    neg = frame.loc[~mask, schema.campaign_col]
    if len(neg) == 0:
        return ""
    return neg.mode().iloc[0] if not neg.mode().empty else ""


def timestamps_as_ns(frame: pd.DataFrame, *, schema: SchemaView | None = None
                     ) -> np.ndarray:
    """Parse the timestamp column to int64 nanoseconds. Raises on unparseable."""
    s = schema or resolve_schema()
    _require_columns(frame, [s.ts_col], "timestamps_as_ns")
    col = frame[s.ts_col]
    if pd.api.types.is_datetime64_any_dtype(col):
        parsed = col
    else:
        parsed = pd.to_datetime(col, format=TS_FORMAT, errors="coerce")
        if parsed.isna().any():
            parsed = pd.to_datetime(col, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise PerturbationError(
            f"{bad} timestamp value(s) in '{s.ts_col}' could not be parsed")
    return _to_ns_int64(parsed)


def with_timestamps_ns(frame: pd.DataFrame, ts_ns: np.ndarray, *,
                       schema: SchemaView | None = None) -> pd.DataFrame:
    """Return a COPY with the timestamp column rewritten from int64 ns."""
    s = schema or resolve_schema()
    ts_ns = np.asarray(ts_ns, dtype="int64")
    if len(ts_ns) != len(frame):
        raise PerturbationError(
            f"timestamp array length {len(ts_ns)} != frame length {len(frame)}")
    out = frame.copy()
    rendered = ns_to_timestamp_strings(ts_ns)
    out[s.ts_col] = rendered.to_numpy()
    return out


def observation_window_ns(frame: pd.DataFrame, *,
                          schema: SchemaView | None = None) -> tuple[int, int]:
    ts = timestamps_as_ns(frame, schema=schema)
    return int(ts.min()), int(ts.max())


def _fresh_ids(frame: pd.DataFrame, n: int, *, schema: SchemaView,
               prefix: str) -> list[str]:
    existing = set(frame[schema.id_col].astype(str))
    out: list[str] = []
    i = 0
    while len(out) < n:
        cand = f"{prefix}{i:08d}"
        if cand not in existing:
            out.append(cand)
            existing.add(cand)
        i += 1
    return out


def _sorted_by_id(frame: pd.DataFrame, schema: SchemaView) -> pd.DataFrame:
    """Stable canonical ordering so results are order-independent."""
    return frame.sort_values(schema.id_col, kind="mergesort").reset_index(drop=True)


# ----------------------------------------------------------------------
# 1. cadence
# ----------------------------------------------------------------------
def scale_group_cadence(frame: pd.DataFrame, *, factor: float,
                        group_col: str | None = None,
                        group_values: Sequence[Any] | None = None,
                        anchor: str = "start",
                        schema: SchemaView | None = None) -> pd.DataFrame:
    """Stretch (factor > 1) or compress (factor < 1) intra-group inter-arrivals.

    Deterministic; no RNG. Each group is rescaled about its own anchor so the
    group stays where it was in absolute time and only its CADENCE changes.
    """
    s = schema or resolve_schema()
    gcol = group_col or s.campaign_col
    _require_columns(frame, [gcol, s.ts_col, s.id_col], "scale_group_cadence")
    if factor <= 0:
        raise PerturbationError(f"factor must be > 0, got {factor}")
    if anchor not in ("start", "end", "median"):
        raise PerturbationError(f"anchor must be start/end/median, got {anchor}")

    ts = timestamps_as_ns(frame, schema=s).astype("float64")
    groups = frame[gcol].astype(str)
    selected = (set(str(v) for v in group_values)
                if group_values is not None
                else {g for g in groups.unique() if g not in ("", "nan", "None")})
    if not selected:
        raise PerturbationError("no groups selected for cadence scaling")

    new_ts = ts.copy()
    touched = 0
    for g in sorted(selected):
        idx = np.flatnonzero((groups == g).to_numpy())
        if idx.size < 2:
            continue
        vals = ts[idx]
        if anchor == "start":
            a = vals.min()
        elif anchor == "end":
            a = vals.max()
        else:
            a = float(np.median(vals))
        new_ts[idx] = a + (vals - a) * float(factor)
        touched += idx.size

    if touched == 0:
        raise PerturbationError(
            "cadence scaling matched no multi-transaction group; check "
            f"group_col='{gcol}' and group_values")

    log.info("scale_group_cadence: factor=%s, groups=%s, rows=%s",
             factor, len(selected), touched)
    return with_timestamps_ns(frame, np.rint(new_ts).astype("int64"), schema=s)


def jitter_timestamps(frame: pd.DataFrame, *, sigma_seconds: float, seed: int,
                      mask: pd.Series | None = None,
                      clip_to_window: bool = True,
                      schema: SchemaView | None = None) -> pd.DataFrame:
    """Add seeded Gaussian noise to timestamps. Breaks perfect scripted rhythm."""
    s = schema or resolve_schema()
    if sigma_seconds < 0:
        raise PerturbationError("sigma_seconds must be >= 0")
    rng = make_rng(seed)
    ts = timestamps_as_ns(frame, schema=s).astype("float64")
    lo, hi = float(ts.min()), float(ts.max())

    sel = (np.ones(len(frame), dtype=bool) if mask is None
           else np.asarray(mask, dtype=bool))
    if sel.shape != (len(frame),):
        raise PerturbationError("mask length does not match frame length")

    noise = rng.normal(0.0, sigma_seconds * NS_PER_SECOND, size=int(sel.sum()))
    ts[sel] = ts[sel] + noise
    if clip_to_window:
        ts = np.clip(ts, lo, hi)
    return with_timestamps_ns(frame, np.rint(ts).astype("int64"), schema=s)


# ----------------------------------------------------------------------
# 2. legitimate volume
# ----------------------------------------------------------------------
def resample_legitimate_transactions(
        frame: pd.DataFrame, *, n_new: int | None = None,
        multiplier: float | None = None, seed: int,
        refresh_entities: bool = True,
        entity_columns: Sequence[str] | None = None,
        time_mode: str = "uniform",
        id_prefix: str = "SYNLEG-",
        schema: SchemaView | None = None) -> pd.DataFrame:
    """Inject additional NEGATIVE background traffic by resampling real rows.

    refresh_entities=True rewrites each entity value on the injected rows with a
    fresh token, so added volume dilutes the population WITHOUT manufacturing
    artificial entity reuse. Set False only if you intend the opposite.
    """
    s = schema or resolve_schema()
    _require_columns(frame, [s.id_col, s.ts_col, s.label_col, s.campaign_col],
                     "resample_legitimate_transactions")
    if (n_new is None) == (multiplier is None):
        raise PerturbationError("supply exactly one of n_new or multiplier")
    if time_mode not in ("uniform", "copy"):
        raise PerturbationError("time_mode must be 'uniform' or 'copy'")

    rng = make_rng(seed)
    mask = attack_mask(frame, schema=s)
    donors = frame.loc[~mask]
    if donors.empty:
        raise PerturbationError("no negative rows available to resample")

    count = int(n_new) if n_new is not None else int(round(len(donors) * float(multiplier)))
    if count <= 0:
        raise PerturbationError(f"resample count must be positive, got {count}")

    take = rng.integers(0, len(donors), size=count)
    new = donors.iloc[take].copy().reset_index(drop=True)
    new[s.id_col] = _fresh_ids(frame, count, schema=s, prefix=id_prefix)
    new[s.label_col] = _negative_label_literal(frame, s)
    new[s.campaign_col] = _empty_campaign_literal(frame, s)

    if refresh_entities:
        cols = list(entity_columns) if entity_columns is not None else list(s.entity)
        for c in cols:
            if c not in new.columns:
                raise PerturbationError(f"entity column '{c}' not in frame")
            new[c] = [f"{c}-syn-{seed}-{i:08d}" for i in range(count)]

    if time_mode == "uniform":
        lo, hi = observation_window_ns(frame, schema=s)
        draws = rng.integers(lo, hi + 1, size=count)
        new = with_timestamps_ns(new, draws, schema=s)

    out = pd.concat([frame, new], ignore_index=True)
    log.info("resample_legitimate_transactions: +%s rows (total %s)",
             count, len(out))
    return _sorted_by_id(out, s)


# ----------------------------------------------------------------------
# 3. entity reuse
# ----------------------------------------------------------------------
def weaken_entity_reuse(frame: pd.DataFrame, *, entity_column: str,
                        fraction: float, seed: int,
                        mask: pd.Series | None = None,
                        schema: SchemaView | None = None) -> pd.DataFrame:
    """Replace a seeded fraction of shared entity values with fresh tokens.

    This severs graph links without deleting transactions, isolating the
    question: how much does detection depend on entity reuse strength?
    """
    s = schema or resolve_schema()
    _require_columns(frame, [entity_column, s.id_col], "weaken_entity_reuse")
    if not 0.0 <= fraction <= 1.0:
        raise PerturbationError(f"fraction must be in [0, 1], got {fraction}")

    rng = make_rng(seed)
    out = frame.copy()
    sel = (attack_mask(frame, schema=s).to_numpy() if mask is None
           else np.asarray(mask, dtype=bool))
    if sel.shape != (len(frame),):
        raise PerturbationError("mask length does not match frame length")

    idx = np.flatnonzero(sel)
    if idx.size == 0:
        raise PerturbationError("weaken_entity_reuse: mask selected no rows")

    k = int(round(idx.size * float(fraction)))
    if k == 0:
        log.info("weaken_entity_reuse: fraction=%s rounds to 0 rows", fraction)
        return out

    chosen = rng.choice(idx, size=k, replace=False)
    chosen.sort()
    out.loc[out.index[chosen], entity_column] = [
        f"{entity_column}-broken-{seed}-{i:08d}" for i in range(k)]
    log.info("weaken_entity_reuse: rewrote %s/%s values of '%s'",
             k, idx.size, entity_column)
    return out


# ----------------------------------------------------------------------
# 4. temporal boundary
# ----------------------------------------------------------------------
def align_group_to_time(frame: pd.DataFrame, *, target_ts_ns: int,
                        group_col: str | None = None,
                        group_values: Sequence[Any] | None = None,
                        anchor: str = "start",
                        schema: SchemaView | None = None) -> pd.DataFrame:
    """Rigidly translate whole groups so their anchor lands on target_ts_ns.

    Intra-group cadence is untouched; only absolute position moves. Use with a
    window boundary as target to test straddling effects.
    """
    s = schema or resolve_schema()
    gcol = group_col or s.campaign_col
    _require_columns(frame, [gcol, s.ts_col], "align_group_to_time")
    if anchor not in ("start", "end", "median"):
        raise PerturbationError(f"anchor must be start/end/median, got {anchor}")

    ts = timestamps_as_ns(frame, schema=s).astype("int64")
    groups = frame[gcol].astype(str)
    selected = (set(str(v) for v in group_values)
                if group_values is not None
                else {g for g in groups.unique() if g not in ("", "nan", "None")})
    if not selected:
        raise PerturbationError("no groups selected for alignment")

    new_ts = ts.copy()
    for g in sorted(selected):
        idx = np.flatnonzero((groups == g).to_numpy())
        if idx.size == 0:
            continue
        vals = ts[idx]
        a = (vals.min() if anchor == "start"
             else vals.max() if anchor == "end"
             else int(np.median(vals)))
        new_ts[idx] = vals + (int(target_ts_ns) - int(a))

    return with_timestamps_ns(frame, new_ts, schema=s)


def straddle_boundary(frame: pd.DataFrame, *, boundary_ts_ns: int,
                      after_fraction: float,
                      group_col: str | None = None,
                      group_values: Sequence[Any] | None = None,
                      schema: SchemaView | None = None) -> pd.DataFrame:
    """Position each group so ~after_fraction of its rows fall past the boundary."""
    s = schema or resolve_schema()
    if not 0.0 <= after_fraction <= 1.0:
        raise PerturbationError("after_fraction must be in [0, 1]")
    gcol = group_col or s.campaign_col
    _require_columns(frame, [gcol, s.ts_col], "straddle_boundary")

    ts = timestamps_as_ns(frame, schema=s).astype("int64")
    groups = frame[gcol].astype(str)
    selected = (set(str(v) for v in group_values)
                if group_values is not None
                else {g for g in groups.unique() if g not in ("", "nan", "None")})

    new_ts = ts.copy()
    for g in sorted(selected):
        idx = np.flatnonzero((groups == g).to_numpy())
        if idx.size == 0:
            continue
        vals = np.sort(ts[idx])
        cut = int(np.clip(round((1.0 - after_fraction) * (vals.size - 1)),
                          0, vals.size - 1))
        pivot = int(vals[cut])
        new_ts[idx] = ts[idx] + (int(boundary_ts_ns) - pivot)

    return with_timestamps_ns(frame, new_ts, schema=s)


# ----------------------------------------------------------------------
# 5. right-censoring
# ----------------------------------------------------------------------
def right_censor(frame: pd.DataFrame, *, cutoff_ts_ns: int | None = None,
                 keep_fraction: float | None = None,
                 schema: SchemaView | None = None) -> pd.DataFrame:
    """Truncate the observation window: drop everything after the cutoff.

    Row count intentionally changes. Deterministic; no RNG.
    """
    s = schema or resolve_schema()
    if (cutoff_ts_ns is None) == (keep_fraction is None):
        raise PerturbationError("supply exactly one of cutoff_ts_ns or "
                                "keep_fraction")
    ts = timestamps_as_ns(frame, schema=s)

    if keep_fraction is not None:
        if not 0.0 < keep_fraction <= 1.0:
            raise PerturbationError("keep_fraction must be in (0, 1]")
        cutoff = int(np.quantile(ts, keep_fraction))
    else:
        cutoff = int(cutoff_ts_ns)

    keep = ts <= cutoff
    if not keep.any():
        raise PerturbationError(
            f"right_censor: cutoff {cutoff} removes every transaction")
    out = frame.loc[keep].copy().reset_index(drop=True)
    log.info("right_censor: cutoff=%s kept %s/%s rows", cutoff, len(out),
             len(frame))
    return _sorted_by_id(out, s)


def truncate_group_tails(frame: pd.DataFrame, *, keep_fraction: float,
                         group_col: str | None = None,
                         group_values: Sequence[Any] | None = None,
                         schema: SchemaView | None = None) -> pd.DataFrame:
    """Keep only the earliest keep_fraction of each group -- partial observation."""
    s = schema or resolve_schema()
    if not 0.0 < keep_fraction <= 1.0:
        raise PerturbationError("keep_fraction must be in (0, 1]")
    gcol = group_col or s.campaign_col
    _require_columns(frame, [gcol, s.ts_col, s.id_col], "truncate_group_tails")

    ts = timestamps_as_ns(frame, schema=s)
    groups = frame[gcol].astype(str)
    selected = (set(str(v) for v in group_values)
                if group_values is not None
                else {g for g in groups.unique() if g not in ("", "nan", "None")})

    drop = np.zeros(len(frame), dtype=bool)
    for g in sorted(selected):
        idx = np.flatnonzero((groups == g).to_numpy())
        if idx.size < 2:
            continue
        order = idx[np.argsort(ts[idx], kind="mergesort")]
        keep_n = max(1, int(np.floor(order.size * keep_fraction)))
        drop[order[keep_n:]] = True

    out = frame.loc[~drop].copy().reset_index(drop=True)
    log.info("truncate_group_tails: dropped %s rows", int(drop.sum()))
    return _sorted_by_id(out, s)


# ----------------------------------------------------------------------
# 6. false-positive stress
# ----------------------------------------------------------------------
def inject_benign_bursts(frame: pd.DataFrame, *, n_bursts: int,
                         burst_size: int, span_seconds: float, seed: int,
                         shared_entity_columns: Sequence[str],
                         vary_columns: Sequence[str] = (),
                         id_prefix: str = "SYNFP-",
                         schema: SchemaView | None = None) -> pd.DataFrame:
    """Inject NEGATIVE clusters that structurally imitate an attack.

    Each burst shares the given entity columns and is compressed into
    span_seconds, so it looks coordinated while being labelled benign. This is
    the adversarial precision test: how many of these does the frozen scorer
    rank above real attacks?
    """
    s = schema or resolve_schema()
    _require_columns(frame, [s.id_col, s.ts_col, s.label_col, s.campaign_col],
                     "inject_benign_bursts")
    if n_bursts <= 0 or burst_size <= 1:
        raise PerturbationError("n_bursts must be > 0 and burst_size > 1")
    if span_seconds < 0:
        raise PerturbationError("span_seconds must be >= 0")
    shared = list(shared_entity_columns)
    if not shared:
        raise PerturbationError(
            "shared_entity_columns is empty; a burst with no shared entity "
            "cannot form a candidate")
    _require_columns(frame, shared, "inject_benign_bursts")

    rng = make_rng(seed)
    mask = attack_mask(frame, schema=s)
    donors = frame.loc[~mask]
    if donors.empty:
        raise PerturbationError("no negative donor rows available")

    lo, hi = observation_window_ns(frame, schema=s)
    span_ns = int(span_seconds * NS_PER_SECOND)
    total = n_bursts * burst_size
    ids = _fresh_ids(frame, total, schema=s, prefix=id_prefix)
    neg_label = _negative_label_literal(frame, s)
    empty_campaign = _empty_campaign_literal(frame, s)

    blocks: list[pd.DataFrame] = []
    cursor = 0
    for b in range(n_bursts):
        take = rng.integers(0, len(donors), size=burst_size)
        blk = donors.iloc[take].copy().reset_index(drop=True)
        blk[s.id_col] = ids[cursor:cursor + burst_size]
        cursor += burst_size
        blk[s.label_col] = neg_label
        blk[s.campaign_col] = empty_campaign

        for c in shared:
            blk[c] = f"{c}-fp-{seed}-{b:05d}"
        for c in vary_columns:
            if c not in blk.columns:
                raise PerturbationError(f"vary column '{c}' not in frame")
            blk[c] = [f"{c}-fp-{seed}-{b:05d}-{i:04d}" for i in range(burst_size)]

        start = int(rng.integers(lo, max(lo + 1, hi - span_ns + 1)))
        offsets = np.sort(rng.integers(0, span_ns + 1, size=burst_size))
        blk = with_timestamps_ns(blk, start + offsets, schema=s)
        blocks.append(blk)

    out = pd.concat([frame, *blocks], ignore_index=True)
    log.info("inject_benign_bursts: +%s rows in %s bursts", total, n_bursts)
    return _sorted_by_id(out, s)


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def summarize_perturbation(before: pd.DataFrame, after: pd.DataFrame, *,
                           name: str,
                           schema: SchemaView | None = None) -> dict[str, Any]:
    """JSON-serializable description of what a perturbation actually did."""
    s = schema or resolve_schema()
    b_ts = timestamps_as_ns(before, schema=s)
    a_ts = timestamps_as_ns(after, schema=s)
    b_pos = int(attack_mask(before, schema=s).sum())
    a_pos = int(attack_mask(after, schema=s).sum())
    b_ids = set(before[s.id_col].astype(str))
    a_ids = set(after[s.id_col].astype(str))
    return {
        "name": name,
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "attack_rows_before": b_pos,
        "attack_rows_after": a_pos,
        "ids_added": len(a_ids - b_ids),
        "ids_removed": len(b_ids - a_ids),
        "window_seconds_before": round((int(b_ts.max()) - int(b_ts.min()))
                                       / NS_PER_SECOND, 3),
        "window_seconds_after": round((int(a_ts.max()) - int(a_ts.min()))
                                      / NS_PER_SECOND, 3),
        "schema_preserved": list(before.columns) == list(after.columns),
    }
