"""CONFLUX behavioural feature layer -- orchestrator, causal window engine, validation.

AUTHORITY
---------
This module implements FEATURE_SPEC.md / DECISIONS.md / AI_WORKING_RULES.md as the
single coherent feature layer. Historical implementations ("File 6", "File 7") were
used only to recover implementation detail; none of their code is trusted as correct.

STATUS
------
NOT EXECUTED by the author of this file. Every quantitative statement about the V4
dataset must come from an actual run of `main()`, which writes all artifacts from ONE
execution (features CSV, feature dictionary, univariate audit, validation report).

CAUSALITY CONTRACT
------------------
For row i, every feature uses only rows j with rank(j) < rank(i) and
ts(j) >= ts(i) - W, i.e. the half-open window [t - W, t). The current row NEVER
contributes to its own aggregate. Timestamps are kept at full nanosecond precision;
they are never truncated. Rows sharing an identical timestamp are ordered
deterministically by transaction_id, and because that order is a prefix-stable total
order, a time-prefix run must reproduce the full run exactly (see `causality_prefix_test`).

OWNERSHIP MAP (prevents the same quantity being emitted under two names)
-----------------------------------------------------------------------
velocity_features : rolling transaction counts + burst density for card / device / ip
device_features   : distinct cards per device, device reuse (lifetime prior count)
merchant_features : every entity -> merchant relationship (count, spread, entropy),
                    related-entities-across-merchants, merchant window context
bin_features      : BIN activity, cards/merchants per BIN, device->BINs, ip->BINs
amount_features   : card and device amount baselines (locked amount entities only)
decline_features  : prior decline rates for card / device / ip / merchant

CONFIGURATION
-------------
`FeatureConfig` is the only source of windows and thresholds. Load it from the
project's existing YAML (`FeatureConfig.from_yaml("configs/default.yaml")`, section
`features:`) or from a mapping supplied by src/conflux/config.py. This module writes
no configuration file; the resolved config is snapshotted into the validation report
as an OUTPUT only.

FORBIDDEN INPUTS
----------------
label, campaign_id, _source_type are dropped before the feature context is built and
are asserted absent. `label` is read separately for the univariate AUDIT only.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from bisect import bisect_left, insort
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("conflux.features")

NS_PER_S = 1_000_000_000
FORBIDDEN_INPUTS = ("label", "campaign_id", "_source_type")

# logical name -> accepted dataset column names (V4 contract first)
SCHEMA_ALIASES: dict[str, tuple[str, ...]] = {
    "row_id":    ("transaction_id",),
    "timestamp": ("timestamp",),
    "amount":    ("amount",),
    "card":      ("card_fingerprint",),
    "device":    ("device_fingerprint",),
    "ip":        ("ip_signature",),
    "bin":       ("bin",),
    "merchant":  ("merchant_id",),
    "auth":      ("auth_outcome",),
    "label":     ("label",),          # ground truth: audit/evaluation only
}
REQUIRED_LOGICAL = ("row_id", "timestamp", "amount", "card", "device", "ip", "bin", "merchant", "auth")

# Identifier-like columns are read as text so that `bin` is never coerced to a numeric
# quantity (DECISIONS.md: BIN is categorical issuer context, never arithmetic).
TEXT_LOGICAL = ("row_id", "card", "device", "ip", "bin", "merchant", "auth")

# ---------------------------------------------------------------------------
# Locked specification coverage. Every emitted feature declares one or more of
# these refs; the build fails if any locked item is left uncovered.
# ---------------------------------------------------------------------------
LOCKED_SPEC_ITEMS: tuple[str, ...] = (
    # 1. Amount
    "AMOUNT.historical_deviation_card",
    "AMOUNT.historical_deviation_device",
    "AMOUNT.similarity_cv",
    "AMOUNT.low_value_ratio",
    "AMOUNT.window_mean",
    "AMOUNT.window_median",
    "AMOUNT.window_std",
    # 2. Device
    "DEVICE.txn_count",
    "DEVICE.distinct_cards",
    "DEVICE.distinct_merchants",
    "DEVICE.velocity",
    "DEVICE.reuse",
    # 3. BIN
    "BIN.activity",
    "BIN.distinct_cards",
    "BIN.distinct_merchants",
    "BIN.overlap_device",
    "BIN.overlap_ip",
    # 4. Merchant pattern
    "MERCHANT.distinct_merchants_card",
    "MERCHANT.distinct_merchants_device",
    "MERCHANT.distinct_merchants_ip",
    "MERCHANT.cross_merchant_spread",
    "MERCHANT.dispersion_entropy",
    "MERCHANT.related_entities",
    # 5. Velocity
    "VELOCITY.rolling_count_card",
    "VELOCITY.rolling_count_device",
    "VELOCITY.rolling_count_ip",
    "VELOCITY.txns_per_minute",
    "VELOCITY.burst_density",
    # 6. Decline
    "DECLINE.rate_card",
    "DECLINE.rate_device",
    "DECLINE.rate_ip",
    "DECLINE.rate_merchant",
)

# Refs that are NOT locked spec items are only allowed if justified here, and the
# justification is printed in the validation report.
DOCUMENTED_DEVIATIONS: dict[str, str] = {
    "CONTEXT.merchant_window_txn_count": (
        "mer_txn_count_{w}s is merchant-side window volume. It is not itself a locked "
        "Merchant Pattern item; it is emitted because MERCHANT.related_entities is a raw "
        "count of related devices/IPs at the current merchant and is uninterpretable "
        "without the merchant's own window volume as scale. Not a dispersion measure and "
        "deliberately NOT merchant->card entropy (that direction was the audited error)."
    ),
}

# Items that are satisfied only when an optional config switch is on. Kept out of the
# hard coverage requirement, reported as 'conditionally covered'.
CONDITIONAL_SPEC_ITEMS: dict[str, str] = {
    "VELOCITY.txns_per_minute": (
        "transactions-per-minute is count * 60 / W, an exact deterministic rescaling of "
        "vel_{entity}_count_{w}s which is already emitted. Emitting both would be the same "
        "quantity under two names (perfectly collinear for the Logistic Regression "
        "baseline). Set features.emit_rate_per_min=true in configuration to emit it."
    ),
}


# ---------------------------------------------------------------------------
# Feature declaration helper
# ---------------------------------------------------------------------------
_SPEC_KEYS = ("name", "group", "entity", "window", "range", "missing", "purpose", "spec_ref", "config")


def feature_spec(name: str, *, group: str, entity: str, window: Any, range_: str,
                 missing: str, purpose: str, spec_ref: Sequence[str],
                 config: Sequence[str] = ()) -> dict[str, Any]:
    """Build one validated feature-dictionary entry.

    FEATURE_SPEC.md requires every feature to declare definition, entity, window,
    causal semantics, purpose, missing-value behaviour and configuration source.
    """
    if not spec_ref:
        raise ValueError(f"{name}: spec_ref is mandatory (feature discipline)")
    for ref in spec_ref:
        if ref not in LOCKED_SPEC_ITEMS and ref not in DOCUMENTED_DEVIATIONS:
            raise ValueError(
                f"{name}: spec_ref '{ref}' is neither a locked spec item nor a documented "
                f"deviation. Add it to LOCKED_SPEC_ITEMS only by an explicit spec decision."
            )
    return dict(
        name=name, group=group, entity=entity, window=window, range=range_,
        missing=missing, purpose=purpose,
        spec_ref=";".join(spec_ref), config=";".join(config),
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureConfig:
    """Single configuration surface for the behavioural feature layer.

    No threshold or window used by a feature may be hardcoded inside a feature
    function (AI_WORKING_RULES.md section 9).
    """

    # FEATURE_SPEC.md starting windows for the simulated campaign durations.
    windows_s: tuple[int, ...] = (30, 120, 300, 1200)
    group_windows_s: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    # Minimum PRIOR observations required before a variability statistic
    # (std / cv / z) or a rate (decline rate, low-value ratio) is defined.
    min_history: int = 3
    # Minimum PRIOR observations before the distinct/total spread ratio is defined.
    min_prior_for_ratio: int = 2
    # Normalised entropy is undefined for a single category.
    min_distinct_for_entropy: int = 2

    # PROJECT ASSUMPTION, not derived from production data and not selected using
    # `label`. Documented in the feature dictionary for every feature that uses it.
    low_value_amount: float = 5.0

    # "related entity across merchants": how many distinct merchants an actor must
    # have touched inside the window before it counts as cross-merchant.
    related_entity_min_merchants: int = 2

    # See CONDITIONAL_SPEC_ITEMS["VELOCITY.txns_per_minute"].
    emit_rate_per_min: bool = False

    # Authorisation outcome vocabulary. Unknown tokens fail loudly rather than being
    # silently mapped to "declined".
    auth_approved_tokens: tuple[str, ...] = ("approved",)
    auth_declined_tokens: tuple[str, ...] = ("declined",)

    eps: float = 1e-12
    fillna: float | None = None          # None => keep NaN, imputation is the model layer's job

    expected_rows: int | None = 31_873   # warn-only sanity check on the frozen dataset
    schema_overrides: Mapping[str, str] = field(default_factory=dict)

    # Validation
    causality_prefix_fraction: float = 0.5
    causality_atol: float = 1e-4
    causality_rtol: float = 1e-5   # 0.0 => require exact reproduction
    auc_flag_threshold: float = 0.75     # flag for INVESTIGATION, never auto-reject
    auc_min_support: int = 200
    missingness_auc_flag: float = 0.65
    range_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        ws = tuple(sorted({int(w) for w in self.windows_s}))
        if not ws or ws[0] <= 0:
            raise ValueError("windows_s must be non-empty positive seconds")
        object.__setattr__(self, "windows_s", ws)
        object.__setattr__(self, "group_windows_s",
                           {k: tuple(sorted({int(w) for w in v})) for k, v in dict(self.group_windows_s).items()})
        if self.min_history < 2:
            raise ValueError("min_history must be >= 2 (std with ddof=1 needs two observations)")
        if self.min_prior_for_ratio < 1:
            raise ValueError("min_prior_for_ratio must be >= 1")
        if not 0.0 < self.causality_prefix_fraction < 1.0:
            raise ValueError("causality_prefix_fraction must be in (0, 1)")
        if self.low_value_amount <= 0:
            raise ValueError("low_value_amount must be positive")

    def windows_for(self, group: str) -> tuple[int, ...]:
        return tuple(self.group_windows_s.get(group, self.windows_s))

    def narrow_wide(self, group: str) -> tuple[int, int]:
        ws = self.windows_for(group)
        return ws[0], ws[-1]

    # -- loaders -------------------------------------------------------------
    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "FeatureConfig":
        data = dict(mapping or {})
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown feature config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path, section: str = "features") -> "FeatureConfig":
        """Load from the project's EXISTING YAML config. No parallel config file."""
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to read the project config. Alternatively pass a "
                "mapping from src/conflux/config.py to FeatureConfig.from_mapping()."
            ) from exc
        doc = yaml.safe_load(Path(path).read_text()) or {}
        node = doc.get(section, {}) if isinstance(doc, dict) else {}
        if node is None:
            node = {}
        if not isinstance(node, dict):
            raise ValueError(f"config section '{section}' must be a mapping")
        return cls.from_mapping(node)

    def snapshot(self) -> dict[str, Any]:
        out = asdict(self)
        out["windows_s"] = list(self.windows_s)
        out["group_windows_s"] = {k: list(v) for k, v in self.group_windows_s.items()}
        out["auth_approved_tokens"] = list(self.auth_approved_tokens)
        out["auth_declined_tokens"] = list(self.auth_declined_tokens)
        out["schema_overrides"] = dict(self.schema_overrides)
        return out


# ---------------------------------------------------------------------------
# Causal window engine
# ---------------------------------------------------------------------------
def _c_log_c(c: int) -> float:
    return 0.0 if c <= 1 else c * math.log(c)


class PastWindow:
    """Strictly-past sliding-window aggregator over a single entity key.

    Rows are supplied already in global causal order (ts, transaction_id). Grouping by
    key with a STABLE argsort therefore preserves causal order inside each group, so a
    single left-to-right pass per group is a correct streaming computation.

    For row i the window is {j : key_j == key_i, rank_j < rank_i, ts_j >= ts_i - W}.
    `w=None` means "all prior rows for this key" (lifetime).
    """

    def __init__(self, ts_ns: np.ndarray, key_codes: np.ndarray) -> None:
        self.n = int(len(ts_ns))
        codes = np.asarray(key_codes, dtype=np.int64)
        if len(codes) != self.n:
            raise ValueError("key length does not match timestamp length")
        order = np.argsort(codes, kind="stable")
        self.order = order
        self.ts = np.asarray(ts_ns, dtype=np.int64)[order]
        c = codes[order]
        self.valid = c >= 0                      # negative code == missing key -> NaN output
        if self.n:
            starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
            self.starts = starts.astype(np.int64)
            self.ends = np.r_[starts[1:], self.n].astype(np.int64)
        else:
            self.starts = self.ends = np.zeros(0, dtype=np.int64)
        self.pos = np.arange(self.n, dtype=np.int64)
        self._lo_cache: dict[Any, np.ndarray] = {}

    # -- window bounds -------------------------------------------------------
    def _lo(self, w: int | None) -> np.ndarray:
        key = None if w is None else int(w)
        cached = self._lo_cache.get(key)
        if cached is not None:
            return cached
        lo = np.empty(self.n, dtype=np.int64)
        for s, e in zip(self.starts, self.ends):
            if key is None:
                lo[s:e] = s
            else:
                t = self.ts[s:e]
                # side="left": ts_j >= ts_i - W is included. Nanoseconds, no truncation.
                lo[s:e] = s + np.searchsorted(t, t - key * NS_PER_S, side="left")
        self._lo_cache[key] = lo
        return lo

    def _scatter(self, vals_sorted: np.ndarray) -> np.ndarray:
        v = np.asarray(vals_sorted, dtype=np.float64).copy()
        v[~self.valid] = np.nan
        out = np.empty(self.n, dtype=np.float64)
        out[self.order] = v
        return out

    def _prep(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(values, dtype=np.float64)[self.order]
        m = np.isfinite(v)
        return np.where(m, v, 0.0), m.astype(np.float64)

    # -- vectorised aggregates ----------------------------------------------
    def count(self, w: int | None = None) -> np.ndarray:
        return self._scatter((self.pos - self._lo(w)).astype(np.float64))

    def sum(self, values: np.ndarray, w: int | None = None) -> np.ndarray:
        v, _ = self._prep(values)
        p = np.r_[0.0, np.cumsum(v)]
        return self._scatter(p[self.pos] - p[self._lo(w)])

    def mean(self, values: np.ndarray, w: int | None = None) -> np.ndarray:
        v, m = self._prep(values)
        pv, pm = np.r_[0.0, np.cumsum(v)], np.r_[0.0, np.cumsum(m)]
        lo = self._lo(w)
        s = pv[self.pos] - pv[lo]
        c = pm[self.pos] - pm[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._scatter(np.where(c > 0, s / np.where(c > 0, c, 1.0), np.nan))

    def std(self, values: np.ndarray, w: int | None = None, ddof: int = 1) -> np.ndarray:
        v, m = self._prep(values)
        pv = np.r_[0.0, np.cumsum(v)]
        pq = np.r_[0.0, np.cumsum(v * v)]
        pm = np.r_[0.0, np.cumsum(m)]
        lo = self._lo(w)
        s = pv[self.pos] - pv[lo]
        q = pq[self.pos] - pq[lo]
        c = pm[self.pos] - pm[lo]
        ok = c > ddof
        with np.errstate(invalid="ignore", divide="ignore"):
            ss = np.where(ok, np.maximum(q - (s * s) / np.where(c > 0, c, 1.0), 0.0), np.nan)
            var = np.where(ok, ss / np.where(ok, c - ddof, 1.0), np.nan)
        return self._scatter(np.sqrt(var))

    # -- streaming aggregates ------------------------------------------------
    def median(self, values: np.ndarray, w: int | None = None) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)[self.order]
        lo = self._lo(w)
        res = np.full(self.n, np.nan)
        for s, e in zip(self.starts, self.ends):
            buf: list[float] = []
            head = int(s)
            for i in range(int(s), int(e)):
                while head < lo[i]:
                    x = v[head]
                    if np.isfinite(x):
                        buf.pop(bisect_left(buf, x))
                    head += 1
                k = len(buf)                      # read BEFORE inserting current row
                if k:
                    res[i] = buf[k // 2] if k % 2 else 0.5 * (buf[k // 2 - 1] + buf[k // 2])
                x = v[i]
                if np.isfinite(x):
                    insort(buf, x)
        return self._scatter(res)

    def nunique(self, target_codes: np.ndarray, w: int | None = None) -> np.ndarray:
        codes = np.asarray(target_codes, dtype=np.int64)[self.order]
        lo = self._lo(w)
        res = np.full(self.n, np.nan)
        for s, e in zip(self.starts, self.ends):
            counter: dict[int, int] = {}
            head = int(s)
            for i in range(int(s), int(e)):
                while head < lo[i]:
                    k = int(codes[head])
                    if k >= 0:
                        c = counter[k] - 1
                        if c:
                            counter[k] = c
                        else:
                            del counter[k]
                    head += 1
                res[i] = float(len(counter))      # 0 prior observations is a real zero
                k = int(codes[i])
                if k >= 0:
                    counter[k] = counter.get(k, 0) + 1
        return self._scatter(res)

    def entropy(self, target_codes: np.ndarray, w: int | None = None,
                min_distinct: int = 2) -> np.ndarray:
        """Normalised Shannon entropy of prior target categories, in [0, 1].

        H = log(n) - (1/n) * sum(c_i log c_i); normalised by log(k), k = distinct count.
        NaN while k < min_distinct (a single category has no dispersion to measure).
        """
        codes = np.asarray(target_codes, dtype=np.int64)[self.order]
        lo = self._lo(w)
        res = np.full(self.n, np.nan)
        for s, e in zip(self.starts, self.ends):
            counter: dict[int, int] = {}
            total = 0
            s_clogc = 0.0
            head = int(s)
            for i in range(int(s), int(e)):
                while head < lo[i]:
                    k = int(codes[head])
                    if k >= 0:
                        c = counter[k]
                        s_clogc -= _c_log_c(c)
                        if c > 1:
                            counter[k] = c - 1
                            s_clogc += _c_log_c(c - 1)
                        else:
                            del counter[k]
                        total -= 1
                    head += 1
                nd = len(counter)
                if total > 0 and nd >= min_distinct:
                    h = math.log(total) - s_clogc / total
                    res[i] = min(max(h / math.log(nd), 0.0), 1.0)
                k = int(codes[i])
                if k >= 0:
                    c = counter.get(k, 0)
                    if c:
                        s_clogc -= _c_log_c(c)
                    counter[k] = c + 1
                    s_clogc += _c_log_c(c + 1)
                    total += 1
        return self._scatter(res)


# ---------------------------------------------------------------------------
# Feature context (memoised: a given aggregate is computed at most once per run)
# ---------------------------------------------------------------------------
class FeatureContext:
    def __init__(self, df: pd.DataFrame, cfg: FeatureConfig, schema: Mapping[str, str]) -> None:
        leaked = [c for c in FORBIDDEN_INPUTS if c in df.columns]
        if leaked:
            raise AssertionError(f"forbidden inputs reached the feature layer: {leaked}")
        self.df = df
        self.cfg = cfg
        self.schema = dict(schema)
        self.n = len(df)
        self.ts_ns = df["__ts_ns"].to_numpy(np.int64)
        self._codes: dict[str, np.ndarray] = {}
        self._channels: dict[str, np.ndarray] = {}
        self._win: dict[str, PastWindow] = {}
        self._cache: dict[tuple, np.ndarray] = {}
        self.cache_hits = 0

    # -- primitives ----------------------------------------------------------
    def codes(self, logical: str) -> np.ndarray:
        c = self._codes.get(logical)
        if c is None:
            col = self.schema.get(logical)
            if col is None:
                raise KeyError(f"entity '{logical}' is not present in the dataset schema")
            codes, _ = pd.factorize(self.df[col], use_na_sentinel=True)
            c = codes.astype(np.int64)
            self._codes[logical] = c
        return c

    def channel(self, name: str) -> np.ndarray:
        ch = self._channels.get(name)
        if ch is None:
            if name == "amount":
                ch = self.df[self.schema["amount"]].to_numpy(np.float64)
            elif name == "declined":
                ch = self.df["__declined"].to_numpy(np.float64)
            elif name == "low_value":
                amt = self.channel("amount")
                ch = np.where(np.isfinite(amt), (amt < self.cfg.low_value_amount).astype(np.float64), np.nan)
            else:
                raise KeyError(f"unknown numeric channel '{name}'")
            self._channels[name] = ch
        return ch

    def window(self, actor: str) -> PastWindow:
        w = self._win.get(actor)
        if w is None:
            w = PastWindow(self.ts_ns, self.codes(actor))
            self._win[actor] = w
        return w

    def _memo(self, key: tuple, fn) -> np.ndarray:
        hit = self._cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        val = fn()
        self._cache[key] = val
        return val

    # -- cached aggregates ---------------------------------------------------
    def count(self, actor: str, w: int | None) -> np.ndarray:
        return self._memo(("count", actor, w), lambda: self.window(actor).count(w))

    def nunique(self, actor: str, target: str, w: int | None) -> np.ndarray:
        return self._memo(("nunique", actor, target, w),
                          lambda: self.window(actor).nunique(self.codes(target), w))

    def entropy(self, actor: str, target: str, w: int | None) -> np.ndarray:
        md = self.cfg.min_distinct_for_entropy
        return self._memo(("entropy", actor, target, w, md),
                          lambda: self.window(actor).entropy(self.codes(target), w, md))

    def sum(self, actor: str, channel: str, w: int | None) -> np.ndarray:
        return self._memo(("sum", actor, channel, w),
                          lambda: self.window(actor).sum(self.channel(channel), w))

    def mean(self, actor: str, channel: str, w: int | None) -> np.ndarray:
        return self._memo(("mean", actor, channel, w),
                          lambda: self.window(actor).mean(self.channel(channel), w))

    def std(self, actor: str, channel: str, w: int | None) -> np.ndarray:
        return self._memo(("std", actor, channel, w),
                          lambda: self.window(actor).std(self.channel(channel), w))

    def median(self, actor: str, channel: str, w: int | None) -> np.ndarray:
        return self._memo(("median", actor, channel, w),
                          lambda: self.window(actor).median(self.channel(channel), w))

    # -- helpers -------------------------------------------------------------
    def safe_div(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.isfinite(b) & (np.abs(b) > self.cfg.eps), a / b, np.nan)

    def gate(self, values: np.ndarray, support: np.ndarray, minimum: int) -> np.ndarray:
        support = np.asarray(support, dtype=np.float64)
        return np.where(np.isfinite(support) & (support >= minimum),
                        np.asarray(values, dtype=np.float64), np.nan)

    def related_entity_counts(self, actor: str, w_s: int) -> np.ndarray:
        """For the CURRENT merchant: how many distinct `actor` entities with PRIOR
        activity at this merchant inside the window also touched at least
        cfg.related_entity_min_merchants distinct merchants inside that window.

        Behavioural precursor to the graph layer only: no edges, no components, no
        campaign identity is constructed here (ARCHITECTURE.md layer boundaries).
        """
        min_m = int(self.cfg.related_entity_min_merchants)
        key = ("related", actor, int(w_s), min_m)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        a_codes = self.codes(actor)
        m_codes = self.codes("merchant")
        lo = np.searchsorted(self.ts_ns, self.ts_ns - int(w_s) * NS_PER_S, side="left")

        actor_merchants: dict[int, dict[int, int]] = {}
        related: dict[int, int] = {}
        out = np.full(self.n, np.nan)

        def contribute(a: int, sign: int) -> None:
            am = actor_merchants.get(a)
            if am is None or len(am) < min_m:
                return
            for m in am:
                related[m] = related.get(m, 0) + sign

        def apply(m: int, a: int, delta: int) -> None:
            contribute(a, -1)                     # withdraw old contribution
            am = actor_merchants.setdefault(a, {})
            c = am.get(m, 0) + delta
            if c:
                am[m] = c
            else:
                am.pop(m, None)
                if not am:
                    actor_merchants.pop(a, None)
            contribute(a, +1)                     # re-add under the new merchant set

        head = 0
        for i in range(self.n):
            while head < lo[i]:                   # evict rows that fell out of the window
                if a_codes[head] >= 0 and m_codes[head] >= 0:
                    apply(int(m_codes[head]), int(a_codes[head]), -1)
                head += 1
            if m_codes[i] >= 0:
                out[i] = float(related.get(int(m_codes[i]), 0))
            if a_codes[i] >= 0 and m_codes[i] >= 0:
                apply(int(m_codes[i]), int(a_codes[i]), +1)   # insert AFTER reading

        self._cache[key] = out
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def resolve_schema(columns: Iterable[str], cfg: FeatureConfig) -> dict[str, str]:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    schema: dict[str, str] = {}
    for logical, aliases in SCHEMA_ALIASES.items():
        override = cfg.schema_overrides.get(logical)
        if override:
            if override not in cols:
                raise KeyError(f"schema override {logical}->{override} is not a dataset column")
            schema[logical] = override
            continue
        for alias in aliases:
            if alias.lower() in lower:
                schema[logical] = lower[alias.lower()]
                break
    missing = [k for k in REQUIRED_LOGICAL if k not in schema]
    if missing:
        raise KeyError(
            f"required field(s) {missing} could not be resolved. Dataset columns: {cols}. "
            f"The V4 dataset contract guarantees these columns; refusing to derive or guess "
            f"a substitute (in particular, BIN is never derived from card_fingerprint)."
        )
    return schema


def _coerce_declined(series: pd.Series, cfg: FeatureConfig) -> pd.Series:
    tok = series.astype("string").str.strip().str.lower()
    approved = {t.lower() for t in cfg.auth_approved_tokens}
    declined = {t.lower() for t in cfg.auth_declined_tokens}
    unknown = sorted(set(tok.dropna().unique()) - approved - declined)
    if unknown:
        raise ValueError(
            f"unexpected auth_outcome token(s) {unknown}: refusing to guess an "
            f"authorisation semantics. Extend auth_approved_tokens / auth_declined_tokens."
        )
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[tok.isin(approved)] = 0.0
    out[tok.isin(declined)] = 1.0
    return out


def load_frame(data_path: str | Path, cfg: FeatureConfig) -> tuple[pd.DataFrame, dict[str, str], pd.Series | None, dict[str, Any]]:
    """Read the frozen dataset, establish deterministic causal order, drop ground truth.

    The frozen CSV is only ever read. All identifier columns are read as text so that
    `bin` cannot be silently turned into a numeric quantity.
    """
    raw = pd.read_csv(data_path, dtype=str, low_memory=False)
    n_in = len(raw)
    schema = resolve_schema(raw.columns, cfg)

    diagnostics: dict[str, Any] = {"input_rows": n_in, "input_columns": list(raw.columns)}
    if cfg.expected_rows is not None and n_in != cfg.expected_rows:
        log.warning("row count %s != expected %s (frozen dataset changed?)", n_in, cfg.expected_rows)
        diagnostics["expected_rows_mismatch"] = {"observed": n_in, "expected": cfg.expected_rows}

    label = raw[schema["label"]].copy() if "label" in schema else None

    df = raw.copy()

    # Amount: the only numeric raw field.
    amount = pd.to_numeric(df[schema["amount"]], errors="coerce")
    if amount.isna().any():
        raise ValueError(f"{int(amount.isna().sum())} unparseable amount value(s); refusing to drop rows")
    df[schema["amount"]] = amount.astype(np.float64)

    # Timestamps: full precision, never truncated.
    ts = pd.to_datetime(df[schema["timestamp"]], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} unparseable timestamp(s); refusing to drop rows")
    ts_ns = ts.values.astype("datetime64[ns]").astype("int64")
    df["__ts_ns"] = ts_ns
    diagnostics["timestamp_span"] = {"min": str(ts.min()), "max": str(ts.max())}
    diagnostics["distinct_timestamps"] = int(pd.Series(ts_ns).nunique())
    diagnostics["rows_sharing_a_timestamp"] = int(n_in - diagnostics["distinct_timestamps"])

    df["__row_key"] = df[schema["row_id"]].astype("string")
    if df["__row_key"].isna().any() or df["__row_key"].duplicated().any():
        raise ValueError("transaction_id must be present and unique (it is the join key)")

    df["__declined"] = _coerce_declined(df[schema["auth"]], cfg)
    diagnostics["declined_rows"] = int(np.nansum(df["__declined"].to_numpy(np.float64)))

    df["__input_pos"] = np.arange(n_in, dtype=np.int64)

    # Deterministic causal order: (timestamp, transaction_id). Prefix-stable, so a
    # time-prefix run reproduces the full run exactly.
    df = df.sort_values(["__ts_ns", "__row_key"], kind="mergesort").reset_index(drop=True)
    df = df.drop(columns=[c for c in FORBIDDEN_INPUTS if c in df.columns])
    schema.pop("label", None)

    if len(df) != n_in:
        raise AssertionError("row count changed during preparation")
    return df, schema, label, diagnostics


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def _modules():
    from . import (amount_features, device_features, bin_features,
                   merchant_features, velocity_features, decline_features)
    return (amount_features, device_features, bin_features,
            merchant_features, velocity_features, decline_features)


def compute_features(df: pd.DataFrame, cfg: FeatureConfig, schema: Mapping[str, str]
                     ) -> tuple[pd.DataFrame, list[dict[str, Any]], FeatureContext]:
    """Run all six locked signal groups against one shared causal context."""
    ctx = FeatureContext(df, cfg, schema)
    n = len(df)
    blocks: list[pd.DataFrame] = []
    dictionary: list[dict[str, Any]] = []
    seen: dict[str, str] = {}

    for mod in _modules():
        group = getattr(mod, "GROUP")
        specs = mod.declare(cfg, ctx)              # declare(cfg, ctx) for EVERY module
        built = mod.build(ctx)

        declared = [s["name"] for s in specs]
        if len(set(declared)) != len(declared):
            raise AssertionError(f"{group}: declare() emits duplicate names")
        if set(built) != set(declared):
            raise AssertionError(
                f"{group}: declare()/build() contract mismatch. "
                f"declared_not_built={sorted(set(declared) - set(built))} "
                f"built_not_declared={sorted(set(built) - set(declared))}"
            )
        for name in declared:
            if name in seen:
                raise AssertionError(f"duplicate feature name '{name}' in {group} and {seen[name]}")
            seen[name] = group
            arr = np.asarray(built[name], dtype=np.float64)
            if arr.shape != (n,):
                raise AssertionError(f"{group}.{name}: shape {arr.shape} != ({n},)")
            built[name] = arr

        blocks.append(pd.DataFrame({k: built[k] for k in declared}, index=df.index))
        dictionary.extend(specs)
        log.info("%-9s %3d features", group, len(specs))

    feats = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)
    if len(feats) != n:
        raise AssertionError("feature block row count diverged from input")
    feats.insert(0, "transaction_id", df["__row_key"].to_numpy())
    return feats, dictionary, ctx


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_RANGE_BOUNDS = {
    "[0, inf)": (0.0, math.inf, True, False),
    "[0, 1]": (0.0, 1.0, True, True),
    "(-inf, inf)": (-math.inf, math.inf, False, False),
}


def check_ranges(feats: pd.DataFrame, dictionary: Sequence[Mapping[str, Any]],
                 cfg: FeatureConfig) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for spec in dictionary:
        bounds = _RANGE_BOUNDS.get(spec["range"])
        if bounds is None:
            violations.append({"feature": spec["name"], "problem": f"undeclared range '{spec['range']}'"})
            continue
        lo, hi, _, _ = bounds
        x = feats[spec["name"]].to_numpy(np.float64)
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            continue
        if finite.min() < lo - cfg.range_tolerance or finite.max() > hi + cfg.range_tolerance:
            violations.append({
                "feature": spec["name"], "declared_range": spec["range"],
                "observed_min": float(finite.min()), "observed_max": float(finite.max()),
            })
    return violations


def check_spec_coverage(dictionary: Sequence[Mapping[str, Any]], cfg: FeatureConfig) -> dict[str, Any]:
    covered: dict[str, list[str]] = {}
    for spec in dictionary:
        for ref in spec["spec_ref"].split(";"):
            covered.setdefault(ref, []).append(spec["name"])
    hard = [i for i in LOCKED_SPEC_ITEMS if i not in CONDITIONAL_SPEC_ITEMS]
    uncovered = [i for i in hard if i not in covered]
    conditional = {i: (i in covered) for i in CONDITIONAL_SPEC_ITEMS}
    return {
        "locked_items": len(LOCKED_SPEC_ITEMS),
        "covered_by": {k: sorted(v) for k, v in sorted(covered.items())},
        "uncovered_locked_items": uncovered,
        "conditional_items": conditional,
        "conditional_rationale": CONDITIONAL_SPEC_ITEMS,
        "documented_deviations": DOCUMENTED_DEVIATIONS,
        "passed": not uncovered,
    }


def causality_prefix_test(df: pd.DataFrame, cfg: FeatureConfig, schema: Mapping[str, str],
                          full_feats: pd.DataFrame) -> dict[str, Any]:
    """THE definitive causality test: features recomputed on a strict time-prefix of the
    dataset must reproduce the full-dataset values for the overlapping transactions.
    """
    n = len(df)
    k = max(2, int(math.ceil(cfg.causality_prefix_fraction * n)))
    if k >= n:
        return {"executed": False, "reason": "prefix fraction leaves no held-out rows"}
    prefix = df.iloc[:k].copy().reset_index(drop=True)
    pf, _, _ = compute_features(prefix, cfg, schema)

    a = pf.set_index("transaction_id").sort_index()
    b = full_feats.set_index("transaction_id").loc[a.index].sort_index()
    cols = [c for c in a.columns]

    mismatches: list[dict[str, Any]] = []
    worst = 0.0
    for c in cols:
        x, y = a[c].to_numpy(np.float64), b[c].to_numpy(np.float64)
        nan_x, nan_y = ~np.isfinite(x), ~np.isfinite(y)
        bad_nan = nan_x != nan_y
        both = ~nan_x & ~nan_y
        diff = np.zeros_like(x)
        diff[both] = np.abs(x[both] - y[both])
        col_worst = float(diff.max()) if diff.size else 0.0
        worst = max(worst, col_worst)
        close = np.ones_like(x, dtype=bool)
        close[both] = np.isclose(
            x[both],
            y[both],
            rtol=cfg.causality_rtol,
            atol=cfg.causality_atol,
        )
        bad = bad_nan | (both & ~close)
        if bad.any():
            mismatches.append({"feature": c, "rows": int(bad.sum()), "max_abs_diff": col_worst})
    return {
        "executed": True,
        "method": ("features recomputed on the first "
                   f"{cfg.causality_prefix_fraction:.0%} of time-sorted rows and compared to the "
                   "full-dataset run for every feature and every overlapping transaction_id"),
        "prefix_rows": int(k), "full_rows": int(n),
        "columns_compared": len(cols),
        "tolerance": {
            "atol": cfg.causality_atol,
            "rtol": cfg.causality_rtol,
        },
        "max_abs_diff_observed": worst,
        "columns_with_mismatches": len(mismatches),
        "mismatch_detail": mismatches[:25],
        "passed": not mismatches,
    }


def rank_auc(y: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    yy, xx = y[m], x[m]
    if yy.size == 0 or np.unique(yy).size < 2:
        return float("nan"), int(yy.size)
    r = pd.Series(xx).rank(method="average").to_numpy()
    n1 = float((yy == 1).sum())
    n0 = float((yy == 0).sum())
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)), int(yy.size)


def univariate_audit(feats: pd.DataFrame, label: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    """AUDIT ONLY. DECISIONS.md: AUC is not an objective and never a selection rule.
    A high value is a signal to INVESTIGATE (artifact? missingness? leakage?).
    """
    y = pd.to_numeric(label, errors="coerce").to_numpy(np.float64)
    rows = []
    for name in feats.columns:
        if name == "transaction_id":
            continue
        x = feats[name].to_numpy(np.float64)
        auc, n = rank_auc(y, x)
        miss = (~np.isfinite(x)).astype(np.float64)
        miss_auc, _ = rank_auc(y, miss)
        directed = max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")
        rows.append({
            "feature": name, "auc": auc, "auc_directed": directed, "n_scored": n,
            "missing_rate": float(miss.mean()),
            "missing_indicator_auc": miss_auc,
            "flag_investigate_high_auc": bool(np.isfinite(directed) and directed > cfg.auc_flag_threshold
                                              and n >= cfg.auc_min_support),
            "flag_missingness_signal": bool(np.isfinite(miss_auc)
                                            and max(miss_auc, 1.0 - miss_auc) > cfg.missingness_auc_flag),
        })
    return pd.DataFrame(rows).sort_values("auc_directed", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_feature_table(data_path: str | Path, cfg: FeatureConfig | None = None, *,
                        run_causality_check: bool = True) -> dict[str, Any]:
    """Build the complete behavioural feature table and its validation evidence.

    Returns a dict with keys: features, dictionary, audit, validation.
    All returned artifacts come from THIS single execution.
    """
    cfg = cfg or FeatureConfig()
    df, schema, label, diagnostics = load_frame(data_path, cfg)
    n = len(df)

    feats, dictionary, ctx = compute_features(df, cfg, schema)

    validation: dict[str, Any] = {
        "status": "NOT EXECUTED",
        "config_resolved": cfg.snapshot(),
        "dataset": diagnostics,
        "schema_resolved": dict(schema),
        "aggregate_cache_hits": ctx.cache_hits,
    }
    failures: list[str] = []

    # -- row / identity alignment -------------------------------------------
    in_order = df.sort_values("__input_pos", kind="mergesort")
    feats = feats.iloc[in_order.index.to_numpy()].reset_index(drop=True)   # restore input row order
    expected_ids = in_order["__row_key"].to_numpy()
    aligned = bool(len(feats) == n and np.array_equal(feats["transaction_id"].to_numpy(), expected_ids))
    validation["row_alignment"] = {
        "input_rows": int(diagnostics["input_rows"]), "output_rows": int(len(feats)),
        "transaction_id_sequence_matches_input": aligned,
        "row_order": "input CSV order (restored after causal-order computation)",
    }
    if not aligned:
        failures.append("transaction_id alignment with the input dataset failed")

    # -- forbidden columns ---------------------------------------------------
    present = [c for c in FORBIDDEN_INPUTS if c in feats.columns]
    validation["forbidden_columns_present"] = present
    if present:
        failures.append(f"forbidden columns present in the feature table: {present}")

    fcols = [c for c in feats.columns if c != "transaction_id"]

    # -- declared/built contract + duplicates already enforced in compute_features
    validation["feature_count_total"] = len(fcols)
    validation["feature_count_by_group"] = (
        pd.DataFrame(dictionary).groupby("group")["name"].count().sort_index().to_dict()
    )

    # -- NaN / Inf -----------------------------------------------------------
    arr = feats[fcols].to_numpy(np.float64)
    inf_cells = int(np.isinf(arr).sum())
    validation["nan_cells_total"] = int(np.isnan(arr).sum())
    validation["inf_cells_total"] = inf_cells
    validation["nan_policy"] = (
        "NaN means 'undefined given the causal window', e.g. insufficient prior history. "
        "It is NOT imputed here; cfg.fillna is available and imputation is the model layer's decision."
    )
    if inf_cells:
        failures.append(f"{inf_cells} non-finite Inf cells present")

    # -- declared ranges -----------------------------------------------------
    range_violations = check_ranges(feats, dictionary, cfg)
    validation["range_violations"] = range_violations
    if range_violations:
        failures.append(f"{len(range_violations)} feature(s) violate their declared range")

    # -- locked spec coverage ------------------------------------------------
    coverage = check_spec_coverage(dictionary, cfg)
    validation["spec_coverage"] = coverage
    if not coverage["passed"]:
        failures.append(f"locked spec items uncovered: {coverage['uncovered_locked_items']}")

    # -- causality -----------------------------------------------------------
    if run_causality_check:
        causal = causality_prefix_test(df, cfg, schema, feats)
        validation["causality_prefix_test"] = causal
        if causal.get("executed") and not causal["passed"]:
            failures.append(f"causality prefix test failed for {causal['columns_with_mismatches']} column(s)")
    else:
        validation["causality_prefix_test"] = {"executed": False, "reason": "disabled via --no-causality-check"}

    if cfg.fillna is not None:
        feats[fcols] = feats[fcols].fillna(cfg.fillna)
        validation["fillna_applied"] = cfg.fillna

    # -- feature dictionary with OBSERVED statistics from this run ------------
    observed = pd.DataFrame({
        "name": fcols,
        "observed_min": [float(feats[c].min()) if feats[c].notna().any() else np.nan for c in fcols],
        "observed_max": [float(feats[c].max()) if feats[c].notna().any() else np.nan for c in fcols],
        "missing_rate": [float(feats[c].isna().mean()) for c in fcols],
        "n_unique": [int(feats[c].nunique(dropna=True)) for c in fcols],
    })
    dictionary_df = pd.DataFrame(dictionary).merge(observed, on="name", how="left")

    # -- audit ---------------------------------------------------------------
    if label is not None:
        audit = univariate_audit(feats, label, cfg)
        flagged = audit[audit.flag_investigate_high_auc]
        validation["audit"] = {
            "note": ("ROC-AUC is an audit metric only (DECISIONS.md). Features above the "
                     "threshold require investigation for synthetic artifact, missingness "
                     "artifact, temporal leakage or actual leakage before being trusted."),
            "auc_flag_threshold": cfg.auc_flag_threshold,
            "features_flagged_for_investigation": flagged[["feature", "auc_directed", "n_scored", "missing_rate"]]
                .to_dict(orient="records"),
            "features_with_missingness_signal": audit[audit.flag_missingness_signal]["feature"].tolist(),
        }
    else:
        audit = None
        validation["audit"] = {"status": "NOT AVAILABLE", "reason": "no label column in the input dataset"}

    validation["failures"] = failures
    validation["status"] = "PASSED" if not failures else "FAILED"
    return {"features": feats, "dictionary": dictionary_df, "audit": audit, "validation": validation}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the CONFLUX behavioural feature table.")
    ap.add_argument("--data", default="data/raw/dataset_v4_final.csv",
                    help="frozen input dataset (read-only)")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--config", default=None,
                    help="project YAML config (e.g. configs/default.yaml); reads the 'features' section")
    ap.add_argument("--config-section", default="features")
    ap.add_argument("--no-causality-check", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s %(message)s")

    cfg = FeatureConfig.from_yaml(args.config, args.config_section) if args.config else FeatureConfig()

    result = build_feature_table(args.data, cfg, run_causality_check=not args.no_causality_check)
    feats, dictionary, audit, validation = (result["features"], result["dictionary"],
                                            result["audit"], result["validation"])

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # ONE execution -> ALL artifacts. These are OUTPUTS, never inputs or source code.
    feats.to_csv(out / "features_v4.csv", index=False)
    dictionary.to_csv(out / "feature_dictionary.csv", index=False)
    if audit is not None:
        audit.to_csv(out / "univariate_auc.csv", index=False)
    (out / "validation_report.json").write_text(json.dumps(validation, indent=2, default=str))

    print(f"rows={len(feats)}  features={validation['feature_count_total']}  "
          f"by_group={validation['feature_count_by_group']}")
    print(f"causality_prefix_test={validation['causality_prefix_test'].get('passed', 'NOT EXECUTED')}")
    print(f"validation={validation['status']}")
    for f in validation["failures"]:
        print(f"  FAILURE: {f}")
    return 0 if validation["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
