"""CONFLUX Phase 4A -- scoring feature construction.

Reuses conflux.evaluation.candidate_diagnostics.build_candidate_features for
every structural quantity. It is called with attributes=None ON PURPOSE: in
that mode it computes only per-candidate structural features, none of which
depends on any population-level statistic, so nothing it returns can leak
across a temporal boundary.

The three behavioural features Phase 4 needs are derived HERE, and each is
also a pure per-candidate function:

  max_transactions_per_shared_card   -- Phase 3B writes device and ip variants
                                        but no card variant; derived by joining
                                        assignments to card_fingerprint.
  max_identical_amount_share         -- largest repeated amount / size.
  auth_share_<value>                 -- ablation only, vocabulary discovered
                                        at runtime and printed.

Ground truth never enters this module.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from conflux.evaluation.campaign_evaluation import GROUND_TRUTH_COLUMNS, ID_COL
from conflux.evaluation.candidate_diagnostics import (
    DiagnosticInputError, build_candidate_features, load_candidate_artifacts,
)
from conflux.graph.config import ENTITY_COLUMNS
from conflux.scoring.config import (
    ABLATION_FEATURES, ALLOWED_RAW_COLUMNS, AUTH_FEATURE_PREFIX,
    CORE_FEATURE_NAMES, CORRELATION_CAP, EXCLUDED_FROM_SCORING,
    MIN_SPAN_SECONDS, PRECEDENCE,
)

log = logging.getLogger("conflux.scoring.candidate_features")

CARD_COL = ENTITY_COLUMNS["card"]
AMOUNT_COL = "amount"
AUTH_COL = "auth_outcome"


class ScoringFeatureError(ValueError):
    """A required input for Phase 4 feature construction is missing."""


@dataclass
class ScoringFeatures:
    frame: pd.DataFrame
    core_features: tuple[str, ...]
    ablation_features: tuple[str, ...]
    auth_features: tuple[str, ...]
    notes: dict[str, Any] = field(default_factory=dict)

    def matrix(self, names: Sequence[str]) -> pd.DataFrame:
        missing = [n for n in names if n not in self.frame.columns]
        if missing:
            raise ScoringFeatureError(f"feature(s) not built: {missing}")
        return self.frame[list(names)].copy()


def _slug(value: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return s or "blank"


def load_structural_attributes(dataset_path: str | Path) -> pd.DataFrame:
    """Read ONLY the allowlisted structural/attribute columns from the raw CSV.

    card_fingerprint is a graph entity column. amount and auth_outcome are the
    graph layer's declared ATTRIBUTE_COLUMNS. label / campaign_id / _source_type
    are never in the allowlist and the read is asserted clean.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise ScoringFeatureError(f"dataset not found: {path}")

    leaked = [c for c in GROUND_TRUTH_COLUMNS if c in ALLOWED_RAW_COLUMNS]
    if leaked:  # defensive; the allowlist is a literal in config
        raise ScoringFeatureError(f"forbidden column in read allowlist: {leaked}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in ALLOWED_RAW_COLUMNS if c in header]
    if ID_COL not in usecols or CARD_COL not in usecols:
        raise ScoringFeatureError(
            f"dataset must expose {ID_COL} and {CARD_COL}; found {usecols}")

    df = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False,
                     na_values=[], low_memory=False)
    out = pd.DataFrame({ID_COL: df[ID_COL].astype(str).str.strip()})
    out[CARD_COL] = df[CARD_COL].astype(str).str.strip()
    out[AMOUNT_COL] = (pd.to_numeric(df[AMOUNT_COL], errors="coerce")
                       if AMOUNT_COL in df.columns else np.nan)
    out[AUTH_COL] = (df[AUTH_COL].astype(str).str.strip()
                     if AUTH_COL in df.columns else "")
    return out


def build_scoring_features(candidates: pd.DataFrame, assignments: pd.DataFrame,
                           attributes: pd.DataFrame, *,
                           min_size: int = 2) -> ScoringFeatures:
    """One row per multi-transaction candidate. Ground truth is not an argument."""
    # ---- structural, delegated. attributes=None is deliberate: that mode has
    # no population statistics at all, so nothing here can leak across time.
    fs = build_candidate_features(candidates, assignments, None, min_size=min_size)
    f = fs.frame.copy()

    cand_ids = f["candidate_id"].astype(str)
    size = f["size"].to_numpy(dtype=float)

    asg = assignments.copy()
    asg["candidate_id"] = asg["candidate_id"].astype(str)
    asg[ID_COL] = asg[ID_COL].astype(str)
    asg = asg.loc[asg["candidate_id"].isin(set(cand_ids))]
    asg["ts_ns"] = asg["ts_ns"].astype("int64")

    # ---- timing / metadata used by the splitters and the reports
    ts = asg.groupby("candidate_id")["ts_ns"].agg(["min", "max"]).reindex(cand_ids)
    f["first_ts_ns"] = ts["min"].to_numpy(dtype="int64")
    f["last_ts_ns"] = ts["max"].to_numpy(dtype="int64")

    # ---- burst rate with the granularity floor (no NaN by construction)
    span = np.maximum(f["time_span_seconds"].to_numpy(dtype=float), MIN_SPAN_SECONDS)
    f["burst_rate_per_minute"] = (size - 1.0) * 60.0 / span

    m = asg[[ID_COL, "candidate_id"]].merge(attributes, on=ID_COL, how="left")
    if m[CARD_COL].isna().any() or (m[CARD_COL].astype(str) == "").any():
        raise ScoringFeatureError(
            "some assigned transactions have no card_fingerprint; refusing to "
            "impute a structural identifier.")

    # ---- shared-card evidence (the Phase 3B API gap, closed here)
    per_card = m.groupby(["candidate_id", CARD_COL]).size()
    f["max_transactions_per_shared_card"] = (
        per_card.groupby(level=0).max().reindex(cand_ids).fillna(1).to_numpy(dtype=float))

    # ---- repeated-amount behaviour
    rounded = m[AMOUNT_COL].round(2)
    per_amt = m.assign(_a=rounded).groupby(["candidate_id", "_a"]).size()
    f["max_identical_amount_share"] = (
        per_amt.groupby(level=0).max().reindex(cand_ids).to_numpy(dtype=float) / size)

    # ---- auth shares: ABLATION ONLY (locked decision). Vocabulary is a
    # property of the dataset, so it is discovered and reported, not assumed.
    auth_features: list[str] = []
    auth_vocabulary: list[str] = []
    if AUTH_COL in m.columns and m[AUTH_COL].astype(str).str.strip().ne("").any():
        auth_vocabulary = sorted(m[AUTH_COL].astype(str).unique().tolist())
        shares = (pd.crosstab(m["candidate_id"], m[AUTH_COL], normalize="index")
                    .reindex(cand_ids).fillna(0.0))
        for col in shares.columns:
            name = f"{AUTH_FEATURE_PREFIX}{_slug(col)}"
            f[name] = shares[col].to_numpy(dtype=float)
            auth_features.append(name)

    # ---- integrity assertions
    leaked = [c for c in GROUND_TRUTH_COLUMNS if c in f.columns]
    if leaked:
        raise ScoringFeatureError(f"ground-truth column(s) reached features: {leaked}")

    missing_core = [c for c in CORE_FEATURE_NAMES if c not in f.columns]
    if missing_core:
        raise ScoringFeatureError(f"core feature(s) not built: {missing_core}")

    bad = {c: int(f[c].isna().sum() + np.isinf(f[c].to_numpy(dtype=float)).sum())
           for c in CORE_FEATURE_NAMES}
    nonfinite = {k: v for k, v in bad.items() if v}
    if nonfinite:
        raise ScoringFeatureError(
            f"core features contain NaN/Inf, which would force an invented "
            f"imputation rule: {nonfinite}")

    ablation = tuple(c for c in ABLATION_FEATURES if c in f.columns)

    log.info("scoring features: %s candidates, %s core, %s ablation, %s auth",
             len(f), len(CORE_FEATURE_NAMES), len(ablation), len(auth_features))

    return ScoringFeatures(
        frame=f.reset_index(drop=True),
        core_features=tuple(CORE_FEATURE_NAMES),
        ablation_features=ablation,
        auth_features=tuple(auth_features),
        notes={
            "min_size": int(min_size),
            "structural_source": "candidate_diagnostics.build_candidate_features"
                                 " (attributes=None -> no population statistics)",
            "phase4_derived": ["burst_rate_per_minute",
                               "max_transactions_per_shared_card",
                               "max_identical_amount_share",
                               f"{AUTH_FEATURE_PREFIX}*"],
            "raw_columns_read": list(ALLOWED_RAW_COLUMNS),
            "auth_vocabulary_observed": auth_vocabulary,
            "auth_policy": "ablation only; excluded from the core scorer",
            "excluded_from_scoring": list(EXCLUDED_FROM_SCORING),
            "min_span_seconds_floor": MIN_SPAN_SECONDS,
            "ground_truth_used": False,
        },
    )


# ----------------------------------------------------------------------
# decorrelation -- computed on the UNLABELLED training population
# ----------------------------------------------------------------------
def spearman_matrix(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in names if c in frame.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return frame[cols].corr(method="spearman").round(4)


def prune_correlated(frame: pd.DataFrame, names: Sequence[str], *,
                     cap: float = CORRELATION_CAP,
                     precedence: Sequence[str] = PRECEDENCE
                     ) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Drop the lower-precedence member of any pair above the cap.

    Labels are not an argument. Precedence is fixed in config by strength of
    the domain argument, so the outcome cannot be steered by results.
    """
    corr = spearman_matrix(frame, names)
    if corr.empty:
        return tuple(names), []

    rank = {n: i for i, n in enumerate(precedence)}
    ordered = sorted(corr.columns, key=lambda n: rank.get(n, len(rank)))
    kept: list[str] = []
    dropped: list[dict[str, Any]] = []
    for name in ordered:
        clash = next(((k, float(corr.loc[name, k])) for k in kept
                      if abs(corr.loc[name, k]) > cap), None)
        if clash is None:
            kept.append(name)
        else:
            dropped.append({"dropped": name, "kept_instead": clash[0],
                            "spearman": clash[1], "cap": cap})
    return tuple(n for n in names if n in set(kept)), dropped
