"""CONFLUX Phase 3C -- diagnostic comparison of attack-containing vs non-campaign
candidate groups.

Module: conflux.evaluation.candidate_diagnostics

SCOPE (locked to this phase): DESCRIPTIVE DIAGNOSTICS ONLY.
No model is trained. No candidate is created, altered, merged or re-derived. No
threshold produced here is a decision rule, a feature, a weight, or an input to
anything upstream. Phase 3B is read-only.

WHAT THIS MODULE DOES
---------------------
1. Reads the Phase 3B artifacts (campaign_candidates.csv +
   campaign_candidate_assignments.csv) and the two non-ground-truth transaction
   attributes the graph layer already allows (amount, auth_outcome).
2. Derives a per-candidate STRUCTURAL + BEHAVIOURAL description of every
   multi-transaction candidate. Every quantity is computable at decision time;
   none of them uses label or campaign_id.
3. THEN, and only then, reads ground truth to split those candidates into two
   groups:
       group A  attack-containing  (>= 1 campaign transaction)   -> expected 81
       group B  non-campaign       (0 campaign transactions)     -> expected 4291
4. Compares the two distributions feature by feature with rank-based,
   distribution-free statistics (Mann-Whitney AUC, Cliff's delta, KS, BH-FDR)
   and ranks the strongest separations.

GROUND TRUTH POLICY
-------------------
label / campaign_id enter this module in exactly one function, attach_groups(),
which receives the already-built feature table as an argument. The group flag is
excluded from the feature list by construction, and build_candidate_features()
is asserted free of forbidden columns.

INTERPRETATION POLICY
---------------------
A large AUC here means "this quantity separates the two groups in this dataset,
in-sample, univariately". It does NOT mean it will work as a detector, it does
not survive multiplicity or correlation on its own, and 81 positives is a small
sample. The descriptive best-threshold block is reported with precision and lift
against the 1.85% base rate precisely so nobody reads it as a ready rule.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from conflux.evaluation.campaign_evaluation import (
    CAMPAIGN_COL,
    GROUND_TRUTH_COLUMNS,
    ID_COL,
    LABEL_COL,
    GroundTruthError,
    load_ground_truth,
)

log = logging.getLogger("conflux.evaluation.candidate_diagnostics")

DIAGNOSTIC_SCHEMA_VERSION = "conflux.evaluation.candidate_diagnostics.v1"

TS_COL = "timestamp"
AMOUNT_COL = "amount"
AUTH_COL = "auth_outcome"

# columns that describe the group split, never features
GROUP_COLUMNS: tuple[str, ...] = (
    "group", "is_attack_containing", "campaign_transactions",
    "non_campaign_transactions", "n_distinct_campaigns", "dominant_campaign_id",
    "campaign_share", "purity_class", "label1_transactions",
)
META_COLUMNS: tuple[str, ...] = ("candidate_id", "dominant_link_type",
                                 "size_bucket", "span_bucket")

GROUP_BY_CHOICES: tuple[str, ...] = ("campaign_id", "label", "either")


class DiagnosticInputError(ValueError):
    """A required Phase 3B artifact or column is missing or unusable."""


# ----------------------------------------------------------------------
# small distribution-free statistics (no scipy dependency)
# ----------------------------------------------------------------------
def normal_sf(z: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def mann_whitney(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Rank-sum comparison. AUC = P(random a > random b) with ties at 0.5."""
    n1, n2 = int(a.size), int(b.size)
    if n1 == 0 or n2 == 0:
        return {"auc": float("nan"), "u": float("nan"), "z": float("nan"),
                "p_value": float("nan")}

    combined = np.concatenate([a, b])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2.0
    auc = u1 / (n1 * n2)

    n = n1 + n2
    _, counts = np.unique(combined, return_counts=True)
    tie_term = float(((counts.astype(float) ** 3) - counts).sum())
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    sigma = math.sqrt(var) if var > 0 else 0.0
    mu = n1 * n2 / 2.0
    z = (u1 - mu) / sigma if sigma > 0 else 0.0
    return {"auc": float(auc), "u": float(u1), "z": float(z),
            "p_value": float(2.0 * normal_sf(abs(z)))}


def cliffs_delta(auc: float) -> float:
    """Cliff's delta is a linear reparametrisation of the rank AUC."""
    return float(2.0 * auc - 1.0) if not math.isnan(auc) else float("nan")


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    sa, sb = np.sort(a), np.sort(b)
    grid = np.sort(np.concatenate([sa, sb]))
    c1 = np.searchsorted(sa, grid, side="right") / sa.size
    c2 = np.searchsorted(sb, grid, side="right") / sb.size
    return float(np.max(np.abs(c1 - c2)))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> dict[str, float]:
    """Comparison of two rates. Pooled-variance normal approximation."""
    if n1 == 0 or n2 == 0:
        return {"z": float("nan"), "p_value": float("nan")}
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if 0 < p < 1 else 0.0
    z = (p1 - p2) / se if se > 0 else 0.0
    return {"z": float(z), "p_value": float(2.0 * normal_sf(abs(z)))}


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """BH-FDR adjusted values, NaN-safe, order preserving."""
    p = np.asarray(p_values, dtype=float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan, dtype=float)
    if not ok.any():
        return q.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx], kind="mergesort")]
    m = order.size
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, p[i] * m / rank)
        q[i] = val
        prev = val
    return q.tolist()


def descriptive_best_threshold(a: np.ndarray, b: np.ndarray, *,
                               direction: int, grid: int = 200) -> dict[str, Any]:
    """In-sample single-cut description. NOT a rule, NOT a model, NOT tuned.

    direction >= 0 -> 'feature >= t' marks the attack-containing side
    direction <  0 -> 'feature <= t' marks the attack-containing side
    """
    if a.size == 0 or b.size == 0:
        return {}
    allv = np.concatenate([a, b])
    qs = np.unique(np.quantile(allv, np.linspace(0.0, 1.0, grid + 1)))
    base_rate = a.size / (a.size + b.size)
    best: dict[str, Any] = {}
    best_j = -np.inf
    for t in qs:
        if direction >= 0:
            tp = int((a >= t).sum())
            fp = int((b >= t).sum())
            rule = f">= {float(t):.6g}"
        else:
            tp = int((a <= t).sum())
            fp = int((b <= t).sum())
            rule = f"<= {float(t):.6g}"
        tn = int(b.size - fp)
        recall = tp / a.size
        specificity = tn / b.size
        j = recall + specificity - 1.0
        if j > best_j:
            best_j = j
            flagged = tp + fp
            precision = tp / flagged if flagged else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            best = {
                "threshold": float(t),
                "rule": f"flag when feature {rule}",
                "candidates_flagged": int(flagged),
                "attack_candidates_flagged": tp,
                "recall": round(recall, 6),
                "specificity": round(specificity, 6),
                "precision": round(precision, 6),
                "f1": round(f1, 6),
                "lift_over_base_rate": round(precision / base_rate, 4) if base_rate else 0.0,
                "youden_j": round(j, 6),
            }
    best["caveat"] = ("in-sample, univariate, not cross-validated, not a decision "
                      "rule and not carried into any later phase")
    return best


def _quantiles(v: np.ndarray) -> dict[str, float]:
    if v.size == 0:
        return {k: float("nan") for k in
                ("min", "p05", "p25", "median", "p75", "p95", "max", "mean", "std")}
    return {
        "min": float(np.min(v)), "p05": float(np.quantile(v, 0.05)),
        "p25": float(np.quantile(v, 0.25)), "median": float(np.median(v)),
        "p75": float(np.quantile(v, 0.75)), "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)), "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
    }


# ----------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------
def load_candidate_artifacts(candidates_path: str | Path,
                             assignments_path: str | Path
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the stored Phase 3B output. Read-only; nothing is re-derived."""
    cpath, apath = Path(candidates_path), Path(assignments_path)
    for p in (cpath, apath):
        if not p.exists():
            raise DiagnosticInputError(
                f"Phase 3B artifact not found: {p}. Run "
                "`python -m conflux.graph.build_candidates` first.")

    candidates = pd.read_csv(cpath, dtype={"candidate_id": str}, low_memory=False)
    assignments = pd.read_csv(
        apath, dtype={ID_COL: str, "candidate_id": str}, low_memory=False)

    need_c = {"candidate_id", "size", "time_span_seconds", "link_edge_count"}
    missing = sorted(need_c - set(candidates.columns))
    if missing:
        raise DiagnosticInputError(f"campaign_candidates.csv missing {missing}")
    need_a = {ID_COL, "candidate_id", "ts_ns"}
    missing = sorted(need_a - set(assignments.columns))
    if missing:
        raise DiagnosticInputError(f"assignments CSV missing {missing}")

    leaked = [c for c in GROUND_TRUTH_COLUMNS
              if c in candidates.columns or c in assignments.columns]
    if leaked:
        raise DiagnosticInputError(
            f"ground-truth column(s) {leaked} found in the Phase 3B artifacts; "
            "refusing to run diagnostics on contaminated inputs.")

    log.info("loaded %s candidates and %s assignments",
             len(candidates), len(assignments))
    return candidates, assignments


def load_transaction_attributes(dataset_path: str | Path) -> pd.DataFrame:
    """amount + auth_outcome only. These are the graph layer's declared
    ATTRIBUTE_COLUMNS: transaction metadata, not ground truth, not identifiers."""
    path = Path(dataset_path)
    if not path.exists():
        raise DiagnosticInputError(f"dataset not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [ID_COL] + [c for c in (AMOUNT_COL, AUTH_COL) if c in header]
    df = pd.read_csv(path, usecols=usecols, dtype=str,
                     keep_default_na=False, na_values=[], low_memory=False)
    out = pd.DataFrame({ID_COL: df[ID_COL].astype(str).str.strip()})
    out[AMOUNT_COL] = (pd.to_numeric(df[AMOUNT_COL], errors="coerce")
                       if AMOUNT_COL in df.columns else np.nan)
    out[AUTH_COL] = (df[AUTH_COL].astype(str).str.strip()
                     if AUTH_COL in df.columns else "")
    return out


# ----------------------------------------------------------------------
# per-candidate description (NO ground truth anywhere in here)
# ----------------------------------------------------------------------
def _slug(value: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return s or "blank"


def _pipe_count(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    return s.apply(lambda v: 0 if not v else len([x for x in v.split("|") if x]))


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    d = den.astype(float)
    return np.where(d > 0, num.astype(float) / d.replace(0, np.nan), np.nan)


@dataclass
class FeatureSet:
    frame: pd.DataFrame
    numeric_features: tuple[str, ...]
    boolean_features: tuple[str, ...]
    notes: dict[str, Any] = field(default_factory=dict)


def build_candidate_features(candidates: pd.DataFrame,
                             assignments: pd.DataFrame,
                             attributes: pd.DataFrame | None = None,
                             *, min_size: int = 2) -> FeatureSet:
    """One row per multi-transaction candidate, structural + behavioural.

    Everything here is a function of Phase 3B output and decision-time
    transaction metadata. Ground truth is not an argument to this function.
    """
    cand = candidates.loc[candidates["size"].astype(int) >= int(min_size)].copy()
    cand["candidate_id"] = cand["candidate_id"].astype(str)
    keep_ids = set(cand["candidate_id"])
    asg = assignments.loc[assignments["candidate_id"].astype(str).isin(keep_ids)].copy()
    asg["candidate_id"] = asg["candidate_id"].astype(str)
    asg["ts_ns"] = asg["ts_ns"].astype("int64")

    f = pd.DataFrame({"candidate_id": cand["candidate_id"].to_numpy()})
    size = cand["size"].astype(float).to_numpy()
    f["size"] = size
    f["time_span_seconds"] = cand["time_span_seconds"].astype(float).to_numpy()

    # ---- timing ------------------------------------------------------
    asg = asg.sort_values(["candidate_id", "ts_ns"], kind="mergesort")
    gaps = asg.groupby("candidate_id", sort=False)["ts_ns"].diff() / 1e9
    gap_stats = (gaps.groupby(asg["candidate_id"]).agg(
        ["mean", "median", "min", "max", "std"]).reindex(f["candidate_id"]))
    f["mean_inter_arrival_seconds"] = gap_stats["mean"].to_numpy()
    f["median_inter_arrival_seconds"] = gap_stats["median"].to_numpy()
    f["min_inter_arrival_seconds"] = gap_stats["min"].to_numpy()
    f["max_inter_arrival_seconds"] = gap_stats["max"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        f["inter_arrival_cv"] = np.where(
            gap_stats["mean"].to_numpy() > 0,
            gap_stats["std"].to_numpy() / gap_stats["mean"].to_numpy(), np.nan)
        span = f["time_span_seconds"].to_numpy()
        f["transactions_per_minute"] = np.where(
            span > 0, (size - 1) * 60.0 / span, np.nan)
        f["burstiness_ratio"] = np.where(
            gap_stats["median"].to_numpy() > 0,
            gap_stats["max"].to_numpy() / gap_stats["median"].to_numpy(), np.nan)

    # ---- link topology ----------------------------------------------
    edges = cand["link_edge_count"].astype(float).to_numpy()
    f["link_edge_count"] = edges
    max_edges = size * (size - 1.0) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        f["link_density"] = np.where(max_edges > 0, edges / max_edges, np.nan)
    f["links_per_transaction"] = edges / size

    multi = cand.get("links_multi_entity", pd.Series(0, index=cand.index)).astype(float).to_numpy()
    f["multi_entity_link_count"] = multi
    with np.errstate(invalid="ignore", divide="ignore"):
        f["multi_entity_link_fraction"] = np.where(edges > 0, multi / edges, np.nan)

    lc = cand.get("links_card", pd.Series(0, index=cand.index)).astype(float).to_numpy()
    ld = cand.get("links_device", pd.Series(0, index=cand.index)).astype(float).to_numpy()
    li = cand.get("links_ip", pd.Series(0, index=cand.index)).astype(float).to_numpy()
    tot = lc + ld + li
    f["links_card"], f["links_device"], f["links_ip"] = lc, ld, li
    with np.errstate(invalid="ignore", divide="ignore"):
        f["link_share_card"] = np.where(tot > 0, lc / tot, np.nan)
        f["link_share_device"] = np.where(tot > 0, ld / tot, np.nan)
        f["link_share_ip"] = np.where(tot > 0, li / tot, np.nan)
    f["n_link_entity_types_used"] = _pipe_count(
        cand.get("link_entity_types", pd.Series("", index=cand.index))).to_numpy()

    stacked = np.vstack([lc, ld, li])
    dom_idx = np.argmax(stacked, axis=0)
    dom = np.array(["card", "device", "ip"])[dom_idx]
    dom = np.where(tot <= 0, "none", dom)
    ties = (stacked.max(axis=0) == stacked).sum(axis=0) > 1
    f["dominant_link_type"] = np.where(ties & (tot > 0), "mixed", dom)

    # ---- entity spread / concentration --------------------------------
    for et in ("cards", "devices", "ips", "merchants"):
        col = f"distinct_{et}"
        vals = cand.get(col, pd.Series(np.nan, index=cand.index)).astype(float).to_numpy()
        f[col] = vals
        with np.errstate(invalid="ignore", divide="ignore"):
            f[f"{col}_per_transaction"] = np.where(size > 0, vals / size, np.nan)
            f[f"transactions_per_distinct_{et[:-1]}"] = np.where(
                vals > 0, size / vals, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        f["cards_per_device"] = np.where(
            f["distinct_devices"] > 0, f["distinct_cards"] / f["distinct_devices"], np.nan)
        f["cards_per_ip"] = np.where(
            f["distinct_ips"] > 0, f["distinct_cards"] / f["distinct_ips"], np.nan)
        f["merchants_per_card"] = np.where(
            f["distinct_cards"] > 0, f["distinct_merchants"] / f["distinct_cards"], np.nan)

    f["max_transactions_per_shared_device"] = cand.get(
        "max_transactions_per_shared_device", pd.Series(0, index=cand.index)).astype(float).to_numpy()
    f["max_transactions_per_shared_ip"] = cand.get(
        "max_transactions_per_shared_ip", pd.Series(0, index=cand.index)).astype(float).to_numpy()
    f["n_shared_cards"] = _pipe_count(
        cand.get("shared_card_ids", pd.Series("", index=cand.index))).to_numpy()
    f["n_shared_devices"] = _pipe_count(
        cand.get("shared_device_ids", pd.Series("", index=cand.index))).to_numpy()
    f["n_shared_ips"] = _pipe_count(
        cand.get("shared_ip_ids", pd.Series("", index=cand.index))).to_numpy()

    # BIN stays context: described, never a connectivity mechanism.
    bins = cand.get("distinct_bins", pd.Series(np.nan, index=cand.index)).astype(float).to_numpy()
    f["distinct_bins"] = bins
    with np.errstate(invalid="ignore", divide="ignore"):
        f["distinct_bins_per_transaction"] = np.where(size > 0, bins / size, np.nan)

    # ---- behavioural (amount / auth_outcome) --------------------------
    auth_cols: list[str] = []
    if attributes is not None and len(attributes):
        m = asg[[ID_COL, "candidate_id"]].merge(attributes, on=ID_COL, how="left")

        amt = m.groupby("candidate_id")[AMOUNT_COL]
        agg = amt.agg(["mean", "median", "std", "min", "max", "nunique"]).reindex(f["candidate_id"])
        f["amount_mean"] = agg["mean"].to_numpy()
        f["amount_median"] = agg["median"].to_numpy()
        f["amount_std"] = agg["std"].to_numpy()
        f["amount_min"] = agg["min"].to_numpy()
        f["amount_max"] = agg["max"].to_numpy()
        f["amount_range"] = f["amount_max"] - f["amount_min"]
        with np.errstate(invalid="ignore", divide="ignore"):
            f["amount_cv"] = np.where(agg["mean"].to_numpy() > 0,
                                      agg["std"].to_numpy() / agg["mean"].to_numpy(), np.nan)
            f["distinct_amounts_per_transaction"] = np.where(
                size > 0, agg["nunique"].to_numpy() / size, np.nan)

        rounded = m[AMOUNT_COL].round(2)
        repeats = (m.assign(_a=rounded).groupby(["candidate_id", "_a"]).size()
                     .groupby(level=0).max().reindex(f["candidate_id"]))
        with np.errstate(invalid="ignore", divide="ignore"):
            f["max_identical_amount_share"] = np.where(
                size > 0, repeats.to_numpy(dtype=float) / size, np.nan)

        is_round = np.isclose(rounded.to_numpy() % 10.0, 0.0, atol=1e-9)
        f["round_amount_share"] = (pd.Series(is_round, index=m["candidate_id"])
                                   .groupby(level=0).mean()
                                   .reindex(f["candidate_id"]).to_numpy())

        global_p10 = float(np.nanquantile(attributes[AMOUNT_COL].to_numpy(dtype=float), 0.10))
        low = m[AMOUNT_COL].to_numpy(dtype=float) <= global_p10
        f["low_amount_share"] = (pd.Series(low, index=m["candidate_id"])
                                 .groupby(level=0).mean()
                                 .reindex(f["candidate_id"]).to_numpy())

        if AUTH_COL in m.columns and m[AUTH_COL].astype(str).str.strip().ne("").any():
            shares = (pd.crosstab(m["candidate_id"], m[AUTH_COL], normalize="index")
                        .reindex(f["candidate_id"]).fillna(0.0))
            for col in shares.columns:
                name = f"auth_share_{_slug(col)}"
                f[name] = shares[col].to_numpy()
                auth_cols.append(name)

    # ---- boolean flags -------------------------------------------------
    f["has_card_links"] = f["links_card"] > 0
    f["has_device_links"] = f["links_device"] > 0
    f["has_ip_links"] = f["links_ip"] > 0
    f["has_multi_entity_link"] = f["multi_entity_link_count"] > 0
    f["uses_multiple_link_types"] = f["n_link_entity_types_used"] >= 2
    f["has_shared_device"] = f["max_transactions_per_shared_device"] >= 2
    f["has_shared_ip"] = f["max_transactions_per_shared_ip"] >= 2
    f["has_shared_card"] = f["n_shared_cards"] >= 1
    f["single_device_group"] = f["distinct_devices"] == 1
    f["single_ip_group"] = f["distinct_ips"] == 1
    f["single_card_group"] = f["distinct_cards"] == 1
    f["all_distinct_cards"] = np.isclose(f["distinct_cards"].to_numpy(), size)
    f["spans_multiple_bins"] = f["distinct_bins"] > 1
    f["spans_multiple_merchants"] = f["distinct_merchants"] > 1
    f["fully_connected"] = np.isclose(f["link_density"].to_numpy(), 1.0)

    # ---- buckets (reporting only) --------------------------------------
    f["size_bucket"] = pd.cut(
        f["size"], bins=[1, 2, 3, 5, 10, 20, 50, np.inf], right=True,
        labels=["2", "3", "4-5", "6-10", "11-20", "21-50", "50+"]).astype(str)
    f["span_bucket"] = pd.cut(
        f["time_span_seconds"], bins=[-0.001, 60, 300, 900, 1800, 3600, np.inf],
        labels=["<=1m", "1-5m", "5-15m", "15-30m", "30-60m", ">60m"]).astype(str)

    boolean_features = tuple(c for c in f.columns if f[c].dtype == bool)
    numeric_features = tuple(
        c for c in f.columns
        if c not in META_COLUMNS and c not in boolean_features
        and pd.api.types.is_numeric_dtype(f[c]))

    leaked = [c for c in GROUND_TRUTH_COLUMNS if c in f.columns]
    if leaked:  # defensive: structurally impossible, asserted anyway
        raise DiagnosticInputError(
            f"ground-truth column(s) {leaked} reached the feature table")

    log.info("built %s candidate rows x %s numeric + %s boolean features",
             len(f), len(numeric_features), len(boolean_features))
    return FeatureSet(
        frame=f.reset_index(drop=True),
        numeric_features=numeric_features,
        boolean_features=boolean_features,
        notes={"min_size": int(min_size),
               "auth_share_columns": auth_cols,
               "behavioural_inputs": [AMOUNT_COL, AUTH_COL],
               "ground_truth_used_in_feature_construction": False},
    )


# ----------------------------------------------------------------------
# ground truth enters HERE, and only to label the two groups
# ----------------------------------------------------------------------
def attach_groups(features: pd.DataFrame, assignments: pd.DataFrame,
                  ground_truth: pd.DataFrame, *,
                  group_by: str = "campaign_id") -> pd.DataFrame:
    """Split existing candidates into attack-containing vs non-campaign.

    The feature table is an INPUT. This function adds group columns and cannot
    change any feature value or any candidate's membership.
    """
    if group_by not in GROUP_BY_CHOICES:
        raise ValueError(f"group_by must be one of {list(GROUP_BY_CHOICES)}")

    asg = assignments[[ID_COL, "candidate_id"]].copy()
    asg[ID_COL] = asg[ID_COL].astype(str)
    asg["candidate_id"] = asg["candidate_id"].astype(str)

    gt = ground_truth[[ID_COL, CAMPAIGN_COL, "is_campaign", "label_int"]].copy()
    gt[ID_COL] = gt[ID_COL].astype(str)

    j = asg.merge(gt, on=ID_COL, how="inner", validate="one_to_one")
    j = j.loc[j["candidate_id"].isin(set(features["candidate_id"]))]

    lab1 = j["label_int"].fillna(-1).astype(int).eq(1)
    if group_by == "campaign_id":
        attack_txn = j["is_campaign"]
    elif group_by == "label":
        attack_txn = lab1
    else:
        attack_txn = j["is_campaign"] | lab1

    per = pd.DataFrame(index=pd.Index(features["candidate_id"], name="candidate_id"))
    per["campaign_transactions"] = (attack_txn.groupby(j["candidate_id"]).sum()
                                    .reindex(per.index).fillna(0).astype(int))
    per["label1_transactions"] = (lab1.groupby(j["candidate_id"]).sum()
                                  .reindex(per.index).fillna(0).astype(int))

    camp = j.loc[j["is_campaign"], ["candidate_id", CAMPAIGN_COL]]
    if len(camp):
        gc = (camp.groupby(["candidate_id", CAMPAIGN_COL]).size()
                  .rename("n").reset_index()
                  .sort_values(["candidate_id", "n", CAMPAIGN_COL],
                               ascending=[True, False, True], kind="mergesort"))
        per["n_distinct_campaigns"] = (gc.groupby("candidate_id")[CAMPAIGN_COL].nunique()
                                       .reindex(per.index).fillna(0).astype(int))
        per["dominant_campaign_id"] = (gc.drop_duplicates("candidate_id", keep="first")
                                         .set_index("candidate_id")[CAMPAIGN_COL]
                                         .reindex(per.index).fillna(""))
    else:
        per["n_distinct_campaigns"] = 0
        per["dominant_campaign_id"] = ""

    out = features.merge(per.reset_index(), on="candidate_id",
                         how="left", validate="one_to_one")
    out["campaign_transactions"] = out["campaign_transactions"].fillna(0).astype(int)
    out["label1_transactions"] = out["label1_transactions"].fillna(0).astype(int)
    out["non_campaign_transactions"] = (out["size"].astype(int)
                                        - out["campaign_transactions"])
    out["campaign_share"] = out["campaign_transactions"] / out["size"].astype(float)
    out["is_attack_containing"] = out["campaign_transactions"] > 0
    out["group"] = np.where(out["is_attack_containing"],
                            "attack_containing", "non_campaign")
    out["purity_class"] = np.select(
        [out["campaign_transactions"] == 0,
         out["n_distinct_campaigns"] >= 2,
         out["non_campaign_transactions"] > 0],
        ["non_campaign", "mixed_campaign", "campaign_with_normal"],
        default="pure_campaign")
    return out


def group_summary(features: pd.DataFrame, *, group_by: str,
                  expected_attack: int | None = None,
                  expected_other: int | None = None) -> dict[str, Any]:
    a = features.loc[features["is_attack_containing"]]
    b = features.loc[~features["is_attack_containing"]]
    total = len(features)
    out: dict[str, Any] = {
        "group_definition": {
            "grouping_column": group_by,
            "attack_containing": ">= 1 transaction flagged by the grouping column",
            "non_campaign": "0 flagged transactions",
            "scope": "multi-transaction candidates only (size >= 2)",
            "note": ("ground truth is used here and nowhere else; it defines the "
                     "two groups and never touches a feature value"),
        },
        "multi_transaction_candidates": int(total),
        "attack_containing_candidates": int(len(a)),
        "non_campaign_candidates": int(len(b)),
        "attack_containing_base_rate": round(len(a) / total, 6) if total else 0.0,
        "transactions_in_attack_containing_candidates": int(a["size"].sum()),
        "campaign_transactions_captured": int(a["campaign_transactions"].sum()),
        "normal_transactions_inside_attack_candidates":
            int(a["non_campaign_transactions"].sum()),
        "purity_class_counts": {
            k: int(v) for k, v in
            features["purity_class"].value_counts().sort_index().items()},
        "campaign_share_distribution_attack_group":
            {k: round(v, 6) for k, v in
             _quantiles(a["campaign_share"].to_numpy(dtype=float)).items()},
        "candidates_with_label1_transaction":
            int((features["label1_transactions"] > 0).sum()),
    }
    if expected_attack is not None or expected_other is not None:
        out["expected_counts_check"] = {
            "expected_attack_containing": expected_attack,
            "expected_non_campaign": expected_other,
            "observed_attack_containing": int(len(a)),
            "observed_non_campaign": int(len(b)),
            "matches": bool((expected_attack in (None, len(a)))
                            and (expected_other in (None, len(b)))),
        }
    return out


# ----------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------
def compare_numeric(features: pd.DataFrame, numeric_features: Iterable[str], *,
                    mask_a: pd.Series, mask_b: pd.Series,
                    label_a: str = "attack_containing",
                    label_b: str = "non_campaign") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in numeric_features:
        col = features[name].to_numpy(dtype=float)
        a = col[mask_a.to_numpy() & np.isfinite(col)]
        b = col[mask_b.to_numpy() & np.isfinite(col)]
        qa, qb = _quantiles(a), _quantiles(b)

        if a.size < 2 or b.size < 2 or np.unique(np.concatenate([a, b])).size < 2:
            rows.append({"feature": name, "n_a": int(a.size), "n_b": int(b.size),
                         "usable": False, "reason": "degenerate or too few values"})
            continue

        mw = mann_whitney(a, b)
        delta = cliffs_delta(mw["auc"])
        direction = 1 if delta >= 0 else -1
        best = descriptive_best_threshold(a, b, direction=direction)

        med_a, med_b = qa["median"], qb["median"]
        rows.append({
            "feature": name, "usable": True,
            "n_a": int(a.size), "n_b": int(b.size),
            "missing_a": int(mask_a.sum() - a.size),
            "missing_b": int(mask_b.sum() - b.size),
            f"median_{label_a}": round(med_a, 6),
            f"median_{label_b}": round(med_b, 6),
            f"mean_{label_a}": round(qa["mean"], 6),
            f"mean_{label_b}": round(qb["mean"], 6),
            f"p25_{label_a}": round(qa["p25"], 6),
            f"p75_{label_a}": round(qa["p75"], 6),
            f"p25_{label_b}": round(qb["p25"], 6),
            f"p75_{label_b}": round(qb["p75"], 6),
            "median_difference": round(med_a - med_b, 6),
            "median_ratio": (round(med_a / med_b, 6)
                             if med_b not in (0.0,) and np.isfinite(med_b) else np.nan),
            "auc": round(mw["auc"], 6),
            "cliffs_delta": round(delta, 6),
            "abs_cliffs_delta": round(abs(delta), 6),
            "effect_size_label": _effect_label(abs(delta)),
            "direction": ("higher in " + label_a) if direction > 0
                         else ("lower in " + label_a),
            "ks_statistic": round(ks_two_sample(a, b), 6),
            "mannwhitney_z": round(mw["z"], 4),
            "p_value": mw["p_value"],
            "threshold": best.get("threshold"),
            "threshold_rule": best.get("rule"),
            "threshold_recall": best.get("recall"),
            "threshold_precision": best.get("precision"),
            "threshold_specificity": best.get("specificity"),
            "threshold_lift": best.get("lift_over_base_rate"),
            "threshold_candidates_flagged": best.get("candidates_flagged"),
        })

    out = pd.DataFrame(rows)
    if "p_value" in out.columns:
        out["q_value_bh"] = benjamini_hochberg(out["p_value"].tolist())
        out["significant_fdr_05"] = out["q_value_bh"] < 0.05
    if "abs_cliffs_delta" in out.columns:
        out = out.sort_values(["abs_cliffs_delta", "feature"],
                              ascending=[False, True], kind="mergesort")
    return out.reset_index(drop=True)


def _effect_label(d: float) -> str:
    if not np.isfinite(d):
        return "undefined"
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def compare_boolean(features: pd.DataFrame, boolean_features: Iterable[str], *,
                    mask_a: pd.Series, mask_b: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    na, nb = int(mask_a.sum()), int(mask_b.sum())
    for name in boolean_features:
        col = features[name].fillna(False).to_numpy(dtype=bool)
        ka = int(col[mask_a.to_numpy()].sum())
        kb = int(col[mask_b.to_numpy()].sum())
        pa = ka / na if na else np.nan
        pb = kb / nb if nb else np.nan
        z = two_proportion_z(ka, na, kb, nb)
        rows.append({
            "feature": name,
            "rate_attack_containing": round(pa, 6),
            "rate_non_campaign": round(pb, 6),
            "count_attack_containing": ka, "count_non_campaign": kb,
            "rate_difference": round(pa - pb, 6),
            "abs_rate_difference": round(abs(pa - pb), 6),
            "risk_ratio": (round(pa / pb, 6) if pb and np.isfinite(pb) else np.nan),
            "z": round(z["z"], 4), "p_value": z["p_value"],
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value_bh"] = benjamini_hochberg(out["p_value"].tolist())
        out["significant_fdr_05"] = out["q_value_bh"] < 0.05
        out = out.sort_values(["abs_rate_difference", "feature"],
                              ascending=[False, True], kind="mergesort")
    return out.reset_index(drop=True)


def attack_rate_crosstab(features: pd.DataFrame, by: str) -> pd.DataFrame:
    total = len(features)
    base = features["is_attack_containing"].mean() if total else 0.0
    g = features.groupby(by, sort=True)["is_attack_containing"]
    out = pd.DataFrame({
        "candidates": g.size(),
        "attack_containing": g.sum().astype(int),
    })
    out["attack_rate"] = (out["attack_containing"] / out["candidates"]).round(6)
    out["lift_over_base_rate"] = (out["attack_rate"] / base).round(4) if base else np.nan
    out["share_of_all_attack_candidates"] = (
        out["attack_containing"] / max(int(features["is_attack_containing"].sum()), 1)
    ).round(6)
    return out.reset_index().rename(columns={by: "bucket"}).assign(dimension=by)


def redundancy_matrix(features: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Spearman correlation among the top separating features.

    Reported so that a list of ten 'strong' features is not mistaken for ten
    independent signals.
    """
    cols = [c for c in names if c in features.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return features[cols].corr(method="spearman").round(4)


def strongest_separations(numeric: pd.DataFrame, boolean: pd.DataFrame, *,
                          top_n: int = 15) -> dict[str, Any]:
    usable = numeric.loc[numeric.get("usable", True) == True]  # noqa: E712
    num_top = usable.head(top_n)
    bool_top = boolean.head(top_n) if len(boolean) else boolean
    return {
        "ranking_criterion": ("absolute Cliff's delta (rank-based, distribution-free); "
                              "BH-FDR q-values reported alongside"),
        "top_numeric": [
            {
                "feature": r["feature"],
                "direction": r["direction"],
                "median_attack_containing": r.get("median_attack_containing"),
                "median_non_campaign": r.get("median_non_campaign"),
                "auc": r["auc"],
                "cliffs_delta": r["cliffs_delta"],
                "effect_size": r["effect_size_label"],
                "ks": r["ks_statistic"],
                "q_value_bh": r.get("q_value_bh"),
                "descriptive_threshold": r.get("threshold"),
                "descriptive_threshold_rule": r.get("threshold_rule"),
                "descriptive_recall": r.get("threshold_recall"),
                "descriptive_precision": r.get("threshold_precision"),
                "descriptive_lift": r.get("threshold_lift"),
            }
            for _, r in num_top.iterrows()
        ],
        "top_boolean": [
            {
                "feature": r["feature"],
                "rate_attack_containing": r["rate_attack_containing"],
                "rate_non_campaign": r["rate_non_campaign"],
                "rate_difference": r["rate_difference"],
                "risk_ratio": r["risk_ratio"],
                "q_value_bh": r.get("q_value_bh"),
            }
            for _, r in bool_top.iterrows()
        ],
        "interpretation_policy": (
            "These are in-sample univariate separations on 81 positives. They are "
            "evidence about which structural signals exist, not a detector, not a "
            "feature set, and not a threshold. Correlated features will appear "
            "repeatedly; see the redundancy matrix before treating them as distinct."),
    }


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------
def render_numeric_table(numeric: pd.DataFrame, *, top_n: int = 20) -> str:
    usable = numeric.loc[numeric.get("usable", True) == True]  # noqa: E712
    cols = [c for c in ("feature", "median_attack_containing", "median_non_campaign",
                        "auc", "cliffs_delta", "effect_size_label", "ks_statistic",
                        "q_value_bh", "threshold_rule", "threshold_recall",
                        "threshold_precision", "threshold_lift")
            if c in usable.columns]
    view = usable[cols].head(top_n).copy()
    if view.empty:
        return "(no usable numeric features)"
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda v: f"{v:.4g}" if pd.notna(v) else "")
    return view.to_string(index=False)


def render_boolean_table(boolean: pd.DataFrame, *, top_n: int = 20) -> str:
    if boolean.empty:
        return "(no boolean features)"
    cols = ["feature", "rate_attack_containing", "rate_non_campaign",
            "rate_difference", "risk_ratio", "q_value_bh"]
    view = boolean[[c for c in cols if c in boolean.columns]].head(top_n).copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda v: f"{v:.4g}" if pd.notna(v) else "")
    return view.to_string(index=False)


def render_crosstab(table: pd.DataFrame) -> str:
    if table.empty:
        return "(empty)"
    return table.to_string(index=False)
