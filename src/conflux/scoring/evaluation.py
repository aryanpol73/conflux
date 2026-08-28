"""CONFLUX Phase 4A -- metrics, nested thresholds, bootstrap, comparators.

Reuses candidate_diagnostics.mann_whitney for ROC-AUC so tie handling matches
Phase 3C exactly. Average precision is implemented here because tie-grouped AP
is not available elsewhere in the project.

No metric in this module may be reported at a threshold selected on the same
rows. select_threshold() is called on training rows; apply_threshold() is
called on held-out rows. The runner never crosses them.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from conflux.evaluation.candidate_diagnostics import mann_whitney
from conflux.scoring.config import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, BUDGET_FRACTIONS,
    PREREGISTERED_CI_LEVEL, PREREGISTERED_MIN_PRAUC_GAIN, PREREGISTERED_RULE,
    THRESHOLD_SWEEP_QUANTILES,
)

log = logging.getLogger("conflux.scoring.evaluation")

# Phase 3C emits "campaign_with_normal"; campaign_evaluation uses
# "pure_campaign_with_normal". Normalised on read rather than editing a frozen
# Phase 3C module.
PURITY_ALIASES = {"campaign_with_normal": "pure_campaign_with_normal"}


def normalize_purity(series: pd.Series) -> pd.Series:
    return series.astype(str).replace(PURITY_ALIASES)


# ----------------------------------------------------------------------
# ranking metrics
# ----------------------------------------------------------------------
def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Tie-grouped AP. Equal scores share one operating point."""
    y = np.asarray(y, dtype=float)
    n_pos = float(y.sum())
    if n_pos == 0 or y.size == 0:
        return float("nan")
    order = np.lexsort((np.arange(y.size), -s))
    ys, ss = y[order], s[order]
    edges = np.r_[np.where(np.diff(ss) != 0)[0], ys.size - 1]
    tp = np.cumsum(ys)[edges]
    k = edges + 1
    precision = tp / k
    recall = tp / n_pos
    dr = np.diff(np.r_[0.0, recall])
    return float(np.sum(dr * precision))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(mann_whitney(pos, neg)["auc"])


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> dict[str, Any]:
    k = int(min(max(k, 0), y.size))
    if k == 0:
        return {"k": 0, "precision": float("nan"), "positives_found": 0}
    order = np.lexsort((np.arange(y.size), -s))[:k]
    hit = int(y[order].sum())
    return {"k": k, "precision": round(hit / k, 6), "positives_found": hit,
            "recall": round(hit / max(float(y.sum()), 1.0), 6)}


def confusion(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = s >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    base = float(y.mean()) if y.size else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6),
        "false_positive_candidate_rate": round(fp / (fp + tn), 6) if (fp + tn) else 0.0,
        "candidates_flagged": tp + fp,
        "base_rate": round(base, 6),
        "lift_over_base_rate": round(prec / base, 4) if base else float("nan"),
    }


# ----------------------------------------------------------------------
# campaign-level metrics
# ----------------------------------------------------------------------
def campaign_metrics(features: pd.DataFrame, s: np.ndarray, threshold: float,
                     *, all_campaigns: int | None = None) -> dict[str, Any]:
    kept = s >= threshold
    pos = features["is_attack_containing"].to_numpy(dtype=bool)
    camp = features["dominant_campaign_id"].fillna("").astype(str).to_numpy()
    ctx = features["campaign_transactions"].to_numpy(dtype=float)

    present = {c for c, p in zip(camp, pos) if p and c}
    retained = {c for c, p, k in zip(camp, pos, kept) if p and k and c}
    total_ct = float(ctx[pos].sum())
    kept_ct = float(ctx[pos & kept].sum())

    frags = pd.Series([c for c, p, k in zip(camp, pos, kept) if p and k and c])
    denom = all_campaigns if all_campaigns is not None else len(present)

    return {
        "campaigns_present_in_evaluation_set": len(present),
        "campaigns_retained_at_threshold": len(retained),
        "campaign_recall": round(len(retained) / denom, 6) if denom else 0.0,
        "campaigns_lost": sorted(present - retained),
        "campaign_transactions_in_set": int(total_ct),
        "campaign_transactions_retained": int(kept_ct),
        "campaign_transaction_recall_after_filtering":
            round(kept_ct / total_ct, 6) if total_ct else 0.0,
        "surviving_fragments_per_campaign_median":
            float(frags.value_counts().median()) if len(frags) else 0.0,
        "note": ("campaign_transaction_recall_after_filtering is the share of "
                 "the campaign transactions that Phase 3B already captured; "
                 "multiply by the Phase 3B input recall (0.9927) for the "
                 "end-to-end figure."),
    }


# ----------------------------------------------------------------------
# nested threshold selection -- TRAINING ROWS ONLY
# ----------------------------------------------------------------------
def select_threshold(y_train: np.ndarray, s_train: np.ndarray, *,
                     rule: str = "max_f1",
                     budget_fraction: float | None = None) -> dict[str, Any]:
    if rule == "budget":
        if budget_fraction is None:
            raise ValueError("budget rule needs budget_fraction")
        thr = float(np.quantile(s_train, 1.0 - budget_fraction))
        return {"rule": f"budget_top_{budget_fraction:.0%}", "threshold": thr,
                "selected_on": "training rows only"}

    cands = np.unique(s_train)
    best_thr, best_f1 = float(cands.max()), -1.0
    for t in cands:
        c = confusion(y_train, s_train, t)
        if c["f1"] > best_f1:
            best_f1, best_thr = c["f1"], float(t)
    return {"rule": "max_f1", "threshold": best_thr,
            "train_f1_at_threshold": round(best_f1, 6),
            "selected_on": "training rows only"}


# ----------------------------------------------------------------------
# campaign-cluster bootstrap
# ----------------------------------------------------------------------
def bootstrap_ci(y: np.ndarray, s: np.ndarray, clusters: np.ndarray,
                 stat: Callable[[np.ndarray, np.ndarray], float], *,
                 n_resamples: int = BOOTSTRAP_RESAMPLES,
                 level: float = PREREGISTERED_CI_LEVEL,
                 seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Resample CLUSTERS, not rows.

    Resampling rows would treat four fragments of one campaign as four
    independent observations and produce dishonestly tight intervals.
    """
    uniq = np.unique(clusters)
    index = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_resamples):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([index[c] for c in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        v = stat(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return {"point": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_effective_resamples": 0}
    a = (1.0 - level) / 2.0
    return {
        "point": round(float(stat(y, s)), 6),
        "ci_low": round(float(np.quantile(vals, a)), 6),
        "ci_high": round(float(np.quantile(vals, 1.0 - a)), 6),
        "level": level, "n_clusters": int(uniq.size),
        "n_effective_resamples": len(vals),
        "resampling_unit": "campaign (positives) / candidate (negatives)",
    }


def paired_difference_ci(diffs: Sequence[float], *,
                         level: float = PREREGISTERED_CI_LEVEL,
                         n_resamples: int = BOOTSTRAP_RESAMPLES,
                         seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    if d.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(d, size=d.size, replace=True).mean())
             for _ in range(n_resamples)]
    a = (1.0 - level) / 2.0
    return {"mean": round(float(d.mean()), 6),
            "ci_low": round(float(np.quantile(means, a)), 6),
            "ci_high": round(float(np.quantile(means, 1.0 - a)), 6),
            "n_folds": int(d.size), "level": level}


def apply_preregistered_rule(uniform_prauc: Sequence[float],
                             tuned_prauc: Sequence[float]) -> dict[str, Any]:
    diffs = [t - u for t, u in zip(tuned_prauc, uniform_prauc)]
    ci = paired_difference_ci(diffs)
    gain_ok = np.isfinite(ci["mean"]) and ci["mean"] >= PREREGISTERED_MIN_PRAUC_GAIN
    ci_ok = np.isfinite(ci["ci_low"]) and ci["ci_low"] > 0.0
    adopt = bool(gain_ok and ci_ok)
    return {
        "rule": PREREGISTERED_RULE,
        "required_min_gain": PREREGISTERED_MIN_PRAUC_GAIN,
        "paired_difference": ci,
        "gain_condition_met": bool(gain_ok),
        "ci_excludes_zero": bool(ci_ok),
        "decision": "adopt_tuned_weights" if adopt else "keep_uniform_weights",
        "note": ("Registered in scoring/config.py before any Phase 4 result "
                 "existed. Not revised after seeing results."),
    }


# ----------------------------------------------------------------------
# comparator rankers -- the scorer is worthless unless it beats these
# ----------------------------------------------------------------------
def comparator_scores(features: pd.DataFrame, *, seed: int = BOOTSTRAP_SEED
                      ) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(features)
    def pct(col: str) -> np.ndarray:
        return features[col].rank(method="average").to_numpy(dtype=float) / n
    return {
        "random": rng.random(n),
        "rank_by_size": pct("size"),
        "rank_by_link_edge_count": pct("link_edge_count"),
    }


# ----------------------------------------------------------------------
# threshold sensitivity / recall-exchange table
# ----------------------------------------------------------------------
def recall_exchange_table(features: pd.DataFrame, s: np.ndarray, *,
                          phase3b_input_recall: float = 0.9927) -> pd.DataFrame:
    y = features["is_attack_containing"].to_numpy(dtype=int)
    rows = []
    for q in THRESHOLD_SWEEP_QUANTILES:
        t = float(np.quantile(s, q))
        c = confusion(y, s, t)
        cm = campaign_metrics(features, s, t)
        rows.append({
            "score_quantile": q, "threshold": c["threshold"],
            "candidates_kept": c["candidates_flagged"],
            "attack_candidates_kept": c["tp"],
            "candidate_precision": c["precision"],
            "candidate_recall": c["recall"],
            "f1": c["f1"],
            "false_positive_candidate_rate": c["false_positive_candidate_rate"],
            "lift": c["lift_over_base_rate"],
            "campaigns_retained": cm["campaigns_retained_at_threshold"],
            "campaign_recall": cm["campaign_recall"],
            "campaign_txn_recall_within_phase3b":
                cm["campaign_transaction_recall_after_filtering"],
            "end_to_end_campaign_txn_recall": round(
                cm["campaign_transaction_recall_after_filtering"]
                * phase3b_input_recall, 6),
        })
    for frac in BUDGET_FRACTIONS:
        t = float(np.quantile(s, 1.0 - frac))
        c = confusion(y, s, t)
        cm = campaign_metrics(features, s, t)
        rows.append({
            "score_quantile": round(1.0 - frac, 4), "threshold": c["threshold"],
            "candidates_kept": c["candidates_flagged"],
            "attack_candidates_kept": c["tp"],
            "candidate_precision": c["precision"],
            "candidate_recall": c["recall"], "f1": c["f1"],
            "false_positive_candidate_rate": c["false_positive_candidate_rate"],
            "lift": c["lift_over_base_rate"],
            "campaigns_retained": cm["campaigns_retained_at_threshold"],
            "campaign_recall": cm["campaign_recall"],
            "campaign_txn_recall_within_phase3b":
                cm["campaign_transaction_recall_after_filtering"],
            "end_to_end_campaign_txn_recall": round(
                cm["campaign_transaction_recall_after_filtering"]
                * phase3b_input_recall, 6),
        })
    return (pd.DataFrame(rows).drop_duplicates("threshold")
              .sort_values("threshold", kind="mergesort").reset_index(drop=True))


def render_exchange_table(t: pd.DataFrame) -> str:
    cols = ["threshold", "candidates_kept", "attack_candidates_kept",
            "candidate_precision", "candidate_recall", "lift",
            "campaigns_retained", "campaign_recall",
            "end_to_end_campaign_txn_recall"]
    v = t[[c for c in cols if c in t.columns]].copy()
    for c in v.columns:
        if pd.api.types.is_float_dtype(v[c]):
            v[c] = v[c].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
    return v.to_string(index=False)
