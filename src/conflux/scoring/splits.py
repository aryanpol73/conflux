"""CONFLUX Phase 4A -- campaign-aware and temporal validation splits.

Ground truth is read HERE only to form fold GROUPS. It never becomes a feature
and never reaches the scorer's fit path.

Protocol A -- campaign-grouped k-fold. Every fragment of one campaign lands in
the same fold, so a held-out campaign is genuinely unseen. Negatives are
interleaved by timestamp so each fold's negative distribution is
representative; Protocol A therefore tests CAMPAIGN generalisation, not
temporal generalisation.

Protocol B -- strict chronological hold-out over candidate last_ts_ns. This is
the one that tests temporal generalisation.

train_baseline.chronological_split is deliberately NOT imported: its signature
was flagged as unverified during design review and was never confirmed.
Guessing it would be worse than the ~15 lines below. If it is confirmed later,
chronological_candidate_split can delegate to it without changing callers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd

from conflux.evaluation.campaign_evaluation import CAMPAIGN_COL
from conflux.scoring.config import (
    CHRONO_TRAIN_FRAC, CHRONO_VAL_FRAC, N_FOLDS, SPLIT_SEEDS,
)

log = logging.getLogger("conflux.scoring.splits")

NEGATIVE_CLUSTER_PREFIX = "NEG::"


@dataclass(frozen=True)
class Fold:
    protocol: str
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    held_out_campaigns: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "repeat": self.repeat,
                "fold": self.fold, "n_train": int(self.train_idx.size),
                "n_test": int(self.test_idx.size),
                "n_held_out_campaigns": len(self.held_out_campaigns),
                "held_out_campaigns": list(self.held_out_campaigns)}


def cluster_ids(features: pd.DataFrame) -> np.ndarray:
    """Bootstrap / grouping cluster per candidate.

    Positives cluster by dominant campaign so that fragments of one campaign
    are one observation. Each negative is its own cluster.
    """
    camp = features["dominant_campaign_id"].fillna("").astype(str).to_numpy()
    cid = features["candidate_id"].astype(str).to_numpy()
    attack = features["is_attack_containing"].to_numpy(dtype=bool)
    return np.where(attack & (camp != ""), camp,
                    np.char.add(NEGATIVE_CLUSTER_PREFIX, cid))


def campaign_grouped_folds(features: pd.DataFrame, *, n_folds: int = N_FOLDS,
                           seeds: tuple[int, ...] = SPLIT_SEEDS) -> list[Fold]:
    """Campaigns are the grouping unit. No campaign spans train and test."""
    pos = features["is_attack_containing"].to_numpy(dtype=bool)
    camp = features["dominant_campaign_id"].fillna("").astype(str).to_numpy()
    order_key = features["first_ts_ns"].to_numpy(dtype="int64")

    # deterministic campaign ordering: earliest first transaction, then id
    campaigns = sorted(
        {c for c, p in zip(camp, pos) if p and c},
        key=lambda c: (int(order_key[(camp == c) & pos].min()), c))
    if not campaigns:
        raise ValueError("no attack-containing candidates; cannot group by campaign")

    neg_idx = np.where(~pos)[0]
    neg_sorted = neg_idx[np.argsort(features["last_ts_ns"].to_numpy()[neg_idx],
                                    kind="mergesort")]

    folds: list[Fold] = []
    for rep, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        shuffled = list(campaigns)
        rng.shuffle(shuffled)
        assign = {c: i % n_folds for i, c in enumerate(shuffled)}

        for k in range(n_folds):
            held = tuple(sorted(c for c in campaigns if assign[c] == k))
            held_set = set(held)
            test_pos = np.where(pos & np.isin(camp, list(held_set)))[0]
            test_neg = neg_sorted[k::n_folds]          # temporally interleaved
            test = np.sort(np.concatenate([test_pos, test_neg]))
            train = np.setdiff1d(np.arange(len(features)), test, assume_unique=False)

            # hard invariant: no held-out campaign may appear in train
            if np.isin(camp[train], list(held_set)).any():
                raise AssertionError(
                    f"campaign leaked into training fold (repeat={rep}, fold={k})")
            folds.append(Fold("campaign_grouped", rep, k, train, test, held))

    log.info("built %s campaign-grouped folds (%s campaigns, %s repeats)",
             len(folds), len(campaigns), len(seeds))
    return folds


def chronological_candidate_split(features: pd.DataFrame, *,
                                  train_frac: float = CHRONO_TRAIN_FRAC,
                                  val_frac: float = CHRONO_VAL_FRAC
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                             dict[str, Any]]:
    """Order candidates by last_ts_ns, cut 70/15/15. Ties broken by id."""
    order = np.lexsort((features["candidate_id"].astype(str).to_numpy(),
                        features["last_ts_ns"].to_numpy(dtype="int64")))
    n = len(order)
    n_tr = int(n * train_frac)
    n_va = int(n * (train_frac + val_frac))
    tr, va, te = order[:n_tr], order[n_tr:n_va], order[n_va:]

    ts = features["last_ts_ns"].to_numpy(dtype="int64")
    meta = {
        "method": "chronological by candidate last_ts_ns, ties by candidate_id",
        "train": int(tr.size), "validation": int(va.size), "test": int(te.size),
        "train_end_ts_ns": int(ts[tr].max()) if tr.size else None,
        "val_end_ts_ns": int(ts[va].max()) if va.size else None,
        "test_end_ts_ns": int(ts[te].max()) if te.size else None,
        "strictly_ordered": bool(
            (tr.size == 0 or va.size == 0 or ts[tr].max() <= ts[va].min())
            and (va.size == 0 or te.size == 0 or ts[va].max() <= ts[te].min())),
    }
    return tr, va, te, meta


def split_feasibility_probe(features: pd.DataFrame) -> dict[str, Any]:
    """Are campaigns temporally contiguous enough for Protocol B to be meaningful?

    If campaigns overlap heavily in time, a chronological boundary is NOT also
    a campaign boundary, and Protocol B's numbers must be read as temporal
    generalisation only -- not as unseen-campaign generalisation.
    """
    pos = features.loc[features["is_attack_containing"]]
    if pos.empty:
        return {"campaigns": 0, "note": "no attack-containing candidates"}

    ext = (pos.groupby("dominant_campaign_id")
              .agg(first=("first_ts_ns", "min"), last=("last_ts_ns", "max"),
                   candidates=("candidate_id", "size"))
              .sort_values("first"))
    spans = (ext["last"] - ext["first"]) / 1e9

    iv = ext[["first", "last"]].to_numpy()
    overlaps = sum(1 for i in range(len(iv)) for j in range(i + 1, len(iv))
                   if iv[i, 0] <= iv[j, 1] and iv[j, 0] <= iv[i, 1])
    total_pairs = len(iv) * (len(iv) - 1) // 2

    return {
        "campaigns": int(len(ext)),
        "campaign_span_seconds_median": float(np.median(spans)),
        "campaign_span_seconds_max": float(spans.max()),
        "fragmented_campaigns": int((ext["candidates"] >= 2).sum()),
        "overlapping_campaign_pairs": overlaps,
        "total_campaign_pairs": total_pairs,
        "overlap_rate": round(overlaps / total_pairs, 6) if total_pairs else 0.0,
        "chronological_boundary_is_also_a_campaign_boundary": overlaps == 0,
        "interpretation": (
            "If overlap_rate is near zero, Protocol B held-out campaigns are "
            "also unseen. If it is high, Protocol B measures temporal "
            "generalisation only and Protocol A is the campaign-generalisation "
            "evidence."),
    }


def iter_folds(folds: list[Fold]) -> Iterator[Fold]:
    yield from folds
