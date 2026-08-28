"""CONFLUX Phase 4A -- transparent deterministic scorer.

The score is an UNSUPERVISED OUTLIER SCORE. Each feature is mapped to its
percentile against a reference population, sign-corrected, and averaged. With
a 1.85% positive base rate the reference population is effectively the noise
distribution, so no label is required to calibrate it.

LEAK GEOMETRY
-------------
fit() takes an UNLABELLED frame and is the only place a population statistic
is computed. transform() is pure. Because reference statistics live in an
immutable object produced by fit(), it is structurally impossible for a
held-out row to influence its own score -- there is no code path that could
do it. fit() additionally refuses any frame carrying a ground-truth or
Phase 3C group column.

Weight tuning is the ONE labelled operation, is confined to tune_weights(),
and the runner only ever calls it with training-fold rows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from conflux.evaluation.campaign_evaluation import GROUND_TRUTH_COLUMNS
from conflux.evaluation.candidate_diagnostics import GROUP_COLUMNS
from conflux.scoring.config import (
    FEATURE_SIGNS, WEIGHT_GRID, WEIGHT_TUNER_PASSES,
)

log = logging.getLogger("conflux.scoring.deterministic_scorer")

FORBIDDEN_IN_FIT: tuple[str, ...] = tuple(
    dict.fromkeys((*GROUND_TRUTH_COLUMNS, *GROUP_COLUMNS)))


class ScorerLeakageError(ValueError):
    """A labelled or group column reached an unlabelled code path."""


@dataclass(frozen=True)
class ScorerReference:
    """Immutable fitted state. Everything transform() needs, nothing else."""
    feature_names: tuple[str, ...]
    signs: tuple[int, ...]
    weights: tuple[float, ...]
    lo: tuple[float, ...]
    hi: tuple[float, ...]
    reference_values: tuple[tuple[float, ...], ...]   # sorted, winsorized
    n_reference: int
    fit_scope: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "signs": list(self.signs),
            "weights": [round(w, 6) for w in self.weights],
            "winsor_lo": [round(v, 6) for v in self.lo],
            "winsor_hi": [round(v, 6) for v in self.hi],
            "n_reference_rows": self.n_reference,
            "fit_scope": self.fit_scope,
            "method": ("per-feature percentile rank vs the reference population, "
                       "sign-corrected, weighted mean; bounded [0,1]"),
        }


def _assert_unlabelled(frame: pd.DataFrame) -> None:
    leaked = [c for c in FORBIDDEN_IN_FIT if c in frame.columns]
    if leaked:
        raise ScorerLeakageError(
            f"fit() received label/group column(s) {leaked}; the scorer must be "
            "fitted on an unlabelled frame.")


class DeterministicScorer:
    """fit / transform. No hidden state, no global state, no randomness."""

    @staticmethod
    def fit(frame: pd.DataFrame, feature_names: Sequence[str], *,
            signs: Mapping[str, int] | None = None,
            weights: Mapping[str, float] | None = None,
            winsor: tuple[float, float] = (0.01, 0.99),
            fit_scope: str = "training_rows_only") -> ScorerReference:
        _assert_unlabelled(frame[list(feature_names)])
        names = tuple(feature_names)
        if not names:
            raise ValueError("at least one feature is required")

        sgn = signs or FEATURE_SIGNS
        wts = weights or {}
        lo_l, hi_l, refs, sg, wl = [], [], [], [], []
        for n in names:
            v = frame[n].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                raise ValueError(f"feature '{n}' has no finite reference values")
            lo = float(np.quantile(v, winsor[0]))
            hi = float(np.quantile(v, winsor[1]))
            if hi < lo:
                lo, hi = hi, lo
            clipped = np.sort(np.clip(v, lo, hi))
            lo_l.append(lo); hi_l.append(hi)
            refs.append(tuple(float(x) for x in clipped))
            sg.append(int(sgn.get(n, +1)))
            wl.append(float(wts.get(n, 1.0)))

        if sum(wl) <= 0:
            raise ValueError("weights must sum to a positive value")

        return ScorerReference(
            feature_names=names, signs=tuple(sg), weights=tuple(wl),
            lo=tuple(lo_l), hi=tuple(hi_l), reference_values=tuple(refs),
            n_reference=int(len(frame)), fit_scope=fit_scope)

    @staticmethod
    def transform(reference: ScorerReference, frame: pd.DataFrame
                  ) -> tuple[np.ndarray, pd.DataFrame]:
        """Return (scores, per-feature weighted contributions)."""
        contribs: dict[str, np.ndarray] = {}
        total = np.zeros(len(frame), dtype=float)
        wsum = float(sum(reference.weights))

        for i, name in enumerate(reference.feature_names):
            ref = np.asarray(reference.reference_values[i], dtype=float)
            x = np.clip(frame[name].to_numpy(dtype=float),
                        reference.lo[i], reference.hi[i])
            n = ref.size
            left = np.searchsorted(ref, x, side="left")
            right = np.searchsorted(ref, x, side="right")
            pct = (left + right) / (2.0 * n)           # tie-safe, in [0, 1]
            if reference.signs[i] < 0:
                pct = 1.0 - pct
            c = reference.weights[i] * pct / wsum
            contribs[name] = c
            total += c

        breakdown = pd.DataFrame(contribs, index=frame.index)
        return np.clip(total, 0.0, 1.0), breakdown

    @staticmethod
    def score_frame(reference: ScorerReference, frame: pd.DataFrame,
                    *, id_col: str = "candidate_id") -> pd.DataFrame:
        scores, breakdown = DeterministicScorer.transform(reference, frame)
        out = pd.DataFrame({id_col: frame[id_col].to_numpy(), "score": scores})
        for c in breakdown.columns:
            out[f"contrib_{c}"] = breakdown[c].to_numpy()
        return out


# ----------------------------------------------------------------------
# the one labelled operation, quarantined
# ----------------------------------------------------------------------
def tune_weights(frame: pd.DataFrame, y: np.ndarray, feature_names: Sequence[str],
                 *, objective, grid: Sequence[float] = WEIGHT_GRID,
                 passes: int = WEIGHT_TUNER_PASSES,
                 signs: Mapping[str, int] | None = None) -> dict[str, float]:
    """Deterministic coordinate ascent from uniform weights.

    MUST only ever be called with TRAINING-fold rows. There is no seed and no
    randomness: identical inputs give identical weights. Ties keep the
    incumbent weight, so the search cannot drift.
    """
    names = list(feature_names)
    weights = {n: 1.0 for n in names}
    ref = DeterministicScorer.fit(frame, names, signs=signs, weights=weights,
                                  fit_scope="weight_tuning_training_rows")
    best, _ = DeterministicScorer.transform(ref, frame)
    best_obj = float(objective(y, best))

    for _ in range(passes):
        improved = False
        for n in names:
            incumbent = weights[n]
            for w in grid:
                if w == incumbent:
                    continue
                trial = dict(weights)
                trial[n] = w
                if sum(trial.values()) <= 0:
                    continue
                r = DeterministicScorer.fit(frame, names, signs=signs,
                                            weights=trial,
                                            fit_scope="weight_tuning_training_rows")
                s, _ = DeterministicScorer.transform(r, frame)
                obj = float(objective(y, s))
                if obj > best_obj + 1e-12:
                    best_obj, weights, improved = obj, trial, True
        if not improved:
            break

    log.info("tuned weights (training folds only): %s", weights)
    return weights
