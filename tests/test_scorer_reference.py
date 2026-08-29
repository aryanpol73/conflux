"""Phase 5.5 tests - scorer reference persistence and validation.

Self-contained: nothing here requires the committed artifact to exist, and the
4372-row production CSV is never loaded by the fast unit tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conflux.scoring.deterministic_scorer import DeterministicScorer, ScorerReference
from conflux.scoring.scorer_reference_io import (
    load_scorer_reference,
    reference_from_dict,
    reference_to_dict,
    references_equal,
    save_scorer_reference,
    validate_scorer_reference,
)

FEATURES = (
    "burst_rate_per_minute",
    "link_density",
    "max_transactions_per_shared_card",
    "multi_entity_link_fraction",
    "distinct_merchants_per_transaction",
    "distinct_bins_per_transaction",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "src/conflux/models/artifacts/scorer_reference_v1.json"
N = 40


def synthetic_frame(n: int = N, *, constant: str | None = None) -> pd.DataFrame:
    """Deterministic, non-degenerate feature frame over the six real names."""
    rng = np.random.default_rng(20240501)
    data = {}
    for i, name in enumerate(FEATURES):
        base = np.linspace(0.1 * (i + 1), 1.0 + i, n)
        data[name] = np.round(base + rng.uniform(0.0, 0.05, n), 6)
    frame = pd.DataFrame(data)
    if constant is not None:
        frame[constant] = 1.0
    return frame


def fit_small(frame: pd.DataFrame, *, signs=None, weights=None) -> ScorerReference:
    s = signs if signs is not None else {f: 1 for f in FEATURES}
    w = weights if weights is not None else {f: 1.0 for f in FEATURES}

    return DeterministicScorer.fit(
        frame[list(FEATURES)],
        list(FEATURES),
        signs=s,
        weights=w,
        winsor=(0.01, 0.99),
        fit_scope="unit_test",
    )


def scores_of(reference: ScorerReference, frame: pd.DataFrame) -> np.ndarray:
    scores, _ = DeterministicScorer.transform(reference, frame)
    return np.asarray(scores, dtype=float)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_frame()


@pytest.fixture(scope="module")
def reference(frame) -> ScorerReference:
    return fit_small(frame)


# --- 1. json round trip ----------------------------------------------------


def test_json_round_trip_preserves_every_field(reference, tmp_path):
    path = save_scorer_reference(reference, tmp_path / "ref.json")
    loaded = load_scorer_reference(path)
    assert tuple(loaded.feature_names) == tuple(reference.feature_names)
    assert tuple(loaded.signs) == tuple(reference.signs)
    assert tuple(loaded.weights) == tuple(reference.weights)
    assert tuple(loaded.lo) == tuple(reference.lo)
    assert tuple(loaded.hi) == tuple(reference.hi)
    assert tuple(loaded.reference_values) == tuple(reference.reference_values)
    assert loaded.n_reference == reference.n_reference
    assert loaded.fit_scope == reference.fit_scope
    assert references_equal(reference, loaded)


def test_round_trip_uses_tuples_not_lists(reference, tmp_path):
    loaded = load_scorer_reference(save_scorer_reference(reference, tmp_path / "r.json"))
    assert isinstance(loaded.feature_names, tuple)
    assert isinstance(loaded.reference_values, tuple)
    assert all(isinstance(c, tuple) for c in loaded.reference_values)


def test_artifact_is_plain_json(reference, tmp_path):
    path = save_scorer_reference(reference, tmp_path / "r.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) >= {
        "feature_names", "signs", "weights", "lo", "hi",
        "reference_values", "n_reference", "fit_scope",
    }


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scorer_reference(tmp_path / "absent.json")


def test_truncated_artifact_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"feature_names": ["a"]}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_scorer_reference(bad)


# --- 2. determinism --------------------------------------------------------


def test_fit_is_deterministic(frame):
    a, b = fit_small(frame), fit_small(frame)
    assert references_equal(a, b)
    assert np.array_equal(scores_of(a, frame), scores_of(b, frame))


def test_scores_survive_round_trip(reference, frame, tmp_path):
    loaded = load_scorer_reference(save_scorer_reference(reference, tmp_path / "r.json"))
    assert np.array_equal(scores_of(reference, frame), scores_of(loaded, frame))


# --- 3. signs by name ------------------------------------------------------


def test_signs_map_by_name_not_position():
    """A sign dict keyed by name must survive a reordered feature list."""
    reordered = tuple(reversed(FEATURES))

    sign_by_name = {
        name: (1 if i % 2 == 0 else -1)
        for i, name in enumerate(FEATURES)
    }

    frame = synthetic_frame()

    expected = tuple(sign_by_name[n] for n in reordered)

    ref = DeterministicScorer.fit(
        frame[list(reordered)],
        list(reordered),
        signs=sign_by_name,
        weights={name: 1.0 for name in reordered},
        winsor=(0.01, 0.99),
        fit_scope="unit_test",
    )

    assert ref.feature_names == reordered
    assert ref.signs == expected

    for name, sign in zip(ref.feature_names, ref.signs):
        assert int(sign) == sign_by_name[name], (
            f"sign misaligned for {name}"
        )


def test_frozen_production_signs_all_positive(reference):
    assert tuple(int(s) for s in reference.signs) == (1,) * 6


# --- 4. weights ------------------------------------------------------------


def test_uniform_weights_are_stored_verbatim(reference):
    assert tuple(float(w) for w in reference.weights) == (1.0,) * 6


# --- 5. score bounds -------------------------------------------------------


def test_scores_finite_and_bounded(reference, frame):
    scores = scores_of(reference, frame)
    assert len(scores) == len(frame)
    assert np.all(np.isfinite(scores))
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


# --- 6. extra columns ------------------------------------------------------


def test_extra_columns_do_not_change_scores(reference, frame):
    padded = frame.copy()
    padded["candidate_id"] = [f"C{i:04d}" for i in range(len(padded))]
    padded["unrelated_metric"] = np.linspace(-5.0, 5.0, len(padded))
    padded["note"] = "ignored"
    assert np.array_equal(scores_of(reference, frame), scores_of(reference, padded))


def test_column_order_does_not_change_scores(reference, frame):
    shuffled = frame[list(reversed(FEATURES))].copy()
    assert np.array_equal(scores_of(reference, frame), scores_of(reference, shuffled))


# --- 7. missing feature ----------------------------------------------------


def test_missing_feature_raises(reference, frame):
    with pytest.raises((KeyError, ValueError)):
        scores_of(reference, frame.drop(columns=[FEATURES[0]]))


# --- 8. validation ---------------------------------------------------------


def test_valid_reference_passes(reference):
    assert validate_scorer_reference(reference) is reference


def test_degenerate_bounds_rejected_by_name(reference):
    lo = list(reference.lo)
    hi = list(reference.hi)
    hi[2] = lo[2]
    broken = replace(reference, lo=tuple(lo), hi=tuple(hi))
    with pytest.raises(ValueError) as exc:
        validate_scorer_reference(broken)
    assert FEATURES[2] in str(exc.value)


def test_non_finite_reference_value_rejected(reference):
    columns = [list(c) for c in reference.reference_values]
    columns[1][0] = float("nan")
    broken = replace(
        reference, reference_values=tuple(tuple(c) for c in columns)
    )
    with pytest.raises(ValueError) as exc:
        validate_scorer_reference(broken)
    assert FEATURES[1] in str(exc.value)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_non_finite_bounds_rejected(reference, bad):
    broken = replace(reference, hi=tuple([bad] + list(reference.hi)[1:]))
    with pytest.raises(ValueError):
        validate_scorer_reference(broken)


def test_wrong_feature_count_rejected(reference):
    broken = replace(
        reference,
        feature_names=tuple(reference.feature_names)[:5],
        signs=tuple(reference.signs)[:5],
        weights=tuple(reference.weights)[:5],
        lo=tuple(reference.lo)[:5],
        hi=tuple(reference.hi)[:5],
        reference_values=tuple(reference.reference_values)[:5],
    )
    with pytest.raises(ValueError):
        validate_scorer_reference(broken, n_features=6)


def test_invalid_sign_rejected(reference):
    broken = replace(reference, signs=tuple([0] + list(reference.signs)[1:]))
    with pytest.raises(ValueError):
        validate_scorer_reference(broken)


def test_zero_weights_rejected(reference):
    broken = replace(reference, weights=(0.0,) * 6)
    with pytest.raises(ValueError):
        validate_scorer_reference(broken)


def test_save_refuses_invalid_reference(reference, tmp_path):
    broken = replace(reference, hi=tuple(reference.lo))
    with pytest.raises(ValueError):
        save_scorer_reference(broken, tmp_path / "never.json")
    assert not (tmp_path / "never.json").exists()


def test_dict_helpers_round_trip(reference):
    assert references_equal(reference, reference_from_dict(reference_to_dict(reference)))


# --- production artifact (conditional) -------------------------------------


@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not built yet")
def test_production_artifact_contract():
    ref = load_scorer_reference(ARTIFACT, n_features=6)
    assert tuple(ref.feature_names) == FEATURES
    assert tuple(int(s) for s in ref.signs) == (1,) * 6
    assert tuple(float(w) for w in ref.weights) == (1.0,) * 6
    assert int(ref.n_reference) == 4372
    for name, lo, hi in zip(ref.feature_names, ref.lo, ref.hi):
        assert hi > lo, f"degenerate bounds for {name}"
    for name, column in zip(ref.feature_names, ref.reference_values):
        assert len(column) > 0, f"empty reference values for {name}"
        assert all(math.isfinite(v) for v in column)


@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not built yet")
def test_production_metadata_matches_artifact():
    meta_path = ARTIFACT.parent / "scorer_reference_v1_metadata.json"
    assert meta_path.exists(), "metadata missing beside the artifact"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ref = load_scorer_reference(ARTIFACT, n_features=6)
    assert meta["retained_features"] == list(ref.feature_names)
    assert meta["signs"] == {f: int(s) for f, s in zip(ref.feature_names, ref.signs)}
    assert meta["weights"] == {f: float(w) for f, w in zip(ref.feature_names, ref.weights)}
    assert meta["n_reference"] == int(ref.n_reference)
    assert meta["min_size"] == 2
    assert meta["phase4a_population_expected"] == 4372
    assert meta["ground_truth_used_for_reference_fit"] is False
