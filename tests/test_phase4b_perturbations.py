"""Phase 4B -- smoke/regression tests for the perturbation primitives.

Fixtures are built from resolve_schema() so no raw column name is hard-coded.

NOTE: SchemaView.entity holds LOGICAL entity types (card, bin, device, ip,
merchant), NOT DataFrame columns. Every call that touches entity columns
therefore passes explicit raw column names resolved from schema.structural.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conflux.robustness.perturbations import (
    NS_PER_SECOND, PerturbationError, SchemaView, attack_mask, describe_schema,
    find_entity_column, inject_benign_bursts, jitter_timestamps, make_rng,
    resample_legitimate_transactions, resolve_schema, right_censor,
    scale_group_cadence, summarize_perturbation, timestamps_as_ns,
    truncate_group_tails, weaken_entity_reuse, with_timestamps_ns,
)

BASE_NS = int(pd.Timestamp("2024-01-01 00:00:00.000000").value)
N_BENIGN, N_ATTACK = 8, 4


@pytest.fixture(scope="module")
def schema() -> SchemaView:
    return resolve_schema()


@pytest.fixture(scope="module")
def entity_cols(schema: SchemaView) -> list[str]:
    """Raw entity-bearing columns: structural minus id and timestamp."""
    return [c for c in schema.structural if c not in (schema.id_col, schema.ts_col)]


@pytest.fixture()
def frame(schema: SchemaView) -> pd.DataFrame:
    n = N_BENIGN + N_ATTACK
    ts = [BASE_NS + i * 10 * NS_PER_SECOND for i in range(n)]
    rows: dict[str, list] = {}
    for col in (*schema.structural, *schema.attribute):
        if col == schema.id_col:
            rows[col] = [f"TX-{i:04d}" for i in range(n)]
        elif col == schema.ts_col:
            rows[col] = list(pd.to_datetime(pd.Series(ts), unit="ns")
                             .dt.strftime("%Y-%m-%d %H:%M:%S.%f"))
        elif col == schema.amount_col:
            rows[col] = [round(1.0 + i, 2) for i in range(n)]
        elif col == schema.auth_col:
            rows[col] = ["approved" if i % 3 else "declined" for i in range(n)]
        else:
            rows[col] = ([f"{col}-b{i:03d}" for i in range(N_BENIGN)]
                         + [f"{col}-shared"] * N_ATTACK)
    rows[schema.label_col] = [False] * N_BENIGN + [True] * N_ATTACK
    rows[schema.campaign_col] = [""] * N_BENIGN + [
        f"CMP-{i % 2:02d}" for i in range(N_ATTACK)]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def test_schema_resolves_cleanly(schema: SchemaView, frame: pd.DataFrame):
    assert schema.unresolved == ()
    assert schema.id_col in frame.columns
    assert schema.ts_col in frame.columns
    d = describe_schema(frame)
    assert isinstance(d, dict)
    assert d["missing_from_frame"] == []


def test_find_entity_column_needs_disambiguation(schema: SchemaView,
                                                 entity_cols: list[str]):
    # "card" is ambiguous across logical + raw names; restrict the pool.
    resolved = find_entity_column("card", schema=schema, candidates=entity_cols)
    assert resolved in entity_cols
    with pytest.raises(PerturbationError):
        find_entity_column("not_a_column", schema=schema)


def test_make_rng_is_deterministic():
    assert np.array_equal(make_rng(7).integers(0, 999, 20),
                          make_rng(7).integers(0, 999, 20))
    assert not np.array_equal(make_rng(7).integers(0, 999, 20),
                              make_rng(8).integers(0, 999, 20))


def test_attack_mask_separates_rows(schema: SchemaView, frame: pd.DataFrame):
    mask = attack_mask(frame, schema=schema)
    assert int(mask.sum()) == N_ATTACK
    assert int((~mask).sum()) == N_BENIGN


def test_attack_mask_survives_string_labels(schema: SchemaView,
                                            frame: pd.DataFrame):
    text = frame.copy()
    text[schema.label_col] = text[schema.label_col].map({True: "True",
                                                         False: "False"})
    assert int(attack_mask(text, schema=schema).sum()) == N_ATTACK


def test_timestamp_round_trip(schema: SchemaView, frame: pd.DataFrame):
    ns = timestamps_as_ns(frame, schema=schema)
    assert len(ns) == len(frame)
    out = with_timestamps_ns(frame, ns, schema=schema)
    assert np.array_equal(timestamps_as_ns(out, schema=schema), ns)
    with pytest.raises(PerturbationError):
        with_timestamps_ns(frame, ns[:2], schema=schema)


def test_jitter_is_deterministic_and_pure(schema: SchemaView,
                                          frame: pd.DataFrame):
    snap = frame.copy()
    a = jitter_timestamps(frame, sigma_seconds=1.0, seed=42, schema=schema)
    b = jitter_timestamps(frame, sigma_seconds=1.0, seed=42, schema=schema)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == len(frame)
    assert list(a[schema.id_col]) == list(frame[schema.id_col])
    pd.testing.assert_frame_equal(frame, snap)
    with pytest.raises(PerturbationError):
        jitter_timestamps(frame, sigma_seconds=-1.0, seed=1, schema=schema)


def test_cadence_scaling_stretches_one_group(schema: SchemaView,
                                             frame: pd.DataFrame):
    snap = frame.copy()
    idx = np.flatnonzero((frame[schema.campaign_col] == "CMP-00").to_numpy())
    before = timestamps_as_ns(frame, schema=schema)
    span0 = int(before[idx].max() - before[idx].min())

    out = scale_group_cadence(frame, factor=2.0, group_values=["CMP-00"],
                              anchor="start", schema=schema)
    after = timestamps_as_ns(out, schema=schema)
    assert int(after[idx].max() - after[idx].min()) > span0
    assert int(after[idx].min()) == int(before[idx].min())
    assert len(out) == len(frame)
    pd.testing.assert_frame_equal(frame, snap)


def test_resample_adds_unique_benign_rows(schema: SchemaView,
                                          frame: pd.DataFrame,
                                          entity_cols: list[str]):
    snap = frame.copy()
    out = resample_legitimate_transactions(
        frame, n_new=6, seed=3, entity_columns=entity_cols, schema=schema)
    assert len(out) == len(frame) + 6
    assert out[schema.id_col].is_unique
    added = out[~out[schema.id_col].isin(frame[schema.id_col])]
    assert len(added) == 6
    assert not attack_mask(added, schema=schema).any()
    assert int(attack_mask(out, schema=schema).sum()) == N_ATTACK
    pd.testing.assert_frame_equal(frame, snap)


def test_resample_is_deterministic(schema: SchemaView, frame: pd.DataFrame,
                                   entity_cols: list[str]):
    kw = dict(n_new=4, entity_columns=entity_cols, schema=schema)
    pd.testing.assert_frame_equal(
        resample_legitimate_transactions(frame, seed=9, **kw),
        resample_legitimate_transactions(frame, seed=9, **kw))
    with pytest.raises(PerturbationError):
        resample_legitimate_transactions(frame, seed=1, **kw,
                                         multiplier=0.5)  # both size args


def test_weaken_entity_reuse_breaks_shared_values(schema: SchemaView,
                                                  frame: pd.DataFrame,
                                                  entity_cols: list[str]):
    col = find_entity_column("card", schema=schema, candidates=entity_cols)
    snap = frame.copy()
    mask = attack_mask(frame, schema=schema)
    out = weaken_entity_reuse(frame, entity_column=col, fraction=1.0, seed=2,
                              schema=schema)
    assert len(out) == len(frame)
    assert out.loc[mask, col].nunique() > frame.loc[mask, col].nunique()
    pd.testing.assert_series_equal(out.loc[~mask, col], frame.loc[~mask, col])
    pd.testing.assert_frame_equal(frame, snap)


def test_censoring_removes_rows_without_mutating(schema: SchemaView,
                                                 frame: pd.DataFrame):
    snap = frame.copy()
    ns = timestamps_as_ns(frame, schema=schema)
    cutoff = int(np.quantile(ns, 0.5))
    censored = right_censor(frame, cutoff_ts_ns=cutoff, schema=schema)
    assert 0 < len(censored) < len(frame)
    assert (timestamps_as_ns(censored, schema=schema) <= cutoff).all()

    truncated = truncate_group_tails(frame, keep_fraction=0.5, schema=schema)
    assert len(truncated) < len(frame)
    assert int((~attack_mask(truncated, schema=schema)).sum()) == N_BENIGN
    pd.testing.assert_frame_equal(frame, snap)


def test_inject_benign_bursts(schema: SchemaView, frame: pd.DataFrame,
                              entity_cols: list[str]):
    shared = [find_entity_column("card", schema=schema, candidates=entity_cols)]
    snap = frame.copy()
    kw = dict(n_bursts=2, burst_size=3, span_seconds=5.0,
              shared_entity_columns=shared, schema=schema)
    out = inject_benign_bursts(frame, seed=11, **kw)
    assert len(out) == len(frame) + 6
    assert out[schema.id_col].is_unique
    added = out[~out[schema.id_col].isin(frame[schema.id_col])]
    assert not attack_mask(added, schema=schema).any()
    assert added[shared[0]].nunique() == 2
    pd.testing.assert_frame_equal(out, inject_benign_bursts(frame, seed=11, **kw))
    pd.testing.assert_frame_equal(frame, snap)


def test_summarize_perturbation_keys(schema: SchemaView, frame: pd.DataFrame,
                                     entity_cols: list[str]):
    after = resample_legitimate_transactions(
        frame, n_new=5, seed=6, entity_columns=entity_cols, schema=schema)
    rep = summarize_perturbation(frame, after, name="volume", schema=schema)
    assert rep["rows_before"] == len(frame)
    assert rep["rows_after"] == len(frame) + 5
    assert rep["ids_added"] == 5 and rep["ids_removed"] == 0
    assert rep["attack_rows_before"] == rep["attack_rows_after"]
    assert rep["schema_preserved"] is True
