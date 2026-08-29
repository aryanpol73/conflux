"""Phase 4B S5 tests - right censoring robustness."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import conflux.robustness.perturbations as p
import conflux.robustness.scenario_right_censoring as s5
from conflux.robustness.perturbations import PerturbationError
from conflux.robustness.scenario_right_censoring import (
    CONTROL_ARM_ID,
    RightCensoringConfig,
    RightCensoringError,
)

ROWS = 48
BASE_NS = int(pd.Timestamp("2024-03-01T00:00:00Z").value)
STEP_NS = 300 * p.NS_PER_SECOND
ENTITY_HINTS = ("card_fingerprint", "bin", "device_fingerprint", "ip_signature")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _timestamp_strings(ts_ns: np.ndarray) -> list[str]:
    try:
        return list(p.ns_to_timestamp_strings(ts_ns))
    except TypeError:  # pragma: no cover
        idx = pd.to_datetime(pd.Series(ts_ns), unit="ns", utc=True)
        return [t.strftime(p.TS_FORMAT) for t in idx]


@pytest.fixture(scope="module")
def schema():
    return p.resolve_schema()


@pytest.fixture
def frame(schema) -> pd.DataFrame:
    """Synthetic baseline, deliberately NOT pre-sorted by transaction id."""
    n = ROWS
    ts_ns = BASE_NS + np.arange(n, dtype=np.int64) * STEP_NS
    ids = [f"TX{n - i:05d}" for i in range(n)]  # descending -> unsorted

    id_col = s5.resolve_id_column(schema=schema)
    ts_col = s5.resolve_timestamp_column(schema=schema) or "timestamp"
    label_col = s5.resolve_label_column(schema=schema) or "label"
    campaign_col = s5.resolve_campaign_column(schema=schema) or "campaign_id"

    labels = [1 if (i % 4 == 0) else 0 for i in range(n)]
    campaigns = [f"CMP{i % 5:03d}" if labels[i] else "" for i in range(n)]

    data = {
        id_col: ids,
        ts_col: _timestamp_strings(ts_ns),
        label_col: labels,
        campaign_col: campaigns,
        "amount": np.round(np.linspace(5.0, 900.0, n), 2),
    }
    for k, hint in enumerate(ENTITY_HINTS):
        data[hint] = [f"{hint[:3].upper()}-{(i + k) % 7:03d}" for i in range(n)]

    df = pd.DataFrame(data)

    for column in s5.schema_entity_columns(schema=schema):
        if column not in df.columns:
            df[column] = [f"ENT-{i % 6:03d}" for i in range(n)]
    return df


@pytest.fixture
def config() -> RightCensoringConfig:
    return RightCensoringConfig()


# ---------------------------------------------------------------------------
# fixture sanity
# ---------------------------------------------------------------------------


def test_fixture_timestamps_parse_with_repository_helper(frame, schema):
    ts = np.asarray(p.timestamps_as_ns(frame, schema=schema))
    assert len(ts) == len(frame)
    assert np.all(np.diff(np.sort(ts)) > 0)


def test_fixture_is_not_sorted_by_id(frame, schema):
    id_col = s5.resolve_id_column(frame, schema=schema)
    assert list(frame[id_col]) != sorted(frame[id_col])


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_default_config_is_valid(config):
    assert config.validate() is config
    assert config.severity_mode == "keep_fraction"
    assert len(config.severity_levels()) >= 2
    assert config.include_control is True


def test_config_is_json_serializable(config):
    json.dumps(config.as_dict())


def test_config_rejects_both_severity_modes():
    cfg = RightCensoringConfig(keep_fractions=(0.5,), cutoff_ts_ns=(BASE_NS,))
    with pytest.raises(RightCensoringError):
        cfg.validate()


def test_config_rejects_neither_severity_mode():
    cfg = RightCensoringConfig(keep_fractions=(), cutoff_ts_ns=())
    with pytest.raises(RightCensoringError):
        cfg.validate()


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, 2.0])
def test_config_rejects_out_of_range_keep_fractions(bad):
    with pytest.raises(RightCensoringError):
        RightCensoringConfig(keep_fractions=(bad,)).validate()


def test_config_rejects_duplicate_levels():
    with pytest.raises(RightCensoringError):
        RightCensoringConfig(keep_fractions=(0.5, 0.5))


@pytest.mark.parametrize("bad", ["0.5", None, {"a": 1}])
def test_config_rejects_non_sequence_fractions(bad):
    with pytest.raises(RightCensoringError):
        RightCensoringConfig(keep_fractions=bad)


def test_config_rejects_non_numeric_fraction_entries():
    with pytest.raises(RightCensoringError):
        RightCensoringConfig(keep_fractions=("half",))


def test_config_rejects_non_integer_cutoffs():
    with pytest.raises(RightCensoringError):
        RightCensoringConfig(keep_fractions=(), cutoff_ts_ns=(1.5,))


def test_cutoff_mode_config_validates():
    cfg = RightCensoringConfig(keep_fractions=(), cutoff_ts_ns=(BASE_NS,)).validate()
    assert cfg.severity_mode == "cutoff_ts_ns"
    assert cfg.arm_id(BASE_NS) == f"cutoff_ts_ns={BASE_NS}"


# ---------------------------------------------------------------------------
# primitive adapter
# ---------------------------------------------------------------------------


def test_primitive_signature_is_live():
    params = s5.primitive_signature().parameters
    assert "keep_fraction" in params
    assert "cutoff_ts_ns" in params
    assert "seed" not in params  # deterministic primitive


def test_call_kwargs_requires_exactly_one_mode():
    with pytest.raises(RightCensoringError):
        s5.censor_call_kwargs()
    with pytest.raises(RightCensoringError):
        s5.censor_call_kwargs(keep_fraction=0.5, cutoff_ts_ns=BASE_NS)


def test_call_kwargs_builds_exact_arguments(schema):
    kwargs = s5.censor_call_kwargs(keep_fraction=0.5, schema=schema)
    assert kwargs["keep_fraction"] == 0.5
    assert "cutoff_ts_ns" not in kwargs
    s5.primitive_signature().bind(pd.DataFrame(), **kwargs)


def test_call_kwargs_rejects_unknown_extra():
    with pytest.raises(RightCensoringError):
        s5.censor_call_kwargs(keep_fraction=0.5, extra={"nope": 1})


def test_primitive_rejects_invalid_keep_fraction_itself(frame, schema):
    with pytest.raises(PerturbationError):
        p.right_censor(frame, keep_fraction=1.5, schema=schema)


def test_primitive_rejects_both_or_neither(frame, schema):
    with pytest.raises(PerturbationError):
        p.right_censor(frame, schema=schema)
    with pytest.raises(PerturbationError):
        p.right_censor(frame, keep_fraction=0.5, cutoff_ts_ns=BASE_NS, schema=schema)


# ---------------------------------------------------------------------------
# core censoring behaviour
# ---------------------------------------------------------------------------


def test_censoring_reduces_rows_monotonically(frame, schema):
    counts = []
    for kf in (1.0, 0.75, 0.5, 0.25):
        out, _ = s5.apply_right_censor(frame, keep_fraction=kf, schema=schema)
        counts.append(len(out))
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == len(frame)
    assert counts[-1] < len(frame)


def test_surviving_rows_are_unchanged(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.6, schema=schema)
    report = s5.assert_surviving_rows_unchanged(frame, out, schema=schema)
    assert report["surviving_rows_identical"] is True
    assert report["removed_rows"] > 0


def test_no_new_transaction_ids_appear(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.5, schema=schema)
    assert s5.id_set(out, schema=schema) <= s5.id_set(frame, schema=schema)


def test_surviving_labels_and_campaigns_unchanged(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.7, schema=schema)
    id_col = s5.resolve_id_column(frame, schema=schema)
    for resolver in (s5.resolve_label_column, s5.resolve_campaign_column):
        column = resolver(frame, schema=schema)
        if column is None:
            continue
        expected = frame.set_index(id_col)[column].loc[out[id_col].tolist()]
        assert list(expected) == list(out[column])


def test_surviving_entity_columns_unchanged(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.7, schema=schema)
    id_col = s5.resolve_id_column(frame, schema=schema)
    columns = [c for c in ENTITY_HINTS if c in frame.columns]
    columns += [c for c in s5.schema_entity_columns(frame, schema=schema) if c not in columns]
    assert columns, "fixture should expose at least one entity column"
    for column in columns:
        expected = frame.set_index(id_col)[column].loc[out[id_col].tolist()]
        assert list(expected) == list(out[column])


def test_column_set_and_dtypes_preserved(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.5, schema=schema)
    assert list(out.columns) == list(frame.columns)
    assert out.dtypes.astype(str).to_dict() == frame.dtypes.astype(str).to_dict()


def test_censoring_is_right_sided(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.5, schema=schema)
    report = s5.assert_cutoff_consistency(frame, out, schema=schema)
    assert report["right_sided"] is True
    assert report["max_kept_ts_ns"] < report["min_removed_ts_ns"]


def test_observed_cutoff_brackets_predicted_cutoff(frame, schema):
    kf = 0.5
    out, _ = s5.apply_right_censor(frame, keep_fraction=kf, schema=schema)
    predicted = s5.predicted_cutoff_ns(frame, kf, schema=schema)
    report = s5.assert_cutoff_consistency(frame, out, schema=schema)
    assert report["max_kept_ts_ns"] <= predicted < report["min_removed_ts_ns"]


def test_keep_fraction_one_keeps_everything(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=1.0, schema=schema)
    assert len(out) == len(frame)
    assert s5.frame_fingerprint(out, schema=schema) == s5.frame_fingerprint(
        frame, schema=schema
    )


def test_cutoff_removing_everything_raises(frame, schema):
    with pytest.raises(PerturbationError):
        s5.apply_right_censor(frame, cutoff_ts_ns=BASE_NS - STEP_NS, schema=schema)


def test_explicit_cutoff_mode_works(frame, schema):
    ts = np.asarray(p.timestamps_as_ns(frame, schema=schema))
    cutoff = int(np.median(ts))
    out, used = s5.apply_right_censor(frame, cutoff_ts_ns=cutoff, schema=schema)
    assert used["cutoff_ts_ns"] == cutoff
    assert 0 < len(out) < len(frame)
    kept_ts = np.asarray(p.timestamps_as_ns(out, schema=schema))
    assert kept_ts.max() <= cutoff


def test_output_is_sorted_by_id(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.5, schema=schema)
    id_col = s5.resolve_id_column(out, schema=schema)
    assert list(out[id_col]) == sorted(out[id_col])
    assert list(out.index) == list(range(len(out)))


# ---------------------------------------------------------------------------
# immutability & determinism
# ---------------------------------------------------------------------------


def test_input_frame_is_not_mutated(frame, schema):
    before = s5.frame_fingerprint(frame, schema=schema)
    snapshot = frame.copy(deep=True)
    s5.apply_right_censor(frame, keep_fraction=0.4, schema=schema)
    assert s5.frame_fingerprint(frame, schema=schema) == before
    pd.testing.assert_frame_equal(frame, snapshot)


def test_censoring_is_deterministic(frame, schema):
    a, _ = s5.apply_right_censor(frame, keep_fraction=0.55, schema=schema)
    b, _ = s5.apply_right_censor(frame, keep_fraction=0.55, schema=schema)
    assert s5.frame_fingerprint(a, schema=schema) == s5.frame_fingerprint(b, schema=schema)
    pd.testing.assert_frame_equal(a, b)


def test_scenario_is_deterministic(frame, config, schema):
    first = s5.run_right_censoring_scenario(frame, config, schema=schema)
    second = s5.run_right_censoring_scenario(frame, config, schema=schema)
    assert [a["fingerprint"] for a in first["arms"]] == [
        a["fingerprint"] for a in second["arms"]
    ]


def test_fingerprint_is_order_insensitive(frame, schema):
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert s5.frame_fingerprint(shuffled, schema=schema) == s5.frame_fingerprint(
        frame, schema=schema
    )


def test_fingerprint_detects_value_change(frame, schema):
    tampered = frame.copy()
    tampered.loc[0, "amount"] = float(tampered.loc[0, "amount"]) + 1.0
    assert s5.frame_fingerprint(tampered, schema=schema) != s5.frame_fingerprint(
        frame, schema=schema
    )


# ---------------------------------------------------------------------------
# invariant guard tampering detection
# ---------------------------------------------------------------------------


def test_guard_detects_modified_surviving_row(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.6, schema=schema)
    tampered = out.copy()
    tampered.loc[0, "amount"] = float(tampered.loc[0, "amount"]) + 42.0
    with pytest.raises(RightCensoringError):
        s5.assert_surviving_rows_unchanged(frame, tampered, schema=schema)


def test_guard_detects_new_transaction_id(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.6, schema=schema)
    tampered = out.copy()
    id_col = s5.resolve_id_column(out, schema=schema)
    tampered.loc[0, id_col] = "TX-SMUGGLED"
    with pytest.raises(RightCensoringError):
        s5.assert_surviving_rows_unchanged(frame, tampered, schema=schema)


def test_guard_detects_duplicated_rows(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.6, schema=schema)
    tampered = pd.concat([out, out.iloc[[0]]], ignore_index=True)
    with pytest.raises(RightCensoringError):
        s5.assert_surviving_rows_unchanged(frame, tampered, schema=schema)


def test_guard_detects_dropped_column(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.6, schema=schema)
    with pytest.raises(RightCensoringError):
        s5.assert_surviving_rows_unchanged(
            frame, out.drop(columns=["amount"]), schema=schema
        )


def test_guard_detects_label_flip(frame, schema):
    label_col = s5.resolve_label_column(frame, schema=schema)
    if label_col is None:
        pytest.fail("fixture must expose a schema label column")
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.6, schema=schema)
    tampered = out.copy()
    tampered[label_col] = tampered[label_col].map(lambda v: 1 - int(v))
    with pytest.raises(RightCensoringError):
        s5.assert_surviving_rows_unchanged(frame, tampered, schema=schema)


def test_guard_tolerates_pure_dtype_drift(frame, schema):
    out, _ = s5.apply_right_censor(frame, keep_fraction=0.6, schema=schema)
    drifted = out.copy()
    drifted["amount"] = drifted["amount"].astype(object)
    report = s5.assert_surviving_rows_unchanged(frame, drifted, schema=schema)
    assert report["surviving_rows_identical"] is True


def test_guard_detects_non_right_sided_removal(frame, schema):
    id_col = s5.resolve_id_column(frame, schema=schema)
    ts = np.asarray(p.timestamps_as_ns(frame, schema=schema))
    earliest_id = frame[id_col].iloc[int(np.argmin(ts))]
    bad = frame[frame[id_col] != earliest_id].copy()
    with pytest.raises(RightCensoringError):
        s5.assert_cutoff_consistency(frame, bad, schema=schema)


# ---------------------------------------------------------------------------
# nesting
# ---------------------------------------------------------------------------


def test_retained_ids_are_nested_across_levels(frame, schema):
    sets = []
    for kf in (0.25, 0.5, 0.75, 1.0):
        out, _ = s5.apply_right_censor(frame, keep_fraction=kf, schema=schema)
        sets.append((kf, s5.id_set(out, schema=schema)))
    report = s5.check_monotone_nesting(sets)
    assert report["nested"] is True
    assert report["retained_counts"] == sorted(report["retained_counts"])


def test_nesting_check_flags_violation():
    report = s5.check_monotone_nesting([(0.5, {"a", "b"}), (0.9, {"c", "d", "e"})])
    assert report["nested"] is False
    assert report["violations"]


# ---------------------------------------------------------------------------
# arm runner
# ---------------------------------------------------------------------------


def test_control_arm_matches_baseline(frame, config, schema):
    record, arm_frame = s5.run_right_censoring_arm(
        frame, arm_id=CONTROL_ARM_ID, config=config, schema=schema, control=True
    )
    assert record["kind"] == "control"
    assert record["primitive"] is None
    assert record["fingerprint"] == record["baseline_fingerprint"]
    assert len(arm_frame) == len(frame)


def test_perturbed_arm_record_shape(frame, config, schema):
    record, arm_frame = s5.run_right_censoring_arm(
        frame, arm_id="keep_fraction=0.5", keep_fraction=0.5, config=config, schema=schema
    )
    assert record["kind"] == "perturbed"
    assert record["primitive"] == "right_censor"
    assert record["keep_fraction"] == 0.5
    assert record["profile"]["rows_removed"] > 0
    assert record["invariants"]["no_new_ids"] is True
    assert len(arm_frame) < len(frame)
    json.dumps(record)


def test_arm_rejects_empty_baseline(config, schema):
    with pytest.raises(RightCensoringError):
        s5.run_right_censoring_arm(
            pd.DataFrame(), arm_id="x", keep_fraction=0.5, config=config, schema=schema
        )


def test_arm_rejects_non_dataframe(config, schema):
    with pytest.raises(RightCensoringError):
        s5.run_right_censoring_arm(
            [1, 2, 3], arm_id="x", keep_fraction=0.5, config=config, schema=schema
        )


# ---------------------------------------------------------------------------
# rebuild integration
# ---------------------------------------------------------------------------


def test_scenario_module_binds_the_real_rebuild_function():
    assert s5.rebuild_world.__name__ == "rebuild_world"
    assert s5.rebuild_world.__module__.endswith("rebuild")


def test_every_arm_reaches_the_rebuild_layer(frame, config, schema, monkeypatch):
    seen: list[int] = []

    def fake_rebuild_world(target, *args, **kwargs):
        rows = len(target) if isinstance(target, pd.DataFrame) else -1
        seen.append(rows)
        return {"rows_seen": rows, "ok": True}

    fake_rebuild_world.__name__ = "rebuild_world"
    monkeypatch.setattr(s5, "rebuild_world", fake_rebuild_world)
    monkeypatch.setattr(s5, "build_world", None)

    result = s5.run_right_censoring_scenario(frame, config, schema=schema)
    expected_calls = len(config.severity_levels()) + (1 if config.include_control else 0)
    assert len(seen) == expected_calls
    assert all(
        (arm["rebuild"]["invoked"] and arm["rebuild"]["function"] == "rebuild_world")
        for arm in result["arms"]
    )
    assert result["control"]["rebuild"]["function"] == "rebuild_world"


def test_control_and_perturbed_frames_differ_at_rebuild(frame, config, schema, monkeypatch):
    sizes: list[int] = []

    def fake_rebuild_world(target, *args, **kwargs):
        sizes.append(len(target) if isinstance(target, pd.DataFrame) else -1)
        return {}

    fake_rebuild_world.__name__ = "rebuild_world"
    monkeypatch.setattr(s5, "rebuild_world", fake_rebuild_world)
    monkeypatch.setattr(s5, "build_world", None)

    s5.run_right_censoring_scenario(frame, config, schema=schema)
    assert sizes[0] == len(frame)          # control sees the full frame
    assert all(size < len(frame) for size in sizes[1:])
    assert sizes[1:] == sorted(sizes[1:], reverse=True)


def test_rebuild_error_is_recorded_not_raised(frame, config, schema, monkeypatch):
    def boom(target, *args, **kwargs):
        raise ValueError("synthetic rebuild rejection")

    boom.__name__ = "rebuild_world"
    monkeypatch.setattr(s5, "rebuild_world", boom)
    monkeypatch.setattr(s5, "build_world", None)

    result = s5.run_right_censoring_scenario(frame, config, schema=schema)
    assert all(arm["rebuild"]["ok"] is False for arm in result["arms"])
    assert all("synthetic rebuild rejection" in arm["rebuild"]["error"] for arm in result["arms"])
    json.dumps(result)


def test_real_rebuild_smoke(frame, schema):
    smoke_frame = frame.copy(deep=True)

    # The synthetic fixture does not include every column required by the
    # real rebuild_world pipeline, so add the missing required columns.
    if "merchant_id" not in smoke_frame.columns:
        smoke_frame["merchant_id"] = "MERCHANT-SMOKE"

    if "auth_outcome" not in smoke_frame.columns:
        smoke_frame["auth_outcome"] = "approved"

    cfg = RightCensoringConfig(keep_fractions=(0.75,))
    result = s5.run_right_censoring_scenario(
        smoke_frame,
        cfg,
        schema=schema,
    )

    arm = result["arms"][0]

    assert arm["rebuild"]["invoked"] is True
    assert arm["rebuild"]["function"] == "rebuild_world"

    if not arm["rebuild"]["ok"]:
        pytest.fail(
            "real rebuild_world rejected the censored frame: "
            f"{arm['rebuild']['error']}"
        )

# ---------------------------------------------------------------------------
# scenario result contract
# ---------------------------------------------------------------------------


def test_result_structure(frame, config, schema):
    result = s5.run_right_censoring_scenario(frame, config, schema=schema)
    for key in (
        "scenario_id",
        "scenario_name",
        "phase",
        "primitive",
        "primitive_signature",
        "config",
        "baseline",
        "control",
        "arms",
        "comparisons",
        "nesting",
        "summary",
    ):
        assert key in result, key
    assert result["scenario_id"] == s5.SCENARIO_ID
    assert result["primitive"] == "right_censor"
    assert result["deterministic"] is True
    assert result["seeded"] is False
    assert len(result["arms"]) == len(config.severity_levels())
    assert len(result["comparisons"]) == len(result["arms"])


def test_result_is_json_serializable(frame, config, schema):
    result = s5.run_right_censoring_scenario(frame, config, schema=schema)
    round_tripped = json.loads(json.dumps(result))
    assert round_tripped["scenario_id"] == s5.SCENARIO_ID


def test_summary_fields(frame, config, schema):
    summary = s5.run_right_censoring_scenario(frame, config, schema=schema)["summary"]
    assert summary["arm_count"] == len(config.severity_levels())
    assert summary["has_control"] is True
    assert summary["severity_mode"] == "keep_fraction"
    assert summary["monotone_nesting_ok"] is True
    assert summary["baseline_input_unchanged"] is True
    assert 0.0 < summary["min_retained_fraction"] <= 1.0
    assert 0.0 <= summary["max_removed_fraction"] < 1.0


def test_comparisons_show_shrinking_arms(frame, config, schema):
    result = s5.run_right_censoring_scenario(frame, config, schema=schema)
    for comparison in result["comparisons"]:
        assert comparison["rows_delta"] <= 0
        assert comparison["retained_fraction"] <= 1.0
    deltas = [c["rows_delta"] for c in result["comparisons"]]
    assert deltas == sorted(deltas, reverse=True)


def test_control_can_be_disabled(frame, schema):
    cfg = RightCensoringConfig(keep_fractions=(0.5,), include_control=False)
    result = s5.run_right_censoring_scenario(frame, cfg, schema=schema)
    assert result["control"] is None
    assert result["comparisons"] == []
    assert result["summary"]["has_control"] is False


def test_scenario_preserves_input_frame(frame, config, schema):
    snapshot = frame.copy(deep=True)
    s5.run_right_censoring_scenario(frame, config, schema=schema)
    pd.testing.assert_frame_equal(frame, snapshot)


def test_scenario_reports_window_shrinkage(frame, config, schema):
    result = s5.run_right_censoring_scenario(frame, config, schema=schema)
    ratios = [arm["profile"]["span_retained_ratio"] for arm in result["arms"]]
    if any(r is None for r in ratios):
        pytest.fail(f"observation_window_ns did not yield spans: {ratios!r}")
    assert all(0.0 <= r <= 1.0 for r in ratios)
    assert ratios == sorted(ratios, reverse=True)


def test_cutoff_mode_scenario_runs(frame, schema):
    ts = np.asarray(p.timestamps_as_ns(frame, schema=schema))
    cutoffs = (int(np.quantile(ts, 0.8)), int(np.quantile(ts, 0.4)))
    cfg = RightCensoringConfig(keep_fractions=(), cutoff_ts_ns=cutoffs)
    result = s5.run_right_censoring_scenario(frame, cfg, schema=schema)
    assert result["summary"]["severity_mode"] == "cutoff_ts_ns"
    assert [arm["cutoff_ts_ns"] for arm in result["arms"]] == list(cutoffs)
    json.dumps(result)


def test_scenario_rejects_invalid_config(frame, schema):
    with pytest.raises(RightCensoringError):
        s5.run_right_censoring_scenario(
            frame, RightCensoringConfig(keep_fractions=(), cutoff_ts_ns=()), schema=schema
        )


def test_scenario_rejects_non_dataframe(config, schema):
    with pytest.raises(RightCensoringError):
        s5.run_right_censoring_scenario("not a frame", config, schema=schema)
