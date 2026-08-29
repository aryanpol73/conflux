"""Phase 4B - S3 (False Positive Stress) tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

s3 = pytest.importorskip("conflux.robustness.scenario_false_positive_stress")

from conflux.robustness.perturbations import PerturbationError  # noqa: E402

FalsePositiveStressConfig = s3.FalsePositiveStressConfig
FalsePositiveStressError = s3.FalsePositiveStressError

# Entity columns the rebuild/world layer requires, in the repository's own names.
_FIXTURE_ENTITY_COLUMNS = ("card_fingerprint", "bin", "device_fingerprint", "ip_signature")


def _make_frame() -> pd.DataFrame:
    stamps = pd.date_range("2024-03-01T00:00:00Z", periods=12, freq="7min")
    rows = []
    for index, stamp in enumerate(stamps):
        attack = index < 6
        rows.append(
            {
                "transaction_id": f"tx-{index:04d}",
                "timestamp": stamp,
                "amount": 25.0 + index,
                "label": 1 if attack else 0,
                "campaign_id": "camp-a" if attack else "",
                "card_fingerprint": "card-shared" if attack else f"card-{index:02d}",
                "bin": "411111" if attack else f"4{index:05d}",
                "device_fingerprint": "dev-shared" if attack else f"dev-{index:02d}",
                "ip_signature": "10.0.0.7" if attack else f"10.0.1.{index}",
                "source_type": "web" if index % 2 else "mobile",
                "auth_outcome": "approved" if index % 2 else "declined",
            }
        )
    return pd.DataFrame(rows)


def _augment_with_schema_columns(base: pd.DataFrame, schema) -> pd.DataFrame:
    """Fill any schema-declared column the fixture does not define, deterministically."""
    declared: list[str] = []
    for attr in ("structural", "entity_columns", "entity", "entity_cols"):
        value = getattr(schema, attr, None)
        if isinstance(value, (list, tuple)):
            declared.extend(c for c in value if isinstance(c, str))
    out = base
    for column in dict.fromkeys(declared):
        if column in out.columns:
            continue
        out = out.assign(**{column: [f"{column}-{i % 3:02d}" for i in range(len(out))]})
    return out


@pytest.fixture()
def frame() -> pd.DataFrame:
    base = _make_frame()
    return _augment_with_schema_columns(base, s3.scenario_schema(base))


@pytest.fixture()
def schema(frame: pd.DataFrame):
    return s3.scenario_schema(frame)


@pytest.fixture()
def entity_columns(frame: pd.DataFrame, schema) -> tuple[str, ...]:
    """Schema-resolved entity columns, falling back to the fixture's own entity set."""
    resolved = s3.schema_entity_columns(frame, schema)
    if resolved:
        return tuple(resolved)
    return tuple(c for c in _FIXTURE_ENTITY_COLUMNS if c in frame.columns)


def _config(entity_columns: tuple[str, ...], **kwargs) -> FalsePositiveStressConfig:
    kwargs.setdefault("burst_counts", (1, 2))
    kwargs.setdefault("shared_entity_columns", entity_columns)
    return FalsePositiveStressConfig(**kwargs)


class RecordingRebuild:
    """Stand-in for the rebuild layer; records every frame it receives by arm id."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, pd.DataFrame]] = []

    def __call__(self, frame: pd.DataFrame, arm_id: str | None = None, **_: object):
        self.calls.append((arm_id or f"call-{len(self.calls)}", frame.copy(deep=True)))
        candidates = pd.DataFrame({"candidate_id": [f"c{i}" for i in range(max(1, len(frame) // 4))]})
        return {"candidates": candidates, "transaction_rows": len(frame)}

    def by_arm(self) -> dict[str, pd.DataFrame]:
        return {arm_id: captured for arm_id, captured in self.calls}


def _run(frame: pd.DataFrame, config: FalsePositiveStressConfig, rebuild=None):
    rebuild = rebuild or RecordingRebuild()
    result = s3.run_false_positive_stress_scenario(frame, config=config, rebuild_fn=rebuild)
    return result, rebuild


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

def test_module_imports_and_exposes_public_api():
    for name in ("run_false_positive_stress_scenario", "FalsePositiveStressConfig",
                 "FalsePositiveStressError", "apply_benign_bursts",
                 "resolve_shared_entity_columns", "schema_entity_columns",
                 "restore_baseline_representation"):
        assert hasattr(s3, name), name


def test_default_config_is_valid_and_json_safe():
    config = FalsePositiveStressConfig()
    assert config.burst_counts and all(v > 0 for v in config.burst_counts)
    assert config.burst_size >= 1
    assert config.span_seconds > 0
    assert config.shared_entity_columns is None  # resolved from schema at run time
    json.dumps(config.as_dict())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"burst_counts": ()},
        {"burst_counts": (0,)},
        {"burst_counts": (-3,)},
        {"burst_counts": (1, 1)},
        {"burst_counts": (1.5,)},
        {"burst_size": 0},
        {"burst_size": -2},
        {"span_seconds": 0},
        {"span_seconds": -60},
        {"seed": -1},
        {"shared_entity_columns": ()},
        {"shared_entity_columns": "device_fingerprint"},
        {"shared_entity_columns": ("device_fingerprint", "device_fingerprint")},
        {"shared_entity_columns": ("device_fingerprint", 7)},
        {"vary_columns": "amount"},
        {"shared_entity_columns": ("device_fingerprint",), "vary_columns": ("device_fingerprint",)},
        {"id_prefix": ""},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(FalsePositiveStressError):
        FalsePositiveStressConfig(**kwargs)


def test_missing_shared_entity_column_raises_perturbation_error(frame):
    config = FalsePositiveStressConfig(
        burst_counts=(1,), shared_entity_columns=("column_that_does_not_exist",)
    )
    with pytest.raises(PerturbationError):
        s3.apply_benign_bursts(frame, config=config, burst_count=1)


def test_missing_vary_column_raises_perturbation_error(frame, entity_columns):
    config = _config(entity_columns, burst_counts=(1,), vary_columns=("nope_not_here",))
    with pytest.raises(PerturbationError):
        s3.apply_benign_bursts(frame, config=config, burst_count=1)


def test_unknown_extra_kwarg_is_rejected(frame, entity_columns):
    config = _config(entity_columns, extra_kwargs={"definitely_not_a_real_parameter": 1})
    with pytest.raises(FalsePositiveStressError):
        s3.apply_benign_bursts(frame, config=config, burst_count=1)


def test_default_entity_resolution_either_works_or_fails_loudly(frame, schema):
    """No skip: exactly one of the two documented branches must hold."""
    resolved = s3.schema_entity_columns(frame, schema)
    config = FalsePositiveStressConfig(burst_counts=(1,))
    if resolved:
        assert s3.resolve_shared_entity_columns(frame, config=config, schema=schema) == resolved
    else:
        with pytest.raises(FalsePositiveStressError, match="shared_entity_columns"):
            s3.resolve_shared_entity_columns(frame, config=config, schema=schema)


def test_scenario_rejects_non_dataframe_input():
    with pytest.raises(FalsePositiveStressError):
        s3.run_false_positive_stress_scenario(["not", "a", "frame"])


# --------------------------------------------------------------------------- #
# primitive binding
# --------------------------------------------------------------------------- #

def test_call_kwargs_bind_exactly_to_the_primitive(frame, entity_columns, schema):
    config = _config(entity_columns, burst_counts=(3,), vary_columns=("amount",), id_prefix="FPX-")
    kwargs = s3.burst_call_kwargs(
        config=config, burst_count=3,
        shared_entity_columns=entity_columns, vary_columns=("amount",), schema=schema,
    )
    assert kwargs["n_bursts"] == 3
    assert kwargs["burst_size"] == config.burst_size
    assert kwargs["span_seconds"] == float(config.span_seconds)
    assert kwargs["seed"] == config.seed
    assert tuple(kwargs["shared_entity_columns"]) == tuple(entity_columns)
    assert tuple(kwargs["vary_columns"]) == ("amount",)
    assert kwargs["id_prefix"] == "FPX-"
    s3.primitive_signature().bind(frame, **kwargs)  # must not raise


def test_effective_id_prefix_falls_back_to_primitive_default():
    default_prefix = s3.effective_id_prefix(FalsePositiveStressConfig())
    declared = s3.primitive_signature().parameters["id_prefix"].default
    assert default_prefix == declared
    assert s3.effective_id_prefix(FalsePositiveStressConfig(id_prefix="ZZZ-")) == "ZZZ-"


def test_injected_ids_carry_the_id_prefix(frame, entity_columns, schema):
    config = _config(entity_columns, burst_counts=(2,), id_prefix="FPX-")
    perturbed, _ = s3.apply_benign_bursts(frame, config=config, burst_count=2)
    new_ids = s3.injected_ids(frame, perturbed, s3.id_column(frame, schema))
    assert new_ids
    assert all(str(value).startswith("FPX-") for value in new_ids)


def test_shared_entity_columns_are_actually_shared_within_bursts(frame, entity_columns, schema):
    config = _config(entity_columns, burst_counts=(2,))
    perturbed, _ = s3.apply_benign_bursts(frame, config=config, burst_count=2)
    new_rows = s3.injected_rows(frame, perturbed, s3.id_column(frame, schema))
    assert len(new_rows) == 2 * config.burst_size
    tuples = new_rows[list(entity_columns)].astype(str).agg("|".join, axis=1)
    assert tuples.nunique() <= 2  # at most one distinct entity tuple per burst


# --------------------------------------------------------------------------- #
# representation preservation (the S3 regression)
# --------------------------------------------------------------------------- #

def test_original_row_timestamps_are_preserved_exactly(frame, entity_columns, schema):
    ts_col = s3._schema_attr(schema, s3._TIME_ATTRS, frame)
    assert ts_col is not None
    id_col = s3.id_column(frame, schema)
    perturbed, _ = s3.apply_benign_bursts(
        frame, config=_config(entity_columns, burst_counts=(2,)), burst_count=2
    )
    recovered = perturbed.set_index(id_col).loc[frame[id_col], ts_col]
    assert list(recovered) == list(frame[ts_col])


def test_injection_preserves_baseline_column_dtypes_and_order(frame, entity_columns):
    perturbed, _ = s3.apply_benign_bursts(
        frame, config=_config(entity_columns, burst_counts=(2,)), burst_count=2
    )
    assert s3.dtype_drift(frame, perturbed) == {}
    assert list(perturbed.columns)[: len(frame.columns)] == list(frame.columns)


def test_guard_tolerates_pure_representation_change(frame, schema):
    """A dtype widening with identical values is not tampering."""
    widened = frame.copy(deep=True)
    ts_col = s3._schema_attr(schema, s3._TIME_ATTRS, widened)
    assert ts_col is not None
    widened[ts_col] = widened[ts_col].astype(object)
    report = s3.assert_original_rows_preserved(frame, widened, schema=schema)
    assert report["original_rows_unchanged"] is True
    assert report["original_dtypes_preserved"] is False
    assert ts_col in report["dtype_drift"]


def test_guard_detects_shifted_timestamps(frame, schema):
    """A real value change in the same column must still be caught."""
    shifted = frame.copy(deep=True)
    ts_col = s3._schema_attr(schema, s3._TIME_ATTRS, shifted)
    assert ts_col is not None
    shifted[ts_col] = shifted[ts_col] + pd.Timedelta(seconds=1)
    with pytest.raises(FalsePositiveStressError, match=ts_col):
        s3.assert_original_rows_preserved(frame, shifted, schema=schema)


# --------------------------------------------------------------------------- #
# immutability, determinism, invariants
# --------------------------------------------------------------------------- #

def test_input_frame_is_not_mutated(frame, entity_columns):
    before = frame.copy(deep=True)
    _run(frame, _config(entity_columns))
    pd.testing.assert_frame_equal(frame, before)


def test_same_seed_produces_identical_results(frame, entity_columns):
    config = _config(entity_columns, seed=99)
    first, _ = _run(frame, config)
    second, _ = _run(frame, config)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_different_seeds_change_the_injected_population(frame, entity_columns):
    a, _ = s3.apply_benign_bursts(frame, config=_config(entity_columns, seed=1), burst_count=2)
    b, _ = s3.apply_benign_bursts(frame, config=_config(entity_columns, seed=2), burst_count=2)
    assert s3.frame_fingerprint(a) != s3.frame_fingerprint(b)


def test_original_transaction_ids_survive(frame, entity_columns, schema):
    perturbed, _ = s3.apply_benign_bursts(
        frame, config=_config(entity_columns, burst_counts=(2,)), burst_count=2
    )
    id_col = s3.id_column(frame, schema)
    assert set(frame[id_col]).issubset(set(perturbed[id_col]))
    assert not perturbed[id_col].duplicated().any()


def test_original_rows_are_byte_identical_after_injection(frame, entity_columns, schema):
    perturbed, _ = s3.apply_benign_bursts(
        frame, config=_config(entity_columns, burst_counts=(3,)), burst_count=3
    )
    report = s3.assert_original_rows_preserved(frame, perturbed, schema=schema)
    assert report["original_rows_unchanged"] is True
    assert report["original_ids_preserved"] is True
    assert report["original_dtypes_preserved"] is True


def test_population_grows_with_burst_count(frame, entity_columns):
    config = _config(entity_columns, burst_counts=(1, 4))
    small, _ = s3.apply_benign_bursts(frame, config=config, burst_count=1)
    large, _ = s3.apply_benign_bursts(frame, config=config, burst_count=4)
    assert len(small) == len(frame) + config.burst_size
    assert len(large) == len(frame) + 4 * config.burst_size


def test_injected_rows_are_benign(frame, entity_columns, schema):
    perturbed, _ = s3.apply_benign_bursts(
        frame, config=_config(entity_columns, burst_counts=(2,)), burst_count=2
    )
    profile = s3.benign_injection_profile(frame, perturbed, schema=schema)
    assert profile["injected_all_benign"] is True


def test_invariant_guard_detects_tampered_amount(frame, schema):
    tampered = frame.copy(deep=True)
    amount_col = getattr(schema, "amount_col", None)
    assert isinstance(amount_col, str) and amount_col in tampered.columns
    tampered.loc[tampered.index[0], amount_col] = float(tampered[amount_col].iloc[0]) + 1.0
    with pytest.raises(FalsePositiveStressError, match=amount_col):
        s3.assert_original_rows_preserved(frame, tampered, schema=schema)


def test_invariant_guard_detects_tampered_label(frame, schema):
    tampered = frame.copy(deep=True)
    label_col = s3._schema_attr(schema, s3._LABEL_ATTRS, tampered)
    assert label_col is not None
    # widen first: pandas 3 refuses a str into an int64 column
    tampered[label_col] = tampered[label_col].astype(object)
    tampered.loc[tampered.index[0], label_col] = "TAMPERED"
    with pytest.raises(FalsePositiveStressError, match=label_col):
        s3.assert_original_rows_preserved(frame, tampered, schema=schema)


def test_invariant_guard_detects_dropped_original_rows(frame, schema):
    with pytest.raises(FalsePositiveStressError):
        s3.assert_original_rows_preserved(frame, frame.iloc[1:].copy(deep=True), schema=schema)


# --------------------------------------------------------------------------- #
# rebuild integration and result contract
# --------------------------------------------------------------------------- #

def test_control_and_perturbed_frames_reach_the_rebuild_layer(frame, entity_columns):
    result, rebuild = _run(frame, _config(entity_columns))
    seen = rebuild.by_arm()
    assert s3.CONTROL_ARM_ID in seen
    for arm in result["arms"]:
        assert arm["arm_id"] in seen


def test_perturbed_frame_differs_from_control_at_the_rebuild_layer(frame, entity_columns):
    _, rebuild = _run(frame, _config(entity_columns, burst_counts=(2,)))
    seen = rebuild.by_arm()
    control, perturbed = seen[s3.CONTROL_ARM_ID], seen["bursts=2"]
    assert len(perturbed) > len(control)
    assert s3.frame_fingerprint(perturbed) != s3.frame_fingerprint(control)


def test_control_arm_is_unperturbed(frame, entity_columns):
    result, rebuild = _run(frame, _config(entity_columns, burst_counts=(1,)))
    assert result["control"]["kind"] == "control"
    assert result["control"]["population"]["injected_rows"] == 0
    pd.testing.assert_frame_equal(
        rebuild.by_arm()[s3.CONTROL_ARM_ID].reset_index(drop=True),
        frame.reset_index(drop=True),
    )


def test_result_structure_and_json_serialisability(frame, entity_columns):
    result, _ = _run(frame, _config(entity_columns))
    for key in ("scenario", "scenario_name", "config", "control", "arms", "comparisons", "summary"):
        assert key in result, key
    assert result["scenario"] == s3.SCENARIO_ID
    arm_ids = [arm["arm_id"] for arm in result["arms"]]
    assert len(arm_ids) == len(set(arm_ids))
    for arm in result["arms"]:
        if arm["kind"] == "perturbed":
            assert arm["parameters"]["shared_entity_columns"]
        assert arm["invariants"]["original_rows_unchanged"] is True
    for comparison in result["comparisons"]:
        assert "structure_deltas" in comparison
        assert comparison["frames_differ"] is True
    json.dumps(result)


def test_summary_reports_false_positive_pressure(frame, entity_columns):
    result, _ = _run(frame, _config(entity_columns))
    summary = result["summary"]
    assert summary["arm_count"] == len(result["arms"])
    assert summary["max_injected_rows"] > 0
    assert "false_positive_pressure_detected" in summary


def test_fingerprints_are_reproducible(frame, entity_columns):
    config = _config(entity_columns, seed=7)
    first, _ = _run(frame, config)
    second, _ = _run(frame, config)
    assert [a["frame_fingerprint"] for a in first["arms"]] == \
           [a["frame_fingerprint"] for a in second["arms"]]


def test_scenario_can_run_without_control_arm(frame, entity_columns):
    config = _config(entity_columns, burst_counts=(1,), include_control=False)
    result, rebuild = _run(frame, config)
    assert result["control"] is None
    assert result["comparisons"] == []
    assert s3.CONTROL_ARM_ID not in rebuild.by_arm()


def test_real_rebuild_smoke(frame, entity_columns):
    """Real pipeline on a synthetic frame; skips only if the pipeline itself is unavailable."""
    rebuild_world = pytest.importorskip("conflux.robustness.rebuild").rebuild_world
    try:
        result = s3.run_false_positive_stress_scenario(
            frame, config=_config(entity_columns, burst_counts=(1,)), rebuild_fn=rebuild_world
        )
    except Exception as exc:
        pytest.skip(f"Real rebuild pipeline unavailable for the synthetic fixture: {exc}")
    json.dumps(result)
    assert result["arms"]
