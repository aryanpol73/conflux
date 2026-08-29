"""Phase 4B - S4 (Legitimate Volume) tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

s4 = pytest.importorskip("conflux.robustness.scenario_legit_volume")

from conflux.robustness.perturbations import PerturbationError  # noqa: E402

LegitVolumeConfig = s4.LegitVolumeConfig
LegitVolumeError = s4.LegitVolumeError
VolumeLevel = s4.VolumeLevel


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
    for attr in ("structural", "entity", "entity_columns", "entity_cols"):
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
    return _augment_with_schema_columns(base, s4.scenario_schema(base))


@pytest.fixture()
def schema(frame: pd.DataFrame):
    return s4.scenario_schema(frame)


@pytest.fixture()
def donors(frame: pd.DataFrame, schema) -> int:
    return s4.donor_count(frame, schema)


def _config(**kwargs) -> LegitVolumeConfig:
    kwargs.setdefault("multipliers", (0.5, 1.0))
    return LegitVolumeConfig(**kwargs)


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


def _run(frame: pd.DataFrame, config: LegitVolumeConfig, rebuild=None):
    rebuild = rebuild or RecordingRebuild()
    result = s4.run_legit_volume_scenario(frame, config=config, rebuild_fn=rebuild)
    return result, rebuild


def _apply(frame: pd.DataFrame, *, kind: str, value, **cfg):
    if kind == s4.MULTIPLIER_MODE:
        config = LegitVolumeConfig(multipliers=(value,), **cfg)
    else:
        config = LegitVolumeConfig(multipliers=None, absolute_counts=(value,), **cfg)
    return s4.apply_legit_volume(frame, config=config, level=VolumeLevel(kind, value))


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

def test_module_imports_and_exposes_public_api():
    for name in ("run_legit_volume_scenario", "LegitVolumeConfig", "LegitVolumeError",
                 "VolumeLevel", "apply_legit_volume", "donor_count",
                 "expected_injected_rows", "restore_baseline_representation"):
        assert hasattr(s4, name), name


def test_default_config_is_valid_and_json_safe():
    config = LegitVolumeConfig()
    assert config.mode == s4.MULTIPLIER_MODE
    assert config.multipliers and all(v > 0 for v in config.multipliers)
    assert config.absolute_counts is None
    assert config.entity_columns is None  # primitive falls back to schema.entity
    assert config.id_prefix is None       # primitive supplies its own default
    json.dumps(config.as_dict())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"multipliers": None},                                  # neither mode
        {"absolute_counts": (5,)},                              # both modes
        {"multipliers": ()},
        {"multipliers": (0.0,)},
        {"multipliers": (-1.0,)},
        {"multipliers": (1.0, 1.0)},
        {"multipliers": ("1.0",)},
        {"multipliers": None, "absolute_counts": ()},
        {"multipliers": None, "absolute_counts": (0,)},
        {"multipliers": None, "absolute_counts": (-4,)},
        {"multipliers": None, "absolute_counts": (3, 3)},
        {"multipliers": None, "absolute_counts": (2.5,)},
        {"seed": -1},
        {"refresh_entities": "yes"},
        {"entity_columns": ()},
        {"entity_columns": "device_fingerprint"},
        {"entity_columns": ("a", "a")},
        {"time_mode": ""},
        {"id_prefix": ""},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(LegitVolumeError):
        LegitVolumeConfig(**kwargs)


def test_exactly_one_mode_is_enforced_like_the_primitive():
    both = LegitVolumeError
    with pytest.raises(both):
        LegitVolumeConfig(multipliers=(1.0,), absolute_counts=(5,))
    with pytest.raises(both):
        LegitVolumeConfig(multipliers=None, absolute_counts=None)
    assert LegitVolumeConfig(multipliers=None, absolute_counts=(5,)).mode == s4.ABSOLUTE_MODE


def test_volume_levels_carry_distinct_arm_ids():
    multiplier_ids = [l.arm_id for l in LegitVolumeConfig(multipliers=(0.5, 2.0)).volume_levels()]
    assert multiplier_ids == ["multiplier=0.5", "multiplier=2"]
    count_ids = [
        l.arm_id for l in
        LegitVolumeConfig(multipliers=None, absolute_counts=(3, 9)).volume_levels()
    ]
    assert count_ids == ["n_new=3", "n_new=9"]


def test_scenario_rejects_non_dataframe_input():
    with pytest.raises(LegitVolumeError):
        s4.run_legit_volume_scenario(["not", "a", "frame"])


def test_invalid_time_mode_is_rejected_by_the_primitive(frame):
    with pytest.raises(PerturbationError):
        _apply(frame, kind=s4.MULTIPLIER_MODE, value=1.0, time_mode="sideways")


def test_missing_entity_column_is_rejected_by_the_primitive(frame):
    with pytest.raises(PerturbationError):
        _apply(
            frame, kind=s4.MULTIPLIER_MODE, value=1.0,
            entity_columns=("column_that_does_not_exist",),
        )


def test_unknown_extra_kwarg_is_rejected(frame):
    config = _config(extra_kwargs={"definitely_not_a_real_parameter": 1})
    with pytest.raises(LegitVolumeError):
        s4.apply_legit_volume(frame, config=config, level=VolumeLevel(s4.MULTIPLIER_MODE, 1.0))


# --------------------------------------------------------------------------- #
# primitive binding
# --------------------------------------------------------------------------- #

def test_call_kwargs_bind_exactly_to_the_primitive(frame, schema):
    config = LegitVolumeConfig(
        multipliers=(0.5,), seed=11, refresh_entities=False,
        entity_columns=("device_fingerprint",), time_mode="copy", id_prefix="LV-",
    )
    kwargs = s4.volume_call_kwargs(
        config=config, level=VolumeLevel(s4.MULTIPLIER_MODE, 0.5), schema=schema
    )
    assert kwargs["multiplier"] == 0.5
    assert "n_new" not in kwargs
    assert kwargs["seed"] == 11
    assert kwargs["refresh_entities"] is False
    assert tuple(kwargs["entity_columns"]) == ("device_fingerprint",)
    assert kwargs["time_mode"] == "copy"
    assert kwargs["id_prefix"] == "LV-"
    s4.primitive_signature().bind(frame, **kwargs)  # must not raise


def test_absolute_mode_sends_n_new_only(frame, schema):
    config = LegitVolumeConfig(multipliers=None, absolute_counts=(7,))
    kwargs = s4.volume_call_kwargs(
        config=config, level=VolumeLevel(s4.ABSOLUTE_MODE, 7), schema=schema
    )
    assert kwargs["n_new"] == 7
    assert "multiplier" not in kwargs


def test_defaults_are_omitted_so_the_primitive_supplies_them(frame, schema):
    kwargs = s4.volume_call_kwargs(
        config=LegitVolumeConfig(), level=VolumeLevel(s4.MULTIPLIER_MODE, 1.0), schema=schema
    )
    assert "entity_columns" not in kwargs
    assert "id_prefix" not in kwargs


def test_effective_id_prefix_falls_back_to_primitive_default():
    declared = s4.primitive_signature().parameters["id_prefix"].default
    assert s4.effective_id_prefix(LegitVolumeConfig()) == declared
    assert s4.effective_id_prefix(LegitVolumeConfig(id_prefix="LV-")) == "LV-"


def test_unknown_level_kind_is_rejected(schema):
    with pytest.raises(LegitVolumeError):
        s4.volume_call_kwargs(
            config=LegitVolumeConfig(), level=VolumeLevel("percentage", 10), schema=schema
        )


# --------------------------------------------------------------------------- #
# volume arithmetic (donor-pool relative, per the primitive)
# --------------------------------------------------------------------------- #

def test_absolute_count_injects_exactly_that_many_rows(frame):
    perturbed, _ = _apply(frame, kind=s4.ABSOLUTE_MODE, value=5)
    assert len(perturbed) == len(frame) + 5


def test_multiplier_scales_the_donor_pool_not_the_population(frame, schema, donors):
    assert 0 < donors < len(frame)  # fixture really has both classes
    perturbed, _ = _apply(frame, kind=s4.MULTIPLIER_MODE, value=1.0)
    injected = len(perturbed) - len(frame)
    assert injected == donors
    assert injected != len(frame)


def test_expected_injection_count_matches_reality(frame, schema):
    for value in (0.5, 1.0, 2.0):
        level = VolumeLevel(s4.MULTIPLIER_MODE, value)
        expected = s4.expected_injected_rows(frame, level=level, schema=schema)
        perturbed, _ = _apply(frame, kind=s4.MULTIPLIER_MODE, value=value)
        assert len(perturbed) - len(frame) == expected


def test_multiplier_rounding_to_zero_is_rejected_by_the_primitive(frame, donors):
    tiny = 0.4 / float(donors)  # rounds to zero rows
    with pytest.raises(PerturbationError):
        _apply(frame, kind=s4.MULTIPLIER_MODE, value=tiny)


def test_larger_volume_injects_more_rows(frame):
    small, _ = _apply(frame, kind=s4.MULTIPLIER_MODE, value=0.5)
    large, _ = _apply(frame, kind=s4.MULTIPLIER_MODE, value=2.0)
    assert len(large) > len(small) > len(frame)


# --------------------------------------------------------------------------- #
# injected-row properties
# --------------------------------------------------------------------------- #

def test_injected_ids_are_unique_and_prefixed(frame, schema):
    perturbed, _ = _apply(frame, kind=s4.ABSOLUTE_MODE, value=6)
    id_col = s4.id_column(frame, schema)
    new_ids = s4.injected_ids(frame, perturbed, id_col)
    assert len(new_ids) == 6
    assert len(set(new_ids)) == 6
    assert not perturbed[id_col].duplicated().any()
    default_prefix = s4.primitive_signature().parameters["id_prefix"].default
    assert all(str(v).startswith(default_prefix) for v in new_ids)


def test_injected_rows_are_negative_and_uncampaigned(frame, schema):
    perturbed, _ = _apply(frame, kind=s4.ABSOLUTE_MODE, value=4)
    profile = s4.legit_volume_profile(frame, perturbed, schema=schema)
    assert profile["injected_all_negative"] is True
    assert profile["injected_campaigns_empty"] is True


def test_refresh_entities_true_creates_no_entity_reuse(frame, schema):
    entity_columns = s4.schema_entity_columns(frame, schema)
    assert entity_columns
    perturbed, reported = _apply(
        frame, kind=s4.ABSOLUTE_MODE, value=6, refresh_entities=True
    )
    profile = s4.legit_volume_profile(
        frame, perturbed, schema=schema, entity_columns=entity_columns
    )
    assert profile["injected_entities_all_fresh"] is True
    assert profile["injected_entity_tuples_reused_from_baseline"] == 0
    assert profile["distinct_injected_entity_tuples"] == 6


def test_refresh_entities_false_copies_donor_entities(frame, schema):
    entity_columns = s4.schema_entity_columns(frame, schema)
    perturbed, _ = _apply(
        frame, kind=s4.ABSOLUTE_MODE, value=6, refresh_entities=False
    )
    profile = s4.legit_volume_profile(
        frame, perturbed, schema=schema, entity_columns=entity_columns
    )
    assert profile["injected_entities_all_fresh"] is False
    assert profile["injected_entity_tuples_reused_from_baseline"] == 6


def test_time_mode_copy_reuses_baseline_timestamps(frame, schema):
    perturbed, _ = _apply(frame, kind=s4.ABSOLUTE_MODE, value=6, time_mode="copy")
    profile = s4.legit_volume_profile(frame, perturbed, schema=schema)
    assert profile["injected_timestamps_reused"] is True


def test_time_mode_uniform_stays_inside_the_observation_window(frame, schema):
    perturbed, _ = _apply(frame, kind=s4.ABSOLUTE_MODE, value=8, time_mode="uniform")
    profile = s4.legit_volume_profile(frame, perturbed, schema=schema)
    assert profile["injected_within_observation_window"] is True


# --------------------------------------------------------------------------- #
# preservation of the original population
# --------------------------------------------------------------------------- #

def test_original_rows_are_preserved_value_for_value(frame, schema):
    perturbed, _ = _apply(frame, kind=s4.MULTIPLIER_MODE, value=2.0)
    report = s4.assert_original_rows_preserved(frame, perturbed, schema=schema)
    assert report["original_rows_unchanged"] is True
    assert report["original_ids_preserved"] is True
    assert report["original_dtypes_preserved"] is True


def test_original_ids_all_survive(frame, schema):
    perturbed, _ = _apply(frame, kind=s4.ABSOLUTE_MODE, value=5)
    id_col = s4.id_column(frame, schema)
    assert set(frame[id_col]).issubset(set(perturbed[id_col]))


def test_attack_rows_are_untouched_by_added_volume(frame, schema):
    perturbed, _ = _apply(frame, kind=s4.MULTIPLIER_MODE, value=2.0)
    profile = s4.legit_volume_profile(frame, perturbed, schema=schema)
    attack_before = len(frame) - s4.donor_count(frame, schema)
    attack_after = len(perturbed) - s4.donor_count(perturbed, schema)
    assert profile["attack_rows"] == attack_before == attack_after


def test_resampling_preserves_dtypes_and_column_order(frame):
    perturbed, _ = _apply(frame, kind=s4.ABSOLUTE_MODE, value=4)
    assert s4.dtype_drift(frame, perturbed) == {}
    assert list(perturbed.columns)[: len(frame.columns)] == list(frame.columns)


def test_guard_tolerates_pure_representation_change(frame, schema):
    widened = frame.copy(deep=True)
    ts_col = s4._schema_attr(schema, s4._TIME_ATTRS, widened)
    assert ts_col is not None
    widened[ts_col] = widened[ts_col].astype(object)
    report = s4.assert_original_rows_preserved(frame, widened, schema=schema)
    assert report["original_rows_unchanged"] is True
    assert report["original_dtypes_preserved"] is False


def test_guard_detects_tampered_amount(frame, schema):
    tampered = frame.copy(deep=True)
    amount_col = getattr(schema, "amount_col", None)
    assert isinstance(amount_col, str) and amount_col in tampered.columns
    tampered.loc[tampered.index[0], amount_col] = float(tampered[amount_col].iloc[0]) + 1.0
    with pytest.raises(LegitVolumeError, match=amount_col):
        s4.assert_original_rows_preserved(frame, tampered, schema=schema)


def test_guard_detects_dropped_original_rows(frame, schema):
    with pytest.raises(LegitVolumeError):
        s4.assert_original_rows_preserved(frame, frame.iloc[1:].copy(deep=True), schema=schema)


def test_guard_survives_the_primitive_row_reordering(frame, schema):
    """The primitive sorts by id, so original rows move; that is not a change."""
    perturbed, _ = _apply(frame, kind=s4.ABSOLUTE_MODE, value=4)
    id_col = s4.id_column(frame, schema)
    assert list(perturbed[id_col])[: len(frame)] != list(frame[id_col])
    s4.assert_original_rows_preserved(frame, perturbed, schema=schema)


# --------------------------------------------------------------------------- #
# determinism and immutability
# --------------------------------------------------------------------------- #

def test_input_frame_is_not_mutated(frame):
    before = frame.copy(deep=True)
    _run(frame, _config())
    pd.testing.assert_frame_equal(frame, before)


def test_same_seed_produces_identical_results(frame):
    config = _config(seed=99)
    first, _ = _run(frame, config)
    second, _ = _run(frame, config)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_different_seeds_change_the_injected_population(frame):
    a, _ = _apply(frame, kind=s4.ABSOLUTE_MODE, value=6, seed=1)
    b, _ = _apply(frame, kind=s4.ABSOLUTE_MODE, value=6, seed=2)
    assert s4.frame_fingerprint(a) != s4.frame_fingerprint(b)


def test_fingerprints_are_reproducible(frame):
    config = _config(seed=7)
    first, _ = _run(frame, config)
    second, _ = _run(frame, config)
    assert [a["frame_fingerprint"] for a in first["arms"]] == \
           [a["frame_fingerprint"] for a in second["arms"]]


# --------------------------------------------------------------------------- #
# rebuild integration and result contract
# --------------------------------------------------------------------------- #

def test_control_and_perturbed_frames_reach_the_rebuild_layer(frame):
    result, rebuild = _run(frame, _config())
    seen = rebuild.by_arm()
    assert s4.CONTROL_ARM_ID in seen
    for arm in result["arms"]:
        assert arm["arm_id"] in seen


def test_rebuild_receives_the_actual_perturbed_frame(frame):
    _, rebuild = _run(frame, _config(multipliers=(1.0,)))
    seen = rebuild.by_arm()
    control, perturbed = seen[s4.CONTROL_ARM_ID], seen["multiplier=1"]
    assert len(perturbed) > len(control)
    assert s4.frame_fingerprint(perturbed) != s4.frame_fingerprint(control)


def test_control_arm_is_unperturbed(frame):
    result, rebuild = _run(frame, _config(multipliers=(1.0,)))
    assert result["control"]["kind"] == "control"
    assert result["control"]["population"]["injected_rows"] == 0
    pd.testing.assert_frame_equal(
        rebuild.by_arm()[s4.CONTROL_ARM_ID].reset_index(drop=True),
        frame.reset_index(drop=True),
    )


def test_result_structure_and_json_serialisability(frame):
    result, _ = _run(frame, _config())
    for key in ("scenario", "scenario_name", "config", "control", "arms", "comparisons", "summary"):
        assert key in result, key
    assert result["scenario"] == s4.SCENARIO_ID
    arm_ids = [arm["arm_id"] for arm in result["arms"]]
    assert len(arm_ids) == len(set(arm_ids))
    for arm in result["arms"]:
        assert arm["invariants"]["original_rows_unchanged"] is True
        if arm["kind"] == "perturbed":
            assert arm["parameters"]["level"]["kind"] == s4.MULTIPLIER_MODE
            assert arm["population"]["injected_rows_match_expected"] is True
    for comparison in result["comparisons"]:
        assert "structure_deltas" in comparison
        assert comparison["frames_differ"] is True
    json.dumps(result)


def test_summary_reports_volume_pressure(frame, donors):
    result, _ = _run(frame, _config())
    summary = result["summary"]
    assert summary["arm_count"] == len(result["arms"])
    assert summary["mode"] == s4.MULTIPLIER_MODE
    assert summary["donor_rows"] == donors
    assert summary["max_injected_rows"] > 0
    assert "structures_added_by_volume" in summary


def test_absolute_mode_runs_end_to_end(frame):
    config = LegitVolumeConfig(multipliers=None, absolute_counts=(3, 6))
    result, rebuild = _run(frame, config)
    assert result["summary"]["mode"] == s4.ABSOLUTE_MODE
    assert set(rebuild.by_arm()) >= {s4.CONTROL_ARM_ID, "n_new=3", "n_new=6"}
    json.dumps(result)


def test_scenario_can_run_without_control_arm(frame):
    config = _config(multipliers=(1.0,), include_control=False)
    result, rebuild = _run(frame, config)
    assert result["control"] is None
    assert result["comparisons"] == []
    assert s4.CONTROL_ARM_ID not in rebuild.by_arm()


def test_real_rebuild_smoke(frame):
    """Real pipeline on a synthetic frame; skips only if the pipeline itself is unavailable."""
    rebuild_world = pytest.importorskip("conflux.robustness.rebuild").rebuild_world
    try:
        result = s4.run_legit_volume_scenario(
            frame, config=_config(multipliers=(1.0,)), rebuild_fn=rebuild_world
        )
    except Exception as exc:
        pytest.skip(f"Real rebuild pipeline unavailable for the synthetic fixture: {exc}")
    json.dumps(result)
    assert result["arms"]
