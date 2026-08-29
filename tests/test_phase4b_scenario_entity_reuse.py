"""Phase 4B S2 -- entity reuse scenario tests. Schema-driven, deterministic.

weaken_entity_reuse is NEVER mocked. Only the rebuild boundary is replaced, by
a recording stub returning a genuine RebuiltWorld, so metric collection runs
for real without the production dataset.

Rebuild calls are identified BY NAME, never by list position.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conflux.robustness import scenario_entity_reuse as se
from conflux.robustness.perturbations import (
    PerturbationError, resolve_schema, weaken_entity_reuse,
)
from conflux.robustness.rebuild import (
    FEAT_CAMPAIGN_COL, FEAT_ID_COL, FEAT_LABEL_COL, RebuiltWorld,
)
from conflux.robustness.world import (
    entity_column, linking_entity_columns, world_frame_columns,
)

SCHEMA = resolve_schema()
TS_FMT = "%Y-%m-%d %H:%M:%S.%f"
TARGET_COL = entity_column(se.DEFAULT_ENTITY_TYPE)


# ----------------------------------------------------------------------
# synthetic fixture -- attack rows share every linking entity per campaign
# ----------------------------------------------------------------------
def _make_frame(n_benign: int = 8,
                campaigns: tuple[tuple[str, int, int], ...] =
                (("CMP-1", 5, 30), ("CMP-2", 4, 60))) -> pd.DataFrame:
    cols = list(world_frame_columns(SCHEMA))
    shared = linking_entity_columns()
    t0 = pd.Timestamp("2024-01-01 00:00:00.000000")
    rows: list[dict] = []
    i = 0

    def base_row(idx: int, ts: pd.Timestamp) -> dict:
        row = {c: f"{c}-{idx:03d}" for c in cols}
        row[SCHEMA.id_col] = f"TX{idx:05d}"
        row[SCHEMA.ts_col] = ts.strftime(TS_FMT)
        row["amount"] = float(10 + idx)
        row["auth_outcome"] = "declined" if idx % 3 else "approved"
        return row

    for b in range(n_benign):
        row = base_row(i, t0 + pd.Timedelta(minutes=7 * b))
        row[SCHEMA.label_col] = "0"
        row[SCHEMA.campaign_col] = ""
        rows.append(row)
        i += 1

    for offset, (cid, size, step) in enumerate(campaigns):
        start = t0 + pd.Timedelta(hours=2 + offset)
        for k in range(size):
            row = base_row(i, start + pd.Timedelta(seconds=step * k))
            row[SCHEMA.label_col] = "1"
            row[SCHEMA.campaign_col] = cid
            for c in shared:
                row[c] = f"{c}-{cid}"
            rows.append(row)
            i += 1

    return pd.DataFrame(rows, columns=cols)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return _make_frame()


class RecordingRebuild:
    """Narrow rebuild-boundary stub returning a real RebuiltWorld."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, frame: pd.DataFrame, *, name: str, **kwargs) -> RebuiltWorld:
        self.calls.append({"frame": frame.copy(), "name": name, "kwargs": kwargs})
        groups = frame[SCHEMA.campaign_col].astype(str)
        camps = sorted(g for g in groups.unique() if g not in ("", "nan", "None"))
        rows = [{FEAT_ID_COL: f"C{n:04d}", FEAT_LABEL_COL: True,
                 FEAT_CAMPAIGN_COL: c} for n, c in enumerate(camps)]
        rows.append({FEAT_ID_COL: f"C{len(camps):04d}", FEAT_LABEL_COL: False,
                     FEAT_CAMPAIGN_COL: ""})
        labelled = pd.DataFrame(rows)
        return RebuiltWorld(
            name=name, world_path=Path("synthetic-world.csv"),
            n_transactions=int(len(frame)),
            graph_summary={"transactions": int(len(frame))},
            candidate_set=None,
            candidate_frame=labelled[[FEAT_ID_COL]].copy(),
            assignments=pd.DataFrame(),
            scoring_features=None,
            labelled_features=labelled,
            ground_truth=pd.DataFrame(),
            grouping_metrics={"n_candidates": int(len(labelled))})


def _reuse(frame: pd.DataFrame, column: str = TARGET_COL) -> dict:
    return se.entity_reuse_profile(frame, entity_column=column, schema=SCHEMA)


def _by_name(stub: RecordingRebuild) -> dict[str, dict]:
    return {call["name"]: call for call in stub.calls}


# ----------------------------------------------------------------------
# 1. defaults
# ----------------------------------------------------------------------
def test_default_configuration_is_valid():
    cfg = se.EntityReuseScenarioConfig()
    assert cfg.fractions == se.DEFAULT_REUSE_FRACTIONS
    assert cfg.fractions[0] == se.IDENTITY_FRACTION == 0.0
    assert cfg.resolve_entity_column() == TARGET_COL
    assert TARGET_COL in linking_entity_columns()
    assert isinstance(cfg.seed, int)


# ----------------------------------------------------------------------
# 2. invalid severity / configuration
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", [(), (-0.1,), (1.5,), (0.5, 2.0),
                                 (float("nan"),)])
def test_invalid_fractions_rejected(bad):
    with pytest.raises(se.EntityReuseScenarioError):
        se.EntityReuseScenarioConfig(fractions=bad)


def test_missing_seed_rejected():
    with pytest.raises(se.EntityReuseScenarioError, match="seed"):
        se.EntityReuseScenarioConfig(seed=None)


def test_primitive_also_rejects_out_of_range_fraction(frame):
    with pytest.raises(PerturbationError, match="fraction"):
        weaken_entity_reuse(frame, entity_column=TARGET_COL, fraction=1.5,
                            seed=1, schema=SCHEMA)


# ----------------------------------------------------------------------
# 3. invalid entity column selection
# ----------------------------------------------------------------------
def test_unknown_entity_type_rejected_at_config_time():
    with pytest.raises(se.EntityReuseScenarioError, match="unknown entity type"):
        se.EntityReuseScenarioConfig(entity_type="wallet")


def test_unknown_entity_column_rejected_by_scenario(frame):
    cfg = se.EntityReuseScenarioConfig(fractions=(0.5,),
                                       entity_column="not_a_column")
    with pytest.raises(se.EntityReuseScenarioError, match="not in frame"):
        se.run_entity_reuse_scenario(frame, config=cfg,
                                     rebuild_fn=RecordingRebuild())


def test_unknown_entity_column_rejected_by_primitive(frame):
    with pytest.raises(PerturbationError, match="missing required column"):
        weaken_entity_reuse(frame, entity_column="not_a_column", fraction=0.5,
                            seed=1, schema=SCHEMA)


# ----------------------------------------------------------------------
# 4. reuse actually weakens as severity rises
# ----------------------------------------------------------------------
def test_reuse_weakens_monotonically_with_fraction(frame):
    baseline = _reuse(frame)
    assert baseline["targeted_rows"] == 9          # 5 + 4 attack rows
    assert baseline["distinct_values"] == 2        # one shared value per campaign
    assert baseline["reuse_index"] > 0.7

    indices: list[float] = []
    for frac in (0.0, 0.25, 0.5, 1.0):
        world = se.build_entity_reuse_world(frame, fraction=frac, schema=SCHEMA)
        indices.append(_reuse(world.frame)["reuse_index"])

    assert indices[0] == pytest.approx(baseline["reuse_index"])
    assert all(b >= a for b, a in zip(indices, indices[1:])), indices
    assert indices[-1] < indices[0]


def test_full_severance_leaves_no_shared_values(frame):
    world = se.build_entity_reuse_world(frame, fraction=1.0, schema=SCHEMA)
    after = _reuse(world.frame)
    assert after["distinct_values"] == after["targeted_rows"]
    assert after["shared_value_rows"] == 0
    assert after["reuse_index"] == pytest.approx(0.0)
    assert after["max_transactions_per_value"] == 1


def test_identity_fraction_changes_nothing(frame):
    world = se.build_entity_reuse_world(frame, fraction=0.0, schema=SCHEMA)
    assert (world.frame[TARGET_COL].to_numpy()
            == frame[TARGET_COL].to_numpy()).all()
    assert _reuse(world.frame) == _reuse(frame)


# ----------------------------------------------------------------------
# 5-9. structural invariants
# ----------------------------------------------------------------------
def test_structural_invariants_are_preserved(frame):
    world = se.build_entity_reuse_world(frame, fraction=0.5, schema=SCHEMA)
    out = world.frame

    assert len(out) == len(frame)                                   # count
    assert list(out.columns) == list(frame.columns)
    b = frame.set_index(frame[SCHEMA.id_col].astype(str)).sort_index()
    a = out.set_index(out[SCHEMA.id_col].astype(str)).sort_index()
    assert list(b.index) == list(a.index)                           # IDs

    untouched = [SCHEMA.label_col, SCHEMA.campaign_col, SCHEMA.ts_col,
                 "amount", "auth_outcome",
                 *(c for c in linking_entity_columns() if c != TARGET_COL)]
    for col in untouched:                                           # labels,
        assert (b[col].astype(str).to_numpy()                       # campaigns,
                == a[col].astype(str).to_numpy()).all(), col        # other entities

    assert (b[TARGET_COL].astype(str).to_numpy()
            != a[TARGET_COL].astype(str).to_numpy()).any()


def test_only_attack_rows_are_weakened(frame):
    world = se.build_entity_reuse_world(frame, fraction=1.0, schema=SCHEMA)
    benign = frame[SCHEMA.campaign_col].astype(str) == ""
    assert benign.any()
    b = frame.set_index(frame[SCHEMA.id_col])[TARGET_COL]
    a = world.frame.set_index(world.frame[SCHEMA.id_col])[TARGET_COL]
    ids = frame.loc[benign, SCHEMA.id_col]
    assert (b.loc[ids].to_numpy() == a.loc[ids].to_numpy()).all()


# ----------------------------------------------------------------------
# 10. no mutation of the caller's frame
# ----------------------------------------------------------------------
def test_input_frame_is_not_mutated(frame):
    original = frame.copy(deep=True)
    se.run_entity_reuse_scenario(frame, rebuild_fn=RecordingRebuild())
    pd.testing.assert_frame_equal(frame, original)


# ----------------------------------------------------------------------
# 11 & 15. determinism and fingerprints
# ----------------------------------------------------------------------
def test_scenario_is_deterministic(frame):
    a = se.run_entity_reuse_scenario(frame, rebuild_fn=RecordingRebuild())
    b = se.run_entity_reuse_scenario(frame, rebuild_fn=RecordingRebuild())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_world_fingerprints_are_reproducible_and_seed_sensitive(frame):
    cfg_a = se.EntityReuseScenarioConfig(seed=11)
    cfg_b = se.EntityReuseScenarioConfig(seed=12)
    one = se.build_entity_reuse_world(frame, fraction=0.5, config=cfg_a,
                                      schema=SCHEMA)
    two = se.build_entity_reuse_world(frame, fraction=0.5, config=cfg_a,
                                      schema=SCHEMA)
    other = se.build_entity_reuse_world(frame, fraction=0.5, config=cfg_b,
                                        schema=SCHEMA)
    assert one.fingerprint == two.fingerprint
    assert one.fingerprint != other.fingerprint


# ----------------------------------------------------------------------
# 12-14. what actually reaches the rebuild layer (identified BY NAME)
# ----------------------------------------------------------------------
def test_control_and_perturbed_frames_reach_the_rebuild_layer(frame):
    stub = RecordingRebuild()
    cfg = se.EntityReuseScenarioConfig(fractions=(1.0,))
    res = se.run_entity_reuse_scenario(frame, config=cfg, rebuild_fn=stub)

    assert res["control_source"] == "unperturbed_control"
    calls = _by_name(stub)
    control_name = f"{cfg.scenario_id}_control"
    perturbed_name = f"{cfg.scenario_id}_fraction_1"
    assert set(calls) == {control_name, perturbed_name}

    control_frame = calls[control_name]["frame"]
    perturbed_frame = calls[perturbed_name]["frame"]

    assert len(control_frame) == len(perturbed_frame) == len(frame)
    assert set(control_frame[SCHEMA.id_col]) == set(frame[SCHEMA.id_col])

    # the rebuild layer must receive the WEAKENED frame, not the original
    assert not control_frame[TARGET_COL].equals(perturbed_frame[TARGET_COL])
    assert _reuse(perturbed_frame)["reuse_index"] < _reuse(control_frame)["reuse_index"]
    assert _reuse(control_frame) == _reuse(frame)


def test_identity_arm_serves_as_control_when_present(frame):
    stub = RecordingRebuild()
    res = se.run_entity_reuse_scenario(frame, rebuild_fn=stub)
    assert res["control_source"] == "identity_arm"
    assert res["control"]["fraction"] == 0.0
    assert len(stub.calls) == len(se.DEFAULT_REUSE_FRACTIONS)
    assert f"{se.SCENARIO_ID}_control" not in _by_name(stub)


# ----------------------------------------------------------------------
# 16 & 17. result structure and JSON safety
# ----------------------------------------------------------------------
def test_result_structure_is_complete_and_json_serializable(frame):
    stub = RecordingRebuild()
    res = se.run_entity_reuse_scenario(frame, rebuild_fn=stub)

    for key in ("schema", "scenario_id", "config", "arms", "control",
                "comparisons", "all_arms_ok", "scoring", "baseline"):
        assert key in res
    assert res["scenario_id"] == se.SCENARIO_ID
    assert res["all_arms_ok"] is True
    assert res["scoring"]["performed"] is False
    assert [a["fraction"] for a in res["arms"]] == list(se.DEFAULT_REUSE_FRACTIONS)
    assert res["config"]["entity_column"] == TARGET_COL

    for arm in res["arms"]:
        assert arm["entity_column"] == TARGET_COL
        assert arm["reuse"]["targeted_rows"] > 0
        assert arm["invariants"]["rewritten_outside_target"] == 0
        pop = arm["rebuild"]["population"]
        assert pop["transactions"] == len(frame)
        assert pop["attack_containing_candidates"] > 0

    for comp in res["comparisons"]:
        assert "population_delta" in comp
        assert "reuse_index_delta" in comp

    json.dumps(res)                      # must not need default=


def test_reduction_tracks_the_configured_fraction(frame):
    res = se.run_entity_reuse_scenario(frame, rebuild_fn=RecordingRebuild())
    deltas = {a["fraction"]: a["reuse"]["reduction"]["reuse_index_delta"]
              for a in res["arms"]}
    assert deltas[0.0] == pytest.approx(0.0)
    assert deltas[1.0] < deltas[0.25] <= 0.0
    rewritten = {a["fraction"]: a["invariants"]["entity_values_rewritten"]
                 for a in res["arms"]}
    assert rewritten[0.0] == 0
    assert rewritten[1.0] == res["arms"][0]["reuse"]["targeted_rows"]


# ----------------------------------------------------------------------
# 18. the invariant guard actually catches tampering
# ----------------------------------------------------------------------
def test_guard_detects_a_tampered_non_target_column(frame):
    tampered = frame.copy()
    tampered.loc[tampered.index[0], SCHEMA.label_col] = "1"
    with pytest.raises(se.EntityReuseScenarioError, match="modified column"):
        se.assert_only_entity_column_changed(frame, tampered,
                                             entity_column=TARGET_COL,
                                             schema=SCHEMA)


def test_guard_detects_weakening_of_an_untargeted_row(frame):
    tampered = frame.copy()
    benign_idx = tampered.index[tampered[SCHEMA.campaign_col].astype(str) == ""][0]
    tampered.loc[benign_idx, TARGET_COL] = "stray-token"
    mask = se.target_mask(frame, schema=SCHEMA)
    with pytest.raises(se.EntityReuseScenarioError, match="untargeted"):
        se.assert_only_entity_column_changed(frame, tampered,
                                             entity_column=TARGET_COL,
                                             mask=mask, schema=SCHEMA)


# ----------------------------------------------------------------------
# campaign targeting
# ----------------------------------------------------------------------
def test_campaign_values_restrict_the_targeted_rows(frame):
    cfg = se.EntityReuseScenarioConfig(fractions=(1.0,), campaign_values=("CMP-1",))
    world = se.build_entity_reuse_world(frame, fraction=1.0, config=cfg,
                                        schema=SCHEMA)
    b = frame.set_index(frame[SCHEMA.id_col])[TARGET_COL]
    a = world.frame.set_index(world.frame[SCHEMA.id_col])[TARGET_COL]
    other = frame.loc[frame[SCHEMA.campaign_col] == "CMP-2", SCHEMA.id_col]
    assert (b.loc[other].to_numpy() == a.loc[other].to_numpy()).all()
    hit = frame.loc[frame[SCHEMA.campaign_col] == "CMP-1", SCHEMA.id_col]
    assert (b.loc[hit].to_numpy() != a.loc[hit].to_numpy()).all()


# ----------------------------------------------------------------------
# opt-in: real rebuild pipeline (same pattern as the cadence suite)
# ----------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("CONFLUX_4B_INTEGRATION"),
                    reason="set CONFLUX_4B_INTEGRATION=1 to run the real pipeline")
def test_real_rebuild_pipeline_on_synthetic_frame(frame):
    cfg = se.EntityReuseScenarioConfig(fractions=(0.0, 1.0))
    res = se.run_entity_reuse_scenario(frame, config=cfg)
    assert res["all_arms_ok"] is True
    assert len(res["arms"]) == 2
