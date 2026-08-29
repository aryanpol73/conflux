"""Phase 4B S1 -- cadence scenario tests. Schema-driven, deterministic.

The cadence primitive is NEVER mocked. Only the rebuild boundary is replaced,
by a recording stub that returns a genuine RebuiltWorld dataclass, so metric
collection is exercised for real without needing the production dataset.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conflux.robustness import scenario_cadence as sc
from conflux.robustness.perturbations import (
    PerturbationError, resolve_schema, scale_group_cadence, timestamps_as_ns,
)
from conflux.robustness.rebuild import (
    FEAT_CAMPAIGN_COL, FEAT_ID_COL, FEAT_LABEL_COL, RebuiltWorld,
)
from conflux.robustness.world import linking_entity_columns, world_frame_columns

SCHEMA = resolve_schema()
TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


# ----------------------------------------------------------------------
# synthetic fixture -- conforms to the resolved schema, no hard-coded names
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


def _ns(frame: pd.DataFrame) -> np.ndarray:
    return timestamps_as_ns(frame, schema=SCHEMA)


def _span_by_group(frame: pd.DataFrame) -> dict[str, float]:
    return {g: p["span_seconds"]
            for g, p in sc.cadence_profile(frame, schema=SCHEMA).items()}


# ----------------------------------------------------------------------
# 1. defaults
# ----------------------------------------------------------------------
def test_default_factor_grid():
    assert sc.DEFAULT_CADENCE_FACTORS == (0.5, 1.0, 2.0)
    assert sc.CadenceScenarioConfig().factors == (0.5, 1.0, 2.0)


# ----------------------------------------------------------------------
# 2. invalid factors
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad", [(), (0.0,), (-1.0,), (0.5, -2.0), (float("nan"),)])
def test_invalid_factors_rejected(bad):
    with pytest.raises(sc.CadenceScenarioError):
        sc.CadenceScenarioConfig(factors=bad)


def test_primitive_also_rejects_non_positive_factor(frame):
    with pytest.raises(PerturbationError):
        scale_group_cadence(frame, factor=0.0, schema=SCHEMA)


# ----------------------------------------------------------------------
# 3. invalid anchors
# ----------------------------------------------------------------------
def test_invalid_anchor_rejected_by_config():
    with pytest.raises(sc.CadenceScenarioError, match="anchor"):
        sc.CadenceScenarioConfig(anchor="middle")


def test_invalid_anchor_rejected_by_primitive(frame):
    with pytest.raises(PerturbationError, match="anchor"):
        scale_group_cadence(frame, factor=2.0, anchor="middle", schema=SCHEMA)


# ----------------------------------------------------------------------
# 4 & 5. faster / slower cadence
# ----------------------------------------------------------------------
@pytest.mark.parametrize("factor", [0.25, 0.5])
def test_faster_cadence_compresses_groups(frame, factor):
    world = sc.build_cadence_world(frame, factor=factor, schema=SCHEMA)
    before, after = _span_by_group(frame), _span_by_group(world.frame)
    assert before and set(before) == set(after)
    for g, b in before.items():
        assert after[g] < b
        assert after[g] == pytest.approx(b * factor, rel=1e-6)


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_slower_cadence_stretches_groups(frame, factor):
    world = sc.build_cadence_world(frame, factor=factor, schema=SCHEMA)
    before, after = _span_by_group(frame), _span_by_group(world.frame)
    for g, b in before.items():
        assert after[g] > b
        assert after[g] == pytest.approx(b * factor, rel=1e-6)


def test_benign_rows_are_not_rescaled(frame):
    """Only campaign groups move; background traffic keeps its timing."""
    world = sc.build_cadence_world(frame, factor=3.0, schema=SCHEMA)
    benign = frame[SCHEMA.campaign_col].astype(str) == ""
    assert benign.any()
    b = frame.set_index(frame[SCHEMA.id_col])[SCHEMA.ts_col]
    a = world.frame.set_index(world.frame[SCHEMA.id_col])[SCHEMA.ts_col]
    ids = frame.loc[benign, SCHEMA.id_col]
    assert (b.loc[ids].to_numpy() == a.loc[ids].to_numpy()).all()


# ----------------------------------------------------------------------
# 6. non-temporal invariants
# ----------------------------------------------------------------------
def test_non_temporal_columns_are_untouched(frame):
    original = frame.copy(deep=True)
    world = sc.build_cadence_world(frame, factor=2.0, schema=SCHEMA)
    out = world.frame

    report = sc.assert_only_timestamps_changed(frame, out, schema=SCHEMA)
    assert report["rows"] == len(frame)
    assert report["timestamps_changed"] > 0

    assert len(out) == len(frame)
    assert list(out.columns) == list(frame.columns)
    b = frame.set_index(frame[SCHEMA.id_col].astype(str)).sort_index()
    a = out.set_index(out[SCHEMA.id_col].astype(str)).sort_index()
    assert list(b.index) == list(a.index)
    checked = [SCHEMA.label_col, SCHEMA.campaign_col, "amount", "auth_outcome",
               *linking_entity_columns()]
    for col in checked:
        assert (b[col].astype(str).to_numpy() == a[col].astype(str).to_numpy()).all()

    # input frame was not mutated
    pd.testing.assert_frame_equal(frame, original)


def test_invariant_guard_detects_a_tampered_column(frame):
    tampered = frame.copy()
    tampered.loc[tampered.index[0], "amount"] = 999999.0
    with pytest.raises(sc.CadenceScenarioError, match="non-temporal"):
        sc.assert_only_timestamps_changed(frame, tampered, schema=SCHEMA)


# ----------------------------------------------------------------------
# 7 & 8. timestamps change; identity arm does not
# ----------------------------------------------------------------------
def test_timestamps_change_for_non_identity_factor(frame):
    world = sc.build_cadence_world(frame, factor=2.0, schema=SCHEMA)
    assert (_ns(frame) != _ns(world.frame)).any()


def test_identity_factor_preserves_timestamps_semantically(frame):
    world = sc.build_cadence_world(frame, factor=1.0, schema=SCHEMA)
    assert (_ns(frame) == _ns(world.frame)).all()
    assert _span_by_group(frame) == pytest.approx(_span_by_group(world.frame))


# ----------------------------------------------------------------------
# 9. determinism
# ----------------------------------------------------------------------
def test_scenario_is_deterministic(frame):
    a = sc.run_cadence_scenario(frame, rebuild_fn=RecordingRebuild())
    b = sc.run_cadence_scenario(frame, rebuild_fn=RecordingRebuild())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_world_fingerprints_are_reproducible(frame):
    one = sc.build_cadence_world(frame, factor=0.5, schema=SCHEMA)
    two = sc.build_cadence_world(frame, factor=0.5, schema=SCHEMA)
    assert one.fingerprint == two.fingerprint


# ----------------------------------------------------------------------
# 10. result structure
# ----------------------------------------------------------------------
def test_result_structure_is_complete_and_json_serializable(frame):
    stub = RecordingRebuild()
    res = sc.run_cadence_scenario(frame, rebuild_fn=stub)

    for key in ("schema", "scenario_id", "config", "arms", "control",
                "comparisons", "all_arms_ok", "scoring", "baseline"):
        assert key in res
    assert res["all_arms_ok"] is True
    assert res["scoring"]["performed"] is False
    assert [a["factor"] for a in res["arms"]] == [0.5, 1.0, 2.0]
    assert res["control_source"] == "identity_arm"
    assert res["control"]["factor"] == 1.0

    for arm in res["arms"]:
        pop = arm["rebuild"]["population"]
        assert pop["transactions"] == len(frame)
        assert pop["multi_transaction_candidates"] > 0
        assert pop["attack_containing_candidates"] > 0
        assert pop["distinct_campaigns_represented"] > 0

    for comp in res["comparisons"]:
        assert "population_delta" in comp
        assert "multi_transaction_candidates" in comp["population_delta"]

    json.dumps(res)                      # must not need default=
    assert len(stub.calls) == 3


def test_perturbed_frame_reaches_the_rebuild_layer(frame):
    stub = RecordingRebuild()
    cfg = sc.CadenceScenarioConfig(factors=(4.0,))
    sc.run_cadence_scenario(frame, config=cfg, rebuild_fn=stub)

    # One control rebuild + one perturbed rebuild.
        # One perturbed rebuild + one control rebuild.
    assert len(stub.calls) == 2

    calls_by_name = {call["name"]: call for call in stub.calls}

    assert "S1_cadence_factor_4" in calls_by_name
    assert "S1_cadence_control" in calls_by_name

    control_call = calls_by_name["S1_cadence_control"]
    perturbed_call = calls_by_name["S1_cadence_factor_4"]

    # The perturbed timestamps must actually reach the rebuild layer.
    assert not control_call["frame"]["timestamp"].equals(
        perturbed_call["frame"]["timestamp"]
    )
    assert "4" in perturbed_call["name"]

    control_frame = control_call["frame"]
    perturbed_frame = perturbed_call["frame"]

    # Same transaction population reaches both rebuilds.
    assert len(control_frame) == len(perturbed_frame)

    # Perturbed arm must actually differ temporally.
    assert not control_frame["timestamp"].equals(
        perturbed_frame["timestamp"]
    )
    passed = stub.calls[0]["frame"]
    assert len(passed) == len(frame)
    assert set(passed[SCHEMA.id_col]) == set(frame[SCHEMA.id_col])
    assert (_ns(passed) != _ns(frame)).any()
    for g, span in _span_by_group(frame).items():
        assert _span_by_group(passed)[g] == pytest.approx(span * 4.0, rel=1e-6)


def test_median_span_ratio_tracks_the_factor(frame):
    res = sc.run_cadence_scenario(frame, rebuild_fn=RecordingRebuild())
    ratios = {a["factor"]: a["cadence"]["median_span_ratio"] for a in res["arms"]}
    assert ratios[0.5] == pytest.approx(0.5, rel=1e-6)
    assert ratios[1.0] == pytest.approx(1.0, rel=1e-6)
    assert ratios[2.0] == pytest.approx(2.0, rel=1e-6)


# ----------------------------------------------------------------------
# opt-in: real rebuild pipeline
# ----------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("CONFLUX_4B_INTEGRATION"),
                    reason="set CONFLUX_4B_INTEGRATION=1 to run the real pipeline")
def test_real_rebuild_pipeline_on_synthetic_frame(frame):
    res = sc.run_cadence_scenario(frame,
                                  config=sc.CadenceScenarioConfig(factors=(1.0, 2.0)))
    assert res["all_arms_ok"] is True
    assert len(res["arms"]) == 2
