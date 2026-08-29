"""Phase 5 detection pipeline tests."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import conflux.pipeline as pl
from conflux.pipeline import (
    DEFAULT_ACTIONS,
    PipelineError,
    RiskThresholds,
    classify_tier,
    run_detection_pipeline,
    select_action,
    top_signals,
)
from conflux.scoring.deterministic_scorer import DeterministicScorer

ROWS = 60
BASE_TS = pd.Timestamp("2024-05-01T00:00:00Z")
ENTITY_HINTS = ("card_fingerprint", "bin", "device_fingerprint", "ip_signature")


# ---------------------------------------------------------------------------
# synthetic population (module-level so the determinism subprocess can reuse it)
# ---------------------------------------------------------------------------


def synthetic_frame(n: int = ROWS) -> pd.DataFrame:
    """Small clustered population: shared entities so candidates actually form."""
    import conflux.robustness.perturbations as p

    schema = p.resolve_schema()

    def col(*attrs, fallback):
        for attr in attrs:
            value = getattr(schema, attr, None)
            if isinstance(value, str) and value:
                return value
        return fallback

    id_col = col("id_column", "id_col", "transaction_id_column", fallback="transaction_id")
    ts_col = col("timestamp_column", "timestamp_col", "ts_column", fallback="timestamp")
    label_col = col("label_column", "label_col", fallback="label")
    campaign_col = col("campaign_column", "campaign_id_column", fallback="campaign_id")

    ts = [BASE_TS + pd.Timedelta(seconds=120 * i) for i in range(n)]
    try:
        stamps = list(p.ns_to_timestamp_strings(np.array([t.value for t in ts], dtype=np.int64)))
    except Exception:  # pragma: no cover
        stamps = [t.strftime(p.TS_FORMAT) for t in ts]

    cluster = [i % 6 for i in range(n)]  # 6 tight entity clusters -> real candidates
    data = {
        id_col: [f"TX{i:05d}" for i in range(n)],
        ts_col: stamps,
        label_col: [1 if c < 2 else 0 for c in cluster],
        campaign_col: [f"CMP{c:03d}" if c < 2 else "" for c in cluster],
        "amount": np.round(np.linspace(10.0, 800.0, n), 2),
    }
    for k, hint in enumerate(ENTITY_HINTS):
        data[hint] = [f"{hint[:3].upper()}-{cluster[i]:03d}" for i in range(n)]

    df = pd.DataFrame(data)
    if "merchant_id" not in df.columns:
        df["merchant_id"] = [
            f"MERCHANT-{cluster[i]:03d}" for i in range(n)
        ]
    for extra in ("entity_columns", "entity_cols"):
        names = getattr(schema, extra, None)
        if isinstance(names, (list, tuple)):
            for name in names:
                if isinstance(name, str) and name and name not in df.columns:
                    df[name] = [f"ENT-{cluster[i]:03d}" for i in range(n)]
    return df


def fit_reference(frame: pd.DataFrame):
    """Fit a reference through the SAME stages inference uses (column agreement)."""
    graph = pl.build_graph(frame)
    cset = pl.generate_candidates(graph)
    candidates = cset.candidate_frame()
    assignments = cset.assignments
    features = pl.build_features(
        frame=frame,
        graph=graph,
        cset=cset,
        candidates=candidates,
        assignments=assignments,
        min_size=2,
    )
    table = pl.features_frame(features)
    names = pl.feature_names(features, table)
    try:
        return DeterministicScorer.fit(table, names)
    except TypeError:
        return DeterministicScorer().fit(table, names)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_frame()


@pytest.fixture(scope="module")
def reference(frame):
    return fit_reference(frame)


@pytest.fixture(scope="module")
def result(frame, reference):
    return run_detection_pipeline(frame, scorer_reference=reference)


# ---------------------------------------------------------------------------
# 1. tier boundaries (pure function)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [(0.0, "LOW"), (0.39, "LOW"), (0.3999, "LOW"), (0.40, "MEDIUM"),
     (0.69, "MEDIUM"), (0.6999, "MEDIUM"), (0.70, "HIGH"), (1.0, "HIGH")],
)
def test_tier_boundaries(score, expected):
    assert classify_tier(score, RiskThresholds()) == expected


def test_tiers_configurable():
    strict = RiskThresholds(medium=0.20, high=0.50)
    assert classify_tier(0.30, strict) == "MEDIUM"
    assert classify_tier(0.55, strict) == "HIGH"


def test_thresholds_reject_inverted():
    with pytest.raises(ValueError):
        RiskThresholds(medium=0.9, high=0.1)


# ---------------------------------------------------------------------------
# 2-3. non-finite scores
# ---------------------------------------------------------------------------


def test_nan_score_raises():
    with pytest.raises(ValueError):
        classify_tier(float("nan"), RiskThresholds())


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), np.float64("nan")])
def test_infinite_score_raises(bad):
    with pytest.raises(ValueError):
        classify_tier(bad, RiskThresholds())


def test_non_numeric_score_raises_type_error():
    with pytest.raises(TypeError):
        classify_tier("0.5", RiskThresholds())


# ---------------------------------------------------------------------------
# 4. action mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tier,action", [("LOW", "flag"), ("MEDIUM", "review"), ("HIGH", "block")]
)
def test_action_mapping(tier, action):
    assert select_action(tier) == action
    assert DEFAULT_ACTIONS[tier] == action


def test_action_mapping_is_configurable():
    assert select_action("HIGH", {"HIGH": "step_up"}) == "step_up"


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        select_action("CRITICAL")


# ---------------------------------------------------------------------------
# 5-6. input contracts
# ---------------------------------------------------------------------------


def test_empty_frame_returns_ok(frame, reference):
    empty = frame.iloc[0:0].copy()
    out = run_detection_pipeline(empty, scorer_reference=reference)
    assert out["status"] == "ok"
    assert out["campaigns"] == []
    assert all(v == 0 for v in out["summary"].values())


def test_missing_columns_raise_value_error(frame, reference):
    required = pl.resolve_required_columns()
    if not required:
        pytest.fail("schema exposed no required columns to test against")
    broken = frame.drop(columns=[required[0]])
    with pytest.raises(ValueError) as exc:
        run_detection_pipeline(broken, scorer_reference=reference)
    assert required[0] in str(exc.value)


def test_non_dataframe_raises_type_error(reference):
    with pytest.raises(TypeError):
        run_detection_pipeline([1, 2, 3], scorer_reference=reference)


def test_missing_reference_raises(frame):
    with pytest.raises(ValueError):
        run_detection_pipeline(frame, scorer_reference=None)


def test_column_validation_precedes_emptiness(reference):
    with pytest.raises(ValueError):
        run_detection_pipeline(pd.DataFrame(), scorer_reference=reference)


# ---------------------------------------------------------------------------
# 7. immutability
# ---------------------------------------------------------------------------


def test_input_frame_unchanged(frame, reference):
    snapshot = frame.copy(deep=True)
    run_detection_pipeline(frame, scorer_reference=reference)
    pd.testing.assert_frame_equal(frame, snapshot, check_dtype=True)


# ---------------------------------------------------------------------------
# 8-9. counts and serialization
# ---------------------------------------------------------------------------


def test_n_scored_not_greater_than_candidates(result):
    s = result["summary"]
    assert s["n_scored"] <= s["n_candidates"]
    assert s["n_transactions"] == ROWS


def test_tier_counts_sum_to_scored(result):
    s = result["summary"]
    assert s["n_low_risk"] + s["n_medium_risk"] + s["n_high_risk"] == s["n_scored"]


def test_counts_are_derived_not_hardcoded(frame, reference):
    small = run_detection_pipeline(frame.iloc[:24].copy(), scorer_reference=reference)
    assert small["summary"]["n_transactions"] == 24


def test_json_serializable_without_default(result):
    text = json.dumps(result)
    assert json.loads(text)["status"] == "ok"


def test_no_numpy_scalars_leak(result):
    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        else:
            assert not isinstance(node, (np.generic, pd.Timestamp)), repr(node)

    walk(result)


# ---------------------------------------------------------------------------
# 10-11. deterministic ordering
# ---------------------------------------------------------------------------


def test_campaign_ordering(result):
    keys = [(-c["score"], str(c["candidate_id"])) for c in result["campaigns"]]
    assert keys == sorted(keys)


def test_repeated_runs_identical(frame, reference):
    a = run_detection_pipeline(frame, scorer_reference=reference)
    b = run_detection_pipeline(frame, scorer_reference=reference)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_top_signal_ordering_absolute_desc():
    row = {"contrib_a": 0.1, "contrib_b": -0.9, "contrib_c": 0.5, "contrib_d": 0.5}
    signals = top_signals(row, limit=4)
    assert [s["feature"] for s in signals] == ["b", "c", "d", "a"]
    assert signals[0]["contribution"] == -0.9  # signed value preserved


def test_top_signal_cap():
    row = {f"contrib_f{i}": float(i) for i in range(12)}
    assert len(top_signals(row, limit=3)) == 3
    assert top_signals(row, limit=0) == []


def test_top_signals_skip_non_finite():
    row = {"contrib_a": float("nan"), "contrib_b": 0.4, "contrib_c": float("inf")}
    assert [s["feature"] for s in top_signals(row)] == ["b"]


def test_pipeline_signals_ordered_and_capped(frame, reference):
    out = run_detection_pipeline(frame, scorer_reference=reference, top_n_signals=3)
    scored = [c for c in out["campaigns"] if c["evidence"]["top_signals"]]
    if not scored:
        pytest.fail("no campaign produced contribution signals")
    for campaign in scored:
        signals = campaign["evidence"]["top_signals"]
        assert len(signals) <= 3
        keys = [(-abs(s["contribution"]), s["feature"]) for s in signals]
        assert keys == sorted(keys)
        assert all(isinstance(s["contribution"], float) for s in signals)


# ---------------------------------------------------------------------------
# 12. no refit
# ---------------------------------------------------------------------------


def test_pipeline_never_refits(frame, reference, monkeypatch):
    # reference is built by the module-scoped fixture BEFORE this patch applies
    def explode(*args, **kwargs):
        raise AssertionError("DeterministicScorer.fit called during inference")

    monkeypatch.setattr(DeterministicScorer, "fit", staticmethod(explode))
    out = run_detection_pipeline(frame, scorer_reference=reference)
    assert out["status"] == "ok"


def test_pipeline_module_does_not_reference_fit():
    import inspect as _inspect

    source = _inspect.getsource(pl)
    body = "\n".join(
        line for line in source.splitlines() if "never fits" not in line.lower()
    )
    assert ".fit(" not in body


# ---------------------------------------------------------------------------
# 13. real component binding / integration
# ---------------------------------------------------------------------------


def test_real_components_are_bound():
    from conflux.graph.build_candidates import form_campaign_candidates as real_fcc
    from conflux.scoring.candidate_features import (
        build_scoring_features as real_bsf,
        load_structural_attributes as real_lsa,
    )

    assert pl.form_campaign_candidates is real_fcc
    assert pl.build_scoring_features is real_bsf
    assert pl.load_structural_attributes is real_lsa
    assert pl.DeterministicScorer.__module__.endswith("deterministic_scorer")


def test_candidate_features_builder_not_used():
    import inspect as _inspect

    assert "build_candidate_features" not in _inspect.getsource(pl)


def test_real_integration_smoke(result):
    assert result["status"] == "ok"
    assert result["summary"]["n_candidates"] > 0, "no candidates formed from fixture"
    if result["summary"]["n_scored"] == 0:
        pytest.fail("candidates formed but none survived min_size=2 scoring")
    for campaign in result["campaigns"]:
        assert set(campaign) >= {
            "candidate_id", "score", "tier", "action", "evidence", "transaction_ids"
        }
        assert campaign["tier"] in pl.TIERS
        assert campaign["action"] == DEFAULT_ACTIONS[campaign["tier"]]
        assert math.isfinite(campaign["score"])


def test_min_size_reduces_scored_set(frame, reference):
    loose = run_detection_pipeline(frame, scorer_reference=reference, min_size=1)
    strict = run_detection_pipeline(frame, scorer_reference=reference, min_size=3)
    assert strict["summary"]["n_scored"] <= loose["summary"]["n_scored"]


# ---------------------------------------------------------------------------
# cross-process determinism
# ---------------------------------------------------------------------------

_CHILD = textwrap.dedent(
    """
    import importlib.util, json, sys
    spec = importlib.util.spec_from_file_location("_p5", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frame = mod.synthetic_frame()
    reference = mod.fit_reference(frame)
    result = mod.run_detection_pipeline(frame, scorer_reference=reference)
    sys.stdout.write(json.dumps(result, sort_keys=True))
    """
)


def _run_child(seed: str, tmp_path: Path) -> str:
    script = tmp_path / f"child_{seed}.py"
    script.write_text(_CHILD, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH", "")]))
    proc = subprocess.run(
        [sys.executable, str(script), str(Path(__file__).resolve())],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(f"child (PYTHONHASHSEED={seed}) failed:\n{proc.stderr}")
    return proc.stdout


def test_cross_process_determinism(tmp_path):
    a = _run_child("0", tmp_path)
    b = _run_child("12345", tmp_path)
    if a != b:
        offset = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        pytest.fail(
            f"hash-order nondeterminism at offset {offset}:\n"
            f"  seed0    : ...{a[max(0, offset - 60):offset + 60]}...\n"
            f"  seed12345: ...{b[max(0, offset - 60):offset + 60]}..."
        )
