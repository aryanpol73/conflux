"""Phase 6 API tests -- transport layer only.

Detection mathematics (Phases 3-5) and the scorer artifact (Phase 5.5) have
their own suites and are not retested here. The pipeline is mocked by default
so that no test builds a graph; one opt-in test exercises the real pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from conflux.api import main as api_main
from conflux.api import schemas
from conflux.api import state as api_state
from conflux.api import websocket as api_ws

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    REPO_ROOT / "src" / "conflux" / "models" / "artifacts" / "scorer_reference_v1.json"
)
DATASET_PATH = REPO_ROOT / "data" / "raw" / "dataset_v4_final.csv"

COLUMNS = schemas.TRANSACTION_COLUMNS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sample_transaction(index: int = 0, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "transaction_id": f"TXN-{index:05d}",
        "timestamp": f"2024-03-01T10:{index % 60:02d}:00",
        "merchant_id": f"MER-{index % 7:03d}",
        "card_fingerprint": f"CARD-{index % 5:04d}",
        "bin": "411111",
        "amount": 100.0 + index,
        "device_fingerprint": f"DEV-{index % 4:04d}",
        "ip_signature": f"IP-{index % 3:04d}",
        "auth_outcome": "declined",
    }
    payload.update(overrides)
    return payload


def fake_result(n_transactions: int, n_campaigns: int = 1) -> dict[str, Any]:
    campaigns = [
        {
            "candidate_id": f"cand-{i}",
            "transaction_ids": [f"TXN-{i:05d}"],
            "score": 0.81 - 0.1 * i,
            "tier": "HIGH" if i == 0 else "MEDIUM",
            "action": "step_up" if i == 0 else "review",
            "evidence": {
                "top_signals": [
                    {"feature": "burst_rate_per_minute", "contribution": 0.44},
                    {"feature": "link_density", "contribution": 0.21},
                ]
            },
        }
        for i in range(n_campaigns)
    ]
    return {
        "status": "ok",
        "summary": {
            "n_transactions": n_transactions,
            "n_candidates": n_campaigns,
            "n_scored": n_campaigns,
            "n_high_risk": sum(1 for c in campaigns if c["tier"] == "HIGH"),
            "n_medium_risk": sum(1 for c in campaigns if c["tier"] == "MEDIUM"),
            "n_low_risk": 0,
        },
        "campaigns": campaigns,
    }


class _Sentinel:
    """Stand-in for the frozen ScorerReference; identity is what we assert on."""


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sentinel_reference() -> _Sentinel:
    return _Sentinel()


@pytest.fixture
def state(sentinel_reference: _Sentinel) -> api_state.ApiState:
    """Fresh state with an injected reference, so no artifact file is touched."""
    fresh = api_state.reset_state()
    fresh.set_scorer_reference(sentinel_reference)
    api_ws.manager.reset()
    yield fresh
    api_ws.manager.reset()
    api_state.reset_state()


@pytest.fixture
def calls() -> list[dict[str, Any]]:
    return []


@pytest.fixture
def client(
    state: api_state.ApiState,
    calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """TestClient with the pipeline mocked; records every invocation."""

    def _fake_pipeline(frame: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        calls.append({"frame": frame.copy(), "kwargs": kwargs})
        return fake_result(len(frame.index))

    monkeypatch.setattr(api_state, "run_detection_pipeline", _fake_pipeline)
    with TestClient(api_main.app) as test_client:
        yield test_client


# ===========================================================================
# A. Transaction validation
# ===========================================================================


def test_valid_transaction_accepted() -> None:
    transaction = schemas.TransactionIn(**sample_transaction(0))
    assert transaction.transaction_id == "TXN-00000"
    assert transaction.amount == pytest.approx(100.0)


def test_record_has_exact_dataset_columns() -> None:
    record = schemas.TransactionIn(**sample_transaction(0)).to_record()
    assert tuple(record) == COLUMNS


def test_timestamp_is_normalised() -> None:
    transaction = schemas.TransactionIn(**sample_transaction(0, timestamp="2024-03-01 10:05:00"))
    assert transaction.timestamp == "2024-03-01T10:05:00"


def test_unparseable_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        schemas.TransactionIn(**sample_transaction(0, timestamp="not-a-date"))


def test_blank_identifier_rejected() -> None:
    with pytest.raises(ValidationError):
        schemas.TransactionIn(**sample_transaction(0, merchant_id="   "))


@pytest.mark.parametrize("amount", [0, -5.0, "abc"])
def test_invalid_amount_rejected(amount: Any) -> None:
    with pytest.raises(ValidationError):
        schemas.TransactionIn(**sample_transaction(0, amount=amount))


def test_integer_bin_normalised_to_string() -> None:
    assert schemas.TransactionIn(**sample_transaction(0, bin=411111)).bin == "411111"


# ===========================================================================
# B. Leakage prevention
# ===========================================================================


@pytest.mark.parametrize("field", ["label", "campaign_id"])
def test_ground_truth_field_rejected_by_schema(field: str) -> None:
    payload = sample_transaction(0)
    payload[field] = 1
    with pytest.raises(ValidationError) as excinfo:
        schemas.TransactionIn(**payload)
    assert field in str(excinfo.value)


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        schemas.TransactionIn(**sample_transaction(0, card_id="nope"))


@pytest.mark.parametrize("field", ["label", "campaign_id"])
def test_ground_truth_rejected_over_rest(client: TestClient, field: str) -> None:
    payload = sample_transaction(0)
    payload[field] = 1
    assert client.post("/transactions", json=payload).status_code == 422


def test_state_refuses_ground_truth_record(state: api_state.ApiState) -> None:
    record = sample_transaction(0)
    record["label"] = 1
    with pytest.raises(ValueError, match="ground-truth"):
        state.add_transaction(record)


# ===========================================================================
# C / D. State storage and DataFrame conversion
# ===========================================================================


def test_state_stores_transaction(state: api_state.ApiState) -> None:
    stored = state.add_transaction(sample_transaction(0))
    assert state.transaction_count == 1
    assert stored["transaction_id"] == "TXN-00000"


def test_stored_transactions_are_isolated(state: api_state.ApiState) -> None:
    record = sample_transaction(0)
    state.add_transaction(record)
    record["merchant_id"] = "MUTATED"
    assert state.transactions()[0]["merchant_id"] != "MUTATED"


def test_frame_has_exact_columns(state: api_state.ApiState) -> None:
    state.add_transaction(sample_transaction(0))
    state.add_transaction(sample_transaction(1))
    frame = state.to_frame()
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame.index) == 2
    assert "label" not in frame.columns
    assert "campaign_id" not in frame.columns


def test_empty_frame_still_has_columns(state: api_state.ApiState) -> None:
    frame = state.to_frame()
    assert frame.empty
    assert list(frame.columns) == list(COLUMNS)


def test_frame_is_a_fresh_copy(state: api_state.ApiState) -> None:
    state.add_transaction(sample_transaction(0))
    first = state.to_frame()
    first.loc[0, "merchant_id"] = "MUTATED"
    assert state.to_frame().loc[0, "merchant_id"] != "MUTATED"


def test_amount_column_is_numeric(state: api_state.ApiState) -> None:
    state.add_transaction(sample_transaction(0))
    assert pd.api.types.is_numeric_dtype(state.to_frame()["amount"])


# ===========================================================================
# E. Health
# ===========================================================================


def test_health_ok(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["scorer_loaded"] is True
    assert body["transactions_in_memory"] == 0
    assert body["transaction_columns"] == list(COLUMNS)


def test_health_counts_transactions(client: TestClient) -> None:
    client.post("/transactions", json=sample_transaction(0))
    client.post("/transactions", json=sample_transaction(1))
    assert client.get("/health").json()["transactions_in_memory"] == 2


def test_health_reports_missing_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh = api_state.reset_state()
    monkeypatch.setattr(fresh, "_artifact_path", Path("/nonexistent/scorer.json"))
    try:
        with TestClient(api_main.app) as degraded:
            body = degraded.get("/health").json()
            assert body["scorer_loaded"] is False
            assert body["status"] == "degraded"
            response = degraded.get("/campaigns")
            assert response.status_code == 503
            assert response.json()["error"] == "scorer_unavailable"
    finally:
        api_state.reset_state()


# ===========================================================================
# REST detection surface
# ===========================================================================


def test_campaigns_returns_pipeline_shape(client: TestClient) -> None:
    client.post("/transactions", json=sample_transaction(0))
    body = client.get("/campaigns").json()
    assert body["status"] == "ok"
    assert set(body["summary"]) >= {
        "n_transactions",
        "n_candidates",
        "n_scored",
        "n_high_risk",
        "n_medium_risk",
        "n_low_risk",
    }
    campaign = body["campaigns"][0]
    assert set(campaign) >= {
        "candidate_id",
        "transaction_ids",
        "score",
        "tier",
        "action",
        "evidence",
    }
    assert "campaign_id" not in campaign
    json.dumps(body)


def test_rest_ingest_returns_population_result_not_a_row_score(
    client: TestClient,
) -> None:
    body = client.post("/transactions", json=sample_transaction(0)).json()
    assert set(body) == {"status", "summary", "campaigns"}
    assert "transaction_score" not in body


# ===========================================================================
# F. WebSocket happy path
# ===========================================================================


def test_ws_connection_ack(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ack = ws.receive_json()
    assert ack["type"] == "connection_ack"
    assert ack["data"]["scorer_loaded"] is True
    assert ack["data"]["transaction_columns"] == list(COLUMNS)


def test_ws_transaction_returns_detection_update(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ack
        ws.send_json({"type": "transaction", "data": sample_transaction(0)})
        message = ws.receive_json()
    assert message["type"] == "detection_update"
    assert message["data"]["status"] == "ok"
    assert message["data"]["summary"]["n_transactions"] == 1


def test_ws_ping_pong(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


def test_ws_snapshot_does_not_ingest(
    client: TestClient, state: api_state.ApiState
) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "snapshot"})
        message = ws.receive_json()
    assert message["type"] == "detection_update"
    assert state.transaction_count == 0


def test_ws_accumulates_population(
    client: TestClient, calls: list[dict[str, Any]]
) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for index in range(3):
            ws.send_json({"type": "transaction", "data": sample_transaction(index)})
            ws.receive_json()
    assert [len(call["frame"].index) for call in calls] == [1, 2, 3]


# ===========================================================================
# G. WebSocket error handling (connection must survive)
# ===========================================================================


def test_ws_malformed_json(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text("{not json")
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "invalid_json"
        # still alive
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_ws_non_object_payload(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text("[1, 2, 3]")
        assert ws.receive_json()["code"] == "invalid_message"


def test_ws_unknown_message_type(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "launch_missiles"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "unknown_message_type"


def test_ws_invalid_transaction_then_recovery(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "transaction", "data": {"transaction_id": "T1"}})
        error = ws.receive_json()
        assert error["code"] == "invalid_transaction"
        assert error["detail"]

        ws.send_json({"type": "transaction", "data": sample_transaction(0)})
        assert ws.receive_json()["type"] == "detection_update"


@pytest.mark.parametrize("field", ["label", "campaign_id"])
def test_ws_rejects_ground_truth(client: TestClient, field: str) -> None:
    payload = sample_transaction(0)
    payload[field] = 1
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "transaction", "data": payload})
        error = ws.receive_json()
    assert error["type"] == "error"
    assert error["code"] == "invalid_transaction"


def test_ws_pipeline_failure_does_not_close_socket(
    state: api_state.ApiState, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(frame: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("graph construction exploded")

    monkeypatch.setattr(api_state, "run_detection_pipeline", _boom)
    with TestClient(api_main.app) as failing:
        with failing.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "transaction", "data": sample_transaction(0)})
            error = ws.receive_json()
            assert error["code"] == "detection_failed"
            assert "graph construction exploded" in error["message"]
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"


# ===========================================================================
# H. Pipeline integration contract
# ===========================================================================


def test_pipeline_called_with_frame_and_frozen_reference(
    client: TestClient,
    state: api_state.ApiState,
    sentinel_reference: _Sentinel,
    calls: list[dict[str, Any]],
) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "transaction", "data": sample_transaction(0)})
        ws.receive_json()

    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call["frame"], pd.DataFrame)
    assert list(call["frame"].columns) == list(COLUMNS)
    assert call["kwargs"]["scorer_reference"] is sentinel_reference


def test_reference_is_loaded_once(
    state: api_state.ApiState, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loads: list[Path] = []
    artifact = tmp_path / "scorer_reference_v1.json"
    artifact.write_text("{}", encoding="utf-8")

    def _fake_load(path: Path) -> _Sentinel:
        loads.append(Path(path))
        return _Sentinel()

    monkeypatch.setattr(api_state, "load_scorer_reference", _fake_load)
    fresh = api_state.ApiState(artifact_path=artifact)
    first = fresh.ensure_scorer_loaded()
    second = fresh.ensure_scorer_loaded()
    assert first is second
    assert len(loads) == 1


def test_no_fallback_when_artifact_missing(tmp_path: Path) -> None:
    fresh = api_state.ApiState(artifact_path=tmp_path / "absent.json")
    with pytest.raises(api_state.ScorerUnavailableError):
        fresh.ensure_scorer_loaded()
    assert fresh.scorer_loaded is False


# ===========================================================================
# Connection manager fan-out (unit-tested directly; TestClient runs each
# websocket session in its own event loop, which cannot exercise fan-out)
# ===========================================================================


def test_broadcast_reaches_other_clients_and_survives_a_dead_peer() -> None:
    class FakeSocket:
        def __init__(self, fail: bool = False) -> None:
            self.received: list[dict[str, Any]] = []
            self.fail = fail

        async def accept(self) -> None:
            return None

        async def send_json(self, payload: dict[str, Any]) -> None:
            if self.fail:
                raise RuntimeError("peer gone")
            self.received.append(payload)

    async def scenario() -> tuple[FakeSocket, FakeSocket, FakeSocket, int]:
        local = api_ws.ConnectionManager()
        sender, listener, broken = FakeSocket(), FakeSocket(), FakeSocket(fail=True)
        for socket in (sender, listener, broken):
            await local.connect(socket)  # type: ignore[arg-type]
        await local.broadcast({"type": "detection_update"}, exclude=sender)  # type: ignore[arg-type]
        return sender, listener, broken, local.count

    sender, listener, broken, remaining = asyncio.run(scenario())
    assert sender.received == []
    assert listener.received == [{"type": "detection_update"}]
    assert broken.received == []
    assert remaining == 2  # the dead peer was dropped


# ===========================================================================
# Opt-in real-pipeline smoke test
# ===========================================================================


@pytest.mark.skipif(
    os.environ.get("CONFLUX_RUN_API_INTEGRATION") != "1",
    reason="set CONFLUX_RUN_API_INTEGRATION=1 to run the real-pipeline test",
)
@pytest.mark.skipif(
    not ARTIFACT_PATH.is_file() or not DATASET_PATH.is_file(),
    reason="scorer artifact or raw dataset not present",
)
def test_real_pipeline_over_websocket() -> None:
    rows = pd.read_csv(DATASET_PATH, nrows=40)
    rows = rows.drop(columns=["label", "campaign_id"], errors="ignore")
    records = json.loads(rows.to_json(orient="records"))

    api_state.reset_state()
    api_ws.manager.reset()
    try:
        with TestClient(api_main.app) as real_client:
            assert real_client.get("/health").json()["scorer_loaded"] is True
            with real_client.websocket_connect("/ws") as ws:
                ws.receive_json()
                for record in records:
                    ws.send_json({"type": "transaction", "data": record})
                    message = ws.receive_json()
                    assert message["type"] == "detection_update", message
            body = real_client.get("/campaigns").json()
            assert body["status"] == "ok"
            assert body["summary"]["n_transactions"] == len(records)
            assert body["summary"]["n_scored"] <= body["summary"]["n_candidates"]
            json.dumps(body)
    finally:
        api_ws.manager.reset()
        api_state.reset_state()
