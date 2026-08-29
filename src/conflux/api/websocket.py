"""Real-time WebSocket endpoint -- the primary live channel for the PWA.

Protocol
--------
On connect the server sends ``connection_ack``. Thereafter the client may send:

``{"type": "transaction", "data": {...}}``
    Validate, store, re-run detection over the whole population, reply with
    ``detection_update`` and broadcast the same update to every other client.

``{"type": "snapshot"}``
    Re-run detection without ingesting; reply with ``detection_update``.

``{"type": "ping"}``
    Reply ``{"type": "pong"}``.

Anything else -- malformed JSON, wrong shape, invalid transaction, ground-truth
leakage, pipeline failure -- produces a structured ``error`` message and the
connection stays open.

WebSocket does not make scoring faster. It removes polling, keeps one warm
connection open, and lets the dashboard update the moment detection finishes.
The pipeline computation is identical to the REST path.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from conflux.api.schemas import (
    CLIENT_MESSAGE_ADAPTER,
    PingMessage,
    SnapshotMessage,
    TransactionMessage,
    connection_ack_message,
    detection_update_message,
    error_message,
    format_validation_error,
    pong_message,
)
from conflux.api.state import (
    ApiState,
    DetectionError,
    ScorerUnavailableError,
    get_state,
    run_detection,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter()

__all__ = ["router", "manager", "ConnectionManager", "handle_client_payload"]


class ConnectionManager:
    """Tracks live sockets so detection updates can fan out to every dashboard."""

    def __init__(self) -> None:
        self._active: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def count(self) -> int:
        return len(self._active)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._active.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._active.discard(websocket)

    async def broadcast(
        self, payload: dict[str, Any], *, exclude: WebSocket | None = None
    ) -> None:
        """Best-effort fan-out. A dead peer is dropped, never fatal."""
        async with self._lock:
            targets = [ws for ws in self._active if ws is not exclude]
        dead: list[WebSocket] = []
        for target in targets:
            try:
                await target.send_json(payload)
            except Exception:  # noqa: BLE001 - a broken peer must not break us
                dead.append(target)
        if dead:
            async with self._lock:
                for target in dead:
                    self._active.discard(target)

    def reset(self) -> None:
        """Drop all tracked sockets without closing them (test helper)."""
        self._active.clear()


manager = ConnectionManager()


async def _run_detection_async(state: ApiState) -> dict[str, Any]:
    """Run the (synchronous, CPU-bound) pipeline off the event loop."""
    return await run_in_threadpool(run_detection, state)


async def handle_client_payload(raw: str, state: ApiState) -> dict[str, Any]:
    """Turn one raw client frame into exactly one server message.

    Pure with respect to the socket: it never sends anything itself, which
    makes it directly unit-testable.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return error_message(
            f"message is not valid JSON: {exc.msg}", code="invalid_json"
        )

    if not isinstance(payload, dict):
        return error_message(
            "message must be a JSON object with a 'type' field",
            code="invalid_message",
        )

    try:
        message = CLIENT_MESSAGE_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        received = payload.get("type")
        code = (
            "unknown_message_type"
            if received not in {"transaction", "ping", "snapshot"}
            else "invalid_transaction"
        )
        summary = (
            f"unsupported message type {received!r}; expected one of "
            "'transaction', 'ping', 'snapshot'"
            if code == "unknown_message_type"
            else "transaction failed validation"
        )
        return error_message(summary, code=code, detail=format_validation_error(exc))

    if isinstance(message, PingMessage):
        return pong_message()

    if isinstance(message, TransactionMessage):
        try:
            state.add_transaction(message.data.to_record())
        except ValueError as exc:
            return error_message(str(exc), code="invalid_transaction")
    elif not isinstance(message, SnapshotMessage):  # pragma: no cover - defensive
        return error_message("unhandled message type", code="unknown_message_type")

    try:
        result = await _run_detection_async(state)
    except ScorerUnavailableError as exc:
        return error_message(str(exc), code="scorer_unavailable")
    except DetectionError as exc:
        return error_message(f"detection failed: {exc}", code="detection_failed")

    return detection_update_message(result)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    state = get_state()
    await manager.connect(websocket)
    try:
        await websocket.send_json(
            connection_ack_message(
                transactions_in_memory=state.transaction_count,
                scorer_loaded=state.scorer_loaded,
            )
        )

        while True:
            frame = await websocket.receive()
            if frame.get("type") == "websocket.disconnect":
                break

            raw = frame.get("text")
            if raw is None:
                data = frame.get("bytes")
                if data is None:
                    await websocket.send_json(
                        error_message("empty frame received", code="invalid_message")
                    )
                    continue
                try:
                    raw = data.decode("utf-8")
                except UnicodeDecodeError:
                    await websocket.send_json(
                        error_message(
                            "binary frame is not valid UTF-8 text",
                            code="invalid_message",
                        )
                    )
                    continue

            response = await handle_client_payload(raw, state)
            await websocket.send_json(response)

            # Keep every other dashboard in sync without polling.
            if response.get("type") == "detection_update":
                await manager.broadcast(response, exclude=websocket)

    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - one bad socket must not kill the server
        LOGGER.exception("websocket connection terminated abnormally")
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001 - already gone
            pass
    finally:
        await manager.disconnect(websocket)
