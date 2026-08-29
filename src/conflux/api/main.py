"""FastAPI application for CONFLUX.

WebSocket ``/ws`` is the primary live channel. The REST surface is deliberately
tiny: ``/health`` for status, ``/campaigns`` for initial dashboard load and
debugging, and ``POST /transactions`` as a fallback ingest path. No auth, no
rate limiting, no batch endpoints, no CRUD.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from conflux.api.schemas import (
    TRANSACTION_COLUMNS,
    DetectionResult,
    HealthResponse,
    TransactionIn,
)
from conflux.api.state import (
    DetectionError,
    ScorerUnavailableError,
    get_state,
    run_detection,
)
from conflux.api.websocket import manager
from conflux.api.websocket import router as websocket_router

LOGGER = logging.getLogger(__name__)

__all__ = ["app", "create_app"]


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the frozen scorer artifact once, at startup."""
    state = get_state()
    try:
        state.ensure_scorer_loaded()
        LOGGER.info("CONFLUX API ready; scorer artifact %s", state.artifact_path)
    except ScorerUnavailableError as exc:
        # Do not crash the process: /health must stay reachable so an operator
        # can see exactly what is wrong. Detection endpoints return 503, and we
        # never fall back to fitting a scorer.
        LOGGER.error("scorer reference unavailable at startup: %s", exc)
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="CONFLUX Detection API",
        version="6.0.0",
        summary="Real-time campaign-detection API over the CONFLUX pipeline.",
        lifespan=lifespan,
    )

    # The PWA is served from a different origin during the demo.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(websocket_router)

    @application.exception_handler(ScorerUnavailableError)
    async def _scorer_unavailable(
        request: Request, exc: ScorerUnavailableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "scorer_unavailable", "detail": str(exc)},
        )

    @application.exception_handler(DetectionError)
    async def _detection_failed(request: Request, exc: DetectionError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "detection_failed", "detail": str(exc)},
        )

    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> dict[str, Any]:
        snapshot = get_state().snapshot()
        return {
            "status": "ok" if snapshot["scorer_loaded"] else "degraded",
            "scorer_loaded": snapshot["scorer_loaded"],
            "transactions_in_memory": snapshot["transactions_in_memory"],
            "active_websocket_clients": manager.count,
            "scorer_artifact_path": snapshot["scorer_artifact_path"],
            "transaction_columns": snapshot["transaction_columns"],
            "load_error": snapshot["load_error"],
        }

    @application.get("/campaigns", response_model=DetectionResult, tags=["detection"])
    async def campaigns() -> dict[str, Any]:
        """Detection over the current in-memory population (initial load / fallback)."""
        return await run_in_threadpool(run_detection, get_state())

    @application.post(
        "/transactions",
        response_model=DetectionResult,
        status_code=status.HTTP_200_OK,
        tags=["detection"],
    )
    async def ingest_transaction(transaction: TransactionIn) -> dict[str, Any]:
        """Fallback ingest. WebSocket /ws is the primary live path.

        Returns a population-level detection result, never a per-transaction
        score: CONFLUX scores coordinated campaigns, not isolated transactions.
        """
        state = get_state()
        try:
            state.add_transaction(transaction.to_record())
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        return await run_in_threadpool(run_detection, state)

    return application


app = create_app()

# Exposed for debugging; the API never mutates this.
API_TRANSACTION_COLUMNS = TRANSACTION_COLUMNS
