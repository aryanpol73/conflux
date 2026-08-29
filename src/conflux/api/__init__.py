"""CONFLUX Phase 6 -- real-time API layer.

A thin transport wrapper around the existing detection pipeline
(:func:`conflux.pipeline.run_detection_pipeline`) and the frozen Phase 5.5
``ScorerReference`` artifact.

This package contains **no** detection logic: no graph construction, no
candidate generation, no feature engineering, no scoring, no tier or action
policy. It ingests transactions, keeps them in memory, hands the current
population to the pipeline, and ships the pipeline's own output back over
WebSocket (primary) or REST (fallback).

Honest scope note: detection is re-run over the whole current in-memory
population on each ingest. The pipeline is not incremental, and WebSocket does
not make scoring faster -- it removes polling latency and keeps the dashboard
live.

Imports here are lazy so that ``import conflux.api`` does not pull in FastAPI
unless the application is actually being used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "app",
    "create_app",
    "ApiState",
    "get_state",
    "reset_state",
    "run_detection",
    "TransactionIn",
    "TRANSACTION_COLUMNS",
    "FORBIDDEN_COLUMNS",
]

_LAZY: dict[str, tuple[str, str]] = {
    "app": ("conflux.api.main", "app"),
    "create_app": ("conflux.api.main", "create_app"),
    "ApiState": ("conflux.api.state", "ApiState"),
    "get_state": ("conflux.api.state", "get_state"),
    "reset_state": ("conflux.api.state", "reset_state"),
    "run_detection": ("conflux.api.state", "run_detection"),
    "TransactionIn": ("conflux.api.schemas", "TransactionIn"),
    "TRANSACTION_COLUMNS": ("conflux.api.schemas", "TRANSACTION_COLUMNS"),
    "FORBIDDEN_COLUMNS": ("conflux.api.schemas", "FORBIDDEN_COLUMNS"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial delegation
    try:
        module_name, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    return getattr(import_module(module_name), attribute)


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(__all__)
