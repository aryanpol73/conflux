"""Pydantic schemas for the CONFLUX API and WebSocket protocol.

This is the *only* schema module in the API package. Transaction fields mirror
``data/raw/dataset_v4_final.csv`` exactly, minus the two ground-truth columns.

Accepted production columns
---------------------------
transaction_id, timestamp, merchant_id, card_fingerprint, bin, amount,
device_fingerprint, ip_signature, auth_outcome

Never accepted
--------------
``label`` and ``campaign_id`` are evaluation-only ground truth. They are
rejected with an explicit error rather than silently dropped, so that a
misconfigured producer fails loudly instead of leaking labels into inference.
``extra="forbid"`` additionally rejects any field the dataset does not define,
which prevents invented column names from reaching the pipeline.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Annotated, Any, Literal, Union

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

__all__ = [
    "TRANSACTION_COLUMNS",
    "FORBIDDEN_COLUMNS",
    "TransactionIn",
    "TransactionMessage",
    "PingMessage",
    "SnapshotMessage",
    "ClientMessage",
    "CLIENT_MESSAGE_ADAPTER",
    "TopSignal",
    "CampaignEvidence",
    "Campaign",
    "DetectionSummary",
    "DetectionResult",
    "HealthResponse",
    "detection_update_message",
    "pong_message",
    "connection_ack_message",
    "error_message",
    "format_validation_error",
]

#: Exact production transaction columns, in dataset order.
TRANSACTION_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "timestamp",
    "merchant_id",
    "card_fingerprint",
    "bin",
    "amount",
    "device_fingerprint",
    "ip_signature",
    "auth_outcome",
)

#: Ground-truth columns that must never enter ingestion or inference.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset({"label", "campaign_id"})

_IDENTIFIER_FIELDS: tuple[str, ...] = (
    "transaction_id",
    "merchant_id",
    "card_fingerprint",
    "bin",
    "device_fingerprint",
    "ip_signature",
    "auth_outcome",
)


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------


class TransactionIn(BaseModel):
    """One production transaction.

    ``bin`` is normalised to a string. The dataset column is categorical (a
    card BIN is an identifier, not a quantity) and pandas would otherwise infer
    ``int64`` from JSON integers and ``object`` from JSON strings, giving two
    different grouping keys for the same value depending on the producer.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transaction_id: str
    timestamp: str
    merchant_id: str
    card_fingerprint: str
    bin: str
    amount: float
    device_fingerprint: str
    ip_signature: str
    auth_outcome: str

    @model_validator(mode="before")
    @classmethod
    def _reject_ground_truth(cls, data: Any) -> Any:
        if isinstance(data, dict):
            leaked = sorted(FORBIDDEN_COLUMNS.intersection(data))
            if leaked:
                raise ValueError(
                    "ground-truth field(s) "
                    + ", ".join(repr(name) for name in leaked)
                    + " are not accepted by the production API; they are "
                    "evaluation-only and must never reach inference"
                )
        return data

    @field_validator("bin", mode="before")
    @classmethod
    def _coerce_bin(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("'bin' must be a string or an integer, not a boolean")
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError(f"'bin' is not a valid identifier: {value!r}")
            return str(int(value))
        return value

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_timestamp(cls, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, pd.Timestamp):  # pragma: no cover - defensive
            return value.isoformat()
        return value

    @field_validator(*_IDENTIFIER_FIELDS)
    @classmethod
    def _not_blank(cls, value: str, info: Any) -> str:
        text = value.strip()
        if not text:
            raise ValueError(f"'{info.field_name}' must not be blank")
        return text

    @field_validator("timestamp")
    @classmethod
    def _parse_timestamp(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("'timestamp' must not be blank")
        parsed = pd.to_datetime(text, errors="coerce")
        if parsed is pd.NaT or pd.isna(parsed):
            raise ValueError(f"'timestamp' is not a parseable datetime: {value!r}")
        # Normalised to ISO-8601 so that every stored row shares one textual
        # format; the pipeline owns any further datetime handling.
        return parsed.isoformat()

    @field_validator("amount")
    @classmethod
    def _positive_amount(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("'amount' must be a finite number")
        if value <= 0:
            raise ValueError("'amount' must be strictly positive")
        return float(value)

    def to_record(self) -> dict[str, Any]:
        """Return a plain dict in dataset column order."""
        return {name: getattr(self, name) for name in TRANSACTION_COLUMNS}


# ---------------------------------------------------------------------------
# WebSocket: client -> server
# ---------------------------------------------------------------------------


class TransactionMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["transaction"]
    data: TransactionIn


class PingMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ping"]


class SnapshotMessage(BaseModel):
    """Ask for a detection run over the current population without ingesting."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["snapshot"]


ClientMessage = Annotated[
    Union[TransactionMessage, PingMessage, SnapshotMessage],
    Field(discriminator="type"),
]

CLIENT_MESSAGE_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


# ---------------------------------------------------------------------------
# Pipeline output mirrors (permissive: never drop or rename pipeline fields)
# ---------------------------------------------------------------------------


class _Permissive(BaseModel):
    model_config = ConfigDict(extra="allow")


class TopSignal(_Permissive):
    feature: str
    contribution: float | None = None


class CampaignEvidence(_Permissive):
    top_signals: list[TopSignal] = Field(default_factory=list)


class Campaign(_Permissive):
    candidate_id: str | int
    transaction_ids: list[str | int] = Field(default_factory=list)
    score: float | None = None
    tier: str | None = None
    action: str | None = None
    evidence: CampaignEvidence = Field(default_factory=CampaignEvidence)


class DetectionSummary(_Permissive):
    n_transactions: int = 0
    n_candidates: int = 0
    n_scored: int = 0
    n_high_risk: int = 0
    n_medium_risk: int = 0
    n_low_risk: int = 0


class DetectionResult(_Permissive):
    status: str
    summary: DetectionSummary = Field(default_factory=DetectionSummary)
    campaigns: list[Campaign] = Field(default_factory=list)


class HealthResponse(_Permissive):
    status: str
    scorer_loaded: bool
    transactions_in_memory: int
    active_websocket_clients: int = 0
    scorer_artifact_path: str | None = None
    transaction_columns: list[str] = Field(default_factory=list)
    load_error: str | None = None


# ---------------------------------------------------------------------------
# Server -> client message builders
# ---------------------------------------------------------------------------


def detection_update_message(result: dict[str, Any]) -> dict[str, Any]:
    return {"type": "detection_update", "data": result}


def pong_message() -> dict[str, Any]:
    return {"type": "pong"}


def connection_ack_message(
    *, transactions_in_memory: int, scorer_loaded: bool
) -> dict[str, Any]:
    return {
        "type": "connection_ack",
        "data": {
            "transactions_in_memory": transactions_in_memory,
            "scorer_loaded": scorer_loaded,
            "transaction_columns": list(TRANSACTION_COLUMNS),
            "accepted_message_types": ["transaction", "ping", "snapshot"],
        },
    }


def error_message(
    message: str, *, code: str = "error", detail: Any = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "error", "code": code, "message": message}
    if detail is not None:
        payload["detail"] = detail
    return payload


def format_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    """Compact, JSON-safe rendering of a pydantic validation failure."""
    compact: list[dict[str, Any]] = []
    for error in exc.errors():
        compact.append(
            {
                "field": ".".join(str(part) for part in error.get("loc", ())),
                "message": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
        )
    return compact
