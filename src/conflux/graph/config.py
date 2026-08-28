"""CONFLUX graph layer -- configuration.

Windows and structural rules live here, never inside query functions
(AI_WORKING_RULES §9). This module intentionally does NOT modify
src/conflux/config.py, which belongs to the frozen feature/ML layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# entity_type -> source column in dataset_v4_final.csv (verified against the file)
ENTITY_COLUMNS: dict[str, str] = {
    "card": "card_fingerprint",
    "bin": "bin",
    "device": "device_fingerprint",
    "ip": "ip_signature",
    "merchant": "merchant_id",
}

ID_COL = "transaction_id"
TS_COL = "timestamp"

# Structural inputs allowed into graph construction.
STRUCTURAL_COLUMNS: tuple[str, ...] = (ID_COL, TS_COL, *ENTITY_COLUMNS.values())

# Transaction attributes carried as node metadata for later campaign evidence.
# These are NOT used to build any edge.
ATTRIBUTE_COLUMNS: tuple[str, ...] = ("amount", "auth_outcome")

# Ground truth. Never read during graph construction, never stored on a node or edge.
FORBIDDEN_GRAPH_INPUTS: tuple[str, ...] = ("label", "campaign_id", "_source_type")

# LOCKED STRUCTURAL RULE: BIN is a graph entity, but it may never act as a
# cross-transaction connectivity mechanism. Not configurable.
BLOCKED_CONNECTIVITY_ENTITY_TYPES: tuple[str, ...] = ("bin",)


class GraphConfigError(ValueError):
    """Raised when a configuration violates a locked structural rule."""


@dataclass(frozen=True)
class GraphConfig:
    """Graph-layer configuration.

    campaign_window_seconds
        Default temporal window for relationship queries. 3600 s is the project
        default; every query accepts an override, and no window value is
        hard-coded anywhere else in this package.

    connectivity_entity_types
        Entity types that may link two transactions together. Card, device and
        IP only, by default.

        merchant is excluded by default: the dataset spreads 31,873 transactions
        over ~400 merchants, so merchant co-occurrence is ambient, not evidence.
        Cross-merchant spread is the attack signature, which makes merchant a
        CONTEXT node. This default is overridable; BIN is not.

    context_entity_types
        Entity types represented in the graph and queryable on their own, but
        never used to join transactions in connectivity queries.
    """

    campaign_window_seconds: float = 3600.0
    connectivity_entity_types: tuple[str, ...] = ("card", "device", "ip")
    context_entity_types: tuple[str, ...] = ("bin", "merchant")
    blocked_connectivity_entity_types: tuple[str, ...] = field(
        default=BLOCKED_CONNECTIVITY_ENTITY_TYPES
    )

    def __post_init__(self) -> None:
        if self.campaign_window_seconds <= 0:
            raise GraphConfigError("campaign_window_seconds must be positive")

        known = set(ENTITY_COLUMNS)
        unknown = [t for t in (*self.connectivity_entity_types,
                               *self.context_entity_types) if t not in known]
        if unknown:
            raise GraphConfigError(f"unknown entity type(s): {unknown}")

        violation = sorted(set(self.connectivity_entity_types)
                           & set(self.blocked_connectivity_entity_types))
        if violation:
            raise GraphConfigError(
                f"entity type(s) {violation} may never be used for campaign "
                "connectivity. BIN is issuer context only (DECISIONS.md / graph "
                "task spec); it must not create cross-transaction components."
            )
        if not self.connectivity_entity_types:
            raise GraphConfigError("at least one connectivity entity type is required")

        overlap = sorted(set(self.connectivity_entity_types)
                         & set(self.context_entity_types))
        if overlap:
            raise GraphConfigError(f"entity type(s) {overlap} declared both "
                                   "connectivity and context")

    @property
    def entity_types(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.connectivity_entity_types)
                            | set(self.context_entity_types)))

    @property
    def campaign_window_ns(self) -> int:
        return int(round(self.campaign_window_seconds * 1_000_000_000))


DEFAULT_GRAPH_CONFIG = GraphConfig()
