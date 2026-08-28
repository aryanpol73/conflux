"""CONFLUX graph layer -- Phase 3B: temporal/entity campaign candidate formation.

SCOPE (locked to this phase): candidate formation ONLY.
No campaign risk score. No ML probability integration. No weights. No metrics.
No frontend. The next phase evaluates what this module produces.

WHAT THIS MODULE DOES
---------------------
1. For every transaction it asks the already-verified TemporalEntityGraph for the
   causally prior transactions that share a CONNECTIVITY entity (card / device /
   ip) inside the configured temporal window. All timestamp and window logic is
   delegated to TemporalEntityGraph.temporal_neighbors(mode="causal"); none of it
   is re-implemented here.
2. Those anchor -> prior pairs form a causal link table with ONE ROW PER
   CONNECTED TRANSACTION PAIR. A link answers "are these two transactions
   connected"; the shared entity types and their ids ride on that same row as
   the evidence answering "why". A pair sharing card + device + ip is one link
   carrying three pieces of evidence, not three links. Use
   CandidateSet.explode_links() for the long, one-row-per-entity-type view.
3. Candidate groups are the connected components of that causal link graph
   (union-find). A group is therefore, by construction, held together only by
   card/device/ip co-occurrence inside the window.
4. Transactions with no causal link in either direction become singleton
   ("isolated") candidates. Nothing is dropped: every transaction lands in
   exactly one candidate.

CAUSALITY
---------
Every link is produced with mode="causal", so the prior transaction is strictly
earlier than the anchor in the graph's deterministic (timestamp, transaction_id)
order and never later in wall-clock time. Equal-timestamp pairs are still
captured exactly once, from the perspective of the tie-break-later transaction.
A group's decision time is defined as its latest member timestamp; no member can
postdate it.

GROUND TRUTH
------------
label / campaign_id / _source_type never enter this module. The graph refuses to
load them, and the outputs are asserted free of them.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ENTITY_COLUMNS, FORBIDDEN_GRAPH_INPUTS, ID_COL, TS_COL
from .temporal_graph import BinConnectivityError, TemporalEntityGraph

log = logging.getLogger("conflux.graph.campaign_detection")

# v2: links collapsed to one row per connected transaction pair (was one row
# per shared entity type in v1). Artifacts written by v1 are not compatible.
CANDIDATE_SCHEMA_VERSION = "conflux.graph.campaign_candidates.v2"

# LOCKED STRUCTURAL RULE (PROJECT_CONTEXT / DECISIONS / Phase 3B task spec):
# neither BIN nor Merchant may ever act as the link between two transactions.
# BIN is issuer context. Merchant co-occurrence is ambient (~400 merchants) and
# cross-merchant spread is the attack signature, not the join key.
BLOCKED_CANDIDATE_LINK_TYPES: tuple[str, ...] = ("bin", "merchant")

# One row per connected PAIR. Per-entity evidence is appended as
# shares_<type> / <type>_entity_id columns, generated from the resolved config.
BASE_LINK_COLUMNS: tuple[str, ...] = (
    "anchor_transaction_id", "prior_transaction_id",
    "link_entity_types", "n_link_entity_types",
    "anchor_ts_ns", "prior_ts_ns", "delta_seconds", "age_seconds",
    "anchor_pos", "prior_pos",
)
_BASE_LINK_DTYPES: dict[str, str] = {
    "anchor_transaction_id": "object", "prior_transaction_id": "object",
    "link_entity_types": "object", "n_link_entity_types": "int64",
    "anchor_ts_ns": "int64", "prior_ts_ns": "int64",
    "delta_seconds": "float64", "age_seconds": "float64",
    "anchor_pos": "int64", "prior_pos": "int64",
}

EXPLODED_LINK_COLUMNS: tuple[str, ...] = (
    "anchor_transaction_id", "prior_transaction_id", "link_entity_type",
    "link_entity_id", "anchor_ts_ns", "prior_ts_ns", "delta_seconds",
    "age_seconds", "anchor_pos", "prior_pos",
)


def link_columns(entity_types: Sequence[str]) -> list[str]:
    """Pair-level link table columns for a given connectivity entity set."""
    cols = list(BASE_LINK_COLUMNS)
    for et in entity_types:
        cols += [f"shares_{et}", f"{et}_entity_id"]
    return cols


def link_dtypes(entity_types: Sequence[str]) -> dict[str, str]:
    d = dict(_BASE_LINK_DTYPES)
    for et in entity_types:
        d[f"shares_{et}"] = "bool"
        d[f"{et}_entity_id"] = "object"
    return d


class CandidateConfigError(ValueError):
    """Configuration violates a locked candidate-formation rule."""


class ContextEntityConnectivityError(CandidateConfigError):
    """A context-only entity type was requested as a connectivity mechanism."""


# ----------------------------------------------------------------------
# configuration (AI_WORKING_RULES §9: windows/thresholds live in config)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateConfig:
    """Phase 3B configuration.

    window_seconds
        Causal relationship window. None means "inherit the graph's configured
        campaign_window_seconds" (project default 3600 s). No window value is
        hard-coded in any function in this module.

    connectivity_entity_types
        Entity types allowed to join two transactions. None means "inherit
        graph.config.connectivity_entity_types" (card, device, ip). bin and
        merchant are refused here regardless of how they were supplied.

    context_entity_types
        Entity types recorded as candidate evidence but never used to join.
    """

    window_seconds: float | None = None
    connectivity_entity_types: tuple[str, ...] | None = None
    context_entity_types: tuple[str, ...] = ("bin", "merchant")
    include_singletons: bool = True
    candidate_id_prefix: str = "CAND"

    def resolve(self, graph: TemporalEntityGraph) -> "ResolvedCandidateConfig":
        window = (float(graph.config.campaign_window_seconds)
                  if self.window_seconds is None else float(self.window_seconds))
        if window <= 0:
            raise CandidateConfigError("window_seconds must be positive")

        types = tuple(graph.config.connectivity_entity_types
                      if self.connectivity_entity_types is None
                      else self.connectivity_entity_types)
        if not types:
            raise CandidateConfigError("at least one connectivity entity type is required")

        unknown = [t for t in types if t not in graph.config.entity_types]
        if unknown:
            raise CandidateConfigError(f"unknown entity type(s): {unknown}")

        blocked = tuple(sorted(set(types) & set(BLOCKED_CANDIDATE_LINK_TYPES)))
        if blocked:
            msg = (f"entity type(s) {list(blocked)} may never join two transactions "
                   "into a campaign candidate. BIN is issuer context; Merchant "
                   "co-occurrence is ambient and cross-merchant spread is the "
                   "signature, not the join key.")
            if "bin" in blocked:
                raise BinConnectivityError(msg)
            raise ContextEntityConnectivityError(msg)

        context = tuple(t for t in self.context_entity_types
                        if t in graph.config.entity_types)
        return ResolvedCandidateConfig(
            window_seconds=window,
            window_ns=int(round(window * 1_000_000_000)),
            connectivity_entity_types=tuple(types),
            context_entity_types=context,
            include_singletons=self.include_singletons,
            candidate_id_prefix=self.candidate_id_prefix,
        )


@dataclass(frozen=True)
class ResolvedCandidateConfig:
    window_seconds: float
    window_ns: int
    connectivity_entity_types: tuple[str, ...]
    context_entity_types: tuple[str, ...]
    include_singletons: bool
    candidate_id_prefix: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_seconds": self.window_seconds,
            "window_ns": self.window_ns,
            "connectivity_entity_types": list(self.connectivity_entity_types),
            "context_entity_types": list(self.context_entity_types),
            "blocked_link_entity_types": list(BLOCKED_CANDIDATE_LINK_TYPES),
            "include_singletons": self.include_singletons,
            "link_granularity": "one_row_per_connected_transaction_pair",
        }


DEFAULT_CANDIDATE_CONFIG = CandidateConfig()


# ----------------------------------------------------------------------
# candidate objects
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CampaignCandidate:
    """One temporally/structurally connected candidate group. No score."""

    candidate_id: str
    transaction_ids: tuple[str, ...]          # causal order
    size: int
    is_isolated: bool

    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    first_ts_ns: int
    last_ts_ns: int
    decision_ts_ns: int                       # == last_ts_ns
    time_span_seconds: float

    link_edge_count: int                      # distinct connected transaction PAIRS
    multi_entity_link_count: int              # pairs backed by >= 2 entity types
    link_entity_types: tuple[str, ...]
    link_counts: dict[str, int]               # pairs whose evidence includes type
    links: tuple[dict[str, Any], ...]         # full pair-level evidence

    distinct_cards: int
    distinct_devices: int
    distinct_ips: int
    distinct_merchants: int
    card_ids: tuple[str, ...]
    device_ids: tuple[str, ...]
    ip_ids: tuple[str, ...]
    merchant_ids: tuple[str, ...]

    shared_entities: dict[str, dict[str, int]]   # link type -> entity id -> txn count
    device_overlap: dict[str, Any]
    ip_overlap: dict[str, Any]
    bin_context: dict[str, Any]                  # CONTEXT ONLY - never a join key

    def as_row(self) -> dict[str, Any]:
        """Flat, CSV-safe projection. Lists are pipe-joined."""
        return {
            "candidate_id": self.candidate_id,
            "size": self.size,
            "is_isolated": self.is_isolated,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "time_span_seconds": self.time_span_seconds,
            "link_edge_count": self.link_edge_count,
            "links_multi_entity": self.multi_entity_link_count,
            "link_entity_types": "|".join(self.link_entity_types),
            "links_card": self.link_counts.get("card", 0),
            "links_device": self.link_counts.get("device", 0),
            "links_ip": self.link_counts.get("ip", 0),
            "distinct_cards": self.distinct_cards,
            "distinct_devices": self.distinct_devices,
            "distinct_ips": self.distinct_ips,
            "distinct_merchants": self.distinct_merchants,
            "shared_card_ids": "|".join(sorted(self.shared_entities.get("card", {}))),
            "shared_device_ids": "|".join(sorted(self.shared_entities.get("device", {}))),
            "shared_ip_ids": "|".join(sorted(self.shared_entities.get("ip", {}))),
            "max_transactions_per_shared_device":
                self.device_overlap.get("max_transactions_per_device", 0),
            "max_transactions_per_shared_ip":
                self.ip_overlap.get("max_transactions_per_ip", 0),
            "distinct_bins": self.bin_context.get("distinct_bins", 0),
            "bin_ids_context": "|".join(self.bin_context.get("bin_ids", ())),
            "transaction_ids": "|".join(self.transaction_ids),
        }


@dataclass
class CandidateSet:
    """Everything Phase 3B produces. Inspectable, no score, no ground truth."""

    schema_version: str
    config: ResolvedCandidateConfig
    candidates: tuple[CampaignCandidate, ...]
    links: pd.DataFrame                        # one row per connected pair
    assignments: pd.DataFrame
    n_transactions: int
    _by_id: dict[str, CampaignCandidate] = field(default_factory=dict, repr=False)
    _links_by_anchor: dict[str, list[dict[str, Any]]] = field(default_factory=dict, repr=False)

    # -- lookups -------------------------------------------------------
    def candidate(self, candidate_id: str) -> CampaignCandidate:
        return self._by_id[candidate_id]

    def candidate_of(self, transaction_id: str) -> CampaignCandidate:
        row = self.assignments.loc[
            self.assignments[ID_COL] == transaction_id, "candidate_id"]
        if row.empty:
            raise KeyError(f"unknown transaction_id: {transaction_id}")
        return self._by_id[row.iloc[0]]

    def causal_candidate(self, transaction_id: str) -> dict[str, Any]:
        """Anchor-level view: the anchor plus ONLY its causally prior connected
        transactions. This is the decision-time object; it can never contain a
        transaction later than the anchor."""
        links = self._links_by_anchor.get(transaction_id, [])
        priors = sorted({(l["prior_ts_ns"], l["prior_transaction_id"]) for l in links})
        anchor_ts = int(self.assignments.loc[
            self.assignments[ID_COL] == transaction_id, "ts_ns"].iloc[0])
        return {
            "anchor_transaction_id": transaction_id,
            "anchor_ts_ns": anchor_ts,
            "decision_ts_ns": anchor_ts,
            "prior_transaction_ids": tuple(t for _, t in priors),
            "n_prior": len(priors),
            "links": tuple(links),
            "window_seconds": self.config.window_seconds,
            "connectivity_entity_types": list(self.config.connectivity_entity_types),
        }

    # -- projections ---------------------------------------------------
    def explode_links(self) -> pd.DataFrame:
        """Long view: one row per (connected pair, shared entity type).

        Recovers the per-entity-type representation from the pair-level table
        without making it the primary shape.
        """
        cols = list(EXPLODED_LINK_COLUMNS)
        if self.links.empty:
            return pd.DataFrame(columns=cols)
        frames = []
        for et in self.config.connectivity_entity_types:
            sub = self.links.loc[self.links[f"shares_{et}"]].copy()
            if sub.empty:
                continue
            sub["link_entity_type"] = et
            sub["link_entity_id"] = sub[f"{et}_entity_id"]
            frames.append(sub[cols])
        if not frames:
            return pd.DataFrame(columns=cols)
        return (pd.concat(frames, ignore_index=True)
                  .sort_values(["anchor_pos", "prior_pos", "link_entity_type"],
                               kind="mergesort")
                  .reset_index(drop=True))

    def candidate_frame(self) -> pd.DataFrame:
        rows = [c.as_row() for c in self.candidates]
        return pd.DataFrame(rows).sort_values("candidate_id",
                                              kind="mergesort").reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        sizes = np.array([c.size for c in self.candidates], dtype=np.int64)
        multi = sizes > 1
        spans = np.array([c.time_span_seconds for c in self.candidates
                          if c.size > 1], dtype=np.float64)
        has_links = len(self.links) > 0
        return {
            "schema_version": self.schema_version,
            "config": self.config.as_dict(),
            "transactions": int(self.n_transactions),
            "transactions_assigned": int(self.assignments[ID_COL].nunique()),
            "candidates_total": int(len(self.candidates)),
            "candidates_multi_transaction": int(multi.sum()),
            "candidates_isolated": int((~multi).sum()),
            "transactions_in_multi_transaction_candidates":
                int(sizes[multi].sum()) if multi.any() else 0,
            "largest_candidate_size": int(sizes.max()) if sizes.size else 0,
            "median_multi_candidate_size":
                float(np.median(sizes[multi])) if multi.any() else 0.0,
            "causal_links_total": int(len(self.links)),          # connected pairs
            "causal_links_by_entity_type": {
                et: int(self.links[f"shares_{et}"].sum()) if has_links else 0
                for et in self.config.connectivity_entity_types},
            "causal_links_multi_entity":
                int((self.links["n_link_entity_types"] >= 2).sum()) if has_links else 0,
            "multi_candidate_span_seconds_max": float(spans.max()) if spans.size else 0.0,
            "multi_candidate_span_seconds_median":
                float(np.median(spans)) if spans.size else 0.0,
            "scope_note": ("Candidate formation only. No campaign risk score, no ML "
                           "integration, no weights, no campaign-level metrics."),
        }


# ----------------------------------------------------------------------
# union-find
# ----------------------------------------------------------------------
class _DisjointSet:
    __slots__ = ("_parent",)

    def __init__(self, n: int) -> None:
        self._parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self._parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:      # path compression
            p[x], x = root, p[x]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # deterministic: the smaller position always becomes the root
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        self._parent[hi] = lo

    def roots(self) -> np.ndarray:
        return np.fromiter((self.find(i) for i in range(len(self._parent))),
                           dtype=np.int64, count=len(self._parent))


# ----------------------------------------------------------------------
# step 1: causal connectivity
# ----------------------------------------------------------------------
def build_causal_links(graph: TemporalEntityGraph,
                       config: ResolvedCandidateConfig) -> pd.DataFrame:
    """One row per (anchor, causally prior) transaction PAIR.

    All shared connectivity entities for that pair are recorded on the row.
    All window/causality logic remains inside graph.temporal_neighbors.
    """
    ids = graph.transactions[ID_COL].to_numpy()
    ts_ns = graph.transactions["ts_ns"].to_numpy(dtype=np.int64)
    pos_of = graph.position
    types = config.connectivity_entity_types

    records: list[dict[str, Any]] = []
    for pos in range(len(ids)):
        tid = ids[pos]
        neighbours = graph.temporal_neighbors(
            tid,
            window_seconds=config.window_seconds,
            entity_types=list(types),
            mode="causal",
            as_frame=False,
        )
        if not neighbours:
            continue
        anchor_ts = int(ts_ns[pos])

        # collapse the per-entity rows the graph returns into one per pair
        merged: dict[str, dict[str, Any]] = {}
        for r in neighbours:
            prior_id = r["neighbor_transaction_id"]
            acc = merged.get(prior_id)
            if acc is None:
                acc = merged[prior_id] = {
                    "prior_ts_ns": int(r["neighbor_ts_ns"]),
                    "delta_seconds": float(r["delta_seconds"]),
                    "entity_ids": {},
                }
            # one entity column per type in this schema, so first wins
            acc["entity_ids"].setdefault(r["link_entity_type"], r["link_entity_id"])

        for prior_id, acc in merged.items():
            eids = acc["entity_ids"]
            present = tuple(et for et in types if et in eids)   # canonical order
            prior_ts = acc["prior_ts_ns"]
            rec: dict[str, Any] = {
                "anchor_transaction_id": tid,
                "prior_transaction_id": prior_id,
                "link_entity_types": "|".join(present),
                "n_link_entity_types": len(present),
                "anchor_ts_ns": anchor_ts,
                "prior_ts_ns": prior_ts,
                "delta_seconds": acc["delta_seconds"],
                "age_seconds": (anchor_ts - prior_ts) / 1e9,
                "anchor_pos": pos,
                "prior_pos": pos_of(prior_id),
            }
            for et in types:
                rec[f"shares_{et}"] = et in eids
                rec[f"{et}_entity_id"] = eids.get(et, "")
            records.append(rec)

    cols = link_columns(types)
    links = pd.DataFrame(records, columns=cols).astype(link_dtypes(types))
    if not links.empty:
        # (anchor_pos, prior_pos) is now a unique key -> total, stable order
        links = (links.sort_values(["anchor_pos", "prior_pos"], kind="mergesort")
                      .reset_index(drop=True))
    log.info("causal links: %s connected pairs over %s transactions "
             "(window=%ss, types=%s)",
             len(links), len(ids), config.window_seconds, list(types))
    return links


# ----------------------------------------------------------------------
# step 2: candidate formation
# ----------------------------------------------------------------------
def form_campaign_candidates(
        graph: TemporalEntityGraph,
        config: CandidateConfig = DEFAULT_CANDIDATE_CONFIG) -> CandidateSet:
    """Phase 3B entry point. Deterministic. Nothing is dropped. No score."""
    resolved = config.resolve(graph)

    leaked = [c for c in FORBIDDEN_GRAPH_INPUTS if c in graph.transactions.columns]
    if leaked:
        raise ValueError(
            f"ground-truth column(s) {leaked} present on the graph; candidate "
            "construction refuses to run with label/campaign_id in scope.")

    links = build_causal_links(graph, resolved)

    n = graph.n_transactions
    dsu = _DisjointSet(n)
    if not links.empty:
        for a, b in zip(links["anchor_pos"].to_numpy(), links["prior_pos"].to_numpy()):
            dsu.union(int(a), int(b))
    roots = dsu.roots()

    txns = graph.transactions
    ids = txns[ID_COL].to_numpy()
    ts_ns = txns["ts_ns"].to_numpy(dtype=np.int64)
    ts_val = txns[TS_COL].to_numpy()
    ent_arrays = {et: txns[ENTITY_COLUMNS[et]].astype(str).to_numpy()
                  for et in graph.config.entity_types}

    # deterministic group order: by the earliest member position
    order = np.argsort(roots, kind="mergesort")
    grouped: dict[int, list[int]] = {}
    for pos in order:
        grouped.setdefault(int(roots[pos]), []).append(int(pos))
    root_sequence = sorted(grouped, key=lambda r: min(grouped[r]))

    links_by_root: dict[int, list[dict[str, Any]]] = {}
    if not links.empty:
        for rec in links.to_dict("records"):
            links_by_root.setdefault(int(roots[int(rec["anchor_pos"])]), []).append(rec)

    candidates: list[CampaignCandidate] = []
    assign_rows: list[tuple] = []
    width = max(6, len(str(len(root_sequence))))

    for i, root in enumerate(root_sequence, start=1):
        members = sorted(grouped[root])          # already causal order
        member_ids = tuple(ids[p] for p in members)
        member_ts = ts_ns[members]
        size = len(members)
        isolated = size == 1

        group_links = sorted(links_by_root.get(root, []),
                             key=lambda r: (r["anchor_pos"], r["prior_pos"]))
        link_counts: Counter[str] = Counter()
        for l in group_links:
            for et in resolved.connectivity_entity_types:
                if l.get(f"shares_{et}"):
                    link_counts[et] += 1
        multi_entity_links = sum(1 for l in group_links
                                 if int(l["n_link_entity_types"]) >= 2)

        shared: dict[str, dict[str, int]] = {}
        for et in resolved.connectivity_entity_types:
            counts = Counter(ent_arrays[et][p] for p in members)
            shared[et] = {k: int(v) for k, v in sorted(counts.items()) if v >= 2}

        def _ids_of(et: str) -> tuple[str, ...]:
            return tuple(sorted({ent_arrays[et][p] for p in members})) \
                if et in ent_arrays else ()

        bin_counts = (Counter(ent_arrays["bin"][p] for p in members)
                      if "bin" in ent_arrays else Counter())
        dev_shared = shared.get("device", {})
        ip_shared = shared.get("ip", {})

        candidate_id = f"{resolved.candidate_id_prefix}-{i:0{width}d}"
        candidates.append(CampaignCandidate(
            candidate_id=candidate_id,
            transaction_ids=member_ids,
            size=size,
            is_isolated=isolated,
            first_timestamp=pd.Timestamp(ts_val[members[0]]),
            last_timestamp=pd.Timestamp(ts_val[members[-1]]),
            first_ts_ns=int(member_ts[0]),
            last_ts_ns=int(member_ts[-1]),
            decision_ts_ns=int(member_ts[-1]),
            time_span_seconds=float((int(member_ts[-1]) - int(member_ts[0])) / 1e9),
            link_edge_count=len(group_links),
            multi_entity_link_count=multi_entity_links,
            link_entity_types=tuple(sorted(link_counts)),
            link_counts={k: int(v) for k, v in sorted(link_counts.items())},
            links=tuple(group_links),
            distinct_cards=len(_ids_of("card")),
            distinct_devices=len(_ids_of("device")),
            distinct_ips=len(_ids_of("ip")),
            distinct_merchants=len(_ids_of("merchant")),
            card_ids=_ids_of("card"),
            device_ids=_ids_of("device"),
            ip_ids=_ids_of("ip"),
            merchant_ids=_ids_of("merchant"),
            shared_entities=shared,
            device_overlap={
                "shared_device_ids": tuple(sorted(dev_shared)),
                "n_shared_devices": len(dev_shared),
                "max_transactions_per_device": int(max(dev_shared.values())) if dev_shared else 0,
                "transactions_on_shared_devices": int(sum(dev_shared.values())),
            },
            ip_overlap={
                "shared_ip_ids": tuple(sorted(ip_shared)),
                "n_shared_ips": len(ip_shared),
                "max_transactions_per_ip": int(max(ip_shared.values())) if ip_shared else 0,
                "transactions_on_shared_ips": int(sum(ip_shared.values())),
            },
            bin_context={
                "role": "context_only_never_a_join_key",
                "distinct_bins": len(bin_counts),
                "bin_ids": tuple(sorted(bin_counts)),
                "bin_transaction_counts": {k: int(v) for k, v in sorted(bin_counts.items())},
                "bins_shared_by_multiple_transactions":
                    tuple(sorted(k for k, v in bin_counts.items() if v >= 2)),
            },
        ))
        for p in members:
            assign_rows.append((ids[p], ts_val[p], int(ts_ns[p]), int(p),
                                candidate_id, size, isolated))

    assignments = pd.DataFrame(
        assign_rows,
        columns=[ID_COL, TS_COL, "ts_ns", "txn_pos", "candidate_id",
                 "candidate_size", "is_isolated"],
    ).sort_values("txn_pos", kind="mergesort").reset_index(drop=True)

    if not resolved.include_singletons:
        keep = {c.candidate_id for c in candidates if not c.is_isolated}
        candidates = [c for c in candidates if c.candidate_id in keep]
        assignments = assignments[assignments["candidate_id"].isin(keep)].reset_index(drop=True)

    result = CandidateSet(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        config=resolved,
        candidates=tuple(candidates),
        links=links,
        assignments=assignments,
        n_transactions=n,
    )
    result._by_id = {c.candidate_id: c for c in candidates}
    by_anchor: dict[str, list[dict[str, Any]]] = {}
    if not links.empty:
        for rec in links.to_dict("records"):
            by_anchor.setdefault(rec["anchor_transaction_id"], []).append(rec)
    result._links_by_anchor = by_anchor

    log.info("formed %s candidates (%s multi-transaction) over %s transactions",
             len(candidates), sum(1 for c in candidates if not c.is_isolated), n)
    return result
