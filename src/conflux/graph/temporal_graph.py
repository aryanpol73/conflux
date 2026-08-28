"""CONFLUX graph layer -- heterogeneous temporal entity graph.

SCOPE (locked to the current task): construction + query backend ONLY.
No campaign detection, no components, no thresholds, no scoring, no model.

STRUCTURE
---------
Nodes:  Transaction, Card, BIN, Device, IP, Merchant.
Edges:  Transaction -> entity, only. There are no entity-to-entity edges;
        the transaction is the hub. Every edge carries the transaction's
        timestamp at full source precision (int64 nanoseconds).

CAUSALITY
---------
Rows are ordered deterministically by (timestamp, transaction_id) and every
per-entity index is stored in that order, so a query anchored at transaction t
can be restricted to strictly earlier transactions (mode="causal"). Building the
graph never lets a later row alter an earlier transaction's representation:
each edge is a function of its own row only.

LEAKAGE
-------
The dataset is read with an explicit column allowlist; label, campaign_id and
_source_type are never loaded, and construction asserts they are absent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import (
    ATTRIBUTE_COLUMNS,
    ENTITY_COLUMNS,
    FORBIDDEN_GRAPH_INPUTS,
    ID_COL,
    STRUCTURAL_COLUMNS,
    TS_COL,
    DEFAULT_GRAPH_CONFIG,
    GraphConfig,
)

log = logging.getLogger("conflux.graph.temporal_graph")

GRAPH_SCHEMA_VERSION = "conflux.graph.temporal_entity.v1"
_TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class GraphIntegrityError(ValueError):
    """Raised when the source data cannot be turned into a sound graph."""


class BinConnectivityError(ValueError):
    """Raised when a caller tries to use BIN as a connectivity mechanism."""


def node_key(entity_type: str, entity_id: str) -> str:
    """Typed node key. Prevents collisions between e.g. a BIN and a merchant id."""
    return f"{entity_type}:{entity_id}"


def _parse_timestamps(raw: pd.Series) -> pd.Series:
    """Parse to datetime64[ns] with no silent coercion. Precision preserved."""
    ts = pd.to_datetime(raw, format=_TS_FORMAT, errors="coerce")
    if ts.isna().any():  # fall back for rows written without a fractional part
        ts = pd.to_datetime(raw, errors="coerce")
    bad = int(ts.isna().sum())
    if bad:
        sample = raw[ts.isna()].head(5).tolist()
        raise GraphIntegrityError(
            f"{bad} unparseable timestamp(s); refusing to drop rows. e.g. {sample}"
        )
    # pandas >= 2.0 may infer non-nanosecond resolution (e.g. datetime64[us])
    # depending on input strings / pandas version, and this is the ONLY place
    # TS_COL's unit is decided. Force ns explicitly here so every downstream
    # `.astype("int64")` on this column is guaranteed to be true nanoseconds
    # instead of silently inheriting whatever resolution pd.to_datetime picked.
    ts = ts.dt.as_unit("ns")
    return ts


class TemporalEntityGraph:
    """Transaction-hub heterogeneous temporal graph over the frozen v4 dataset."""

    def __init__(self, transactions: pd.DataFrame, edges: pd.DataFrame,
                 config: GraphConfig) -> None:
        self.config = config
        self.schema_version = GRAPH_SCHEMA_VERSION
        self.transactions = transactions           # sorted causal order, RangeIndex
        self.edges = edges                         # transaction -> entity, tidy
        self._pos: dict[str, int] = {
            t: i for i, t in enumerate(transactions[ID_COL].to_numpy())
        }
        self._ts_ns: np.ndarray = transactions["ts_ns"].to_numpy(dtype=np.int64)
        self._entity_index: dict[tuple[str, str], np.ndarray] = {}
        for (etype, eid), grp in edges.groupby(["entity_type", "entity_id"], sort=False):
            # positions are already ascending: edges inherit causal row order
            self._entity_index[(etype, eid)] = grp["txn_pos"].to_numpy(dtype=np.int64)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: str | Path,
                 config: GraphConfig = DEFAULT_GRAPH_CONFIG) -> "TemporalEntityGraph":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {path}")

        header = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [c for c in (*STRUCTURAL_COLUMNS, *ATTRIBUTE_COLUMNS)
                   if c not in header]
        if missing:
            raise GraphIntegrityError(f"dataset is missing column(s): {missing}")

        usecols = [*STRUCTURAL_COLUMNS, *ATTRIBUTE_COLUMNS]
        leaked = [c for c in FORBIDDEN_GRAPH_INPUTS if c in usecols]
        if leaked:  # defensive: the allowlist must never contain ground truth
            raise GraphIntegrityError(f"forbidden column(s) in read allowlist: {leaked}")

        # bin stays a string: categorical issuer context, never a number.
        dtypes = {c: str for c in STRUCTURAL_COLUMNS}
        dtypes["auth_outcome"] = str
        df = pd.read_csv(path, usecols=usecols, dtype=dtypes, low_memory=False)
        log.info("read %s rows x %s columns from %s", len(df), df.shape[1], path)
        return cls.from_frame(df, config=config, source=str(path))

    @classmethod
    def from_frame(cls, df: pd.DataFrame,
                   config: GraphConfig = DEFAULT_GRAPH_CONFIG,
                   source: str | None = None) -> "TemporalEntityGraph":
        present_forbidden = [c for c in FORBIDDEN_GRAPH_INPUTS if c in df.columns]
        if present_forbidden:
            raise GraphIntegrityError(
                f"ground-truth column(s) {present_forbidden} reached graph "
                "construction; label/campaign_id are evaluation-only."
            )

        df = df.copy()

        # --- structural identifiers: never silently discarded ------------
        problems: dict[str, int] = {}
        for col in STRUCTURAL_COLUMNS:
            s = df[col]
            bad = int(s.isna().sum() + (s.astype(str).str.strip() == "").sum())
            if bad:
                problems[col] = bad
        if problems:
            raise GraphIntegrityError(
                f"missing/blank structural identifier(s): {problems}. Refusing to "
                "drop rows: every transaction must be representable in the graph."
            )

        dup = int(df[ID_COL].duplicated().sum())
        if dup:
            raise GraphIntegrityError(f"{dup} duplicate transaction_id in the dataset")

        df[TS_COL] = _parse_timestamps(df[TS_COL])

        # deterministic causal order, same tie-break rule as the feature/ML layers
        df = df.sort_values([TS_COL, ID_COL], kind="mergesort").reset_index(drop=True)
        df["ts_ns"] = df[TS_COL].astype("int64")
        df["txn_pos"] = np.arange(len(df), dtype=np.int64)
        df["node_key"] = "txn:" + df[ID_COL].astype(str)

        if not df["ts_ns"].is_monotonic_increasing:
            raise GraphIntegrityError("causal ordering failed after sort")

        # --- edges: transaction -> entity, one per entity type -----------
        frames = []
        for etype in config.entity_types:
            col = ENTITY_COLUMNS[etype]
            e = pd.DataFrame({
                "txn_pos": df["txn_pos"].to_numpy(),
                ID_COL: df[ID_COL].to_numpy(),
                "entity_type": etype,
                "entity_id": df[col].astype(str).to_numpy(),
                "ts_ns": df["ts_ns"].to_numpy(),
                TS_COL: df[TS_COL].to_numpy(),
            })
            e["entity_node_key"] = etype + ":" + e["entity_id"]
            e["edge_type"] = f"txn_uses_{etype}"
            frames.append(e)

        edges = (pd.concat(frames, ignore_index=True)
                   .sort_values(["entity_type", "entity_id", "txn_pos"],
                                kind="mergesort")
                   .reset_index(drop=True))

        graph = cls(df, edges, config)
        graph.source = source
        log.info("built graph: %s transactions, %s edges, entity types=%s",
                 len(df), len(edges), list(config.entity_types))
        return graph

    # ------------------------------------------------------------------
    # basic accessors
    # ------------------------------------------------------------------
    @property
    def n_transactions(self) -> int:
        return len(self.transactions)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def has_transaction(self, transaction_id: str) -> bool:
        return transaction_id in self._pos

    def position(self, transaction_id: str) -> int:
        try:
            return self._pos[transaction_id]
        except KeyError as exc:
            raise KeyError(f"unknown transaction_id: {transaction_id}") from exc

    def transaction(self, transaction_id: str) -> dict[str, Any]:
        """Transaction node: identifiers, timestamp and metadata. No ground truth."""
        row = self.transactions.iloc[self.position(transaction_id)]
        out = {
            "transaction_id": row[ID_COL],
            "node_key": row["node_key"],
            "timestamp": row[TS_COL],
            "ts_ns": int(row["ts_ns"]),
            "position": int(row["txn_pos"]),
        }
        for col in ATTRIBUTE_COLUMNS:
            if col in self.transactions.columns:
                out[col] = row[col]
        out["entities"] = self.entities_of(transaction_id)
        return out

    def entities_of(self, transaction_id: str) -> dict[str, str]:
        """All entities this transaction connects to, including bin and merchant."""
        row = self.transactions.iloc[self.position(transaction_id)]
        return {et: str(row[ENTITY_COLUMNS[et]]) for et in self.config.entity_types}

    def entity_ids(self, entity_type: str) -> list[str]:
        self._require_known(entity_type)
        col = ENTITY_COLUMNS[entity_type]
        return sorted(self.transactions[col].astype(str).unique().tolist())

    def entity_counts(self) -> dict[str, int]:
        return {et: int(self.transactions[ENTITY_COLUMNS[et]].nunique())
                for et in self.config.entity_types}

    # ------------------------------------------------------------------
    # temporal entity queries
    # ------------------------------------------------------------------
    def transactions_of_entity(self, entity_type: str, entity_id: str, *,
                               start_ns: int | None = None,
                               end_ns: int | None = None) -> list[str]:
        """Transactions attached to one entity, in causal order.

        Works for every entity type including bin: inspecting BIN activity is
        allowed. What is forbidden is letting BIN *join* transactions in a
        connectivity query -- see temporal_neighbors.
        """
        self._require_known(entity_type)
        pos = self._entity_index.get((entity_type, str(entity_id)))
        if pos is None:
            return []
        if start_ns is not None or end_ns is not None:
            ts = self._ts_ns[pos]
            lo = 0 if start_ns is None else int(np.searchsorted(ts, start_ns, "left"))
            hi = len(pos) if end_ns is None else int(np.searchsorted(ts, end_ns, "right"))
            pos = pos[lo:hi]
        return self.transactions[ID_COL].to_numpy()[pos].tolist()

    def temporal_neighbors(self, transaction_id: str, *,
                           window_seconds: float | None = None,
                           entity_types: Sequence[str] | None = None,
                           mode: str = "symmetric",
                           as_frame: bool = True):
        """Transactions sharing a connectivity entity within a temporal window.

        entity_types defaults to config.connectivity_entity_types (card, device,
        ip). Passing a blocked type raises BinConnectivityError.

        mode="causal"  -> only strictly earlier transactions (no future leakage)
        mode="symmetric" -> +/- window around the anchor
        """
        anchor_pos = self.position(transaction_id)
        types = self._resolve_connectivity_types(entity_types)
        window_ns = (self.config.campaign_window_ns if window_seconds is None
                     else int(round(float(window_seconds) * 1_000_000_000)))
        if window_ns <= 0:
            raise ValueError("window_seconds must be positive")
        if mode not in ("symmetric", "causal"):
            raise ValueError("mode must be 'symmetric' or 'causal'")

        t0 = int(self._ts_ns[anchor_pos])
        lo_ns = t0 - window_ns
        hi_ns = t0 if mode == "causal" else t0 + window_ns

        ent = self.entities_of(transaction_id)
        ids = self.transactions[ID_COL].to_numpy()
        rows = []
        for etype in types:
            pos = self._entity_index.get((etype, ent[etype]))
            if pos is None:
                continue
            ts = self._ts_ns[pos]
            lo = int(np.searchsorted(ts, lo_ns, "left"))
            hi = int(np.searchsorted(ts, hi_ns, "right"))
            for p in pos[lo:hi]:
                p = int(p)
                if p == anchor_pos:
                    continue
                if mode == "causal" and p >= anchor_pos:
                    continue  # deterministic tie-break: never look forward
                rows.append({
                    "anchor_transaction_id": transaction_id,
                    "neighbor_transaction_id": ids[p],
                    "link_entity_type": etype,
                    "link_entity_id": ent[etype],
                    "neighbor_ts_ns": int(self._ts_ns[p]),
                    "delta_seconds": (int(self._ts_ns[p]) - t0) / 1e9,
                })
        if not as_frame:
            return rows
        cols = ["anchor_transaction_id", "neighbor_transaction_id",
                "link_entity_type", "link_entity_id", "neighbor_ts_ns",
                "delta_seconds"]
        return (pd.DataFrame(rows, columns=cols)
                  .sort_values(["neighbor_ts_ns", "neighbor_transaction_id",
                                "link_entity_type"], kind="mergesort")
                  .reset_index(drop=True))

    def shared_entities(self, t1: str, t2: str, *,
                        entity_types: Sequence[str] | None = None) -> list[str]:
        """Entity types on which two transactions coincide (context types included
        only if explicitly requested)."""
        types = (tuple(self.config.entity_types) if entity_types is None
                 else tuple(entity_types))
        for et in types:
            self._require_known(et)
        a, b = self.entities_of(t1), self.entities_of(t2)
        return [et for et in types if a[et] == b[et]]

    # ------------------------------------------------------------------
    # BIN: represented, inspectable, never a connectivity mechanism
    # ------------------------------------------------------------------
    def bin_of(self, transaction_id: str) -> str:
        return self.entities_of(transaction_id)["bin"]

    def bin_activity(self, bin_id: str, *, start_ns: int | None = None,
                     end_ns: int | None = None) -> list[str]:
        """Explicit, separate BIN read path. Never routed through connectivity."""
        return self.transactions_of_entity("bin", str(bin_id),
                                           start_ns=start_ns, end_ns=end_ns)

    def entity_fanout(self, entity_type: str) -> dict[str, float]:
        """Transactions-per-entity statistics. Used to evidence the BIN rule."""
        self._require_known(entity_type)
        counts = self.transactions[ENTITY_COLUMNS[entity_type]].value_counts()
        return {
            "distinct_entities": int(counts.size),
            "transactions_per_entity_mean": float(counts.mean()),
            "transactions_per_entity_median": float(counts.median()),
            "transactions_per_entity_max": int(counts.max()),
            "is_connectivity_entity": entity_type in self.config.connectivity_entity_types,
            "blocked_from_connectivity": entity_type in self.config.blocked_connectivity_entity_types,
        }

    # ------------------------------------------------------------------
    # internals / export
    # ------------------------------------------------------------------
    def _require_known(self, entity_type: str) -> None:
        if entity_type not in self.config.entity_types:
            raise KeyError(f"unknown entity type '{entity_type}'; "
                           f"known: {list(self.config.entity_types)}")

    def _resolve_connectivity_types(self, entity_types: Sequence[str] | None
                                    ) -> tuple[str, ...]:
        types = (tuple(self.config.connectivity_entity_types)
                 if entity_types is None else tuple(entity_types))
        for et in types:
            self._require_known(et)
        blocked = sorted(set(types) & set(self.config.blocked_connectivity_entity_types))
        if blocked:
            raise BinConnectivityError(
                f"entity type(s) {blocked} cannot be used to connect transactions. "
                "BIN is issuer context: shared BIN must never produce campaign "
                "connectivity. Use bin_activity()/entity_fanout('bin') instead."
            )
        return types

    def node_table(self) -> pd.DataFrame:
        """All nodes, one row each: transactions plus every entity."""
        txn = pd.DataFrame({
            "node_key": self.transactions["node_key"],
            "node_type": "transaction",
            "node_id": self.transactions[ID_COL],
            "ts_ns": self.transactions["ts_ns"],
        })
        ent = (self.edges[["entity_node_key", "entity_type", "entity_id"]]
               .drop_duplicates()
               .rename(columns={"entity_node_key": "node_key",
                                "entity_type": "node_type",
                                "entity_id": "node_id"}))
        ent["ts_ns"] = pd.NA
        return pd.concat([txn, ent], ignore_index=True)

    def edge_table(self) -> pd.DataFrame:
        return self.edges[[ID_COL, "entity_node_key", "edge_type", "entity_type",
                           "entity_id", TS_COL, "ts_ns", "txn_pos"]].copy()

    def summary(self) -> dict[str, Any]:
        ec = self.entity_counts()
        return {
            "schema_version": self.schema_version,
            "source": getattr(self, "source", None),
            "transactions": self.n_transactions,
            "entity_node_counts": ec,
            "total_entity_nodes": int(sum(ec.values())),
            "total_nodes": self.n_transactions + int(sum(ec.values())),
            "edges_total": self.n_edges,
            "edges_by_type": {k: int(v) for k, v in
                              self.edges["edge_type"].value_counts().sort_index().items()},
            "edges_per_transaction": self.n_edges / max(self.n_transactions, 1),
            "entity_to_entity_edges": 0,
            "timestamp_min": str(self.transactions[TS_COL].min()),
            "timestamp_max": str(self.transactions[TS_COL].max()),
            "span_seconds": float((int(self._ts_ns[-1]) - int(self._ts_ns[0])) / 1e9),
            "distinct_timestamps": int(self.transactions[TS_COL].nunique()),
            "config": {
                "campaign_window_seconds": self.config.campaign_window_seconds,
                "connectivity_entity_types": list(self.config.connectivity_entity_types),
                "context_entity_types": list(self.config.context_entity_types),
                "blocked_connectivity_entity_types":
                    list(self.config.blocked_connectivity_entity_types),
            },
        }
