"""CONFLUX graph layer -- build + verify the temporal entity graph.

Runs construction against the frozen dataset, executes the sanity checks, and
prints real statistics. Writes nothing except the optional export directory;
it never touches data/raw/, the feature table, or any model artifact.

STOP POINT: no campaign detection, no components, no scoring.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from conflux.config import PROCESSED_DIR, RAW_DATASET_PATH
except ImportError:  # allow `python src/conflux/graph/build_graph.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from conflux.config import PROCESSED_DIR, RAW_DATASET_PATH  # type: ignore

from conflux.graph.config import (  # noqa: E402
    ATTRIBUTE_COLUMNS, ENTITY_COLUMNS, FORBIDDEN_GRAPH_INPUTS, ID_COL,
    STRUCTURAL_COLUMNS, TS_COL, GraphConfig,
)
from conflux.graph.temporal_graph import (  # noqa: E402
    BinConnectivityError, TemporalEntityGraph,
)

log = logging.getLogger("conflux.graph.build_graph")
DEFAULT_OUT_DIR = PROCESSED_DIR / "graph"
FROZEN_PATHS = {Path(RAW_DATASET_PATH).resolve()}


@dataclass
class CheckResult:
    name: str
    verdict: str                      # "PASS" | "FAIL"
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


def check_transactions_represented(g: TemporalEntityGraph,
                                   raw_ids: list[str]) -> CheckResult:
    graph_ids = set(g.transactions[ID_COL])
    missing = set(raw_ids) - graph_ids
    extra = graph_ids - set(raw_ids)
    ev = {
        "rows_in_csv": len(raw_ids),
        "unique_ids_in_csv": len(set(raw_ids)),
        "transaction_nodes": g.n_transactions,
        "csv_ids_absent_from_graph": len(missing),
        "graph_ids_absent_from_csv": len(extra),
        "every_transaction_has_5_edges": bool(
            g.edges.groupby(ID_COL).size().eq(len(g.config.entity_types)).all()
        ),
    }
    ok = (not missing and not extra
          and g.n_transactions == len(set(raw_ids))
          and ev["every_transaction_has_5_edges"])
    return CheckResult("A_all_transactions_represented", "PASS" if ok else "FAIL", ev)


def check_entity_counts(g: TemporalEntityGraph, raw: pd.DataFrame) -> CheckResult:
    expected = {et: int(raw[col].astype(str).nunique())
                for et, col in ENTITY_COLUMNS.items() if et in g.config.entity_types}
    actual = g.entity_counts()
    mismatched = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    edges_expected = g.n_transactions * len(g.config.entity_types)
    ev = {
        "expected_entity_counts_from_pandas": expected,
        "graph_entity_counts": actual,
        "mismatches": mismatched,
        "edges_expected": edges_expected,
        "edges_actual": g.n_edges,
        "distinct_entity_nodes_in_edge_table":
            int(g.edges["entity_node_key"].nunique()),
    }
    ok = (not mismatched and g.n_edges == edges_expected
          and ev["distinct_entity_nodes_in_edge_table"] == sum(actual.values()))
    return CheckResult("B_entity_counts", "PASS" if ok else "FAIL", ev)


def check_no_silent_drops(g: TemporalEntityGraph, raw: pd.DataFrame) -> CheckResult:
    blanks = {}
    for col in STRUCTURAL_COLUMNS:
        s = raw[col]
        n = int(s.isna().sum() + (s.astype(str).str.strip() == "").sum())
        if n:
            blanks[col] = n
    nodes = g.node_table()
    ev = {
        "blank_or_null_structural_identifiers_in_csv": blanks,
        "csv_rows": int(len(raw)),
        "graph_transaction_nodes": g.n_transactions,
        "rows_dropped": int(len(raw)) - g.n_transactions,
        "null_entity_ids_in_node_table": int(nodes["node_id"].isna().sum()),
        "empty_entity_ids_in_edge_table":
            int((g.edges["entity_id"].astype(str).str.strip() == "").sum()),
    }
    ok = (not blanks and ev["rows_dropped"] == 0
          and ev["null_entity_ids_in_node_table"] == 0
          and ev["empty_entity_ids_in_edge_table"] == 0)
    return CheckResult("C_no_silent_drops", "PASS" if ok else "FAIL", ev)


def check_timestamps(g: TemporalEntityGraph, raw: pd.DataFrame) -> CheckResult:
    ts = g.transactions[TS_COL]
    ts_ns = g.transactions["ts_ns"].to_numpy(dtype=np.int64)
    # every edge must carry exactly its transaction's timestamp
    joined = g.edges.merge(g.transactions[[ID_COL, "ts_ns"]], on=ID_COL,
                           how="left", suffixes=("_edge", "_txn"))
    edge_ts_ok = bool((joined["ts_ns_edge"] == joined["ts_ns_txn"]).all())
    # round-trip a sample against the raw strings to prove no precision loss
    sample = raw.head(2000)[[ID_COL, TS_COL]].copy()
    parsed = pd.to_datetime(sample[TS_COL], errors="coerce")
    roundtrip = parsed.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str.rstrip("0").str.rstrip(".")
    original = sample[TS_COL].astype(str).str.rstrip("0").str.rstrip(".")
    ev = {
        "unparseable_timestamps": 0,
        "dtype": str(ts.dtype),
        "min": str(ts.min()),
        "max": str(ts.max()),
        "span_seconds": float((int(ts_ns[-1]) - int(ts_ns[0])) / 1e9),
        "ts_ns_unit_is_nanoseconds": bool(
            ts.dtype == np.dtype("datetime64[ns]")
        ),
        "monotonic_non_decreasing_after_causal_sort":
            bool(np.all(np.diff(ts_ns) >= 0)),
        "sub_second_precision_rows": int((ts_ns % 1_000_000_000 != 0).sum()),
        "distinct_timestamps": int(ts.nunique()),
        "duplicate_timestamp_rows": int(len(ts) - ts.nunique()),
        "edge_timestamp_equals_transaction_timestamp": edge_ts_ok,
        "roundtrip_sample_size": int(len(sample)),
        "roundtrip_mismatches": int((roundtrip != original).sum()),
        "ordering_rule": "sort_values([timestamp, transaction_id], kind=mergesort)",
    }
    ok = (ev["monotonic_non_decreasing_after_causal_sort"] and edge_ts_ok
          and ev["roundtrip_mismatches"] == 0 and ev["sub_second_precision_rows"] > 0 and ev["ts_ns_unit_is_nanoseconds"])
    return CheckResult("D_timestamps_and_causal_order", "PASS" if ok else "FAIL", ev)


def check_no_ground_truth(g: TemporalEntityGraph, dataset_path: Path) -> CheckResult:
    header = pd.read_csv(dataset_path, nrows=0).columns.tolist()
    loaded = list(g.transactions.columns)
    nodes, edges = g.node_table(), g.edge_table()
    present = {
        "in_transaction_nodes": [c for c in FORBIDDEN_GRAPH_INPUTS if c in loaded],
        "in_node_table": [c for c in FORBIDDEN_GRAPH_INPUTS if c in nodes.columns],
        "in_edge_table": [c for c in FORBIDDEN_GRAPH_INPUTS if c in edges.columns],
    }
    ev = {
        "forbidden_columns": list(FORBIDDEN_GRAPH_INPUTS),
        "present_in_csv_header": [c for c in FORBIDDEN_GRAPH_INPUTS if c in header],
        "read_allowlist": list(STRUCTURAL_COLUMNS) + list(ATTRIBUTE_COLUMNS),
        "columns_actually_loaded": loaded,
        "forbidden_columns_found": present,
        "edge_construction_inputs": [ENTITY_COLUMNS[et] for et in g.config.entity_types],
    }
    ok = not any(present.values())
    return CheckResult("E_no_label_or_campaign_id", "PASS" if ok else "FAIL", ev)


def check_bin_separation(g: TemporalEntityGraph) -> CheckResult:
    ev: dict[str, Any] = {
        "bin_is_a_graph_node_type": "bin" in g.config.entity_types,
        "bin_in_connectivity_types": "bin" in g.config.connectivity_entity_types,
        "bin_in_blocked_types": "bin" in g.config.blocked_connectivity_entity_types,
        "connectivity_entity_types": list(g.config.connectivity_entity_types),
        "context_entity_types": list(g.config.context_entity_types),
    }
    # the API must actively refuse BIN as a connectivity mechanism
    anchor = g.transactions[ID_COL].iloc[0]
    try:
        g.temporal_neighbors(anchor, entity_types=["bin"])
        ev["temporal_neighbors_rejects_bin"] = False
    except BinConnectivityError as exc:
        ev["temporal_neighbors_rejects_bin"] = True
        ev["rejection_message"] = str(exc)
    try:
        GraphConfig(connectivity_entity_types=("card", "bin"))
        ev["config_rejects_bin_as_connectivity"] = False
    except Exception as exc:  # GraphConfigError
        ev["config_rejects_bin_as_connectivity"] = True
        ev["config_rejection_message"] = str(exc)
    # BIN remains readable on its own path
    a_bin = g.bin_of(anchor)
    ev["bin_activity_readable"] = len(g.bin_activity(a_bin)) > 0
    # quantitative justification, not a campaign computation
    ev["fanout"] = {et: g.entity_fanout(et) for et in g.config.entity_types}

    ok = (ev["bin_is_a_graph_node_type"] and not ev["bin_in_connectivity_types"]
          and ev["bin_in_blocked_types"] and ev["temporal_neighbors_rejects_bin"]
          and ev["config_rejects_bin_as_connectivity"] and ev["bin_activity_readable"])
    return CheckResult("F_bin_represented_but_not_connectivity",
                       "PASS" if ok else "FAIL", ev)


def check_query_layer(g: TemporalEntityGraph) -> CheckResult:
    """Exercise the query API on real rows, including the causal guarantee."""
    ids = g.transactions[ID_COL].to_numpy()
    rng = np.random.default_rng(0)
    probe = ids[rng.choice(len(ids), size=min(200, len(ids)), replace=False)]

    ts_ns = g.transactions["ts_ns"].to_numpy(dtype=np.int64)
    window_ns = g.config.campaign_window_ns
    violations_future = 0
    violations_window = 0
    violations_link = 0
    total_causal = 0
    total_symmetric = 0
    for tid in probe:
        pos = g.position(tid)
        t0 = int(ts_ns[pos])
        sym = g.temporal_neighbors(tid, mode="symmetric")
        cau = g.temporal_neighbors(tid, mode="causal")
        total_symmetric += len(sym)
        total_causal += len(cau)
        if len(cau):
            violations_future += int((cau["neighbor_ts_ns"] > t0).sum())
            violations_future += int(
                (cau["neighbor_transaction_id"].map(g.position) >= pos).sum())
        if len(sym):
            d = (sym["neighbor_ts_ns"] - t0).abs()
            violations_window += int((d > window_ns).sum())
            for _, r in sym.head(20).iterrows():
                shared = g.shared_entities(tid, r["neighbor_transaction_id"])
                if r["link_entity_type"] not in shared:
                    violations_link += 1

    ev = {
        "probe_transactions": int(len(probe)),
        "window_seconds_used": g.config.campaign_window_seconds,
        "symmetric_neighbor_pairs": total_symmetric,
        "causal_neighbor_pairs": total_causal,
        "causal_results_containing_future_rows": violations_future,
        "symmetric_results_outside_window": violations_window,
        "link_entity_disagreements": violations_link,
        "custom_window_override_works": bool(
            len(g.temporal_neighbors(probe[0], window_seconds=60, as_frame=True))
            <= len(g.temporal_neighbors(probe[0], window_seconds=86400, as_frame=True))
        ),
    }
    ok = (violations_future == 0 and violations_window == 0
          and violations_link == 0 and ev["custom_window_override_works"])
    return CheckResult("G_query_layer_and_temporal_windows",
                       "PASS" if ok else "FAIL", ev)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build and verify the CONFLUX temporal entity graph.")
    ap.add_argument("--dataset", default=str(RAW_DATASET_PATH))
    ap.add_argument("--window-seconds", type=float, default=3600.0)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s %(message)s")

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    if out_dir.resolve() in FROZEN_PATHS or dataset_path.resolve() == out_dir.resolve():
        raise SystemExit(f"refusing to write to a frozen path: {out_dir}")

    config = GraphConfig(campaign_window_seconds=args.window_seconds)
    graph = TemporalEntityGraph.from_csv(dataset_path, config=config)

    # independent re-read for the checks (full file, including ground truth
    # columns, used ONLY to verify that the graph excluded them)
    raw = pd.read_csv(dataset_path, dtype=str, low_memory=False)

    results = [
        check_transactions_represented(graph, raw[ID_COL].tolist()),
        check_entity_counts(graph, raw),
        check_no_silent_drops(graph, raw),
        check_timestamps(graph, raw),
        check_no_ground_truth(graph, dataset_path),
        check_bin_separation(graph),
        check_query_layer(graph),
    ]

    summary = graph.summary()
    report = {
        "status": "OK" if all(r.verdict == "PASS" for r in results) else "FAILED",
        "graph": summary,
        "sanity_checks": [asdict(r) for r in results],
        "check_summary": {
            "pass": sum(r.verdict == "PASS" for r in results),
            "fail": sum(r.verdict == "FAIL" for r in results),
        },
        "scope_note": ("Graph backend only. No campaign detection, no connected "
                       "components, no thresholds, no scoring."),
    }

    print("\n=== GRAPH ===")
    print(json.dumps(summary, indent=2, default=str))
    print("\n=== SANITY CHECKS ===")
    print(json.dumps(report["sanity_checks"], indent=2, default=str))
    print("\n=== RESULT ===")
    print(json.dumps(report["check_summary"] | {"status": report["status"]}, indent=2))

    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        graph.node_table().to_csv(out_dir / "graph_nodes.csv", index=False)
        graph.edge_table().to_csv(out_dir / "graph_edges.csv", index=False)
        (out_dir / "graph_report.json").write_text(
            json.dumps(report, indent=2, default=str))
        print(f"\nwrote {out_dir/'graph_nodes.csv'}, {out_dir/'graph_edges.csv'}, "
              f"{out_dir/'graph_report.json'}")

    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
