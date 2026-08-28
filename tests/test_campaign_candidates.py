"""Phase 3B focused tests. Synthetic graphs only; the frozen dataset is untouched.

LINK CONTRACT UNDER TEST
------------------------
CandidateSet.links holds ONE ROW PER CONNECTED TRANSACTION PAIR. The shared
connectivity entity types are evidence carried on that row. The long,
one-row-per-entity-type view is available via CandidateSet.explode_links().
"""
from __future__ import annotations

import pandas as pd
import pytest

from conflux.graph.campaign_detection import (
    CandidateConfig, ContextEntityConnectivityError, form_campaign_candidates,
)
from conflux.graph.config import GraphConfig
from conflux.graph.temporal_graph import (
    BinConnectivityError, GraphIntegrityError, TemporalEntityGraph,
)

BASE = "2026-08-26 00:00:00.000000"


def _row(tid, secs, card="C1", bin_="400001", dev="D1", ip="I1", merch="M001"):
    ts = pd.Timestamp(BASE) + pd.Timedelta(seconds=secs)
    return {
        "transaction_id": tid,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "merchant_id": merch, "card_fingerprint": card, "bin": bin_,
        "amount": "10.00", "device_fingerprint": dev, "ip_signature": ip,
        "auth_outcome": "approved",
    }


def _graph(rows, window=3600.0):
    return TemporalEntityGraph.from_frame(
        pd.DataFrame(rows), config=GraphConfig(campaign_window_seconds=window))


def _groups(cs):
    return {c.candidate_id: set(c.transaction_ids) for c in cs.candidates}


# 1 --------------------------------------------------------------------
def test_no_future_transaction_in_candidate():
    rows = [_row("t1", 0), _row("t2", 10), _row("t3", 20)]      # same card
    cs = form_campaign_candidates(_graph(rows))
    assert (cs.links["prior_ts_ns"] <= cs.links["anchor_ts_ns"]).all()
    assert (cs.links["prior_pos"] < cs.links["anchor_pos"]).all()
    assert (cs.links["delta_seconds"] <= 0).all()
    assert cs.causal_candidate("t1")["prior_transaction_ids"] == ()
    assert set(cs.causal_candidate("t2")["prior_transaction_ids"]) == {"t1"}
    assert set(cs.causal_candidate("t3")["prior_transaction_ids"]) == {"t1", "t2"}
    for c in cs.candidates:
        for l in c.links:
            assert l["prior_ts_ns"] <= c.decision_ts_ns
            assert l["anchor_ts_ns"] <= c.decision_ts_ns


def test_equal_timestamps_pair_exactly_once():
    rows = [_row("t_b", 0), _row("t_a", 0)]      # same card, device, ip, same ts
    cs = form_campaign_candidates(_graph(rows))
    assert len(cs.links) == 1                    # one PAIR, not one per entity type
    link = cs.links.iloc[0]
    assert link["anchor_transaction_id"] == "t_b"
    assert link["prior_transaction_id"] == "t_a"
    assert link["link_entity_types"] == "card|device|ip"
    assert link["n_link_entity_types"] == 3
    assert bool(link["shares_card"]) and bool(link["shares_device"]) and bool(link["shares_ip"])
    assert len(cs.explode_links()) == 3          # long view still available
    assert len(_groups(cs)) == 1


def test_pair_is_one_link_carrying_all_shared_entity_evidence():
    """Connectivity is a property of a transaction PAIR; shared entity types are
    the evidence explaining it. Regression guard for the 3-rows-per-pair bug."""
    rows = [_row("t1", 0, card="CS", dev="DS", ip="I1", merch="M1", bin_="400001"),
            _row("t2", 30, card="CS", dev="DS", ip="I2", merch="M2", bin_="500002")]
    cs = form_campaign_candidates(_graph(rows))
    assert len(cs.links) == 1
    link = cs.links.iloc[0]
    assert link["link_entity_types"] == "card|device"
    assert bool(link["shares_card"]) and bool(link["shares_device"])
    assert not bool(link["shares_ip"])
    assert link["card_entity_id"] == "CS" and link["device_entity_id"] == "DS"
    assert link["ip_entity_id"] == ""
    c = cs.candidates[0]
    assert c.link_edge_count == 1              # one connected pair
    assert c.multi_entity_link_count == 1      # backed by two entity types
    assert c.link_counts == {"card": 1, "device": 1}
    assert "bin" not in cs.links.columns and "shares_bin" not in cs.links.columns
    assert "merchant" not in cs.links.columns and "shares_merchant" not in cs.links.columns


# 2 --------------------------------------------------------------------
def test_shared_bin_alone_does_not_connect():
    rows = [_row("t1", 0, card="C1", dev="D1", ip="I1", merch="M1", bin_="400001"),
            _row("t2", 60, card="C2", dev="D2", ip="I2", merch="M2", bin_="400001")]
    cs = form_campaign_candidates(_graph(rows))
    assert len(cs.links) == 0
    assert all(c.is_isolated for c in cs.candidates)
    assert len(cs.candidates) == 2
    # bin is still recorded as context
    assert cs.candidate_of("t1").bin_context["distinct_bins"] == 1
    with pytest.raises(BinConnectivityError):
        form_campaign_candidates(_graph(rows),
                                 CandidateConfig(connectivity_entity_types=("bin",)))
    with pytest.raises(BinConnectivityError):
        form_campaign_candidates(
            _graph(rows), CandidateConfig(connectivity_entity_types=("card", "bin")))


# 3 --------------------------------------------------------------------
def test_shared_merchant_alone_does_not_connect():
    rows = [_row("t1", 0, card="C1", dev="D1", ip="I1", merch="M9", bin_="400001"),
            _row("t2", 60, card="C2", dev="D2", ip="I2", merch="M9", bin_="500002")]
    cs = form_campaign_candidates(_graph(rows))
    assert len(cs.links) == 0
    assert all(c.is_isolated for c in cs.candidates)
    with pytest.raises(ContextEntityConnectivityError):
        form_campaign_candidates(
            _graph(rows), CandidateConfig(connectivity_entity_types=("merchant",)))


# 4 --------------------------------------------------------------------
@pytest.mark.parametrize("shared,kw", [
    ("card", dict(card="CX")), ("device", dict(dev="DX")), ("ip", dict(ip="IX"))])
def test_card_device_ip_connectivity(shared, kw):
    a = dict(card="C1", dev="D1", ip="I1", merch="M1", bin_="400001")
    b = dict(card="C2", dev="D2", ip="I2", merch="M2", bin_="500002")
    a.update(kw); b.update(kw)
    cs = form_campaign_candidates(_graph([_row("t1", 0, **a), _row("t2", 30, **b)]))
    assert len(cs.candidates) == 1
    c = cs.candidates[0]
    assert set(c.transaction_ids) == {"t1", "t2"}
    assert c.link_edge_count == 1
    assert c.multi_entity_link_count == 0
    assert c.link_entity_types == (shared,)
    assert c.shared_entities[shared] == {kw[list(kw)[0]]: 2}
    assert c.time_span_seconds == pytest.approx(30.0)
    assert cs.links.iloc[0]["link_entity_types"] == shared


def test_evidence_fields_are_populated():
    rows = [_row("t1", 0, card="C1", dev="DX", ip="I1", merch="M1", bin_="400001"),
            _row("t2", 30, card="C2", dev="DX", ip="I2", merch="M2", bin_="500002")]
    c = form_campaign_candidates(_graph(rows)).candidates[0]
    assert c.distinct_cards == 2 and c.distinct_merchants == 2
    assert c.device_overlap["max_transactions_per_device"] == 2
    assert c.ip_overlap["n_shared_ips"] == 0
    assert c.bin_context["distinct_bins"] == 2
    assert c.link_edge_count == 1 and len(c.links) == 1


# 5 --------------------------------------------------------------------
def test_temporal_window_respected():
    inside = [_row("t1", 0, card="CW"), _row("t2", 3600, card="CW")]
    outside = [_row("t1", 0, card="CW"), _row("t2", 3601, card="CW")]
    cs_in = form_campaign_candidates(_graph(inside))
    assert len(cs_in.links) == 1                                     # inclusive at 3600
    assert cs_in.links.iloc[0]["age_seconds"] == pytest.approx(3600.0)
    assert len(form_campaign_candidates(_graph(outside)).links) == 0
    # explicit override beats the graph default, in both directions
    assert len(form_campaign_candidates(
        _graph(inside), CandidateConfig(window_seconds=60)).links) == 0
    assert len(form_campaign_candidates(
        _graph(outside, window=60), CandidateConfig(window_seconds=7200)).links) == 1


# 6 --------------------------------------------------------------------
def test_deterministic_construction():
    rows = [_row("t1", 0, card="CA"), _row("t2", 5, card="CA"),
            _row("t3", 9, dev="DZ", card="CB"), _row("t4", 11, dev="DZ", card="CC"),
            _row("t5", 4000, card="CD", dev="DD", ip="ID", merch="MD")]
    r1 = form_campaign_candidates(_graph(rows))
    r2 = form_campaign_candidates(_graph(list(reversed(rows))))  # graph re-sorts causally
    assert r1.assignments.equals(r2.assignments)
    assert r1.candidate_frame().equals(r2.candidate_frame())
    assert r1.links.equals(r2.links)
    assert r1.explode_links().equals(r2.explode_links())


# 7 --------------------------------------------------------------------
def test_ground_truth_never_enters_construction():
    rows = [_row("t1", 0), _row("t2", 5)]
    df = pd.DataFrame(rows)
    df["label"] = "1"
    df["campaign_id"] = "CAMP_1"
    with pytest.raises(GraphIntegrityError):
        _graph(df.to_dict("records"))
    cs = form_campaign_candidates(_graph(rows))
    forbidden = {"label", "campaign_id", "_source_type"}
    for frame in (cs.links, cs.explode_links(), cs.assignments, cs.candidate_frame()):
        assert not forbidden & set(frame.columns)


# 8 --------------------------------------------------------------------
def test_all_transactions_accounted_for_including_isolated():
    rows = [_row("t1", 0, card="CA"), _row("t2", 5, card="CA"),
            _row("t3", 100, card="CX", dev="DX", ip="IX", merch="MX", bin_="400001"),
            _row("t4", 200, card="CY", dev="DY", ip="IY", merch="MY", bin_="400001")]
    cs = form_campaign_candidates(_graph(rows))
    members = [t for c in cs.candidates for t in c.transaction_ids]
    assert sorted(members) == ["t1", "t2", "t3", "t4"]
    assert len(members) == len(set(members)) == cs.n_transactions
    assert len(cs.assignments) == 4
    isolated = {c.transaction_ids[0] for c in cs.candidates if c.is_isolated}
    assert isolated == {"t3", "t4"}
    assert cs.summary()["candidates_isolated"] == 2
