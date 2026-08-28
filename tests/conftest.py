"""Synthetic Phase 3B artifacts. Tests never touch the real dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

N_CAMPAIGNS = 6
FRAGMENTED = {"CMP-002", "CMP-005"}      # 2 candidates each -> 8 attack groups
N_NEGATIVES = 120


def _members(cid, n, start_ns, step_ns, cards, devices, ips, merchants, bins,
             amounts, auth, campaign):
    rows = []
    for i in range(n):
        rows.append({
            "transaction_id": f"{cid}-T{i:03d}",
            "ts_ns": int(start_ns + i * step_ns),
            "candidate_id": cid,
            "card_fingerprint": cards[i % len(cards)],
            "device_fingerprint": devices[i % len(devices)],
            "ip_signature": ips[i % len(ips)],
            "merchant_id": merchants[i % len(merchants)],
            "bin": bins[i % len(bins)],
            "amount": float(amounts[i % len(amounts)]),
            "auth_outcome": auth[i % len(auth)],
            "campaign_id": campaign,
            "label": "1" if campaign else "0",
        })
    return rows


def _candidate_row(cid, mem, edges, multi_edges):
    df = pd.DataFrame(mem)
    size = len(df)
    span = (df["ts_ns"].max() - df["ts_ns"].min()) / 1e9

    def shared(col):
        c = df[col].value_counts()
        return sorted(c[c >= 2].index.tolist()), (int(c.max()) if len(c) else 0)

    sc, _ = shared("card_fingerprint")
    sd, maxd = shared("device_fingerprint")
    si, maxi = shared("ip_signature")
    return {
        "candidate_id": cid, "size": size, "is_isolated": False,
        "first_timestamp": pd.Timestamp(int(df["ts_ns"].min())),
        "last_timestamp": pd.Timestamp(int(df["ts_ns"].max())),
        "time_span_seconds": float(span),
        "link_edge_count": int(edges), "links_multi_entity": int(multi_edges),
        "link_entity_types": "card|device|ip",
        "links_card": int(edges), "links_device": int(max(edges - 1, 0)),
        "links_ip": int(max(edges - 2, 0)),
        "distinct_cards": int(df["card_fingerprint"].nunique()),
        "distinct_devices": int(df["device_fingerprint"].nunique()),
        "distinct_ips": int(df["ip_signature"].nunique()),
        "distinct_merchants": int(df["merchant_id"].nunique()),
        "shared_card_ids": "|".join(sc), "shared_device_ids": "|".join(sd),
        "shared_ip_ids": "|".join(si),
        "max_transactions_per_shared_device": maxd if len(sd) else 0,
        "max_transactions_per_shared_ip": maxi if len(si) else 0,
        "distinct_bins": int(df["bin"].nunique()),
        "bin_ids_context": "|".join(sorted(df["bin"].unique())),
        "transaction_ids": "|".join(df["transaction_id"]),
    }


def build_synthetic(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(4)
    base = int(pd.Timestamp("2024-03-01").value)
    members, cand_rows = [], []

    n = 0
    for c in range(1, N_CAMPAIGNS + 1):
        camp = f"CMP-{c:03d}"
        frags = 2 if camp in FRAGMENTED else 1
        for fgt in range(frags):
            n += 1
            cid = f"CAND-{n:06d}"
            size = 5 + (c % 3)
            start = base + (c * 6 * 3600 + fgt * 900) * 10**9
            mem = _members(
                cid, size, start, int(4e9),                    # 4 s apart: bursty
                cards=[f"CARD-A{c}"] * 2 + [f"CARD-B{c}"],     # heavy card reuse
                devices=[f"DEV-{c}"], ips=[f"IP-{c}"],
                merchants=[f"M-{i}" for i in range(size)],     # merchant spread
                bins=[f"BIN-{i}" for i in range(3)],           # issuer spread
                amounts=[1.00, 1.00, 1.00, 2.00],              # repeated probe
                auth=["declined", "declined", "approved"], campaign=camp)
            members += mem
            e = size * (size - 1) // 2                          # near-clique
            cand_rows.append(_candidate_row(cid, mem, e, e - 1))

    for j in range(N_NEGATIVES):
        n += 1
        cid = f"CAND-{n:06d}"
        size = int(rng.integers(2, 6))
        start = base + int(rng.integers(0, 40 * 24 * 3600)) * 10**9
        mem = _members(
            cid, size, start, int(600e9),                       # 10 min apart
            cards=[f"CARD-N{j}-{i}" for i in range(size)],      # no card reuse
            devices=[f"DEVN-{j}"], ips=[f"IPN-{j}"],
            merchants=[f"MN-{j}"],                              # single merchant
            bins=[f"BINN-{j}"],                                 # single BIN
            amounts=[10.0 + i * 7.3 for i in range(size)],
            auth=["approved"], campaign="")
        members += mem
        cand_rows.append(_candidate_row(cid, mem, size - 1, 0))

    mdf = pd.DataFrame(members).sort_values(["ts_ns", "transaction_id"],
                                            kind="mergesort").reset_index(drop=True)
    mdf["txn_pos"] = np.arange(len(mdf))
    mdf["timestamp"] = pd.to_datetime(mdf["ts_ns"]).astype(str)

    cdf = pd.DataFrame(cand_rows)
    sizes = cdf.set_index("candidate_id")["size"]
    asg = mdf[["transaction_id", "timestamp", "ts_ns", "txn_pos", "candidate_id"]].copy()
    asg["candidate_size"] = asg["candidate_id"].map(sizes).astype(int)
    asg["is_isolated"] = False

    ds = mdf[["transaction_id", "timestamp", "card_fingerprint", "bin",
              "device_fingerprint", "ip_signature", "merchant_id", "amount",
              "auth_outcome", "label", "campaign_id"]]

    root.mkdir(parents=True, exist_ok=True)
    paths = {"candidates": root / "campaign_candidates.csv",
             "assignments": root / "campaign_candidate_assignments.csv",
             "dataset": root / "dataset_synth.csv"}
    cdf.to_csv(paths["candidates"], index=False)
    asg.to_csv(paths["assignments"], index=False)
    ds.to_csv(paths["dataset"], index=False)
    return paths


@pytest.fixture(scope="session")
def synth(tmp_path_factory) -> dict[str, Path]:
    return build_synthetic(tmp_path_factory.mktemp("phase3b"))


@pytest.fixture(scope="session")
def scored(synth):
    from conflux.evaluation.campaign_evaluation import load_ground_truth
    from conflux.evaluation.candidate_diagnostics import (
        attach_groups, load_candidate_artifacts)
    from conflux.scoring.candidate_features import (
        build_scoring_features, load_structural_attributes)

    cand, asg = load_candidate_artifacts(synth["candidates"], synth["assignments"])
    attrs = load_structural_attributes(synth["dataset"])
    sf = build_scoring_features(cand, asg, attrs)
    gt = load_ground_truth(synth["dataset"])
    return sf, attach_groups(sf.frame, asg, gt, group_by="campaign_id")
