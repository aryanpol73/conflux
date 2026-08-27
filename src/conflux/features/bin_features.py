"""BIN signal group -- issuer-level pressure and cross-entity BIN overlap.

Locked BIN spec items implemented here:
  BIN activity            -> bin_txn_count_{w}s
  cards per BIN           -> bin_distinct_cards_{w}s
  merchants per BIN       -> bin_distinct_merchants_{w}s
  BIN overlap (device)    -> dev_distinct_bins_{w}s
  BIN overlap (ip)        -> ip_distinct_bins_{w}s

BIN handling (DECISIONS.md, locked):
  * `bin` is read from the dataset as TEXT and factorised to an opaque category code.
    It is never used as a numeric quantity: no differences, no z-scores, no means.
  * BIN is never derived from card_fingerprint. If the dataset ever lacks a `bin`
    column the schema resolver fails loudly rather than fabricating one.

BIN overlap is implemented DIRECTIONALLY (device -> distinct BINs, ip -> distinct BINs)
because that is the direction the attack expresses: one machine or network presenting
cards from several issuer ranges. The reverse direction (BIN -> devices) would be an
issuer-population statistic, not attacker behaviour.

Elevated BIN activity is a population statistic and is expected to be a weak signal:
a popular issuer prefix is busy for legitimate reasons.
"""
from __future__ import annotations

import numpy as np

from .build_feature_table import feature_spec

GROUP = "bin"


def declare(cfg, ctx) -> list[dict]:
    specs: list[dict] = []
    for w in cfg.windows_for(GROUP):
        specs += [
            feature_spec(
                f"bin_txn_count_{w}s", group=GROUP, entity="bin", window=w,
                range_="[0, inf)", missing="never missing; 0 means no prior transaction on this BIN in the window",
                purpose="prior transactions on this issuer BIN inside the window (BIN activity)",
                spec_ref=("BIN.activity",), config=("windows_s",),
            ),
            feature_spec(
                f"bin_distinct_cards_{w}s", group=GROUP, entity="bin", window=w,
                range_="[0, inf)", missing="never missing; 0 means no prior card on this BIN in the window",
                purpose="distinct prior cards on this BIN inside the window (cards per BIN)",
                spec_ref=("BIN.distinct_cards",), config=("windows_s",),
            ),
            feature_spec(
                f"bin_distinct_merchants_{w}s", group=GROUP, entity="bin", window=w,
                range_="[0, inf)", missing="never missing; 0 means no prior merchant on this BIN in the window",
                purpose="distinct prior merchants touched by this BIN inside the window (merchants per BIN)",
                spec_ref=("BIN.distinct_merchants",), config=("windows_s",),
            ),
            feature_spec(
                f"dev_distinct_bins_{w}s", group=GROUP, entity="device_fingerprint", window=w,
                range_="[0, inf)", missing="never missing; 0 means no prior BIN seen on this device in the window",
                purpose="distinct prior BINs presented by THIS device inside the window (BIN overlap, device side)",
                spec_ref=("BIN.overlap_device",), config=("windows_s",),
            ),
            feature_spec(
                f"ip_distinct_bins_{w}s", group=GROUP, entity="ip_signature", window=w,
                range_="[0, inf)", missing="never missing; 0 means no prior BIN seen from this IP in the window",
                purpose="distinct prior BINs seen from THIS ip_signature inside the window (BIN overlap, IP side)",
                spec_ref=("BIN.overlap_ip",), config=("windows_s",),
            ),
        ]
    return specs


def build(ctx) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for w in ctx.cfg.windows_for(GROUP):
        out[f"bin_txn_count_{w}s"] = ctx.count("bin", w)
        out[f"bin_distinct_cards_{w}s"] = ctx.nunique("bin", "card", w)
        out[f"bin_distinct_merchants_{w}s"] = ctx.nunique("bin", "merchant", w)
        out[f"dev_distinct_bins_{w}s"] = ctx.nunique("device", "bin", w)
        out[f"ip_distinct_bins_{w}s"] = ctx.nunique("ip", "bin", w)
    return out
