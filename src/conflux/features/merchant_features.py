"""Merchant Pattern signal group -- cross-merchant behaviour, the central CONFLUX signal.

This module owns EVERY entity -> merchant relationship, for all three actor entities,
so that no other module recomputes or renames the same quantity:

  distinct merchants by entity   -> {card,device,ip}_distinct_merchants_{w}s
  cross-merchant spread          -> {card,device,ip}_merchant_spread_{w}s
  merchant dispersion (entropy)  -> {card,device,ip}_merchant_entropy_{w}s
  related entities across merch. -> merchant_related_devices_{w}s, merchant_related_ips_{w}s
  (merchant window context)      -> mer_txn_count_{w}s   [documented deviation]

DIRECTION IS LOCKED (this is the audited historical error). Dispersion is measured as

    ENTITY -> MERCHANT DISTRIBUTION      (card / device / IP spreading across merchants)

and never as MERCHANT -> CARD DISTRIBUTION (how concentrated card activity is at one
merchant). The previous mer_card_entropy measured the latter while being labelled the
former; it is not reintroduced under any name.

Entropy is normalised Shannon entropy of the entity's prior transactions over distinct
merchants in the window: H = -sum p_i log p_i, normalised by log(k). It is NaN while
k < cfg.min_distinct_for_entropy, because dispersion over one category is undefined.
Low value = concentrated on one merchant; high value = evenly spread across many, which
is the coordinated cross-merchant pattern the project targets. It is the distribution-
shape companion to the raw distinct-merchant counts, not a substitute for them.

LAYER BOUNDARY: merchant_related_* is a plain count of related entities inside a causal
window. It builds no edges, no components, no campaign identity and no campaign-level
statistic. Graph construction and campaign discovery belong to src/conflux/graph/.
"""
from __future__ import annotations

import numpy as np

from .build_feature_table import feature_spec

GROUP = "merchant"
ACTORS = ("card", "device", "ip")
_ACTOR_SOURCE = {"card": "card_fingerprint", "device": "device_fingerprint", "ip": "ip_signature"}
# The Device group's own locked list also requires "distinct merchants per device";
# that item is satisfied here rather than duplicated in device_features.py.
_EXTRA_REFS = {"device": ("DEVICE.distinct_merchants",)}


def declare(cfg, ctx) -> list[dict]:
    specs: list[dict] = []
    min_m = cfg.related_entity_min_merchants
    for w in cfg.windows_for(GROUP):
        specs.append(feature_spec(
            f"mer_txn_count_{w}s", group=GROUP, entity="merchant_id", window=w,
            range_="[0, inf)", missing="never missing; 0 means no prior transaction at this merchant in the window",
            purpose=("prior transactions at THIS merchant inside the window. Emitted as the scale "
                     "for the related-entity counts below, which are raw counts and are not "
                     "interpretable without the merchant's own window volume."),
            spec_ref=("CONTEXT.merchant_window_txn_count",), config=("windows_s",),
        ))
        for a in ACTORS:
            src = _ACTOR_SOURCE[a]
            extra = _EXTRA_REFS.get(a, ())
            specs += [
                feature_spec(
                    f"{a}_distinct_merchants_{w}s", group=GROUP, entity=src, window=w,
                    range_="[0, inf)", missing="never missing; 0 means no prior merchant for this entity in the window",
                    purpose=f"distinct prior merchants touched by this {a} inside the window (cross-merchant reach)",
                    spec_ref=(f"MERCHANT.distinct_merchants_{a}",) + extra, config=("windows_s",),
                ),
                feature_spec(
                    f"{a}_merchant_spread_{w}s", group=GROUP, entity=src, window=w,
                    range_="[0, 1]",
                    missing=f"NaN with fewer than {cfg.min_prior_for_ratio} prior transactions for this entity in the window",
                    purpose=(f"distinct prior merchants divided by prior transactions for this {a}: "
                             "1.0 means every attempt hit a different merchant, low values mean "
                             "repeated activity at the same merchant. Zero denominator is explicitly NaN."),
                    spec_ref=("MERCHANT.cross_merchant_spread",),
                    config=("windows_s", "min_prior_for_ratio", "eps"),
                ),
                feature_spec(
                    f"{a}_merchant_entropy_{w}s", group=GROUP, entity=src, window=w,
                    range_="[0, 1]",
                    missing=f"NaN with fewer than {cfg.min_distinct_for_entropy} distinct prior merchants for this entity",
                    purpose=(f"normalised Shannon entropy of this {a}'s prior transactions across distinct "
                             "merchants inside the window (entity -> merchant dispersion). High = evenly "
                             "spread across merchants; low = concentrated on one merchant."),
                    spec_ref=("MERCHANT.dispersion_entropy",),
                    config=("windows_s", "min_distinct_for_entropy"),
                ),
            ]
        specs += [
            feature_spec(
                f"merchant_related_devices_{w}s", group=GROUP, entity="merchant_id", window=w,
                range_="[0, inf)", missing="never missing; 0 means no such device at this merchant in the window",
                purpose=(f"distinct devices with prior activity at THIS merchant inside the window that also "
                         f"touched at least {min_m} distinct merchants inside the same window. Behavioural "
                         "precursor to the graph layer; a raw count, pair it with mer_txn_count."),
                spec_ref=("MERCHANT.related_entities",),
                config=("windows_s", "related_entity_min_merchants"),
            ),
            feature_spec(
                f"merchant_related_ips_{w}s", group=GROUP, entity="merchant_id", window=w,
                range_="[0, inf)", missing="never missing; 0 means no such IP at this merchant in the window",
                purpose=(f"same measure for ip_signature: distinct IPs with prior activity at this merchant "
                         f"inside the window that also touched at least {min_m} distinct merchants."),
                spec_ref=("MERCHANT.related_entities",),
                config=("windows_s", "related_entity_min_merchants"),
            ),
        ]
    return specs


def build(ctx) -> dict[str, np.ndarray]:
    cfg = ctx.cfg
    out: dict[str, np.ndarray] = {}

    for w in cfg.windows_for(GROUP):
        out[f"mer_txn_count_{w}s"] = ctx.count("merchant", w)

        for a in ACTORS:
            n = ctx.count(a, w)                       # shared with velocity via the memo cache
            distinct = ctx.nunique(a, "merchant", w)
            out[f"{a}_distinct_merchants_{w}s"] = distinct
            out[f"{a}_merchant_spread_{w}s"] = ctx.gate(
                ctx.safe_div(distinct, n), n, cfg.min_prior_for_ratio)
            out[f"{a}_merchant_entropy_{w}s"] = ctx.entropy(a, "merchant", w)

        out[f"merchant_related_devices_{w}s"] = ctx.related_entity_counts("device", w)
        out[f"merchant_related_ips_{w}s"] = ctx.related_entity_counts("ip", w)

    return out
