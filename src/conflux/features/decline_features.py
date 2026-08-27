"""Decline Ratio signal group -- prior authorisation failure rates, four entities.

Locked Decline spec items, all four implemented (the historical gap was device and IP):
  card decline rate     -> dec_card_rate_{w}s
  device decline rate   -> dec_device_rate_{w}s
  ip decline rate       -> dec_ip_rate_{w}s
  merchant decline rate -> dec_merchant_rate_{w}s

Formulation is strictly causal and identical for all four entities:

    prior declined transactions for the entity in the window
    -------------------------------------------------------- , NaN below min_history
    prior transactions for the entity in the window

The CURRENT transaction's auth_outcome never contributes to its own decline features.
That is the only formulation that is valid for a live detector, where the outcome of the
transaction being scored is not yet known.

FORBIDDEN HERE: campaign-level decline rate. campaign_id is evaluation ground truth and
campaigns have not been discovered at this point in the pipeline. Any campaign-level
authorisation statistic belongs after graph-based campaign detection, in
src/conflux/scoring/, and must be derived from discovered candidates rather than labels.

AUDIT WARNING. These features may show a high univariate AUC on V4 because the dataset's
attack campaigns are generated with a much higher decline rate than normal traffic. That
is legitimate observed-signal, not label leakage (auth_outcome is a transaction
attribute, not ground truth) and not temporal leakage (only prior rows are used). It is
still a generator property, so treat the AUC as an audit number, expect it to be flagged
for investigation by the build, and do not let a decline threshold become the detector.

Raw prior-decline COUNTS are deliberately not emitted: the locked spec asks for rates,
and count == rate * window_count would be a near-duplicate of features already present.
"""
from __future__ import annotations

import numpy as np

from .build_feature_table import feature_spec

GROUP = "decline"
ENTITIES = ("card", "device", "ip", "merchant")
_ENTITY_SOURCE = {
    "card": "card_fingerprint",
    "device": "device_fingerprint",
    "ip": "ip_signature",
    "merchant": "merchant_id",
}


def declare(cfg, ctx) -> list[dict]:
    if "auth" not in ctx.schema:
        raise KeyError("no authorisation-outcome column resolved; the Decline group cannot be built")
    specs: list[dict] = []
    k = cfg.min_history
    for w in cfg.windows_for(GROUP):
        for e in ENTITIES:
            specs.append(feature_spec(
                f"dec_{e}_rate_{w}s", group=GROUP, entity=_ENTITY_SOURCE[e], window=w,
                range_="[0, 1]",
                missing=f"NaN with fewer than {k} prior transactions for this entity in the window",
                purpose=(f"prior declined transactions divided by prior transactions for this {e} inside "
                         "the window. The current transaction's own auth_outcome is excluded."),
                spec_ref=(f"DECLINE.rate_{e}",),
                config=("windows_s", "min_history", "auth_approved_tokens", "auth_declined_tokens", "eps"),
            ))
    return specs


def build(ctx) -> dict[str, np.ndarray]:
    cfg = ctx.cfg
    out: dict[str, np.ndarray] = {}
    for w in cfg.windows_for(GROUP):
        for e in ENTITIES:
            attempts = ctx.count(e, w)
            declines = ctx.sum(e, "declined", w)
            out[f"dec_{e}_rate_{w}s"] = ctx.gate(
                ctx.safe_div(declines, attempts), attempts, cfg.min_history)
    return out
