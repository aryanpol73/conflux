"""Device signal group -- device/card coupling and device reuse.

Locked Device spec items and where each is implemented (single ownership, no duplicates):
  device transaction count in window  -> velocity_features.vel_device_count_{w}s
  distinct cards per device           -> HERE: dev_distinct_cards_{w}s
  distinct merchants per device       -> merchant_features.device_distinct_merchants_{w}s
  device velocity                     -> velocity_features (count per fixed window + burst density)
  device reuse                        -> HERE: dev_reuse_prior_total

Why the count and velocity live in velocity_features: a rolling count of prior device
transactions inside window W and "device transaction count in window W" are the same
number. The historical implementation emitted both (dev_txn_count_{w}s and
vel_device_count_{w}s) plus a per-minute rescaling, i.e. one quantity under three names.
That is the "duplicate features under different names" failure, so the count is computed
once, in the velocity module, and referenced from here.

Shared device activity is NOT automatically fraudulent. The V4 dataset deliberately
contains legitimate shared-device clusters, so these features are weak evidence to be
combined with other groups and later with the graph layer -- never a standalone rule.
"""
from __future__ import annotations

import numpy as np

from .build_feature_table import feature_spec

GROUP = "device"


def declare(cfg, ctx) -> list[dict]:
    specs = [feature_spec(
        "dev_reuse_prior_total", group=GROUP, entity="device_fingerprint", window=None,
        range_="[0, inf)", missing="never missing; 0 means first sighting of this device",
        purpose=("count of ALL PRIOR occurrences of this device fingerprint (device reuse). "
                 "Unwindowed but strictly causal: only rows earlier in the causal order count."),
        spec_ref=("DEVICE.reuse",), config=(),
    )]
    for w in cfg.windows_for(GROUP):
        specs.append(feature_spec(
            f"dev_distinct_cards_{w}s", group=GROUP, entity="device_fingerprint", window=w,
            range_="[0, inf)", missing="never missing; 0 means no prior card seen on this device in the window",
            purpose=("distinct PRIOR cards seen on this device inside the window (cards per device). "
                     "One device presenting many cards is the core card-testing coupling."),
            spec_ref=("DEVICE.distinct_cards",), config=("windows_s",),
        ))
    return specs


def build(ctx) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {
        "dev_reuse_prior_total": ctx.count("device", None),
    }
    for w in ctx.cfg.windows_for(GROUP):
        out[f"dev_distinct_cards_{w}s"] = ctx.nunique("device", "card", w)
    return out
