"""Velocity signal group -- temporal concentration, kept separate per entity.

Locked Velocity spec items:
  rolling transaction count (card / device / IP)  -> vel_{entity}_count_{w}s
  transactions per minute                         -> see the note below (config switch)
  burst density                                   -> vel_{entity}_burst_density_{narrow}s_{wide}s
  window consistency                              -> every window uses the one causal engine

Card, device and IP velocity are NEVER collapsed into a single "entity velocity": the
identity of the bursting entity is itself the evidence, and the six-group design relies
on those three being distinguishable.

TRANSACTIONS PER MINUTE. rate = count * 60 / W is an exact deterministic rescaling of
vel_{entity}_count_{w}s, which is already emitted for every configured window. Emitting
both is the same quantity under two names and is perfectly collinear for the Logistic
Regression baseline, so it is OFF by default. Set features.emit_rate_per_min=true in the
project configuration to emit it; the count features carry the identical information.

This module also satisfies the Device group's "device transaction count in window" and
"device velocity" items, so device_features.py does not recompute them.

BURST DENSITY is rate-normalised: (narrow count / narrow width) / (wide count / wide
width). Above 1.0 means activity is accelerating into the present. It is NaN when the
wide window contains no prior transaction, so its missingness is informative and is
audited via missing_indicator_auc rather than silently imputed. Bursts are also a
designed property of the synthetic V4 campaigns, so a high audit AUC here is expected
and must be read as a generator artifact indicator, not as validation of the feature.
"""
from __future__ import annotations

import numpy as np

from .build_feature_table import feature_spec

GROUP = "velocity"
ENTITIES = ("card", "device", "ip")
_ENTITY_SOURCE = {"card": "card_fingerprint", "device": "device_fingerprint", "ip": "ip_signature"}
_EXTRA_COUNT_REFS = {"device": ("DEVICE.txn_count", "DEVICE.velocity")}


def declare(cfg, ctx) -> list[dict]:
    windows = cfg.windows_for(GROUP)
    narrow, wide = cfg.narrow_wide(GROUP)
    specs: list[dict] = []

    for e in ENTITIES:
        src = _ENTITY_SOURCE[e]
        extra = _EXTRA_COUNT_REFS.get(e, ())
        for w in windows:
            specs.append(feature_spec(
                f"vel_{e}_count_{w}s", group=GROUP, entity=src, window=w,
                range_="[0, inf)", missing="never missing; 0 means no prior transaction for this entity in the window",
                purpose=(f"rolling count of PRIOR {e} transactions inside the window. With a fixed window "
                         f"this is also the {e} velocity (transactions per {w}s)."),
                spec_ref=(f"VELOCITY.rolling_count_{e}",) + extra, config=("windows_s",),
            ))
            if cfg.emit_rate_per_min:
                specs.append(feature_spec(
                    f"vel_{e}_rate_per_min_{w}s", group=GROUP, entity=src, window=w,
                    range_="[0, inf)", missing="never missing",
                    purpose=(f"prior {e} transactions per minute inside the window. Exact rescaling of "
                             f"vel_{e}_count_{w}s; emitted only because emit_rate_per_min is enabled."),
                    spec_ref=("VELOCITY.txns_per_minute",), config=("windows_s", "emit_rate_per_min"),
                ))
        if len(windows) >= 2:
            specs.append(feature_spec(
                f"vel_{e}_burst_density_{narrow}s_{wide}s", group=GROUP, entity=src,
                window=f"({narrow}, {wide})", range_="[0, inf)",
                missing=f"NaN when this entity has no prior transaction inside the {wide}s window",
                purpose=(f"rate-normalised ratio of {e} activity in the narrowest window to the widest "
                         "window: > 1 means the entity's activity is accelerating into the present"),
                spec_ref=("VELOCITY.burst_density",), config=("windows_s",),
            ))
    return specs


def build(ctx) -> dict[str, np.ndarray]:
    cfg = ctx.cfg
    windows = cfg.windows_for(GROUP)
    narrow, wide = cfg.narrow_wide(GROUP)
    out: dict[str, np.ndarray] = {}

    for e in ENTITIES:
        counts: dict[int, np.ndarray] = {}
        for w in windows:
            n = ctx.count(e, w)                    # shared with the other groups via the memo cache
            counts[w] = n
            out[f"vel_{e}_count_{w}s"] = n
            if cfg.emit_rate_per_min:
                out[f"vel_{e}_rate_per_min_{w}s"] = n * (60.0 / float(w))
        if len(windows) >= 2:
            # (narrow / narrow_width) / (wide / wide_width) == narrow / (wide * narrow/wide)
            out[f"vel_{e}_burst_density_{narrow}s_{wide}s"] = ctx.safe_div(
                counts[narrow], counts[wide] * (float(narrow) / float(wide)))

    return out
