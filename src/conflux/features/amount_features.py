"""Amount signal group -- causal amount baselines for the two locked amount entities.

FEATURE_SPEC.md / locked spec items implemented here:
  historical amount deviation (card, device)  -> amt_{e}_dev_{w}s, amt_{e}_z_{w}s
  amount similarity / clustering             -> amt_{e}_cv_{w}s
  low-value ratio                            -> amt_card_low_value_ratio_{w}s
  causal window statistics                   -> amt_{e}_mean/median/std_{w}s

Entity scope is deliberately limited to card and device: the locked spec names exactly
these two, and explicitly warns against spontaneously adding merchant / IP / BIN amount
statistics. Nothing here is emitted for a third entity.

ASSUMPTION FLAGGED: "low-value" has no threshold in the spec. cfg.low_value_amount
(default 5.0) is an explicit, documented PROJECT ASSUMPTION -- it is not derived from
production data and was not selected using `label`. The default may produce a feature
that is almost entirely zero on V4; that is a property of the data, not a defect, and
it is left in place because the locked spec requires the concept. Confirm the value
with the spec owner before drawing conclusions from this feature.

Amount is a SUPPORTING signal. It must not become "amount < X -> fraud".
"""
from __future__ import annotations

import numpy as np

from .build_feature_table import feature_spec

GROUP = "amount"
ENTITIES = ("card", "device")          # locked amount entities; do not extend here
_ENTITY_SOURCE = {"card": "card_fingerprint", "device": "device_fingerprint"}


def declare(cfg, ctx) -> list[dict]:
    specs: list[dict] = []
    k = cfg.min_history
    for w in cfg.windows_for(GROUP):
        for e in ENTITIES:
            src = _ENTITY_SOURCE[e]
            specs += [
                feature_spec(
                    f"amt_{e}_mean_{w}s", group=GROUP, entity=src, window=w,
                    range_="[0, inf)", missing="NaN when there is no prior transaction for this entity in the window",
                    purpose=f"causal window mean of PRIOR {e} amounts (baseline for deviation)",
                    spec_ref=("AMOUNT.window_mean",), config=("windows_s",),
                ),
                feature_spec(
                    f"amt_{e}_median_{w}s", group=GROUP, entity=src, window=w,
                    range_="[0, inf)", missing="NaN when there is no prior transaction for this entity in the window",
                    purpose=f"causal window median of PRIOR {e} amounts (outlier-robust baseline)",
                    spec_ref=("AMOUNT.window_median",), config=("windows_s",),
                ),
                feature_spec(
                    f"amt_{e}_std_{w}s", group=GROUP, entity=src, window=w,
                    range_="[0, inf)", missing=f"NaN with fewer than {k} prior transactions in the window",
                    purpose=f"causal window standard deviation of PRIOR {e} amounts",
                    spec_ref=("AMOUNT.window_std",), config=("windows_s", "min_history"),
                ),
                feature_spec(
                    f"amt_{e}_dev_{w}s", group=GROUP, entity=src, window=w,
                    range_="(-inf, inf)", missing="NaN when the prior mean is undefined",
                    purpose=f"current amount minus the prior {e} window mean (absolute deviation form)",
                    spec_ref=(f"AMOUNT.historical_deviation_{e}",), config=("windows_s",),
                ),
                feature_spec(
                    f"amt_{e}_z_{w}s", group=GROUP, entity=src, window=w,
                    range_="(-inf, inf)", missing=f"NaN with fewer than {k} prior transactions in the window",
                    purpose=f"(amount - prior {e} mean) / prior {e} std (scale-free deviation form)",
                    spec_ref=(f"AMOUNT.historical_deviation_{e}",), config=("windows_s", "min_history"),
                ),
                feature_spec(
                    f"amt_{e}_cv_{w}s", group=GROUP, entity=src, window=w,
                    range_="[0, inf)", missing=f"NaN with fewer than {k} prior transactions, or a non-positive prior mean",
                    purpose=(f"coefficient of variation of PRIOR {e} amounts (std/mean): amount "
                             "similarity/clustering, low values mean repeated near-identical amounts"),
                    spec_ref=("AMOUNT.similarity_cv",), config=("windows_s", "min_history", "eps"),
                ),
            ]
        specs.append(feature_spec(
            f"amt_card_low_value_ratio_{w}s", group=GROUP, entity="card_fingerprint", window=w,
            range_="[0, 1]", missing=f"NaN with fewer than {k} prior card transactions in the window",
            purpose=(f"fraction of PRIOR card transactions in the window below "
                     f"cfg.low_value_amount={cfg.low_value_amount} (card-testing probe behaviour). "
                     "The threshold is a documented project ASSUMPTION, not a data-derived value."),
            spec_ref=("AMOUNT.low_value_ratio",),
            config=("windows_s", "min_history", "low_value_amount"),
        ))
    return specs


def build(ctx) -> dict[str, np.ndarray]:
    cfg = ctx.cfg
    amount = ctx.channel("amount")
    out: dict[str, np.ndarray] = {}

    for w in cfg.windows_for(GROUP):
        for e in ENTITIES:
            n = ctx.count(e, w)
            mean = ctx.mean(e, "amount", w)
            std = ctx.std(e, "amount", w)

            out[f"amt_{e}_mean_{w}s"] = mean
            out[f"amt_{e}_median_{w}s"] = ctx.median(e, "amount", w)
            out[f"amt_{e}_std_{w}s"] = ctx.gate(std, n, cfg.min_history)
            # mean is already NaN with zero prior history, so the deviation inherits it.
            out[f"amt_{e}_dev_{w}s"] = amount - mean
            out[f"amt_{e}_z_{w}s"] = ctx.gate(ctx.safe_div(amount - mean, std), n, cfg.min_history)
            out[f"amt_{e}_cv_{w}s"] = ctx.gate(ctx.safe_div(std, mean), n, cfg.min_history)

        n_card = ctx.count("card", w)
        out[f"amt_card_low_value_ratio_{w}s"] = ctx.gate(
            ctx.mean("card", "low_value", w), n_card, cfg.min_history)

    return out
