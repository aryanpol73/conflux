"""CONFLUX -- explainability formatter for the deterministic scorer.

The deterministic scorer is fully transparent: a campaign's score is a
weighted average of per-feature percentiles. ``DeterministicScorer.transform``
returns the weighted contributions but discards the percentiles, so this
module recovers them algebraically:

    contribution_i = weight_i * percentile_i / sum(weights)
    =>  percentile_i = contribution_i * sum(weights) / weight_i

This is an exact inversion, not an approximation. There is no surrogate model
and no SHAP sampling involved -- the numbers shown to an analyst are the same
numbers the scorer used to rank the campaign.

Sign handling: for a feature registered with sign -1, the scorer flips the
percentile (``pct = 1.0 - pct``) before weighting, because a *low* raw value is
the suspicious direction. The recovered quantity is therefore a
*suspicion percentile* -- always "higher means more unusual" -- and the raw
percentile is reconstructed by flipping it back.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

__all__ = [
    "PHRASE",
    "STRONG_PERCENTILE",
    "ORDINARY_PERCENTILE",
    "humanize_feature",
    "percentile_map",
    "explain_campaign",
    "enrich_result",
]


#: (high_text, low_text) per feature. The high text completes
#: "... 97% of scored groups"; the low text completes the inverted phrasing
#: "... 98% of scored groups" when the feature sits below the median.
#: This dictionary is the only place display language lives.
PHRASE: dict[str, tuple[str, str]] = {
    "burst_rate_per_minute": (
        "transactions arrived in a tighter time window than",
        "transactions were more spread out in time than",
    ),
    "link_density": (
        "the group is more tightly interconnected than",
        "the group is more loosely connected than",
    ),
    "max_transactions_per_shared_card": (
        "reuses a single card more heavily than",
        "reuses a single card less heavily than",
    ),
    "multi_entity_link_fraction": (
        "more of its links run through several shared entities than",
        "fewer of its links run through several shared entities than",
    ),
    "distinct_merchants_per_transaction": (
        "spreads across more different merchants than",
        "concentrates on fewer merchants than",
    ),
    "distinct_bins_per_transaction": (
        "uses a wider spread of card BINs than",
        "uses a narrower spread of card BINs than",
    ),
}

#: At or above this suspicion percentile a signal is called out as standing out.
STRONG_PERCENTILE = 0.90

#: Below this suspicion percentile a signal is explicitly called unremarkable.
ORDINARY_PERCENTILE = 0.60


def humanize_feature(feature: str) -> tuple[str, str]:
    """Return (high_text, low_text) for a feature, with a readable fallback."""
    if feature in PHRASE:
        return PHRASE[feature]
    pretty = feature.replace("_", " ")
    return (f"its {pretty} is higher than", f"its {pretty} is lower than")


def _reference_parts(reference: Any) -> tuple[list[str], list[float], list[int]]:
    """Pull the three parallel arrays off a ScorerReference."""
    names = [str(n) for n in getattr(reference, "feature_names")]
    weights = [float(w) for w in getattr(reference, "weights")]
    signs = [int(s) for s in getattr(reference, "signs")]
    if not (len(names) == len(weights) == len(signs)):
        raise ValueError(
            "scorer reference is inconsistent: "
            f"{len(names)} names, {len(weights)} weights, {len(signs)} signs"
        )
    return names, weights, signs


def percentile_map(reference: Any) -> dict[str, dict[str, float]]:
    """Per-feature inversion factor, weight and sign, keyed by feature name.

    ``factor`` is what a weighted contribution must be multiplied by to recover
    the suspicion percentile. Zero-weight features are omitted: they contribute
    nothing, so nothing can be inverted.
    """
    names, weights, signs = _reference_parts(reference)
    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("scorer reference has non-positive total weight")

    out: dict[str, dict[str, float]] = {}
    for name, weight, sign in zip(names, weights, signs):
        if weight <= 0:
            continue
        out[name] = {"factor": wsum / weight, "weight": weight, "sign": float(sign)}
    return out


def _member_count(campaign: Mapping[str, Any]) -> int:
    """Best-effort transaction count for a campaign payload."""
    ids = campaign.get("transaction_ids")
    if isinstance(ids, (list, tuple, set)):
        return len(ids)
    for key in ("size", "n_transactions", "transaction_count", "member_count"):
        value = campaign.get(key)
        # bool is an int subclass; a True here would silently become 1.
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return 0


def _describe_signal(
    feature: str,
    contribution: float,
    info: Mapping[str, float],
) -> dict[str, Any]:
    """Decode one weighted contribution into a percentile plus a sentence."""
    suspicion = float(contribution) * float(info["factor"])
    suspicion = min(1.0, max(0.0, suspicion))

    sign = int(info["sign"])
    raw = (1.0 - suspicion) if sign < 0 else suspicion

    high_text, low_text = humanize_feature(feature)
    if raw >= 0.5:
        sentence = f"{high_text} {raw:.0%} of scored groups."
    else:
        sentence = f"{low_text} {1.0 - raw:.0%} of scored groups."

    return {
        "feature": feature,
        "label": feature.replace("_", " "),
        "suspicion_percentile": round(suspicion, 4),
        "percentile_display": f"{suspicion:.0%}",
        "contribution": round(float(contribution), 4),
        "sign": sign,
        "sentence": sentence,
        "is_strong": suspicion >= STRONG_PERCENTILE,
    }


def explain_campaign(reference: Any, campaign: Mapping[str, Any]) -> dict[str, Any]:
    """Build a plain-English explanation block for one campaign payload.

    Reads ``campaign["evidence"]["top_signals"]``, a list of
    ``{"feature": str, "contribution": float}`` entries produced by the
    pipeline. Render ``all_signals`` in the UI -- it is ordered most-to-least
    unusual and carries an ``is_strong`` flag for styling. ``strong_signals``
    and ``ordinary_signals`` are convenience subsets and do not partition the
    whole list: a mid-range feature belongs to neither.
    """
    lookup = percentile_map(reference)
    total_features = len(lookup)

    evidence = campaign.get("evidence") or {}
    raw_signals: Iterable[Any] = evidence.get("top_signals") or []

    items: list[dict[str, Any]] = []
    for entry in raw_signals:
        if not isinstance(entry, Mapping):
            continue
        feature = str(entry.get("feature", ""))
        info = lookup.get(feature)
        if info is None:
            continue
        try:
            contribution = float(entry.get("contribution", 0.0))
        except (TypeError, ValueError):
            continue
        items.append(_describe_signal(feature, contribution, info))

    items.sort(key=lambda d: -float(d["suspicion_percentile"]))
    strong = [d for d in items if d["is_strong"]]
    ordinary = [
        d for d in items if float(d["suspicion_percentile"]) < ORDINARY_PERCENTILE
    ]

    if not items:
        verdict = "No scored signals were available for this group."
    elif not strong:
        verdict = "Nothing stands out -- this group is unremarkable on every measure."
    elif len(strong) == 1:
        verdict = (
            f"{strong[0]['label']} is the one property that stands out; "
            "the rest are unremarkable."
        )
    else:
        names = ", ".join(d["label"] for d in strong)
        verdict = f"{len(strong)} properties are unusual together: {names}."

    members = _member_count(campaign)
    summary = (
        f"{members} transactions the graph linked together through shared "
        "payment entities (cards, devices, IP signatures)."
        if members
        else "Transactions the graph linked together through shared payment entities."
    )

    return {
        "summary": summary,
        "headline": items[0]["sentence"] if items else "",
        "verdict": verdict,
        "strong_signals": strong,
        "ordinary_signals": ordinary,
        "all_signals": items,
        "signals_shown": len(items),
        "signals_total": total_features,
        "truncated": len(items) < total_features,
        "method": (
            "Percentiles recovered exactly from the scorer's own weighted "
            "contributions. No separate model is involved."
        ),
        "caveat": "This is a ranking signal, not a probability of fraud.",
    }


def enrich_result(reference: Any, result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Attach an ``explanation`` block to every campaign, in place.

    Deliberately non-fatal: if a campaign payload is shaped unexpectedly the
    explanation is skipped and the response is returned untouched. Detection
    must never fail because a display helper did.
    """
    campaigns = result.get("campaigns") if isinstance(result, Mapping) else None
    if not isinstance(campaigns, list):
        return result

    for campaign in campaigns:
        if not isinstance(campaign, dict):
            continue
        try:
            campaign["explanation"] = explain_campaign(reference, campaign)
        except Exception:  # noqa: BLE001 - display helper, never fatal
            continue
    return result
