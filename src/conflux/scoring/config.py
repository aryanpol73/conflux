"""CONFLUX Phase 4A -- scoring configuration.

AUTHORITY OF RECORD
-------------------
Every feature below is justified by a WRITTEN DOMAIN ARGUMENT, not by a
Phase 3C separation statistic. No constant in this file was read off a
diagnostic table. If a feature's argument does not survive scrutiny, the
feature is removed regardless of how well it separated in Phase 3C.

WHAT PHASE 4A IS
----------------
Retrospective candidate triage. It scores and ranks the FINALIZED Phase 3B
connected components, whose evidence spans the whole group. It is NOT a
decision-time transaction score; the anchor-prefix variant is Phase 4D.

WHAT PHASE 4A MAY NOT DO
------------------------
Create, merge, split, re-form or re-order candidates. Read label / campaign_id
/ _source_type as a feature. Use a threshold discovered on the data it reports
metrics for. Modify anything in conflux.graph or conflux.evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from conflux.config import PROCESSED_DIR, RAW_DATASET_PATH

SCORING_SCHEMA_VERSION = "conflux.scoring.phase4a.v1"

# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
GRAPH_ARTIFACT_DIR: Path = PROCESSED_DIR / "graph"
CANDIDATES_PATH: Path = GRAPH_ARTIFACT_DIR / "campaign_candidates.csv"
ASSIGNMENTS_PATH: Path = GRAPH_ARTIFACT_DIR / "campaign_candidate_assignments.csv"
SCORING_OUT_DIR: Path = PROCESSED_DIR / "scoring"
FROZEN_PATHS: frozenset[Path] = frozenset({
    Path(RAW_DATASET_PATH).resolve(), GRAPH_ARTIFACT_DIR.resolve()})

# structural / attribute columns Phase 4 is permitted to read from the raw CSV.
# card_fingerprint is a graph ENTITY column, not ground truth (decision locked).
ALLOWED_RAW_COLUMNS: tuple[str, ...] = (
    "transaction_id", "card_fingerprint", "amount", "auth_outcome")

# ----------------------------------------------------------------------
# granularity floor -- NOT a tuned threshold
# ----------------------------------------------------------------------
# A candidate whose members share one timestamp has span 0, which makes a
# rate undefined. Rather than emit NaN and then invent an imputation rule, the
# span is floored at one second. This is a statement about timestamp
# granularity, fixed before any result was inspected, and it is monotone-safe:
# every sub-second burst collapses to the same maximal rate.
MIN_SPAN_SECONDS: float = 1.0

# ----------------------------------------------------------------------
# feature registry
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    sign: int          # +1 => higher is more suspicious
    source: str        # "phase3c" (read from build_candidate_features) or "phase4"
    rationale: str


CORE_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "burst_rate_per_minute", "timing", +1, "phase4",
        "Coordinated activity is executed by a script and therefore compresses "
        "in time. Organic activity from independent cardholders does not "
        "concentrate, because the arrivals are independent. Rate, not raw count, "
        "so this is not a proxy for candidate size."),
    FeatureSpec(
        "link_density", "structure", +1, "phase3c",
        "An operation reusing one set of infrastructure produces a near-clique: "
        "most member pairs share something. Incidental co-occurrence produces a "
        "sparse chain, because each link is a separate coincidence. Normalised "
        "by the maximum possible edge count, so it is size-independent."),
    FeatureSpec(
        "max_transactions_per_shared_card", "card", +1, "phase4",
        "Repeated presentation of one card inside a short burst is card-testing "
        "behaviour. A legitimate cardholder rarely transacts many times within "
        "one window, and when they do it is at one merchant."),
    FeatureSpec(
        "multi_entity_link_fraction", "connectivity", +1, "phase3c",
        "Two transactions sharing card AND device AND IP is a far stronger join "
        "than any single shared entity, which can arise from NAT, shared "
        "hardware, or a recycled fingerprint. The fraction of links that are "
        "corroborated by two or more entity types measures join strength."),
    FeatureSpec(
        "distinct_merchants_per_transaction", "merchant", +1, "phase4",
        "The project's own graph configuration states that cross-merchant "
        "spread is the attack signature and that merchant co-occurrence is "
        "ambient. Normal bursts concentrate at one merchant; an operation "
        "distributes to avoid per-merchant velocity controls."),
    FeatureSpec(
        "distinct_bins_per_transaction", "bin_context", +1, "phase4",
        "Issuer spread inside a single short window implies a pool of cards "
        "from multiple issuers rather than one cardholder's wallet. BIN is used "
        "here strictly as CONTEXT: it never joins two transactions, and the "
        "no-BIN ablation is mandatory in every report."),
    FeatureSpec(
        "max_identical_amount_share", "amount", +1, "phase4",
        "Scripted probing repeats a fixed low-value amount to test validity. "
        "Independent purchases produce a spread of amounts. Expressed as a "
        "share of the group so it is size-independent."),
)

# Available, computed, and reported -- but NOT in the live scorer without a
# written argument. auth_outcome is ablation-only by locked decision: it is a
# post-authorization field and its decision-time status is unresolved.
ABLATION_FEATURES: tuple[str, ...] = (
    "inter_arrival_cv", "burstiness_ratio", "cards_per_device", "cards_per_ip",
    "link_share_card", "link_share_device", "link_share_ip",
    "distinct_cards_per_transaction", "merchants_per_card",
    "max_transactions_per_shared_device", "max_transactions_per_shared_ip",
)
AUTH_FEATURE_PREFIX = "auth_share_"

# raw `size` is deliberately EXCLUDED from every scored set: it correlates with
# nearly everything and with the one-hour window, so a scorer containing it
# largely becomes a size ranker. It survives as metadata and as a denominator.
EXCLUDED_FROM_SCORING: tuple[str, ...] = ("size", "link_edge_count",
                                          "links_per_transaction")

CORE_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in CORE_FEATURES)
FEATURE_SIGNS: dict[str, int] = {f.name: f.sign for f in CORE_FEATURES}
BIN_FEATURES: tuple[str, ...] = ("distinct_bins_per_transaction",)

# ----------------------------------------------------------------------
# decorrelation
# ----------------------------------------------------------------------
# At most one feature per concept family. If two retained features exceed the
# cap on the UNLABELLED training population, the earlier name in PRECEDENCE
# survives. Precedence is by strength of the domain argument, fixed here.
CORRELATION_CAP: float = 0.70
PRECEDENCE: tuple[str, ...] = (
    "multi_entity_link_fraction", "link_density", "burst_rate_per_minute",
    "max_transactions_per_shared_card", "distinct_merchants_per_transaction",
    "max_identical_amount_share", "distinct_bins_per_transaction",
)

# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------
N_FOLDS: int = 5
N_REPEATS: int = 5                    # 45 campaigns is high-variance; repeat
SPLIT_SEEDS: tuple[int, ...] = (11, 23, 37, 53, 71)
CHRONO_TRAIN_FRAC: float = 0.70
CHRONO_VAL_FRAC: float = 0.15
BUDGET_FRACTIONS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
THRESHOLD_SWEEP_QUANTILES: tuple[float, ...] = tuple(
    round(q, 4) for q in [0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995])
BOOTSTRAP_RESAMPLES: int = 500
BOOTSTRAP_SEED: int = 20240401

# ----------------------------------------------------------------------
# PRE-REGISTERED decision rule: uniform vs tuned weights
# ----------------------------------------------------------------------
# Fixed before any Phase 4 result was produced. Tuned weights are adopted ONLY
# if BOTH conditions hold on held-out folds. Otherwise uniform wins, because
# with 45 campaigns every fitted parameter is expensive.
PREREGISTERED_CI_LEVEL: float = 0.95
PREREGISTERED_MIN_PRAUC_GAIN: float = 0.05   # absolute mean PR-AUC improvement
PREREGISTERED_RULE: str = (
    "Adopt tuned weights only if (a) mean held-out PR-AUC gain over uniform "
    ">= 0.05 absolute AND (b) the 95% campaign-cluster bootstrap CI of the "
    "PAIRED per-fold difference excludes zero. Registered before any Phase 4 "
    "result existed. Not revisable after seeing results.")

# deterministic weight-tuner grid (coordinate ascent, training folds only)
WEIGHT_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
WEIGHT_TUNER_PASSES: int = 3
