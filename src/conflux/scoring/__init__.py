"""CONFLUX Phase 4A -- candidate triage scoring layer.

Scores and ranks the candidates Phase 3B already produced. Never creates,
merges, splits or re-orders them, and never modifies conflux.graph or
conflux.evaluation.
"""
from conflux.scoring.candidate_features import (
    ScoringFeatureError, ScoringFeatures, build_scoring_features,
    load_structural_attributes, prune_correlated, spearman_matrix,
)
from conflux.scoring.config import (
    ABLATION_FEATURES, BIN_FEATURES, CORE_FEATURES, CORE_FEATURE_NAMES,
    CORRELATION_CAP, FEATURE_SIGNS, PREREGISTERED_RULE, SCORING_SCHEMA_VERSION,
    FeatureSpec,
)
from conflux.scoring.deterministic_scorer import (
    DeterministicScorer, ScorerLeakageError, ScorerReference, tune_weights,
)
from conflux.scoring.evaluation import (
    apply_preregistered_rule, average_precision, bootstrap_ci, campaign_metrics,
    comparator_scores, confusion, normalize_purity, paired_difference_ci,
    precision_at_k, recall_exchange_table, render_exchange_table, roc_auc,
    select_threshold,
)
from conflux.scoring.splits import (
    Fold, campaign_grouped_folds, chronological_candidate_split, cluster_ids,
    split_feasibility_probe,
)

__all__ = [
    "SCORING_SCHEMA_VERSION", "FeatureSpec", "CORE_FEATURES",
    "CORE_FEATURE_NAMES", "FEATURE_SIGNS", "ABLATION_FEATURES", "BIN_FEATURES",
    "CORRELATION_CAP", "PREREGISTERED_RULE",
    "ScoringFeatures", "ScoringFeatureError", "build_scoring_features",
    "load_structural_attributes", "prune_correlated", "spearman_matrix",
    "DeterministicScorer", "ScorerReference", "ScorerLeakageError",
    "tune_weights",
    "Fold", "campaign_grouped_folds", "chronological_candidate_split",
    "cluster_ids", "split_feasibility_probe",
    "average_precision", "roc_auc", "confusion", "precision_at_k",
    "campaign_metrics", "select_threshold", "bootstrap_ci",
    "paired_difference_ci", "apply_preregistered_rule", "comparator_scores",
    "recall_exchange_table", "render_exchange_table", "normalize_purity",
]
