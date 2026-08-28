"""CONFLUX evaluation layer: ground-truth evaluation and diagnostics only.

Nothing in this package creates, alters or re-derives campaign candidates. It
reads what the graph layer produced and compares it with label / campaign_id
strictly after the fact.
"""
from conflux.evaluation.campaign_evaluation import (
    BIN_CONTEXT_COLUMNS, EVALUATION_SCHEMA_VERSION, EVIDENCE_COLUMNS,
    PURITY_CLASSES, AlignmentError, AlignmentReport, CandidateEvaluation,
    GroundTruthError, GroupingEvaluation, align, bin_informativeness,
    bin_only_baseline_grouping, candidate_frames_from_candidate_set,
    compare_groupings, evaluate_candidates, evaluate_grouping, evidence_view,
    label_campaign_consistency, load_ground_truth, normalize_ground_truth,
    render_campaign_table, render_candidate_table, render_summary,
)
from conflux.evaluation.candidate_diagnostics import (
    DIAGNOSTIC_SCHEMA_VERSION, GROUP_BY_CHOICES, DiagnosticInputError, FeatureSet,
    attach_groups, attack_rate_crosstab, benjamini_hochberg,
    build_candidate_features, cliffs_delta, compare_boolean, compare_numeric,
    descriptive_best_threshold, group_summary, ks_two_sample,
    load_candidate_artifacts, load_transaction_attributes, mann_whitney,
    redundancy_matrix, render_boolean_table, render_crosstab,
    render_numeric_table, strongest_separations, two_proportion_z,
)

__all__ = [
    # campaign_evaluation
    "BIN_CONTEXT_COLUMNS", "EVALUATION_SCHEMA_VERSION", "EVIDENCE_COLUMNS",
    "PURITY_CLASSES", "AlignmentError", "AlignmentReport", "CandidateEvaluation",
    "GroundTruthError", "GroupingEvaluation", "align", "bin_informativeness",
    "bin_only_baseline_grouping", "candidate_frames_from_candidate_set",
    "compare_groupings", "evaluate_candidates", "evaluate_grouping",
    "evidence_view", "label_campaign_consistency", "load_ground_truth",
    "normalize_ground_truth", "render_campaign_table", "render_candidate_table",
    "render_summary",
    # candidate_diagnostics
    "DIAGNOSTIC_SCHEMA_VERSION", "GROUP_BY_CHOICES", "DiagnosticInputError",
    "FeatureSet", "attach_groups", "attack_rate_crosstab", "benjamini_hochberg",
    "build_candidate_features", "cliffs_delta", "compare_boolean",
    "compare_numeric", "descriptive_best_threshold", "group_summary",
    "ks_two_sample", "load_candidate_artifacts", "load_transaction_attributes",
    "mann_whitney", "redundancy_matrix", "render_boolean_table",
    "render_crosstab", "render_numeric_table", "strongest_separations",
    "two_proportion_z",
]

