"""CONFLUX Phase 3C -- evaluation of campaign candidates against ground truth.

Module: conflux.evaluation.campaign_evaluation

SCOPE: evaluation only. This module never creates, alters, re-derives or
re-orders campaign candidates. It consumes the assignments produced by
conflux.graph.campaign_detection and compares them with label / campaign_id.

GROUND TRUTH POLICY
-------------------
label and campaign_id are read HERE and only here, strictly after candidate
assignments already exist. Nothing computed in this module is fed back into
graph construction, connectivity, candidate formation, thresholds or weights.
This phase produces no score of any kind.

BIN POLICY
----------
BIN is never a connectivity mechanism. Two BIN artefacts exist in this module:

  * evidence_view(...) -- a reporting switch that shows or suppresses BIN
    context. Membership is an input to this function, so it is structurally
    incapable of changing membership; the runner asserts the metrics are
    identical across both views.

  * bin_only_baseline_grouping(...) -- an isolated BIN-only BASELINE, used
    exclusively as a measuring stick. It is NOT a candidate generator. It is
    never persisted as candidates, never unioned with candidates, and the
    candidates under evaluation are never re-derived from it.

DEFINITIONS USED THROUGHOUT
---------------------------
campaign transaction   : campaign_id is a non-empty string.
non-campaign / normal  : campaign_id is the empty string.
group                  : one candidate (Phase 3B) or one baseline group.
multi-transaction group: size >= 2. Singletons are reported separately and are
                         never counted as detections.
dominant campaign      : the most frequent non-empty campaign_id in the group,
                         ties broken by the lexicographically smallest id so the
                         result is deterministic.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("conflux.evaluation.campaign_evaluation")

EVALUATION_SCHEMA_VERSION = "conflux.evaluation.campaign_evaluation.v1"

ID_COL = "transaction_id"
TS_COL = "timestamp"
LABEL_COL = "label"
CAMPAIGN_COL = "campaign_id"
GROUND_TRUTH_COLUMNS: tuple[str, ...] = (LABEL_COL, CAMPAIGN_COL, "_source_type")

# candidate evidence columns that are BIN context and nothing else
BIN_CONTEXT_COLUMNS: tuple[str, ...] = ("distinct_bins", "bin_ids_context")

PURITY_CLASSES: tuple[str, ...] = (
    "pure_campaign",              # exactly 1 campaign, no normal traffic
    "pure_campaign_with_normal",  # exactly 1 campaign, plus normal traffic
    "mixed_campaign",             # >= 2 distinct campaigns
    "non_campaign",               # no campaign transactions at all
)

EVIDENCE_COLUMNS: tuple[str, ...] = (
    "time_span_seconds", "link_edge_count", "links_multi_entity",
    "link_entity_types", "links_card", "links_device", "links_ip",
    "distinct_cards", "distinct_devices", "distinct_ips", "distinct_merchants",
    "max_transactions_per_shared_device", "max_transactions_per_shared_ip",
    "distinct_bins", "bin_ids_context",
)


class GroundTruthError(ValueError):
    """The ground-truth source is unusable."""


class AlignmentError(ValueError):
    """Assignments and ground truth do not line up 1:1."""


# ----------------------------------------------------------------------
# ground truth
# ----------------------------------------------------------------------
def normalize_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical ground-truth frame: transaction_id, campaign_id, label, flags.

    Empty / whitespace campaign_id means 'not part of a known campaign'. No row
    is ever dropped and no value is ever inferred.
    """
    missing = [c for c in (ID_COL, CAMPAIGN_COL) if c not in df.columns]
    if missing:
        raise GroundTruthError(f"ground truth is missing column(s): {missing}")

    out = pd.DataFrame(index=range(len(df)))
    out[ID_COL] = df[ID_COL].astype(str).str.strip().to_numpy()

    camp = df[CAMPAIGN_COL].astype(str)
    camp = camp.where(df[CAMPAIGN_COL].notna(), "").str.strip()
    camp = camp.replace({"nan": "", "None": "", "<NA>": ""})
    out[CAMPAIGN_COL] = camp.to_numpy()
    out["is_campaign"] = out[CAMPAIGN_COL] != ""

    if LABEL_COL in df.columns:
        out["label_int"] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype("Int64").to_numpy()
    else:
        out["label_int"] = pd.array([pd.NA] * len(df), dtype="Int64")

    if out[ID_COL].eq("").any():
        raise GroundTruthError("blank transaction_id in ground truth; refusing to guess")
    return out.reset_index(drop=True)


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Read label/campaign_id from the frozen dataset. Read-only, never written."""
    path = Path(path)
    if not path.exists():
        raise GroundTruthError(f"dataset not found: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in (ID_COL, LABEL_COL, CAMPAIGN_COL) if c not in header]
    if missing:
        raise GroundTruthError(f"dataset is missing ground-truth column(s): {missing}")

    raw = pd.read_csv(
        path,
        usecols=[ID_COL, LABEL_COL, CAMPAIGN_COL],
        dtype=str,
        keep_default_na=False,   # keep "" as "", never as NaN -> deterministic
        na_values=[],
        low_memory=False,
    )
    gt = normalize_ground_truth(raw)
    log.info(
        "ground truth: %s rows, %s campaign transactions, %s distinct campaigns",
        len(gt),
        int(gt["is_campaign"].sum()),
        gt.loc[gt["is_campaign"], CAMPAIGN_COL].nunique(),
    )
    return gt


def label_campaign_consistency(gt: pd.DataFrame) -> dict[str, Any]:
    """Observed relationship between label and campaign_id. Reported, not assumed."""
    if not gt["label_int"].notna().any():
        return {"label_column_available": False}

    lab = gt["label_int"].fillna(-1).astype(int)
    return {
        "label_column_available": True,
        "label_1_rows": int((lab == 1).sum()),
        "label_0_rows": int((lab == 0).sum()),
        "label_other_rows": int((~lab.isin([0, 1])).sum()),
        "campaign_transactions": int(gt["is_campaign"].sum()),
        "label_1_without_campaign_id": int(((lab == 1) & ~gt["is_campaign"]).sum()),
        "label_0_with_campaign_id": int(((lab == 0) & gt["is_campaign"]).sum()),
        "label_and_campaign_id_agree": bool(((lab == 1) == gt["is_campaign"]).all()),
    }


# ----------------------------------------------------------------------
# alignment (no silent drops, ever)
# ----------------------------------------------------------------------
@dataclass
class AlignmentReport:
    assignment_rows: int
    assignment_unique_ids: int
    ground_truth_rows: int
    ground_truth_unique_ids: int
    duplicate_assignment_ids: tuple[str, ...]
    duplicate_ground_truth_ids: tuple[str, ...]
    assigned_ids_missing_from_ground_truth: tuple[str, ...]
    ground_truth_ids_missing_from_assignments: tuple[str, ...]
    aligned: bool

    def as_dict(self, sample: int = 10) -> dict[str, Any]:
        return {
            "aligned": self.aligned,
            "assignment_rows": self.assignment_rows,
            "assignment_unique_ids": self.assignment_unique_ids,
            "ground_truth_rows": self.ground_truth_rows,
            "ground_truth_unique_ids": self.ground_truth_unique_ids,
            "duplicate_assignment_ids": len(self.duplicate_assignment_ids),
            "duplicate_assignment_id_sample": list(self.duplicate_assignment_ids[:sample]),
            "duplicate_ground_truth_ids": len(self.duplicate_ground_truth_ids),
            "duplicate_ground_truth_id_sample": list(self.duplicate_ground_truth_ids[:sample]),
            "assigned_ids_missing_from_ground_truth":
                len(self.assigned_ids_missing_from_ground_truth),
            "assigned_missing_sample":
                list(self.assigned_ids_missing_from_ground_truth[:sample]),
            "ground_truth_ids_missing_from_assignments":
                len(self.ground_truth_ids_missing_from_assignments),
            "ground_truth_missing_sample":
                list(self.ground_truth_ids_missing_from_assignments[:sample]),
        }


def align(assignments: pd.DataFrame, ground_truth: pd.DataFrame, *,
          strict: bool = True) -> AlignmentReport:
    """Audit the join before performing it. Strict mode refuses to evaluate on a
    silently reduced join."""
    a_ids = assignments[ID_COL].astype(str)
    g_ids = ground_truth[ID_COL].astype(str)

    dup_a = tuple(sorted(a_ids[a_ids.duplicated()].unique().tolist()))
    dup_g = tuple(sorted(g_ids[g_ids.duplicated()].unique().tolist()))
    miss_g = tuple(sorted(set(a_ids) - set(g_ids)))
    miss_a = tuple(sorted(set(g_ids) - set(a_ids)))

    rep = AlignmentReport(
        assignment_rows=int(len(a_ids)),
        assignment_unique_ids=int(a_ids.nunique()),
        ground_truth_rows=int(len(g_ids)),
        ground_truth_unique_ids=int(g_ids.nunique()),
        duplicate_assignment_ids=dup_a,
        duplicate_ground_truth_ids=dup_g,
        assigned_ids_missing_from_ground_truth=miss_g,
        ground_truth_ids_missing_from_assignments=miss_a,
        aligned=not (dup_a or dup_g or miss_g or miss_a),
    )
    if strict and not rep.aligned:
        raise AlignmentError(
            "assignments and ground truth are not 1:1; refusing to evaluate on a "
            f"silently reduced join. {json.dumps(rep.as_dict(), indent=2)}"
        )
    return rep


# ----------------------------------------------------------------------
# internal tables
# ----------------------------------------------------------------------
def _quantiles(s: pd.Series | None) -> dict[str, float]:
    if s is None or len(s) == 0:
        return {"n": 0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0,
                "p95": 0.0, "max": 0.0, "mean": 0.0}
    v = pd.Series(s).astype(float)
    return {
        "n": int(v.size),
        "min": float(round(v.min(), 6)),
        "p25": float(round(v.quantile(0.25), 6)),
        "median": float(round(v.median(), 6)),
        "p75": float(round(v.quantile(0.75), 6)),
        "p95": float(round(v.quantile(0.95), 6)),
        "max": float(round(v.max(), 6)),
        "mean": float(round(v.mean(), 6)),
    }


def _join(assignments: pd.DataFrame, ground_truth: pd.DataFrame,
          group_col: str) -> pd.DataFrame:
    if group_col not in assignments.columns:
        raise ValueError(f"assignments frame has no group column '{group_col}'")

    left = assignments[[ID_COL, group_col]].copy()
    left[ID_COL] = left[ID_COL].astype(str)
    left[group_col] = left[group_col].astype(str)
    left = left.rename(columns={group_col: "group_id"})

    right = ground_truth[[ID_COL, CAMPAIGN_COL, "is_campaign"]].copy()
    right[ID_COL] = right[ID_COL].astype(str)

    joined = left.merge(right, on=ID_COL, how="inner", validate="one_to_one")
    return joined.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)


def _group_table(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per group, with campaign composition. Fully deterministic."""
    size = joined.groupby("group_id", sort=True).size().rename("size")
    tbl = size.to_frame()

    camp = joined.loc[joined["is_campaign"], ["group_id", CAMPAIGN_COL]]
    if len(camp):
        gc = (camp.groupby(["group_id", CAMPAIGN_COL], sort=True)
                  .size().rename("n").reset_index())
        gc = gc.sort_values(["group_id", "n", CAMPAIGN_COL],
                            ascending=[True, False, True], kind="mergesort")
        dom = gc.drop_duplicates("group_id", keep="first").set_index("group_id")

        tbl["campaign_transactions"] = (gc.groupby("group_id")["n"].sum()
                                          .reindex(tbl.index).fillna(0).astype(int))
        tbl["n_distinct_campaigns"] = (gc.groupby("group_id")[CAMPAIGN_COL].nunique()
                                         .reindex(tbl.index).fillna(0).astype(int))
        tbl["dominant_campaign_id"] = dom[CAMPAIGN_COL].reindex(tbl.index).fillna("")
        tbl["dominant_campaign_transactions"] = (dom["n"].reindex(tbl.index)
                                                    .fillna(0).astype(int))
    else:
        tbl["campaign_transactions"] = 0
        tbl["n_distinct_campaigns"] = 0
        tbl["dominant_campaign_id"] = ""
        tbl["dominant_campaign_transactions"] = 0

    tbl["non_campaign_transactions"] = tbl["size"] - tbl["campaign_transactions"]
    tbl["is_multi_transaction"] = tbl["size"] >= 2

    with np.errstate(invalid="ignore", divide="ignore"):
        tbl["dominant_fraction_of_group"] = (
            tbl["dominant_campaign_transactions"] / tbl["size"])
        tbl["dominant_fraction_of_campaign_transactions"] = np.where(
            tbl["campaign_transactions"] > 0,
            tbl["dominant_campaign_transactions"]
            / tbl["campaign_transactions"].replace(0, 1),
            0.0,
        )

    # largest single class when non-campaign traffic is treated as its own class
    tbl["largest_class_transactions"] = np.maximum(
        tbl["dominant_campaign_transactions"], tbl["non_campaign_transactions"])

    tbl["purity_class"] = np.select(
        [tbl["campaign_transactions"] == 0,
         tbl["n_distinct_campaigns"] >= 2,
         tbl["non_campaign_transactions"] > 0],
        ["non_campaign", "mixed_campaign", "pure_campaign_with_normal"],
        default="pure_campaign",
    )

    return (tbl.reset_index()
               .sort_values("group_id", kind="mergesort")
               .reset_index(drop=True))


def _campaign_table(joined: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """One row per true campaign: coverage, fragmentation, best containing group."""
    empty_cols = [
        CAMPAIGN_COL, "campaign_transactions", "transactions_in_multi_groups",
        "transactions_isolated", "n_multi_groups", "n_groups_including_isolated",
        "best_group_id", "best_group_transactions", "best_group_share",
        "is_represented", "is_majority_captured",
    ]
    camp = joined.loc[joined["is_campaign"], ["group_id", CAMPAIGN_COL]].copy()
    if camp.empty:
        return pd.DataFrame(columns=empty_cols)

    sizes = groups.set_index("group_id")["size"]
    camp["group_size"] = camp["group_id"].map(sizes).astype(int)
    camp["in_multi"] = camp["group_size"] >= 2

    total = camp.groupby(CAMPAIGN_COL, sort=True).size().rename("campaign_transactions")
    in_multi = (camp.loc[camp["in_multi"]].groupby(CAMPAIGN_COL).size()
                    .rename("transactions_in_multi_groups"))
    n_multi = (camp.loc[camp["in_multi"]].groupby(CAMPAIGN_COL)["group_id"].nunique()
                   .rename("n_multi_groups"))
    n_all = (camp.groupby(CAMPAIGN_COL)["group_id"].nunique()
                 .rename("n_groups_including_isolated"))

    per = (camp.loc[camp["in_multi"]]
               .groupby([CAMPAIGN_COL, "group_id"], sort=True)
               .size().rename("n").reset_index())
    if len(per):
        per = per.sort_values([CAMPAIGN_COL, "n", "group_id"],
                              ascending=[True, False, True], kind="mergesort")
        best = per.drop_duplicates(CAMPAIGN_COL, keep="first").set_index(CAMPAIGN_COL)
    else:
        best = pd.DataFrame(columns=["group_id", "n"])

    out = total.to_frame()
    out["transactions_in_multi_groups"] = in_multi.reindex(out.index).fillna(0).astype(int)
    out["transactions_isolated"] = (out["campaign_transactions"]
                                    - out["transactions_in_multi_groups"])
    out["n_multi_groups"] = n_multi.reindex(out.index).fillna(0).astype(int)
    out["n_groups_including_isolated"] = n_all.reindex(out.index).fillna(0).astype(int)

    if len(best):
        out["best_group_id"] = best["group_id"].reindex(out.index).fillna("")
        out["best_group_transactions"] = best["n"].reindex(out.index).fillna(0).astype(int)
    else:
        out["best_group_id"] = ""
        out["best_group_transactions"] = 0

    out["best_group_share"] = out["best_group_transactions"] / out["campaign_transactions"]
    out["is_represented"] = out["n_multi_groups"] >= 1
    out["is_majority_captured"] = out["best_group_share"] >= 0.5

    return (out.reset_index()
               .sort_values(CAMPAIGN_COL, kind="mergesort")
               .reset_index(drop=True))


# ----------------------------------------------------------------------
# core grouping evaluation (shared by candidates and the BIN baseline)
# ----------------------------------------------------------------------
@dataclass
class GroupingEvaluation:
    """Metrics for ONE grouping (graph candidates, or the BIN-only baseline)."""

    name: str
    schema_version: str
    alignment: AlignmentReport
    group_table: pd.DataFrame
    campaign_table: pd.DataFrame
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "alignment": self.alignment.as_dict(),
            **self.metrics,
        }


def evaluate_grouping(assignments: pd.DataFrame, ground_truth: pd.DataFrame, *,
                      group_col: str = "candidate_id",
                      name: str = "graph_candidates",
                      strict: bool = True) -> GroupingEvaluation:
    """A / B / C / D / E metrics for an arbitrary transaction -> group mapping.

    The grouping is an INPUT. This function cannot and does not modify it.
    """
    alignment = align(assignments, ground_truth, strict=strict)
    joined = _join(assignments, ground_truth, group_col)
    groups = _group_table(joined)
    campaigns = _campaign_table(joined, groups)

    multi = groups.loc[groups["is_multi_transaction"]]
    n_txn = int(len(joined))
    n_camp_txn = int(joined["is_campaign"].sum())
    n_norm_txn = n_txn - n_camp_txn
    txn_in_multi = int(multi["size"].sum())
    camp_in_multi = int(multi["campaign_transactions"].sum())
    norm_in_multi = int(multi["non_campaign_transactions"].sum())

    pc = multi["purity_class"].value_counts()

    merged = multi.loc[multi["n_distinct_campaigns"] >= 2]
    merged_campaign_ids: set[str] = set()
    if len(merged):
        mj = joined.loc[joined["group_id"].isin(set(merged["group_id"]))
                        & joined["is_campaign"], CAMPAIGN_COL]
        merged_campaign_ids = set(mj.unique().tolist())

    represented = campaigns.loc[campaigns["is_represented"]] if len(campaigns) else campaigns
    frag = represented["n_multi_groups"] if len(represented) else pd.Series(dtype=float)

    def _f(x: Any) -> float:
        return float(round(float(x), 6))

    metrics: dict[str, Any] = {
        "grouping": {
            "groups_total": int(len(groups)),
            "groups_multi_transaction": int(len(multi)),
            "groups_singleton": int((~groups["is_multi_transaction"]).sum()),
            "largest_group_size": int(groups["size"].max()) if len(groups) else 0,
            "transactions_in_multi_transaction_groups": txn_in_multi,
        },
        "A_campaign_coverage": {
            "ground_truth_campaigns_total": int(len(campaigns)),
            "campaigns_represented_by_a_multi_transaction_group": int(len(represented)),
            "campaign_coverage_recall":
                _f(len(represented) / len(campaigns)) if len(campaigns) else 0.0,
            "campaigns_majority_captured_by_one_group":
                int(campaigns["is_majority_captured"].sum()) if len(campaigns) else 0,
            "campaign_majority_capture_rate":
                _f(campaigns["is_majority_captured"].mean()) if len(campaigns) else 0.0,
            "campaigns_entirely_isolated":
                int((~campaigns["is_represented"]).sum()) if len(campaigns) else 0,
        },
        "B_purity": {
            "definition": {
                "pure_campaign": "exactly one campaign_id, no normal traffic",
                "pure_campaign_with_normal": "one campaign_id plus normal traffic",
                "mixed_campaign": "two or more campaign_ids",
                "non_campaign": "no campaign transactions",
            },
            "multi_group_purity_class_counts": {
                k: int(pc.get(k, 0)) for k in PURITY_CLASSES},
            "pure_candidate_count_strict": int(pc.get("pure_campaign", 0)),
            "pure_candidate_count_allowing_normal_noise":
                int(pc.get("pure_campaign", 0) + pc.get("pure_campaign_with_normal", 0)),
            "mixed_candidate_count": int(pc.get("mixed_campaign", 0)),
            "non_campaign_group_count": int(pc.get("non_campaign", 0)),
            "campaign_purity_transaction_weighted":
                _f(multi["dominant_campaign_transactions"].sum() / camp_in_multi)
                if camp_in_multi else 0.0,
            "group_purity_transaction_weighted_with_normal_class":
                _f(multi["largest_class_transactions"].sum() / txn_in_multi)
                if txn_in_multi else 0.0,
            "dominant_fraction_of_group_distribution": _quantiles(
                multi.loc[multi["campaign_transactions"] > 0,
                          "dominant_fraction_of_group"]),
        },
        "C_fragmentation": {
            "campaigns_split_across_multiple_groups":
                int((frag >= 2).sum()) if len(frag) else 0,
            "mean_groups_per_represented_campaign": _f(frag.mean()) if len(frag) else 0.0,
            "median_groups_per_represented_campaign":
                _f(frag.median()) if len(frag) else 0.0,
            "max_groups_for_one_campaign": int(frag.max()) if len(frag) else 0,
            "fragmentation_distribution": _quantiles(frag),
            "campaign_transactions_left_isolated":
                int(campaigns["transactions_isolated"].sum()) if len(campaigns) else 0,
        },
        "D_merging": {
            "merged_groups": int(len(merged)),
            "campaigns_involved_in_a_merge": len(merged_campaign_ids),
            "largest_merge_distinct_campaigns":
                int(merged["n_distinct_campaigns"].max()) if len(merged) else 0,
            "transactions_in_merged_groups":
                int(merged["size"].sum()) if len(merged) else 0,
            "merge_rate_over_multi_groups":
                _f(len(merged) / len(multi)) if len(multi) else 0.0,
        },
        "E_transaction_coverage": {
            "transactions_total": n_txn,
            "campaign_transactions": n_camp_txn,
            "non_campaign_transactions": n_norm_txn,
            "campaign_transactions_in_multi_transaction_groups": camp_in_multi,
            "campaign_transaction_recall":
                _f(camp_in_multi / n_camp_txn) if n_camp_txn else 0.0,
            "campaign_transactions_isolated": n_camp_txn - camp_in_multi,
            "non_campaign_transactions_in_multi_transaction_groups": norm_in_multi,
            "non_campaign_inclusion_rate":
                _f(norm_in_multi / n_norm_txn) if n_norm_txn else 0.0,
            "campaign_share_of_transactions_in_multi_groups":
                _f(camp_in_multi / txn_in_multi) if txn_in_multi else 0.0,
        },
    }

    return GroupingEvaluation(
        name=name,
        schema_version=EVALUATION_SCHEMA_VERSION,
        alignment=alignment,
        group_table=groups,
        campaign_table=campaigns,
        metrics=metrics,
    )


# ----------------------------------------------------------------------
# candidate-specific evaluation (adds Phase 3B evidence to the group table)
# ----------------------------------------------------------------------
@dataclass
class CandidateEvaluation:
    grouping: GroupingEvaluation
    candidate_table: pd.DataFrame          # group metrics + Phase 3B evidence
    ground_truth_consistency: dict[str, Any]
    evidence_columns: tuple[str, ...]
    bin_context_included: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.grouping.as_dict(),
            "ground_truth_consistency": self.ground_truth_consistency,
            "evidence_columns": list(self.evidence_columns),
            "bin_context_included": self.bin_context_included,
        }


def candidate_frames_from_candidate_set(candidate_set: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapter: CandidateSet -> (assignments, evidence). Read-only, no re-derivation."""
    assignments = candidate_set.assignments.copy()
    evidence = candidate_set.candidate_frame().copy()
    return assignments, evidence


def evidence_view(evidence: pd.DataFrame, *, include_bin: bool) -> pd.DataFrame:
    """Reporting switch. Membership is not an argument here, so this cannot
    change which transactions belong to which candidate -- that is the point."""
    if include_bin:
        return evidence.copy()
    drop = [c for c in BIN_CONTEXT_COLUMNS if c in evidence.columns]
    return evidence.drop(columns=drop).copy()


def evaluate_candidates(assignments: pd.DataFrame, ground_truth: pd.DataFrame, *,
                        evidence: pd.DataFrame | None = None,
                        include_bin_context: bool = True,
                        name: str = "graph_candidates",
                        strict: bool = True) -> CandidateEvaluation:
    grouping = evaluate_grouping(assignments, ground_truth,
                                 group_col="candidate_id", name=name, strict=strict)
    table = grouping.group_table.rename(columns={"group_id": "candidate_id"}).copy()

    used: tuple[str, ...] = ()
    if evidence is not None and len(evidence):
        ev = evidence_view(evidence, include_bin=include_bin_context)
        ev = ev.copy()
        ev["candidate_id"] = ev["candidate_id"].astype(str)
        keep = [c for c in EVIDENCE_COLUMNS if c in ev.columns]
        used = tuple(keep)
        table = table.merge(ev[["candidate_id", *keep]], on="candidate_id",
                            how="left", validate="one_to_one")

    table = (table.sort_values(["size", "candidate_id"], ascending=[False, True],
                               kind="mergesort")
                  .reset_index(drop=True))

    return CandidateEvaluation(
        grouping=grouping,
        candidate_table=table,
        ground_truth_consistency=label_campaign_consistency(ground_truth),
        evidence_columns=used,
        bin_context_included=bool(include_bin_context
                                  and any(c in used for c in BIN_CONTEXT_COLUMNS)),
    )


# ----------------------------------------------------------------------
# BIN: isolated baseline + informativeness. NEVER a candidate generator.
# ----------------------------------------------------------------------
def bin_only_baseline_grouping(transactions: pd.DataFrame, *,
                               window_seconds: float,
                               key_col: str = "bin",
                               prefix: str = "BINBASE") -> pd.DataFrame:
    """EVALUATION-ONLY baseline: group transactions that share a BIN and sit
    within `window_seconds` of a chain neighbour.

    This exists solely to quantify how far BIN alone gets you. It is not a
    candidate generator: nothing in this function's output is ever written as a
    candidate, unioned with candidates, or used to alter candidate membership.

    Component equivalence note: linking each transaction to the immediately
    preceding transaction of the same BIN within the window yields exactly the
    same connected components as linking every within-window pair, because any
    two transactions inside a window necessarily have all intermediate
    consecutive gaps inside that window too. The baseline is therefore
    component-equivalent to running the Phase 3B algorithm with BIN as the sole
    connectivity type -- which is precisely the configuration the graph layer
    refuses to run.
    """
    missing = [c for c in (ID_COL, key_col) if c not in transactions.columns]
    if missing:
        raise ValueError(f"baseline needs column(s): {missing}")

    if "ts_ns" in transactions.columns:
        ts = transactions["ts_ns"].astype("int64")
    elif TS_COL in transactions.columns:
        ts = (
            pd.to_datetime(transactions[TS_COL], errors="raise")
            .dt.as_unit("ns")
            .astype("int64")
        )
    else:
        raise ValueError("baseline needs ts_ns or timestamp")

    df = pd.DataFrame({
        ID_COL: transactions[ID_COL].astype(str).to_numpy(),
        "key": transactions[key_col].astype(str).to_numpy(),
        "ts_ns": ts.to_numpy(dtype=np.int64),
    })
    df = df.sort_values(["key", "ts_ns", ID_COL], kind="mergesort").reset_index(drop=True)

    window_ns = int(round(float(window_seconds) * 1_000_000_000))
    gap = df.groupby("key", sort=False)["ts_ns"].diff()
    new_run = gap.isna() | (gap > window_ns)
    df["run"] = new_run.groupby(df["key"]).cumsum().astype(int)

    df["baseline_group_id"] = (prefix + "-" + df["key"] + "-"
                               + df["run"].astype(str).str.zfill(4))
    df["baseline_group_size"] = df.groupby("baseline_group_id")[ID_COL].transform("size")

    out = df[[ID_COL, "baseline_group_id", "baseline_group_size"]]
    return out.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)


def bin_informativeness(transactions: pd.DataFrame, ground_truth: pd.DataFrame, *,
                        key_col: str = "bin", top_n: int = 15) -> dict[str, Any]:
    """How much does static BIN identity alone tell you? Descriptive, no model.

    Answers the question that matters for the BIN-vs-graph argument: can a BIN
    delimit a campaign at all, or do campaigns span BINs and BINs span campaigns?
    """
    df = pd.DataFrame({
        ID_COL: transactions[ID_COL].astype(str).to_numpy(),
        "bin": transactions[key_col].astype(str).to_numpy(),
    }).merge(ground_truth[[ID_COL, CAMPAIGN_COL, "is_campaign"]],
             on=ID_COL, how="inner", validate="one_to_one")

    per_bin = df.groupby("bin", sort=True).agg(
        transactions=(ID_COL, "size"),
        campaign_transactions=("is_campaign", "sum"),
        distinct_campaigns=(CAMPAIGN_COL, lambda s: s[s != ""].nunique()),
    )
    per_bin["campaign_transactions"] = per_bin["campaign_transactions"].astype(int)
    per_bin["campaign_rate"] = (per_bin["campaign_transactions"]
                                / per_bin["transactions"]).round(6)

    camp = df.loc[df["is_campaign"]]
    per_campaign = camp.groupby(CAMPAIGN_COL, sort=True).agg(
        transactions=(ID_COL, "size"),
        distinct_bins=("bin", "nunique"),
    )

    n_camp_txn = int(df["is_campaign"].sum())
    pure_attack_bins = per_bin.loc[(per_bin["campaign_rate"] == 1.0)
                                   & (per_bin["transactions"] >= 2)]
    hot_bins = per_bin.loc[per_bin["campaign_rate"] >= 0.5]

    top = (per_bin.sort_values(["campaign_transactions", "campaign_rate", "bin"],
                               ascending=[False, False, True], kind="mergesort")
                  .head(top_n).reset_index())

    return {
        "note": ("Descriptive only. Uses ground truth strictly for measurement; "
                 "no BIN quantity here is or ever becomes a feature, a weight, "
                 "a threshold, or a connectivity mechanism."),
        "distinct_bins": int(len(per_bin)),
        "bins_touched_by_campaign_transactions":
            int((per_bin["campaign_transactions"] > 0).sum()),
        "bins_with_campaign_rate_ge_0.5": int(len(hot_bins)),
        "campaign_transactions_in_bins_with_rate_ge_0.5":
            int(hot_bins["campaign_transactions"].sum()),
        "share_of_campaign_transactions_in_those_bins":
            float(round(hot_bins["campaign_transactions"].sum() / n_camp_txn, 6))
            if n_camp_txn else 0.0,
        "fully_attack_bins_min2_transactions": int(len(pure_attack_bins)),
        "distinct_campaigns_per_bin": _quantiles(
            per_bin.loc[per_bin["distinct_campaigns"] > 0, "distinct_campaigns"]),
        "bins_hosting_multiple_campaigns":
            int((per_bin["distinct_campaigns"] >= 2).sum()),
        "distinct_bins_per_campaign": (_quantiles(per_campaign["distinct_bins"])
                                       if len(per_campaign)
                                       else _quantiles(pd.Series(dtype=float))),
        "campaigns_spanning_multiple_bins":
            int((per_campaign["distinct_bins"] >= 2).sum()) if len(per_campaign) else 0,
        "campaigns_confined_to_one_bin":
            int((per_campaign["distinct_bins"] == 1).sum()) if len(per_campaign) else 0,
        "top_bins_by_campaign_transactions": top.to_dict("records"),
    }


def compare_groupings(primary: GroupingEvaluation,
                      baseline: GroupingEvaluation) -> dict[str, Any]:
    """Side-by-side facts. Deltas only -- no verdict is generated here."""
    def pick(e: GroupingEvaluation) -> dict[str, Any]:
        return {
            "groups_multi_transaction": e.metrics["grouping"]["groups_multi_transaction"],
            "largest_group_size": e.metrics["grouping"]["largest_group_size"],
            "campaign_coverage_recall":
                e.metrics["A_campaign_coverage"]["campaign_coverage_recall"],
            "campaign_majority_capture_rate":
                e.metrics["A_campaign_coverage"]["campaign_majority_capture_rate"],
            "pure_candidate_count_strict":
                e.metrics["B_purity"]["pure_candidate_count_strict"],
            "campaign_purity_transaction_weighted":
                e.metrics["B_purity"]["campaign_purity_transaction_weighted"],
            "merge_rate_over_multi_groups":
                e.metrics["D_merging"]["merge_rate_over_multi_groups"],
            "largest_merge_distinct_campaigns":
                e.metrics["D_merging"]["largest_merge_distinct_campaigns"],
            "max_groups_for_one_campaign":
                e.metrics["C_fragmentation"]["max_groups_for_one_campaign"],
            "campaign_transaction_recall":
                e.metrics["E_transaction_coverage"]["campaign_transaction_recall"],
            "non_campaign_inclusion_rate":
                e.metrics["E_transaction_coverage"]["non_campaign_inclusion_rate"],
            "campaign_share_of_transactions_in_multi_groups":
                e.metrics["E_transaction_coverage"]
                         ["campaign_share_of_transactions_in_multi_groups"],
        }

    a, b = pick(primary), pick(baseline)
    return {
        "primary_name": primary.name,
        "baseline_name": baseline.name,
        "primary": a,
        "baseline": b,
        "delta_primary_minus_baseline": {
            k: (round(a[k] - b[k], 6)
                if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))
                else None)
            for k in a
        },
        "interpretation_policy": (
            "These are measurements, not a ranking. A claim that the graph "
            "outperforms BIN is only warranted if the deltas in this block "
            "support it on coverage AND purity AND merge rate simultaneously."),
    }


# ----------------------------------------------------------------------
# human-readable rendering
# ----------------------------------------------------------------------
def render_candidate_table(table: pd.DataFrame, *, top_n: int = 25) -> str:
    cols = [c for c in ("candidate_id", "size", "campaign_transactions",
                        "non_campaign_transactions", "n_distinct_campaigns",
                        "dominant_campaign_id", "dominant_fraction_of_group",
                        "purity_class", "distinct_cards", "distinct_devices",
                        "distinct_ips", "distinct_merchants", "distinct_bins",
                        "time_span_seconds", "link_edge_count")
            if c in table.columns]
    view = table.loc[table["size"] >= 2, cols].head(top_n).copy()
    if view.empty:
        return "(no multi-transaction candidates)"
    if "dominant_fraction_of_group" in view.columns:
        view["dominant_fraction_of_group"] = view["dominant_fraction_of_group"].round(3)
    if "time_span_seconds" in view.columns:
        view["time_span_seconds"] = view["time_span_seconds"].round(1)
    return view.to_string(index=False)


def render_campaign_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "(no campaigns in ground truth)"
    view = table.copy()
    view["best_group_share"] = view["best_group_share"].round(3)
    return view.to_string(index=False)


def render_summary(metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    for section in ("grouping", "A_campaign_coverage", "B_purity",
                    "C_fragmentation", "D_merging", "E_transaction_coverage"):
        block = metrics.get(section, {})
        lines.append(f"[{section}]")
        for k, v in block.items():
            if k == "definition":
                continue
            if isinstance(v, dict):
                lines.append(f"  {k}: {json.dumps(v)}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")
    return "\n".join(lines)
