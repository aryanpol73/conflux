"""Phase 5 - final deterministic detection pipeline.

Wires the existing components together; adds no new modelling of any kind:

    transactions -> TemporalEntityGraph -> form_campaign_candidates
                 -> load_structural_attributes -> build_scoring_features
                 -> DeterministicScorer.score_frame (pre-fitted reference)
                 -> tier -> action -> structured evidence -> json-safe dict

Contracts
---------
* Inference NEVER fits.  ``DeterministicScorer.fit`` is not called anywhere in
  this module; a pre-fitted :class:`ScorerReference` must be injected.
* Empty input: a DataFrame carrying the required columns but zero rows returns
  ``status="ok"`` with all counts zero and ``campaigns=[]``.  This is decided
  after column validation and *before* graph construction.
* Missing required columns raise ``ValueError``; non-DataFrame input raises
  ``TypeError``.
* Non-finite scores raise ``ValueError`` - they never fall through to MEDIUM.
* Signal ranking is by ABSOLUTE contribution descending, tie-broken by feature
  name ascending, so a strong negative contribution is not discarded.  The raw
  signed value is what gets reported.
* Output ordering is total: campaigns by score descending then candidate_id
  ascending.  Nothing volatile (clock, uuid, host) enters the result.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from conflux.graph.build_candidates import form_campaign_candidates
from conflux.scoring.candidate_features import (
    build_scoring_features,
    load_structural_attributes,
)
from conflux.scoring.deterministic_scorer import DeterministicScorer, ScorerReference

try:  # config objects live alongside the graph builders
    from conflux.graph.build_candidates import CandidateConfig
except Exception:  # pragma: no cover
    from conflux.graph.campaign_detection import CandidateConfig  # type: ignore

_GRAPH_IMPORT_ERROR: str | None = None
try:
    from conflux.graph.campaign_detection import GraphConfig, TemporalEntityGraph
except Exception:  # pragma: no cover - resolved lazily, reported loudly
    try:
        from conflux.graph.build_candidates import (  # type: ignore
            GraphConfig,
            TemporalEntityGraph,
        )
    except Exception as exc:  # pragma: no cover
        GraphConfig = None  # type: ignore[assignment]
        TemporalEntityGraph = None  # type: ignore[assignment]
        _GRAPH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

TIER_LOW = "LOW"
TIER_MEDIUM = "MEDIUM"
TIER_HIGH = "HIGH"
TIERS = (TIER_LOW, TIER_MEDIUM, TIER_HIGH)

DEFAULT_ACTIONS: Mapping[str, str] = {
    TIER_LOW: "flag",
    TIER_MEDIUM: "review",
    TIER_HIGH: "block",
}

CONTRIB_PREFIX = "contrib_"
DEFAULT_ID_COL = "candidate_id"

__all__ = [
    "PipelineError",
    "RiskThresholds",
    "DEFAULT_ACTIONS",
    "TIER_LOW",
    "TIER_MEDIUM",
    "TIER_HIGH",
    "TIERS",
    "classify_tier",
    "select_action",
    "json_safe",
    "resolve_required_columns",
    "validate_transactions",
    "build_graph",
    "generate_candidates",
    "build_features",
    "features_frame",
    "feature_names",
    "score_candidates",
    "resolve_score_column",
    "top_signals",
    "empty_result",
    "run_detection_pipeline",
]


class PipelineError(RuntimeError):
    """Raised when a repository interface does not match what Phase 5 expects."""


# ---------------------------------------------------------------------------
# thresholds / tiers / actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskThresholds:
    """Lower edges of the MEDIUM and HIGH bands (inclusive)."""

    medium: float = 0.40
    high: float = 0.70

    def __post_init__(self) -> None:
        for name in ("medium", "high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"threshold {name!r} must be a number, got {value!r}")
            f = float(value)
            if not math.isfinite(f):
                raise ValueError(f"threshold {name!r} must be finite, got {value!r}")
            object.__setattr__(self, name, f)
        if not self.medium <= self.high:
            raise ValueError(
                f"medium threshold ({self.medium}) must not exceed high ({self.high})"
            )

    def as_dict(self) -> dict[str, float]:
        return {"medium": self.medium, "high": self.high}


def classify_tier(score: Any, thresholds: RiskThresholds | None = None) -> str:
    """Pure tier classifier. ``>=`` on each lower edge; non-finite -> ValueError."""
    t = thresholds or RiskThresholds()
    if isinstance(score, bool) or not isinstance(
        score, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"score must be a real number, got {score!r}")
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"score must be finite, got {score!r}")
    if value >= t.high:
        return TIER_HIGH
    if value >= t.medium:
        return TIER_MEDIUM
    return TIER_LOW


def select_action(tier: str, actions: Mapping[str, str] | None = None) -> str:
    """Pure tier -> action mapping."""
    table = actions or DEFAULT_ACTIONS
    try:
        return table[tier]
    except KeyError:
        raise ValueError(
            f"no action configured for tier {tier!r}; known tiers: {sorted(table)!r}"
        ) from None


# ---------------------------------------------------------------------------
# json safety
# ---------------------------------------------------------------------------


def json_safe(obj: Any) -> Any:
    """Coerce nested structures to plain json types (no ``default=`` needed)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted((json_safe(v) for v in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

_SCHEMA_COLUMN_ATTRS = (
    "id_column",
    "id_col",
    "transaction_id_column",
    "timestamp_column",
    "timestamp_col",
    "ts_column",
)


def _resolve_schema() -> Any | None:
    try:  # the repository's single schema resolver
        from conflux.robustness.perturbations import resolve_schema
    except Exception:  # pragma: no cover
        return None
    try:
        return resolve_schema()
    except Exception:  # pragma: no cover
        return None


def resolve_required_columns(schema: Any | None = None) -> tuple[str, ...]:
    """Structurally required transaction columns, derived from the SchemaView.

    Nothing is hardcoded: if the schema cannot be resolved the caller must pass
    ``required_columns`` explicitly.
    """
    s = schema if schema is not None else _resolve_schema()
    if s is None:
        return ()
    found: list[str] = []
    for attr in _SCHEMA_COLUMN_ATTRS:
        value = getattr(s, attr, None)
        if isinstance(value, str) and value and value not in found:
            found.append(value)
    return tuple(found)


def validate_transactions(
    frame: Any, *, required_columns: Sequence[str] | None = None, schema: Any = None
) -> tuple[str, ...]:
    """Type/column validation. Returns the required columns actually enforced."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"transactions must be a pandas DataFrame, got {type(frame).__name__}"
        )
    required = (
        tuple(required_columns)
        if required_columns is not None
        else resolve_required_columns(schema)
    )
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"transaction frame is missing required column(s) {missing!r}; "
            f"present columns: {list(frame.columns)!r}"
        )
    return required


# ---------------------------------------------------------------------------
# reflective binding against repository interfaces
# ---------------------------------------------------------------------------


def _bind(fn: Any, pool: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Fill a callable's parameters by name from ``pool``.

    Raises PipelineError naming the unsatisfiable parameters instead of
    guessing an argument order.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:  # pragma: no cover
        raise PipelineError(f"cannot introspect {label}: {exc}") from exc

    kwargs: dict[str, Any] = {}
    unmet: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in pool and pool[name] is not None:
            kwargs[name] = pool[name]
        elif param.default is param.empty:
            unmet.append(name)
    if unmet:
        raise PipelineError(
            f"{label}{sig} requires parameter(s) {unmet!r} that the pipeline "
            f"cannot supply; available: {sorted(k for k, v in pool.items() if v is not None)!r}"
        )
    return kwargs


def build_graph(
    frame: pd.DataFrame, config: Any = None, *, schema: Any = None
) -> Any:
    """Construct the temporal entity graph via whichever constructor exists."""
    if TemporalEntityGraph is None:  # pragma: no cover
        raise PipelineError(
            f"TemporalEntityGraph could not be imported: {_GRAPH_IMPORT_ERROR}"
        ) 
    graph_frame = frame.drop(
        columns=["label", "campaign_id"],
        errors="ignore",
    ).copy(deep=True)

    cfg = config if config is not None else (
        GraphConfig() if GraphConfig else None
    )

    pool = {
        "frame": graph_frame,
        "transactions": graph_frame,
        "df": graph_frame,
        "config": cfg,
        "schema": schema,
    }
    attempts: list[tuple[str, Any]] = []
    for name in ("from_transactions", "from_frame", "build", "from_dataframe"):
        factory = getattr(TemporalEntityGraph, name, None)
        if callable(factory):
            attempts.append((f"TemporalEntityGraph.{name}", factory))
    attempts.append(("TemporalEntityGraph", TemporalEntityGraph))

    errors: list[str] = []
    for label, factory in attempts:
        try:
            kwargs = _bind(factory, pool, label=label)
        except PipelineError as exc:
            errors.append(str(exc))
            continue
        try:
            return factory(**kwargs)
        except TypeError as exc:
            errors.append(f"{label}: TypeError: {exc}")
            continue
    raise PipelineError(
        "could not construct TemporalEntityGraph; tried "
        + "; ".join(label for label, _ in attempts)
        + " -> "
        + " | ".join(errors)
    )


def generate_candidates(graph: Any, config: Any = None) -> Any:
    """Call the real ``form_campaign_candidates``."""
    cfg = config if config is not None else CandidateConfig()
    return form_campaign_candidates(graph, cfg)


def _candidate_frame(cset: Any) -> pd.DataFrame:
    getter = getattr(cset, "candidate_frame", None)
    if callable(getter):
        out = getter()
    else:  # pragma: no cover
        out = getattr(cset, "candidates", None)
    if not isinstance(out, pd.DataFrame):
        raise PipelineError(
            f"CandidateSet.candidate_frame() returned {type(out).__name__}, "
            "expected a DataFrame"
        )
    return out


def _assignments(cset: Any) -> pd.DataFrame:
    out = getattr(cset, "assignments", None)
    if not isinstance(out, pd.DataFrame):
        raise PipelineError(
            f"CandidateSet.assignments is {type(out).__name__}, expected a DataFrame"
        )
    return out


def build_features(
    *,
    frame: pd.DataFrame,
    graph: Any,
    cset: Any,
    candidates: pd.DataFrame,
    assignments: pd.DataFrame,
    min_size: int,
    schema: Any = None,
) -> Any:
    """attributes via ``load_structural_attributes``, then ``build_scoring_features``.

    ``# Use the scoring feature builder directly.`` is deliberately NOT called: the scorer consumes
    what ``build_scoring_features`` produces, and calling both would build the
    same features twice.
    """
    pool = {
        "frame": frame,
        "transactions": frame,
        "df": frame,
        "graph": graph,
        "cset": cset,
        "candidate_set": cset,
        "candidates": candidates,
        "candidate_frame": candidates,
        "assignments": assignments,
        "schema": schema,
    }
    attributes = pd.DataFrame(
    {
        "transaction_id": frame["transaction_id"].astype(str).str.strip(),
        "card_fingerprint": frame["card_fingerprint"].astype(str).str.strip(),
        "amount": (
            pd.to_numeric(frame["amount"], errors="coerce")
            if "amount" in frame.columns
            else float("nan")
        ),
        "auth_outcome": (
            frame["auth_outcome"].astype(str).str.strip()
            if "auth_outcome" in frame.columns
            else ""
        ),
    }
)

    return build_scoring_features(
        candidates, assignments, attributes, min_size=int(min_size)
    )


_FRAME_ATTRS = ("frame", "features", "feature_frame", "scoring_frame", "table", "data")
_NAME_ATTRS = ("feature_names", "features", "names", "columns", "feature_columns")


def features_frame(features: Any) -> pd.DataFrame:
    """Extract the scoreable DataFrame from whatever build_scoring_features returns."""
    if isinstance(features, pd.DataFrame):
        return features
    for attr in _FRAME_ATTRS:
        value = getattr(features, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:  # pragma: no cover
                continue
        if isinstance(value, pd.DataFrame):
            return value
    raise PipelineError(
        f"cannot locate a DataFrame on {type(features).__name__}; "
        f"probed {_FRAME_ATTRS!r}; available: "
        f"{[a for a in dir(features) if not a.startswith('_')]!r}"
    )


def feature_names(features: Any, frame: pd.DataFrame | None = None) -> list[str]:
    """Extract the feature-name list the scorer was fitted against."""
    for attr in _NAME_ATTRS:
        value = getattr(features, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:  # pragma: no cover
                continue
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            return list(value)
        if isinstance(value, pd.Index):
            return [str(v) for v in value]
    if frame is not None:
        return [
            str(c)
            for c in frame.columns
            if c != DEFAULT_ID_COL and pd.api.types.is_numeric_dtype(frame[c])
        ]
    raise PipelineError(
        f"cannot locate feature names on {type(features).__name__}; probed {_NAME_ATTRS!r}"
    )


def score_candidates(
    reference: ScorerReference,
    frame: pd.DataFrame,
    *,
    id_col: str = DEFAULT_ID_COL,
) -> pd.DataFrame:
    """Score with the injected reference. Never fits."""
    if reference is None:
        raise ValueError(
            "scorer_reference is required; the inference pipeline never fits the "
            "DeterministicScorer. Supply a pre-fitted ScorerReference."
        )
    scored = DeterministicScorer.score_frame(reference, frame, id_col=id_col)
    if not isinstance(scored, pd.DataFrame):
        raise PipelineError(
            f"score_frame returned {type(scored).__name__}, expected a DataFrame"
        )
    return scored


_SCORE_CANDIDATES = ("score", "risk_score", "campaign_score", "final_score", "total")


def resolve_score_column(scored: pd.DataFrame) -> str:
    for name in _SCORE_CANDIDATES:
        if name in scored.columns:
            return name
    numeric = [
        str(c)
        for c in scored.columns
        if not str(c).startswith(CONTRIB_PREFIX)
        and c != DEFAULT_ID_COL
        and pd.api.types.is_numeric_dtype(scored[c])
    ]
    if len(numeric) == 1:
        return numeric[0]
    raise PipelineError(
        f"cannot identify the score column; probed {_SCORE_CANDIDATES!r}, "
        f"non-contrib numeric columns are {numeric!r}"
    )


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------


def top_signals(row: Mapping[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    """Rank contrib_* by |contribution| desc, then feature name asc.

    Absolute magnitude is used so a strongly negative contribution is surfaced
    rather than buried; the reported value stays signed.
    """
    if limit < 0:
        raise ValueError(f"top_n_signals must be >= 0, got {limit}")
    signals: list[tuple[float, str, float]] = []
    for key, value in row.items():
        name = str(key)
        if not name.startswith(CONTRIB_PREFIX):
            continue
        if value is None:
            continue
        try:
            contribution = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(contribution):
            continue
        signals.append((abs(contribution), name[len(CONTRIB_PREFIX) :], contribution))
    signals.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"feature": feature, "contribution": contribution}
        for _, feature, contribution in signals[:limit]
    ]


def _transaction_ids(assignments: pd.DataFrame, id_col: str) -> dict[Any, list[Any]]:
    """Map candidate_id -> sorted transaction ids, if assignments exposes them."""
    if id_col not in assignments.columns:
        return {}
    txn_col = None
    for column in assignments.columns:
        name = str(column).lower()
        if column == id_col:
            continue
        if "transaction" in name or name in ("txn_id", "tx_id", "member", "node"):
            txn_col = column
            break
    if txn_col is None:
        return {}
    grouped: dict[Any, list[Any]] = {}
    for candidate_id, group in assignments.groupby(id_col, sort=False):
        grouped[candidate_id] = sorted(
            (json_safe(v) for v in group[txn_col].tolist()), key=repr
        )
    return grouped


# ---------------------------------------------------------------------------
# result assembly
# ---------------------------------------------------------------------------


def empty_result(n_transactions: int = 0) -> dict[str, Any]:
    return {
        "status": "ok",
        "summary": {
            "n_transactions": int(n_transactions),
            "n_candidates": 0,
            "n_scored": 0,
            "n_high_risk": 0,
            "n_medium_risk": 0,
            "n_low_risk": 0,
        },
        "campaigns": [],
    }


def run_detection_pipeline(
    frame: pd.DataFrame,
    *,
    scorer_reference: ScorerReference,
    graph_config: Any = None,
    candidate_config: Any = None,
    min_size: int = 2,
    thresholds: RiskThresholds | None = None,
    actions: Mapping[str, str] | None = None,
    top_n_signals: int = 5,
    id_col: str = DEFAULT_ID_COL,
    required_columns: Sequence[str] | None = None,
    schema: Any = None,
) -> dict[str, Any]:
    """Run the full detection pipeline and return a json-safe result."""
    validate_transactions(frame, required_columns=required_columns, schema=schema)
    tiers = thresholds or RiskThresholds()
    action_table = actions or DEFAULT_ACTIONS

    n_transactions = int(len(frame))
    if n_transactions == 0:
        return empty_result(0)

    if scorer_reference is None:
        raise ValueError(
            "scorer_reference is required; the inference pipeline never fits the "
            "DeterministicScorer."
        )

    work = frame.copy(deep=True)  # the caller's frame is never touched

    graph = build_graph(work, graph_config, schema=schema)
    cset = generate_candidates(graph, candidate_config)
    candidates = _candidate_frame(cset)
    assignments = _assignments(cset)
    n_candidates = int(len(candidates))

    features = build_features(
        frame=work,
        graph=graph,
        cset=cset,
        candidates=candidates,
        assignments=assignments,
        min_size=min_size,
        schema=schema,
    )
    feature_table = features_frame(features)

    if len(feature_table) == 0:
        result = empty_result(n_transactions)
        result["summary"]["n_candidates"] = n_candidates
        return json_safe(result)

    scored = score_candidates(scorer_reference, feature_table, id_col=id_col)
    score_col = resolve_score_column(scored)
    n_scored = int(len(scored))
    if n_scored > n_candidates:
        raise PipelineError(
            f"scored {n_scored} rows from {n_candidates} candidates; "
            "scoring must not create candidates"
        )

    txn_index = _transaction_ids(assignments, id_col)

    campaigns: list[dict[str, Any]] = []
    counts = {TIER_LOW: 0, TIER_MEDIUM: 0, TIER_HIGH: 0}
    for row in scored.to_dict(orient="records"):
        raw_score = row.get(score_col)
        tier = classify_tier(raw_score, tiers)  # ValueError on NaN/Inf
        counts[tier] += 1
        candidate_id = row.get(id_col)
        campaigns.append(
            {
                "candidate_id": json_safe(candidate_id),
                "transaction_ids": txn_index.get(candidate_id, []),
                "score": float(raw_score),
                "tier": tier,
                "action": select_action(tier, action_table),
                "evidence": {"top_signals": top_signals(row, limit=top_n_signals)},
            }
        )

    campaigns.sort(key=lambda c: (-c["score"], str(c["candidate_id"])))

    result = {
        "status": "ok",
        "summary": {
            "n_transactions": n_transactions,
            "n_candidates": n_candidates,
            "n_scored": n_scored,
            "n_high_risk": counts[TIER_HIGH],
            "n_medium_risk": counts[TIER_MEDIUM],
            "n_low_risk": counts[TIER_LOW],
        },
        "campaigns": campaigns,
    }
    return json_safe(result)
