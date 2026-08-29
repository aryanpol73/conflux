"""JSON persistence and validation for :class:`ScorerReference`.

ScorerReference holds only strings, ints, floats and tuples thereof, so JSON
round-trips it exactly (Python's float repr is shortest-round-trip).  JSON is
used instead of pickle because the artifact is diffable, auditable in review,
and carries no code-execution risk on load.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from conflux.scoring.deterministic_scorer import ScorerReference

ARTIFACT_SCHEMA_VERSION = 1
EXPECTED_N_FEATURES = 6

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "EXPECTED_N_FEATURES",
    "reference_to_dict",
    "reference_from_dict",
    "save_scorer_reference",
    "load_scorer_reference",
    "validate_scorer_reference",
    "references_equal",
]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_scorer_reference(
    reference: ScorerReference, *, n_features: int | None = EXPECTED_N_FEATURES
) -> ScorerReference:
    """Structural validation. Raises ValueError naming the offending feature.

    ``n_features=None`` skips only the fixed-arity check; every other invariant
    is always enforced.
    """
    if not isinstance(reference, ScorerReference):
        raise ValueError(
            f"expected a ScorerReference, got {type(reference).__name__}"
        )

    names = tuple(reference.feature_names)
    if n_features is not None and len(names) != n_features:
        raise ValueError(
            f"expected exactly {n_features} features, got {len(names)}: {list(names)!r}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names: {list(names)!r}")
    if not all(isinstance(n, str) and n for n in names):
        raise ValueError(f"feature names must be non-empty strings: {list(names)!r}")

    for field in ("signs", "weights", "lo", "hi", "reference_values"):
        value = getattr(reference, field)
        if len(value) != len(names):
            raise ValueError(
                f"{field} has length {len(value)}, expected {len(names)} "
                f"to match feature_names"
            )

    if int(reference.n_reference) <= 0:
        raise ValueError(f"n_reference must be > 0, got {reference.n_reference}")
    if not isinstance(reference.fit_scope, str) or not reference.fit_scope:
        raise ValueError(f"fit_scope must be a non-empty string, got {reference.fit_scope!r}")

    for name, sign in zip(names, reference.signs):
        if int(sign) not in (1, -1):
            raise ValueError(f"sign for feature {name!r} must be +1 or -1, got {sign!r}")

    for name, weight in zip(names, reference.weights):
        if not _finite(weight):
            raise ValueError(f"weight for feature {name!r} is not finite: {weight!r}")
    total = float(sum(float(w) for w in reference.weights))
    if not total > 0.0:
        raise ValueError(f"weights must sum to > 0, got {total!r}")

    for name, lo, hi in zip(names, reference.lo, reference.hi):
        if not _finite(lo) or not _finite(hi):
            raise ValueError(
                f"winsor bounds for feature {name!r} are not finite: lo={lo!r} hi={hi!r}"
            )
        if not float(hi) > float(lo):
            raise ValueError(
                f"degenerate reference distribution for feature {name!r}: "
                f"winsor bounds lo={lo!r}, hi={hi!r} (require hi > lo)"
            )

    for name, values in zip(names, reference.reference_values):
        if len(values) == 0:
            raise ValueError(f"reference_values for feature {name!r} is empty")
        for value in values:
            if not _finite(value):
                raise ValueError(
                    f"reference_values for feature {name!r} contains a non-finite "
                    f"value: {value!r}"
                )
    return reference


def reference_to_dict(reference: ScorerReference) -> dict[str, Any]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_names": [str(n) for n in reference.feature_names],
        "signs": [int(s) for s in reference.signs],
        "weights": [float(w) for w in reference.weights],
        "lo": [float(v) for v in reference.lo],
        "hi": [float(v) for v in reference.hi],
        "reference_values": [
            [float(v) for v in column] for column in reference.reference_values
        ],
        "n_reference": int(reference.n_reference),
        "fit_scope": str(reference.fit_scope),
    }


def reference_from_dict(data: dict[str, Any]) -> ScorerReference:
    required = (
        "feature_names", "signs", "weights", "lo", "hi",
        "reference_values", "n_reference", "fit_scope",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"artifact is missing required field(s): {missing!r}")
    return ScorerReference(
        feature_names=tuple(str(n) for n in data["feature_names"]),
        signs=tuple(int(s) for s in data["signs"]),
        weights=tuple(float(w) for w in data["weights"]),
        lo=tuple(float(v) for v in data["lo"]),
        hi=tuple(float(v) for v in data["hi"]),
        reference_values=tuple(
            tuple(float(v) for v in column) for column in data["reference_values"]
        ),
        n_reference=int(data["n_reference"]),
        fit_scope=str(data["fit_scope"]),
    )


def save_scorer_reference(
    reference: ScorerReference,
    path: str | Path,
    *,
    n_features: int | None = EXPECTED_N_FEATURES,
) -> Path:
    """Validate then persist as JSON. Returns the written path."""
    validate_scorer_reference(reference, n_features=n_features)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(reference_to_dict(reference), indent=2, sort_keys=True)
    target.write_text(payload, encoding="utf-8")
    return target


def load_scorer_reference(
    path: str | Path, *, n_features: int | None = None
) -> ScorerReference:
    """Load and validate. Never fits, never rebuilds.

    ``n_features`` defaults to None so that loading is arity-agnostic; the
    builder enforces the six-feature production contract explicitly.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"scorer reference artifact not found: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"artifact at {source} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"artifact at {source} must be a JSON object, got {type(data).__name__}"
        )
    reference = reference_from_dict(data)
    return validate_scorer_reference(reference, n_features=n_features)


def references_equal(a: ScorerReference, b: ScorerReference) -> bool:
    """Exact field-by-field equality (no tolerance)."""
    return (
        tuple(a.feature_names) == tuple(b.feature_names)
        and tuple(int(s) for s in a.signs) == tuple(int(s) for s in b.signs)
        and tuple(float(w) for w in a.weights) == tuple(float(w) for w in b.weights)
        and tuple(float(v) for v in a.lo) == tuple(float(v) for v in b.lo)
        and tuple(float(v) for v in a.hi) == tuple(float(v) for v in b.hi)
        and tuple(tuple(float(v) for v in c) for c in a.reference_values)
        == tuple(tuple(float(v) for v in c) for c in b.reference_values)
        and int(a.n_reference) == int(b.n_reference)
        and str(a.fit_scope) == str(b.fit_scope)
    )
