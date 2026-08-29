"""Phase 5.5 - build and persist the production ScorerReference artifact.

Freezes the approved Phase 4A deterministic scorer configuration against the
Phase 4A candidate feature population.  Reads retained features and signs from
the Phase 4A report by name; never touches labels or campaign ids.

    $env:PYTHONPATH="$PWD\\src"
    py -3.14 tools/build_scorer_reference.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conflux.scoring.deterministic_scorer import DeterministicScorer  # noqa: E402
from conflux.scoring.scorer_reference_io import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    load_scorer_reference,
    references_equal,
    save_scorer_reference,
    validate_scorer_reference,
)

PHASE4A_REPORT = REPO_ROOT / "data/processed/scoring/phase4a_scoring_report.json"
FEATURE_CSV = REPO_ROOT / "data/processed/scoring/candidate_scoring_features.csv"
ARTIFACT_PATH = REPO_ROOT / "src/conflux/models/artifacts/scorer_reference_v1.json"
METADATA_PATH = REPO_ROOT / "src/conflux/models/artifacts/scorer_reference_v1_metadata.json"

EXPECTED_POPULATION = 4372
WINSOR = (0.01, 0.99)
FIT_SCOPE = "phase4a_candidate_scoring_features_csv"
MIN_SIZE = 2
LEAKAGE_COLUMNS = ("label", "campaign_id")
UNIFORM_WEIGHT = 1.0


class BuildError(RuntimeError):
    """Raised when a build invariant fails."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_reference(frame, features, signs, weights):
    """Call fit, adapting signs/weights to whichever container it accepts.

    The tuples are built by comprehension over ``features``, so the name->sign
    mapping is preserved even when the scorer wants positional sequences.
    """
    sign_tuple = tuple(int(signs[f]) for f in features)
    weight_tuple = tuple(float(weights[f]) for f in features)
    attempts = (
        ("mapping", dict(signs), dict(weights)),
        ("sequence", sign_tuple, weight_tuple),
    )
    errors = []
    for label, s, w in attempts:
        try:
            return DeterministicScorer.fit(
                frame, list(features), signs=s, weights=w,
                winsor=WINSOR, fit_scope=FIT_SCOPE,
            ), label
        except (TypeError, ValueError) as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise BuildError(
        "DeterministicScorer.fit rejected both mapping and sequence forms of "
        f"signs/weights; signature is {inspect.signature(DeterministicScorer.fit)}. "
        + " | ".join(errors)
    )


def main() -> int:
    print("Building production ScorerReference (Phase 5.5)...\n")

    for path in (PHASE4A_REPORT, FEATURE_CSV):
        if not path.exists():
            raise BuildError(f"required input not found: {path}")

    report = json.loads(PHASE4A_REPORT.read_text(encoding="utf-8"))
    design = report["feature_design"]
    features = list(design["retained_after_decorrelation"])
    if len(features) != 6:
        raise BuildError(
            f"expected 6 retained features, report lists {len(features)}: {features!r}"
        )

    # signs BY NAME - core_features has 7 entries, retained has 6
    sign_by_name = {item["name"]: int(item["sign"]) for item in design["core_features"]}
    unresolved = [f for f in features if f not in sign_by_name]
    if unresolved:
        raise BuildError(f"no sign in core_features for retained feature(s): {unresolved!r}")
    signs = {f: sign_by_name[f] for f in features}
    weights = {f: UNIFORM_WEIGHT for f in features}

    df = pd.read_csv(FEATURE_CSV)

    present_leakage = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if present_leakage:
        raise BuildError(
            f"leakage column(s) {present_leakage!r} present in {FEATURE_CSV.name}; "
            "the reference must never be fitted against ground truth"
        )
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise BuildError(f"feature CSV is missing retained feature(s): {missing!r}")
    if "candidate_id" not in df.columns:
        raise BuildError("feature CSV has no candidate_id column")

    n_rows, n_unique = len(df), int(df["candidate_id"].nunique())
    if n_rows != EXPECTED_POPULATION or n_unique != EXPECTED_POPULATION:
        raise BuildError(
            f"population gate failed: rows={n_rows}, unique candidate_id={n_unique}, "
            f"expected {EXPECTED_POPULATION} (Phase 4A multi_transaction_candidates)"
        )

    fit_frame = df[features].copy()
    non_finite = {
        f: int((~np.isfinite(fit_frame[f].to_numpy(dtype=float))).sum())
        for f in features
    }
    offenders = {f: n for f, n in non_finite.items() if n}
    if offenders:
        raise BuildError(f"non-finite values in projected features: {offenders!r}")

    reference, binding = fit_reference(fit_frame, features, signs, weights)
    validate_scorer_reference(reference, n_features=6)

    if tuple(reference.feature_names) != tuple(features):
        raise BuildError(
            f"fitted feature_names {tuple(reference.feature_names)!r} do not match "
            f"retained order {tuple(features)!r}"
        )
    if tuple(int(s) for s in reference.signs) != tuple(signs[f] for f in features):
        raise BuildError("fitted signs do not match the Phase 4A frozen signs")
    if int(reference.n_reference) != EXPECTED_POPULATION:
        raise BuildError(
            f"n_reference is {reference.n_reference}, expected {EXPECTED_POPULATION}; "
            "fit may have dropped rows"
        )

    saved = save_scorer_reference(reference, ARTIFACT_PATH, n_features=6)
    reloaded = load_scorer_reference(saved, n_features=6)
    if not references_equal(reference, reloaded):
        raise BuildError("round trip failed: reloaded reference differs from the original")

    scores_a, _ = DeterministicScorer.transform(reference, fit_frame)
    scores_b, _ = DeterministicScorer.transform(reloaded, fit_frame)
    arr_a = np.asarray(scores_a, dtype=float)
    arr_b = np.asarray(scores_b, dtype=float)
    if not np.array_equal(arr_a, arr_b):
        raise BuildError("scores from the reloaded reference differ from the original")
    if not np.all(np.isfinite(arr_a)):
        raise BuildError("scorer produced non-finite scores on the reference population")
    if float(arr_a.min()) < 0.0 or float(arr_a.max()) > 1.0:
        raise BuildError(f"scores outside [0,1]: min={arr_a.min()}, max={arr_a.max()}")

    bounds = {
        f: {"lo": float(lo), "hi": float(hi)}
        for f, lo, hi in zip(reference.feature_names, reference.lo, reference.hi)
    }
    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "phase": "5.5",
        "artifact_file": ARTIFACT_PATH.name,
        "reference_representation": "phase4a_candidate_scoring_features_csv",
        "source_feature_file": str(FEATURE_CSV.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_phase4a_report": str(PHASE4A_REPORT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_phase4a_report_sha256": _sha256(PHASE4A_REPORT),
        "source_feature_file_sha256": _sha256(FEATURE_CSV),
        "phase3b_artifact_sha256": report.get("phase3b_artifact_sha256"),
        "retained_features": list(reference.feature_names),
        "signs": {f: int(s) for f, s in zip(reference.feature_names, reference.signs)},
        "sign_source": "feature_design.core_features[].sign (matched by name)",
        "ground_truth_used_for_reference_fit": False,
        "phase4a_ground_truth_used": design.get("notes", {}).get("ground_truth_used"),
        "weights": {f: float(w) for f, w in zip(reference.feature_names, reference.weights)},
        "weight_decision": report.get("weight_decision_preregistered", {}).get("decision"),
        "weight_finding": (
            "Phase 4A preregistration adopted the tuned-weight branch, but the "
            "resulting selected fold weights were uniform (1.0) for all six "
            "retained features. Production therefore uses uniform weights."
        ),
        "winsor": list(WINSOR),
        "feature_bounds": bounds,
        "fit_scope": reference.fit_scope,
        "signs_weights_binding": binding,
        "n_reference": int(reference.n_reference),
        "min_size": MIN_SIZE,
        "phase4a_population_expected": EXPECTED_POPULATION,
        "phase4a_population_actual": n_unique,
        "score_range_on_reference": [float(arr_a.min()), float(arr_a.max())],
        "artifact_size_bytes": saved.stat().st_size,
        "builder": "tools/build_scorer_reference.py",
        "method": (
            "Fitted DeterministicScorer on the six Phase 4A retained features "
            "projected from the Phase 4A candidate scoring feature population. "
            "Signs inherited by name from the Phase 4A report; weights uniform; "
            "no labels, campaign ids, or supervised computation involved."
        ),
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("REFERENCE BUILD: PASS\n")
    print(f"candidate population: {n_unique} / expected {EXPECTED_POPULATION}")
    print(f"features: {len(reference.feature_names)}")
    print(f"signs:   {list(reference.signs)}")
    print(f"weights: {list(reference.weights)}")
    print(f"n_reference: {reference.n_reference}")
    print(f"fit_scope: {reference.fit_scope}")
    print(f"signs/weights binding: {binding}")
    print(f"score range: [{arr_a.min():.6f}, {arr_a.max():.6f}]\n")
    print("per-feature winsor bounds:")
    width = max(len(f) for f in reference.feature_names)
    for f, s, w in zip(reference.feature_names, reference.signs, reference.weights):
        b = bounds[f]
        print(f"  {f:<{width}}  sign={s:+d}  weight={w:.1f}  "
              f"lo={b['lo']:.6f}  hi={b['hi']:.6f}")
    print(f"\nartifact path: {saved}")
    print(f"artifact size: {saved.stat().st_size} bytes")
    print(f"metadata path: {METADATA_PATH}")
    print("round-trip: PASS")
    print("determinism: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"\nREFERENCE BUILD: FAIL\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
