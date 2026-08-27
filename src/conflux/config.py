"""CONFLUX project configuration -- the shared interface every layer imports from.

AUTHORITY
---------
`FeatureConfig` remains defined, validated, and authoritative in
`conflux.features.build_feature_table`. That module states this itself:

    "`FeatureConfig` is the only source of windows and thresholds. Load it from the
    project's existing YAML ... or from a mapping supplied by src/conflux/config.py."

This module does not define a second `FeatureConfig`, does not redeclare any of its
fields, and does not change how it is constructed. It only re-exports it and adds the
one loader (`get_feature_config`) that the docstring above already anticipates, so
`models/`, `evaluation/`, and `scoring/` have a single shared import instead of each
reaching into the features package directly or hardcoding their own defaults.

WHAT IS (AND ISN'T) CENTRALIZED HERE
-------------------------------------
Only paths and constants that already exist, literally, in build_feature_table.py:
- `--data` / `--outdir` CLI defaults ("data/raw/dataset_v4_final.csv", "data/processed")
- the exact output filenames its `main()` writes (features_v4.csv, feature_dictionary.csv,
  univariate_auc.csv, validation_report.json)
- `FORBIDDEN_INPUTS`, re-exported as-is (not redefined) from the same module

Nothing here is a new configuration value invented for this refactor.
"""
from __future__ import annotations

from pathlib import Path

from .features.build_feature_table import FeatureConfig, FORBIDDEN_INPUTS

# ---------------------------------------------------------------------------
# FeatureConfig access -- re-exported, never redefined.
# ---------------------------------------------------------------------------
def get_feature_config() -> FeatureConfig:
    """Return the project's active `FeatureConfig`.

    Currently this is `FeatureConfig()` with library defaults -- the same defaults
    build_feature_table.main() uses when no --config YAML is passed, and what the
    existing frozen v4 feature table (156 features, 31,873 rows) was generated with.
    Every future layer should call this instead of constructing `FeatureConfig()`
    itself, so there remains exactly one place that decides how it's built.
    """
    return FeatureConfig()


# Re-exported unchanged from build_feature_table.py: the columns that must never be
# used as model/feature inputs. Not redefined or extended here.
FORBIDDEN_MODEL_INPUTS: tuple[str, ...] = FORBIDDEN_INPUTS

# ---------------------------------------------------------------------------
# Project paths -- taken directly from build_feature_table.py's own CLI defaults
# and the literal output filenames its main() writes. Nothing added.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

RAW_DATASET_PATH: Path = PROJECT_ROOT / "data" / "raw" / "dataset_v4_final.csv"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

FEATURES_TABLE_PATH: Path = PROCESSED_DIR / "features_v4.csv"
FEATURE_DICTIONARY_PATH: Path = PROCESSED_DIR / "feature_dictionary.csv"
UNIVARIATE_AUDIT_PATH: Path = PROCESSED_DIR / "univariate_auc.csv"
VALIDATION_REPORT_PATH: Path = PROCESSED_DIR / "validation_report.json"


__all__ = [
    "FeatureConfig",
    "get_feature_config",
    "FORBIDDEN_MODEL_INPUTS",
    "PROJECT_ROOT",
    "RAW_DATASET_PATH",
    "PROCESSED_DIR",
    "FEATURES_TABLE_PATH",
    "FEATURE_DICTIONARY_PATH",
    "UNIVARIATE_AUDIT_PATH",
    "VALIDATION_REPORT_PATH",
]