"""In-memory transaction state and the single detection entry point.

Deliberately minimal: a list of transaction records plus a cached frozen
``ScorerReference``. No database, no cache server, no persistence -- state is
process-local and disappears on restart, which is the intended behaviour for a
demo surface.

Both the WebSocket handler and the REST routes call :func:`run_detection` here,
so there is exactly one place where the pipeline is invoked and exactly one
symbol to patch in tests.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from conflux.api.schemas import FORBIDDEN_COLUMNS, TRANSACTION_COLUMNS
from conflux.pipeline import run_detection_pipeline
from conflux.scoring.scorer_reference_io import load_scorer_reference

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ScorerUnavailableError",
    "DetectionError",
    "ApiState",
    "get_state",
    "reset_state",
    "run_detection",
    "default_artifact_path",
    "ARTIFACT_FILENAME",
]

ARTIFACT_FILENAME = "scorer_reference_v1.json"
_ARTIFACT_ENV_VAR = "CONFLUX_SCORER_ARTIFACT"

#: object dtype for identifiers, float for the single numeric column.
_COLUMN_DTYPES: dict[str, str] = {
    name: ("float64" if name == "amount" else "object") for name in TRANSACTION_COLUMNS
}


class ScorerUnavailableError(RuntimeError):
    """The frozen Phase 5.5 artifact could not be loaded. Never fall back."""


class DetectionError(RuntimeError):
    """The existing detection pipeline raised while scoring the population."""


def default_artifact_path() -> Path:
    """Locate ``scorer_reference_v1.json`` without depending on the CWD."""
    override = os.environ.get(_ARTIFACT_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()

    package_root = Path(__file__).resolve().parent.parent  # .../src/conflux
    return package_root / "models" / "artifacts" / ARTIFACT_FILENAME


class ApiState:
    """Thread-safe in-memory store: transactions + the frozen scorer reference."""

    def __init__(self, artifact_path: Path | str | None = None) -> None:
        self._lock = threading.RLock()
        self._artifact_path = (
            Path(artifact_path) if artifact_path is not None else default_artifact_path()
        )
        self._reference: Any | None = None
        self._load_error: str | None = None
        self._rows: list[dict[str, Any]] = []
        #: Optional pass-through kwargs for run_detection_pipeline (min_size,
        #: thresholds, actions, top_n_signals, ...). Empty by default so the
        #: pipeline's own defaults win.
        self.pipeline_kwargs: dict[str, Any] = {}

    # -- scorer reference ---------------------------------------------------

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    @property
    def scorer_loaded(self) -> bool:
        with self._lock:
            return self._reference is not None

    @property
    def load_error(self) -> str | None:
        with self._lock:
            return self._load_error

    def set_scorer_reference(self, reference: Any) -> None:
        """Inject a reference directly (tests, or an embedding application)."""
        with self._lock:
            self._reference = reference
            self._load_error = None

    def ensure_scorer_loaded(self) -> Any:
        """Load the frozen artifact once and cache it.

        Raises :class:`ScorerUnavailableError` on any failure. There is no
        fallback path: the API must never fit a scorer, tune weights, or
        compute reference distributions at runtime.
        """
        with self._lock:
            if self._reference is not None:
                return self._reference

            path = self._artifact_path
            if not path.is_file():
                self._load_error = f"scorer artifact not found at {path}"
                raise ScorerUnavailableError(
                    f"{self._load_error}. Build it with "
                    "'py -3.14 tools/build_scorer_reference.py', or point "
                    f"{_ARTIFACT_ENV_VAR} at an existing artifact."
                )
            try:
                self._reference = load_scorer_reference(path)
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise ScorerUnavailableError(
                    f"failed to load scorer artifact {path}: {self._load_error}"
                ) from exc

            self._load_error = None
            LOGGER.info("loaded frozen scorer reference from %s", path)
            return self._reference

    # -- transaction store --------------------------------------------------

    @property
    def transaction_count(self) -> int:
        with self._lock:
            return len(self._rows)

    def add_transaction(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Store one transaction record; returns the stored copy.

        The record is copied on the way in and on the way out, so a caller
        holding the original mapping cannot mutate stored state.
        """
        leaked = sorted(FORBIDDEN_COLUMNS.intersection(record))
        if leaked:
            raise ValueError(
                "refusing to store ground-truth field(s): " + ", ".join(leaked)
            )
        missing = [name for name in TRANSACTION_COLUMNS if name not in record]
        if missing:
            raise ValueError(
                "transaction record is missing required column(s): "
                + ", ".join(missing)
            )
        stored = {name: record[name] for name in TRANSACTION_COLUMNS}
        with self._lock:
            self._rows.append(stored)
        return dict(stored)

    def transactions(self) -> list[dict[str, Any]]:
        """Return a deep-enough copy of the stored records."""
        with self._lock:
            return [dict(row) for row in self._rows]

    def to_frame(self) -> pd.DataFrame:
        """Return the current population as a fresh DataFrame.

        Columns are always exactly ``TRANSACTION_COLUMNS`` in dataset order,
        even when the store is empty -- an empty-but-well-formed frame exercises
        the pipeline's documented empty-input path instead of tripping its
        column validation.
        """
        rows = self.transactions()
        if not rows:
            return pd.DataFrame(
                {name: pd.Series(dtype=dtype) for name, dtype in _COLUMN_DTYPES.items()}
            )
        frame = pd.DataFrame(rows, columns=list(TRANSACTION_COLUMNS))
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
        return frame

    def clear_transactions(self) -> None:
        with self._lock:
            self._rows.clear()

    def reset(self) -> None:
        """Clear transactions *and* the cached scorer reference."""
        with self._lock:
            self._rows.clear()
            self._reference = None
            self._load_error = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scorer_loaded": self._reference is not None,
                "scorer_artifact_path": str(self._artifact_path),
                "transactions_in_memory": len(self._rows),
                "transaction_columns": list(TRANSACTION_COLUMNS),
                "load_error": self._load_error,
            }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_STATE: ApiState | None = None
_STATE_LOCK = threading.Lock()


def get_state() -> ApiState:
    """Return the process-wide state singleton."""
    global _STATE
    if _STATE is None:
        with _STATE_LOCK:
            if _STATE is None:
                _STATE = ApiState()
    return _STATE


def reset_state() -> ApiState:
    """Replace the singleton with a fresh instance (tests and demo resets)."""
    global _STATE
    with _STATE_LOCK:
        _STATE = ApiState()
    return _STATE


# ---------------------------------------------------------------------------
# The one and only detection entry point
# ---------------------------------------------------------------------------


def run_detection(state: ApiState | None = None, **overrides: Any) -> dict[str, Any]:
    """Run the existing pipeline over the current in-memory population.

    Not incremental: the whole population is re-scored on every call, because
    ``run_detection_pipeline`` is not incremental. This is the honest,
    architecture-respecting behaviour for the demo.
    """
    state = get_state() if state is None else state
    reference = state.ensure_scorer_loaded()  # raises ScorerUnavailableError
    frame = state.to_frame()

    kwargs: dict[str, Any] = dict(state.pipeline_kwargs)
    kwargs.update(overrides)

    try:
        result = run_detection_pipeline(frame, scorer_reference=reference, **kwargs)
    except Exception as exc:  # noqa: BLE001 - wrapped, never swallowed
        LOGGER.exception(
            "detection pipeline failed over %d transaction(s)", len(frame.index)
        )
        raise DetectionError(f"{type(exc).__name__}: {exc}") from exc

    if not isinstance(result, dict):
        raise DetectionError(
            f"detection pipeline returned {type(result).__name__}, expected a dict"
        )
    return result
