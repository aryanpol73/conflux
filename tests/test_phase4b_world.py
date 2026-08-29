"""Phase 4B -- world layer regression tests. Schema-driven, no hard-coded names."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conflux.robustness import world as W
from conflux.robustness.perturbations import (
    attack_mask, resolve_schema, right_censor,
)

SCHEMA = resolve_schema()
NS = 1_000_000_000


def _synthetic(n_benign: int = 12, n_attack: int = 4) -> pd.DataFrame:
    """A tiny frame with exactly the columns the frozen pipeline requires."""
    cols = W.world_frame_columns(SCHEMA)
    rows = []
    t0 = pd.Timestamp("2024-01-01 00:00:00.000000")
    for i in range(n_benign + n_attack):
        attack = i >= n_benign
        ts = t0 + pd.Timedelta(seconds=30 * i)
        row = {c: f"{c}-{i:03d}" for c in cols}
        row[SCHEMA.id_col] = f"TX{i:05d}"
        row[SCHEMA.ts_col] = ts.strftime("%Y-%m-%d %H:%M:%S.%f")
        row["amount"] = 10.0 + i
        row["auth_outcome"] = "approved"
        row[SCHEMA.label_col] = "1" if attack else "0"
        row[SCHEMA.campaign_col] = "CMP-001" if attack else ""
        if attack:  # give the attack rows shared entities
            for c in W.linking_entity_columns():
                row[c] = f"{c}-shared"
        rows.append(row)
    return pd.DataFrame(rows, columns=list(cols))


def test_entity_column_map_returns_real_columns():
    cols = W.linking_entity_columns() + W.context_entity_columns()
    assert len(set(cols)) == 5
    for c in cols:
        assert c in W.STRUCTURAL_COLUMNS
    # the logical keys must NOT be mistaken for columns
    assert "card" not in cols and W.entity_column("card") == "card_fingerprint"


def test_baseline_world_is_identity():
    frame = _synthetic()
    w = W.baseline_world(frame)
    assert w.is_baseline
    assert w.n_transactions == len(frame)
    assert w.perturbation_summary["rows_before"] == w.perturbation_summary["rows_after"]


def test_fingerprint_is_row_order_independent():
    frame = _synthetic()
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    assert W.frame_fingerprint(frame) == W.frame_fingerprint(shuffled)


def test_build_world_is_deterministic_and_does_not_mutate():
    frame = _synthetic()
    before = W.frame_fingerprint(frame)

    def bump(f: pd.DataFrame) -> pd.DataFrame:
        out = f.copy()
        out["amount"] = out["amount"] * 2.0
        return out

    a = W.build_world(frame, scenario_id="T", name="t", transform=bump)
    b = W.build_world(frame, scenario_id="T", name="t", transform=bump)
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != a.baseline_fingerprint
    assert W.frame_fingerprint(frame) == before


def test_build_world_rejects_in_place_mutation():
    frame = _synthetic()

    def bad(f: pd.DataFrame) -> pd.DataFrame:
        f["amount"] = 0.0          # mutates the caller's frame
        return f.copy()

    with pytest.raises(W.WorldError, match="MUTATED"):
        W.build_world(frame, scenario_id="T", name="t", transform=bad)


def test_build_world_rejects_label_rewrite():
    frame = _synthetic()

    def flip(f: pd.DataFrame) -> pd.DataFrame:
        out = f.copy()
        out.loc[out.index[0], SCHEMA.label_col] = "1"
        return out

    with pytest.raises(W.WorldError, match="changed on"):
        W.build_world(frame, scenario_id="T", name="t", transform=flip)


def test_build_world_rejects_positive_injection():
    frame = _synthetic()

    def inject(f: pd.DataFrame) -> pd.DataFrame:
        extra = f.iloc[[0]].copy()
        extra[SCHEMA.id_col] = "TX99999"
        extra[SCHEMA.label_col] = "1"
        return pd.concat([f, extra], ignore_index=True)

    with pytest.raises(W.WorldError, match="benign"):
        W.build_world(frame, scenario_id="T", name="t", transform=inject)


def test_removals_require_explicit_permission():
    frame = _synthetic()
    cut = lambda f: right_censor(f, keep_fraction=0.6, schema=SCHEMA)

    with pytest.raises(W.WorldError, match="allow_removals"):
        W.build_world(frame, scenario_id="T", name="t", transform=cut)

    w = W.build_world(frame, scenario_id="T", name="t", transform=cut,
                      allow_removals=True, allow_additions=False)
    assert w.n_transactions < len(frame)
    assert w.label_delta["transactions_removed"] > 0
    assert w.label_delta["transactions_added"] == 0


def test_duplicate_ids_are_rejected():
    frame = _synthetic()
    dupe = lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True)
    with pytest.raises(W.WorldError, match="duplicate"):
        W.build_world(frame, scenario_id="T", name="t", transform=dupe)


def test_describe_is_json_serializable():
    frame = _synthetic()
    w = W.build_world(frame, scenario_id="T", name="t", seed=7,
                      parameters={"factor": np.float64(2.0),
                                  "cols": W.linking_entity_columns()},
                      transform=lambda f: f.copy())
    text = json.dumps(w.describe())          # no default= allowed
    assert "__dataframe__" not in text
    assert json.loads(text)["seed"] == 7


@pytest.mark.skipif(not W.Path(W.RAW_DATASET_PATH).is_file(),
                    reason="raw dataset not present")
def test_real_baseline_loads_and_validates():
    frame = W.load_baseline_frame()
    W.validate_world_frame(frame, schema=SCHEMA)
    assert attack_mask(frame, schema=SCHEMA).sum() > 0
    w = W.baseline_world(frame)
    assert w.is_baseline and w.n_transactions == len(frame)
