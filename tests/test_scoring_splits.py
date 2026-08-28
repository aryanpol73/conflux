import numpy as np

from conflux.scoring.splits import (
    campaign_grouped_folds, chronological_candidate_split, cluster_ids,
    split_feasibility_probe)


def test_no_campaign_appears_in_train_and_test(scored):
    _, feats = scored
    camp = feats["dominant_campaign_id"].fillna("").astype(str).to_numpy()
    pos = feats["is_attack_containing"].to_numpy(dtype=bool)
    for f in campaign_grouped_folds(feats, n_folds=3, seeds=(1, 2)):
        tr = set(camp[f.train_idx][pos[f.train_idx]])
        te = set(camp[f.test_idx][pos[f.test_idx]])
        assert not (tr & te), f"campaign leaked: {tr & te}"


def test_all_fragments_of_a_campaign_share_a_fold(scored):
    _, feats = scored
    camp = feats["dominant_campaign_id"].fillna("").astype(str).to_numpy()
    for f in campaign_grouped_folds(feats, n_folds=3, seeds=(5,)):
        for c in f.held_out_campaigns:
            assert not np.isin(camp[f.train_idx], [c]).any()


def test_every_fold_partitions_the_population(scored):
    _, feats = scored
    for f in campaign_grouped_folds(feats, n_folds=3, seeds=(7,)):
        assert set(f.train_idx) | set(f.test_idx) == set(range(len(feats)))
        assert not set(f.train_idx) & set(f.test_idx)


def test_chronological_split_is_ordered(scored):
    _, feats = scored
    tr, va, te, meta = chronological_candidate_split(feats)
    assert meta["strictly_ordered"]
    ts = feats["last_ts_ns"].to_numpy()
    assert ts[tr].max() <= ts[va].min() <= ts[va].max() <= ts[te].min()


def test_clusters_group_fragments(scored):
    _, feats = scored
    cl = cluster_ids(feats)
    pos = feats["is_attack_containing"].to_numpy(dtype=bool)
    assert len(set(cl[pos])) == feats.loc[pos, "dominant_campaign_id"].nunique()
    assert len(set(cl[~pos])) == int((~pos).sum())


def test_feasibility_probe_reports_overlap(scored):
    _, feats = scored
    p = split_feasibility_probe(feats)
    assert p["campaigns"] > 0
    assert "chronological_boundary_is_also_a_campaign_boundary" in p
