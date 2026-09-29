from dataclasses import replace

import pandas as pd

from sae_feature_atlas.config.registry import make_config
from sae_feature_atlas.config.schema import PathsConfig
from sae_feature_atlas.inspection.feature_cards import enrich_feature_cards


def test_refreshing_empty_diagnostics_removes_old_scores_examples_and_labels(tmp_path):
    cfg = replace(make_config(run_name="refresh"), paths=PathsConfig(data_root=tmp_path))
    cfg.run_data_dir.mkdir()
    pd.DataFrame([{
        "feature_id": 7, "p99_activation": None,
        "bimodality_score": 9999., "max_coactivation_jaccard": .8, "max_decoder_cosine": .99,
        "inspection_labels": "bimodal_candidate,coactivation_hub,decoder_geometry_dense",
        "manual_priority": "high",
        "bimodal_low_examples_json": "[]", "bimodal_high_examples_json": "[]",
        "top_decoder_neighbors_json": "[]", "top_coactivation_neighbors_json": "[]",
    }]).to_parquet(cfg.feature_cards_path)
    for path in [cfg.bimodal_candidates_path, cfg.bimodal_peak_examples_path,
                 cfg.coactivation_pairs_path, cfg.decoder_neighbors_path]:
        pd.DataFrame().to_parquet(path)
    cards = enrich_feature_cards(cfg)
    assert cards.primary_label.tolist() == ["unlabeled"]
    assert cards.manual_priority.tolist() == ["unreviewed"]
    for column in ["bimodality_score", "max_coactivation_jaccard", "max_decoder_cosine",
                   "bimodal_low_examples_json", "bimodal_high_examples_json",
                   "top_decoder_neighbors_json", "top_coactivation_neighbors_json"]:
        assert column not in cards or cards[column].isna().all()
