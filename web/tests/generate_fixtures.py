"""Small synthetic artifact fixtures for browser integration tests. No model downloads."""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from sae_feature_atlas.config.registry import make_config
from sae_feature_atlas.config.schema import PathsConfig
from sae_feature_atlas.pipeline.lineage import write_lineage
from sae_feature_atlas.report.explorer import write_explorer

root = Path(__file__).resolve().parents[2] / "tmp" / "browser-fixtures"
cfg = replace(
    make_config(run_name="synthetic-browser-test"),
    paths=PathsConfig(
        data_root=root / "data",
        reports_root=root / "reports",
    ),
)
cfg.run_data_dir.mkdir(parents=True, exist_ok=True)
write_lineage(cfg, "cards")
pd.DataFrame({"text_id": [0] * 60 + [1] * 60, "token_pos": list(range(60)) * 2}).to_parquet(
    cfg.token_metadata_path
)
pd.DataFrame(
    [
        {
            "feature_id": fid,
            "primary_label": "manual_review",
            "analysis_token_frequency": freq,
            "analysis_activation_count": support,
            "analysis_token_denominator": 100,
            "stored_token_denominator": 120,
            "analysis_text_count": 2,
            "p99_activation": activation,
        }
        for fid, freq, support, activation in [(7, 0.1, 10, 3.0), (9, 0.05, 5, 2.0)]
        + [(fid, 0.01, 1, 0.1) for fid in range(100, 130)]
    ]
).to_parquet(cfg.feature_cards_path)
pd.DataFrame(
    [
        {
            "feature_id": fid,
            "activation": activation,
            "text_id": 0,
            "token_pos": i,
            "left_context": (
                "Earlier saved context. " + "Some surrounding text. " * 6 if fid == 7 else ""
            ) + "A quantum ",
            "center_token": center,
            "right_context": " example.",
            "source": "synthetic",
        }
        for fid, activation, center in [
            (7, 3.0, "<script>globalThis.injected=true</script>"),
            (9, 2.0, "neighbor"),
        ]
        for i in range(6)
    ]
).to_parquet(cfg.top_examples_path)
pd.DataFrame([{"feature_id": 7, "activation": float(i)} for i in range(5)]).to_parquet(
    cfg.sae_activations_path
)
pd.DataFrame([{"feature_i": 7, "feature_j": 9, "decoder_cosine": 0.8}]).to_parquet(
    cfg.decoder_neighbors_path
)
pd.DataFrame(
    [{"feature_i": 7, "feature_j": 9, "coactivation_count": 4, "jaccard": 0.4}]
).to_parquet(cfg.coactivation_pairs_path)
write_explorer(cfg.run_data_dir, root / "report.html")

partial = root / "partial"
partial.mkdir(exist_ok=True)
(partial / "feature_cards.parquet").write_bytes(b"corrupt artifact")
pd.DataFrame().to_parquet(partial / "decoder_neighbors.parquet")
write_explorer(partial, root / "partial.html")

legacy = root / "legacy"
legacy.mkdir(exist_ok=True)
pd.DataFrame(
    [{"feature_id": 7, "token_frequency": 0.1, "n_token_activations": 10, "n_texts": 2}]
).to_parquet(legacy / "feature_stats.parquet")
write_explorer(legacy, root / "legacy.html")

empty_selection = root / "empty-selection"
empty_selection.mkdir(exist_ok=True)
pd.DataFrame(columns=["feature_id"]).to_parquet(empty_selection / "feature_cards.parquet")
pd.DataFrame([{"feature_id": 7, "n_token_activations": 1}]).to_parquet(empty_selection / "feature_stats.parquet")
write_explorer(empty_selection, root / "empty-selection.html")
