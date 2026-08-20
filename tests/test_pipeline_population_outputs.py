from __future__ import annotations

from dataclasses import replace

import pandas as pd

from sae_feature_atlas.analysis.populations import build_activation_populations
from sae_feature_atlas.config.registry import make_config
from sae_feature_atlas.config.schema import (
    AnalysisConfig,
    FeatureFilterConfig,
    PathsConfig,
)
from sae_feature_atlas.inspection.context import ContextRenderer
from sae_feature_atlas.inspection.feature_cards import build_and_save_feature_outputs


TOKEN_TEXT = {
    0: "<bos>",
    1: '"',
    2: " clean",
    3: ".",
}


def _decode(ids: list[int]) -> str:
    return "".join(TOKEN_TEXT[token_id] for token_id in ids)


def test_feature_outputs_keep_stored_artifacts_but_clean_example_centers(tmp_path) -> None:
    tokens = pd.DataFrame(
        [
            {
                "text_id": 0,
                "token_pos": pos,
                "source": "synthetic",
                "token_id": pos,
                "token_str": TOKEN_TEXT[pos],
            }
            for pos in range(4)
        ]
    )
    activations = pd.DataFrame(
        [
            {
                "text_id": 0,
                "token_pos": 1,
                "source": "synthetic",
                "token_str": '"',
                "feature_id": 7,
                "activation": 100.0,
            },
            {
                "text_id": 0,
                "token_pos": 2,
                "source": "synthetic",
                "token_str": " clean",
                "feature_id": 7,
                "activation": 1.0,
            },
        ]
    )
    cfg = replace(
        make_config(run_name="population-output-test"),
        feature_filter=FeatureFilterConfig(
            min_feature_token_count=1,
            min_feature_text_count=1,
            max_feature_token_frequency=1.0,
        ),
        analysis=AnalysisConfig(top_examples_per_feature=5, context_window=2),
        paths=PathsConfig(
            raw_texts_path=tmp_path / "raw.jsonl",
            data_root=tmp_path / "data",
            reports_root=tmp_path / "reports",
        ),
    )
    populations = build_activation_populations(
        activations,
        tokens,
        cfg.activation_filter,
    )

    build_and_save_feature_outputs(
        populations,
        ContextRenderer(tokens, _decode),
        cfg,
    )

    stats = pd.read_parquet(cfg.feature_stats_path).set_index("feature_id").loc[7]
    examples = pd.read_parquet(cfg.top_examples_path)
    cards = pd.read_parquet(cfg.feature_cards_path)

    assert stats["stored_activation_count"] == 2
    assert stats["analysis_activation_count"] == 1
    assert stats["stored_frequency_semantics"] == "retained_topk_membership_per_collected_token"
    assert examples["target_token_str"].tolist() == [" clean"]
    assert examples["center_token"].tolist() == [" clean"]
    assert examples["target_quality"].tolist() == ["clean"]
    assert '"' in examples.loc[0, "display_context"]
    assert cards.loc[0, "activation_population"] == "analysis_activations"
    assert cards.loc[0, "feature_population"] == "analysis_features"
