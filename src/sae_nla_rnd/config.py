from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "google/gemma-3-1b-pt"
    sae_release: str = "gemma-scope-2-1b-pt-res"
    sae_id: str = "layer_13_width_16k_l0_medium"
    layer: int = 13
    hook_name: str = "blocks.13.hook_resid_post"
    d_model: int = 1152
    d_sae: int = 16384


@dataclass(frozen=True)
class CollectionConfig:
    run_name: str = "gemma3_1b_l13_res16k_tinystories"
    max_texts: int = 508
    max_seq_len: int = 256
    top_k_features_per_token: int = 10
    residual_sample_stride: int = 8
    random_seed: int = 42


@dataclass(frozen=True)
class ActivationRowFilterConfig:
    exclude_token_positions: tuple[int, ...] = (0,)
    exclude_token_strings: tuple[str, ...] = ("<bos>",)
    exclude_token_substrings: tuple[str, ...] = ()
    include_sources: tuple[str, ...] | None = None
    exclude_sources: tuple[str, ...] = ()
    min_activation: float | None = None
    require_finite_activation: bool = True


@dataclass(frozen=True)
class FeatureFilterConfig:
    min_feature_token_count: int = 20
    min_feature_text_count: int = 5
    max_feature_token_frequency: float = 0.20


@dataclass(frozen=True)
class PathsConfig:
    raw_texts_path: Path = Path("data/raw/texts_sample.jsonl")
    data_root: Path = Path("data/processed")
    reports_root: Path = Path("reports")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = ModelConfig()
    collection: CollectionConfig = CollectionConfig()
    activation_filter: ActivationRowFilterConfig = ActivationRowFilterConfig()
    feature_filter: FeatureFilterConfig = FeatureFilterConfig()
    paths: PathsConfig = PathsConfig()

    @property
    def run_data_dir(self) -> Path:
        return self.paths.data_root / self.collection.run_name

    @property
    def run_reports_dir(self) -> Path:
        return self.paths.reports_root / self.collection.run_name

    @property
    def token_metadata_path(self) -> Path:
        return self.run_data_dir / "token_metadata.parquet"

    @property
    def sae_activations_path(self) -> Path:
        return self.run_data_dir / "sae_activations_topk.parquet"

    @property
    def residual_vectors_path(self) -> Path:
        return self.run_data_dir / "residual_vectors_sample.npy"

    @property
    def residual_metadata_path(self) -> Path:
        return self.run_data_dir / "residual_vectors_metadata.parquet"

    @property
    def feature_stats_path(self) -> Path:
        return self.run_data_dir / "feature_stats.parquet"

    @property
    def filtered_features_path(self) -> Path:
        return self.run_data_dir / "filtered_features.parquet"

    @property
    def top_examples_path(self) -> Path:
        return self.run_data_dir / "top_feature_examples.parquet"

    @property
    def feature_cards_path(self) -> Path:
        return self.run_data_dir / "feature_cards.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.run_reports_dir / "manifest.json"


CFG = ExperimentConfig()
