from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ActivationMode = Literal["topk", "positive"]


@dataclass(frozen=True)
class ModelSelection:
    model: str = "gemma-3-1b-pt"
    layer: int | None = None
    site: str = "resid_post"
    width: str = "16k"
    l0: str = "medium"


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    sae_release: str
    sae_id: str
    layer: int
    hook_name: str
    site: str = "resid_post"
    width: str = "16k"
    l0: str = "medium"
    d_model: int | None = None
    d_sae: int | None = None


@dataclass(frozen=True)
class CollectionConfig:
    run_name: str
    corpus: str = "pile-10k"
    max_texts: int = 1500
    max_seq_len: int = 256
    activation_mode: ActivationMode = "topk"
    top_k_features_per_token: int = 64
    residual_sample_stride: int = 8
    random_seed: int = 42


@dataclass(frozen=True)
class ActivationRowFilterConfig:
    exclude_token_positions: tuple[int, ...] = (0,)
    exclude_token_positions_ge: int | None = None
    exclude_token_strings: tuple[str, ...] = ("<bos>",)
    exclude_token_substrings: tuple[str, ...] = ("â", "€", "™")
    include_sources: tuple[str, ...] | None = None
    exclude_sources: tuple[str, ...] = ()
    min_activation: float | None = None
    require_finite_activation: bool = True


@dataclass(frozen=True)
class FeatureFilterConfig:
    min_feature_token_count: int = 50
    min_feature_text_count: int = 10
    max_feature_token_frequency: float = 0.20


@dataclass(frozen=True)
class AnalysisConfig:
    coactivation_max_pairs: int = 100_000
    decoder_neighbors_top_k: int = 20
    decoder_neighbors_batch_size: int = 512
    bimodality_min_points: int = 100
    top_examples_per_feature: int = 20
    context_window: int = 20
    inspection_top_features: int = 30
    inspection_top_pairs: int = 30
    pca_components: int = 20
    pca_scatter_sample: int = 5000
    umap_neighbors: int = 30
    umap_min_dist: float = 0.10
    umap_metric: str = "cosine"
    umap_random_state: int = 42
    lda_min_class_size: int = 20


@dataclass(frozen=True)
class PathsConfig:
    raw_texts_path: Path = Path("data/raw/texts_sample.jsonl")
    data_root: Path = Path("data/processed")
    reports_root: Path = Path("reports")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    collection: CollectionConfig
    activation_filter: ActivationRowFilterConfig = ActivationRowFilterConfig()
    feature_filter: FeatureFilterConfig = FeatureFilterConfig()
    analysis: AnalysisConfig = AnalysisConfig()
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
        suffix = "positive" if self.collection.activation_mode == "positive" else "topk"
        return self.run_data_dir / f"sae_activations_{suffix}.parquet"

    @property
    def token_activation_summary_path(self) -> Path:
        return self.run_data_dir / "token_activation_summary.parquet"

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
    def coactivation_pairs_path(self) -> Path:
        return self.run_data_dir / "coactivation_pairs.parquet"

    @property
    def decoder_neighbors_path(self) -> Path:
        return self.run_data_dir / "decoder_neighbors.parquet"

    @property
    def geometry_vs_coactivation_path(self) -> Path:
        return self.run_data_dir / "geometry_vs_coactivation.parquet"

    @property
    def bimodal_candidates_path(self) -> Path:
        return self.run_data_dir / "bimodal_feature_candidates.parquet"

    @property
    def residual_pca_summary_path(self) -> Path:
        return self.run_data_dir / "residual_pca_summary.parquet"

    @property
    def decoder_pca_summary_path(self) -> Path:
        return self.run_data_dir / "decoder_pca_summary.parquet"

    @property
    def decoder_feature_pca_path(self) -> Path:
        return self.run_data_dir / "decoder_feature_pca.parquet"

    @property
    def decoder_feature_umap_path(self) -> Path:
        return self.run_data_dir / "decoder_feature_umap.parquet"

    @property
    def decoder_feature_lda_path(self) -> Path:
        return self.run_data_dir / "decoder_feature_lda.parquet"

    @property
    def inspection_feature_summaries_path(self) -> Path:
        return self.run_data_dir / "inspection_feature_summaries.parquet"

    @property
    def inspection_pair_summaries_path(self) -> Path:
        return self.run_data_dir / "inspection_pair_summaries.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.run_reports_dir / "manifest.json"

    @property
    def summary_md_path(self) -> Path:
        return self.run_reports_dir / "summary.md"

    @property
    def html_report_path(self) -> Path:
        return self.run_reports_dir / "index.html"

    @property
    def inspection_report_md_path(self) -> Path:
        return self.run_reports_dir / "inspection_report.md"

    @property
    def inspection_report_json_path(self) -> Path:
        return self.run_reports_dir / "inspection_report.json"
