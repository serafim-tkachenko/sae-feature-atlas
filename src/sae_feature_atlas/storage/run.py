
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AtlasRun:
    """Notebook-friendly reader for a generated `data/processed/<run_name>/` directory."""

    path: Path

    @classmethod
    def from_dir(cls, path: str | Path) -> "AtlasRun":
        run_dir = Path(path)
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        return cls(run_dir)

    def _parquet(self, name: str, *, required: bool = False) -> pd.DataFrame:
        path = self.path / name
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def token_metadata(self) -> pd.DataFrame:
        return self._parquet("token_metadata.parquet")

    def sae_activations(self) -> pd.DataFrame:
        topk = self.path / "sae_activations_topk.parquet"
        positive = self.path / "sae_activations_positive.parquet"
        if topk.exists():
            return pd.read_parquet(topk)
        if positive.exists():
            return pd.read_parquet(positive)
        return pd.DataFrame()

    def residual_vectors(self) -> np.ndarray:
        path = self.path / "residual_vectors_sample.npy"
        if not path.exists():
            return np.empty((0, 0))
        return np.load(path)

    def residual_metadata(self) -> pd.DataFrame:
        return self._parquet("residual_vectors_metadata.parquet")

    def feature_stats(self) -> pd.DataFrame:
        return self._parquet("feature_stats.parquet")

    def analysis_features(self) -> pd.DataFrame:
        return self._parquet("analysis_features.parquet")

    def feature_cards(self) -> pd.DataFrame:
        return self._parquet("feature_cards.parquet")

    def top_examples(self) -> pd.DataFrame:
        return self._parquet("top_feature_examples.parquet")

    def coactivation_pairs(self) -> pd.DataFrame:
        return self._parquet("coactivation_pairs.parquet")

    def bimodality_evaluated(self) -> pd.DataFrame:
        return self._parquet("bimodality_evaluated_features.parquet")

    def decoder_neighbors(self) -> pd.DataFrame:
        return self._parquet("decoder_neighbors.parquet")

    def geometry_vs_coactivation(self) -> pd.DataFrame:
        return self._parquet("geometry_vs_coactivation.parquet")

    def bimodal_candidates(self) -> pd.DataFrame:
        return self._parquet("bimodal_feature_candidates.parquet")

    def bimodal_peak_examples(self) -> pd.DataFrame:
        return self._parquet("bimodal_peak_examples.parquet")

    def decoder_residual_pc_alignment(self) -> pd.DataFrame:
        return self._parquet("decoder_residual_pc_alignment.parquet")

    def coverage_profiles(self) -> pd.DataFrame:
        """Compatibility alias for the renamed diagnostic artifact."""
        return self.decoder_residual_pc_alignment()

    def graph_alignment(self) -> pd.DataFrame:
        return self._parquet("feature_graph_alignment.parquet")

    def feature_examples(self, feature_id: int, n: int = 20) -> pd.DataFrame:
        examples = self.top_examples()
        if examples.empty or "feature_id" not in examples.columns:
            return pd.DataFrame()
        out = examples[examples["feature_id"].astype(int).eq(int(feature_id))].copy()
        if "activation" in out.columns:
            out = out.sort_values("activation", ascending=False)
        return out.head(n).reset_index(drop=True)

    def feature_card(self, feature_id: int) -> pd.Series:
        cards = self.feature_cards()
        if cards.empty or "feature_id" not in cards.columns:
            return pd.Series(dtype="object")
        rows = cards[cards["feature_id"].astype(int).eq(int(feature_id))]
        if rows.empty:
            return pd.Series(dtype="object")
        return rows.iloc[0]

    def artifact_status(self) -> pd.DataFrame:
        names = [
            "source_texts.jsonl",
            "token_metadata.parquet",
            "sae_activations_topk.parquet",
            "sae_activations_positive.parquet",
            "token_activation_summary.parquet",
            "residual_vectors_sample.npy",
            "residual_vectors_metadata.parquet",
            "feature_stats.parquet",
            "analysis_features.parquet",
            "top_feature_examples.parquet",
            "coactivation_pairs.parquet",
            "coactivation_metadata.json",
            "decoder_neighbors.parquet",
            "geometry_vs_coactivation.parquet",
            "bimodality_evaluated_features.parquet",
            "bimodal_feature_candidates.parquet",
            "bimodal_peak_examples.parquet",
            "decoder_residual_pc_alignment.parquet",
            "feature_graph_alignment.parquet",
            "graph_alignment_summary.parquet",
            "feature_cards.parquet",
            "lineage.json"]
        rows = []
        for name in names:
            path = self.path / name
            rows.append({"artifact": name, "exists": path.exists(), "path": str(path)})
        return pd.DataFrame(rows)
