from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sae_feature_atlas.analysis.geometry import get_decoder_weight


def compute_residual_pca(residual_vectors_path, n_components: int = 20) -> pd.DataFrame:
    """PCA over sampled residual-stream activations.

    This is an activation-space diagnostic: it answers how anisotropic the sampled
    residual stream is and how many directions explain large variance
    """

    X = np.load(residual_vectors_path)
    if X.shape[0] < 2:
        return pd.DataFrame()
    n = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n, random_state=0).fit(X)
    return pd.DataFrame(
        {
            "component": range(1, n + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )


def _decoder_matrix(sae, normalize: bool = True) -> np.ndarray:
    W_dec = get_decoder_weight(sae).detach().float().cpu().numpy()
    if normalize:
        norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
        W_dec = W_dec / np.clip(norms, 1e-8, None)
    return W_dec


def compute_decoder_pca(sae, feature_stats: pd.DataFrame | None = None, n_components: int = 20):
    """PCA over normalized SAE decoder vectors.

    This is a linear baseline for SAE feature-space geometry. It is useful for
    eigenvalue/anisotropy diagnostics and as a sanity check before nonlinear methods
    """

    W_dec = _decoder_matrix(sae, normalize=True)
    n = min(n_components, W_dec.shape[0], W_dec.shape[1])
    pca = PCA(n_components=n, random_state=0)
    coords = pca.fit_transform(W_dec)

    summary = pd.DataFrame(
        {
            "component": range(1, n + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    projection = pd.DataFrame(
        {
            "feature_id": np.arange(W_dec.shape[0], dtype=int),
            "decoder_pc1": coords[:, 0],
            "decoder_pc2": coords[:, 1] if n > 1 else 0.0,
        }
    )
    if feature_stats is not None and not feature_stats.empty:
        keep = [
            c
            for c in ["feature_id", "token_frequency", "p99_activation", "n_token_activations"]
            if c in feature_stats.columns
        ]
        projection = projection.merge(feature_stats[keep], on="feature_id", how="left")
    return summary, projection


def compute_decoder_umap(
    sae,
    feature_cards: pd.DataFrame | None = None,
    n_neighbors: int = 30,
    min_dist: float = 0.10,
    metric: str = "cosine",
    random_state: int = 42,
) -> pd.DataFrame:
    """UMAP over normalized SAE decoder vectors.

    UMAP is used only as a visualization/triage tool. It can reveal local
    neighborhoods, but it should not be treated as proof of semantic clustering
    """

    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "UMAP support requires the optional dependency `umap-learn` "
            "Install it with `uv sync` or `uv add umap-learn`"
        ) from exc

    W_dec = _decoder_matrix(sae, normalize=True)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(W_dec)
    df = pd.DataFrame(
        {
            "feature_id": np.arange(W_dec.shape[0], dtype=int),
            "decoder_umap_x": coords[:, 0],
            "decoder_umap_y": coords[:, 1],
        }
    )
    if feature_cards is not None and not feature_cards.empty:
        keep = [
            c
            for c in [
                "feature_id",
                "token_frequency",
                "p99_activation",
                "artifact_score",
                "bimodality_score",
                "manual_priority",
                "primary_label",
                "inspection_labels",
            ]
            if c in feature_cards.columns
        ]
        df = df.merge(feature_cards[keep], on="feature_id", how="left")
    return df
