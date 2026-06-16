from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sae_feature_atlas.analysis.geometry import get_decoder_weight


_EPS = 1e-12


def _normalized_decoder_matrix(sae) -> np.ndarray:
    """Return decoder directions as row-normalized vectors.

    SAE-Lens stores decoder weights in the SAE object; `get_decoder_weight`
    hides version-specific details. We normalize rows so squared projections
    onto an orthonormal PCA basis can be interpreted as mass fractions.
    """
    w_dec = get_decoder_weight(sae).detach().float().cpu().numpy()
    norms = np.linalg.norm(w_dec, axis=1, keepdims=True)
    return w_dec / np.clip(norms, 1e-8, None)


def _entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _effective_dimension(probabilities: np.ndarray) -> float:
    denom = float(np.square(probabilities).sum())
    if denom <= 0:
        return 0.0
    return float(1.0 / denom)


def _coverage_bucket(
    *,
    observed_mass: float,
    norm_top1: float,
    norm_top5: float,
    effective_dim: float,
    center_of_mass: float,
    n_components: int,
) -> str:
    """Assign a human-readable residual-coverage bucket.

    Important distinction:
    - `observed_mass` is absolute squared norm captured by the sampled residual
      PCA subspace. If it is small, the decoder direction mostly lies outside
      the observed PCA basis.
    - `norm_top*` values describe where the captured mass lies *inside* the
      observed PCA subspace. If only 20 PCs are available, norm_top_20 is always
      1.0 and should not by itself be treated as evidence of high-variance
      alignment.
    """
    if observed_mass < 0.05:
        return "mostly_outside_observed_pca"

    if norm_top1 >= 0.50:
        return "single_head_pc_aligned"

    if norm_top5 >= 0.65 and effective_dim <= 6.0:
        return "compact_high_variance_aligned"

    if center_of_mass <= max(5.0, 0.25 * n_components):
        return "broad_high_variance_aligned"

    if center_of_mass >= max(8.0, 0.65 * n_components):
        return "mid_tail_variance_aligned"

    if effective_dim >= max(8.0, 0.50 * n_components):
        return "distributed_across_components"

    return "mid_variance_aligned"


def compute_feature_coverage_profiles(
    sae,
    residual_vectors_path: Path,
    feature_ids: Iterable[int],
    n_components: int = 64,
    top_components: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """Measure how SAE decoder directions sit in residual activation PCA space.

    For a normalized SAE decoder direction d_i and residual PCA component v_k,
    the basic quantity is

        p_ik = <d_i, v_k>^2.

    Raw masses such as `pc_mass_top_5` are absolute squared projection mass in
    the full residual space. Normalized masses such as `pc_norm_mass_top_5`
    describe the distribution *inside the sampled PCA subspace*.

    This distinction is essential. If the run only has 20 PCA components, then
    `pc_norm_mass_top_20 == 1.0` for every feature by construction; it must not
    be used as a strong scientific or steering signal.
    """
    feature_ids = [int(fid) for fid in feature_ids]
    if not feature_ids:
        return pd.DataFrame()

    if not residual_vectors_path.exists():
        raise FileNotFoundError(f"Missing residual vectors: {residual_vectors_path}")

    residual = np.load(residual_vectors_path)
    if residual.ndim != 2 or min(residual.shape) < 2:
        return pd.DataFrame()

    n = min(int(n_components), residual.shape[0], residual.shape[1])
    if n < 1:
        return pd.DataFrame()

    pca = PCA(n_components=n, random_state=0)
    pca.fit(residual)
    components = pca.components_
    explained_variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)

    w_dec = _normalized_decoder_matrix(sae)
    valid_feature_ids = [fid for fid in feature_ids if 0 <= fid < w_dec.shape[0]]
    if not valid_feature_ids:
        return pd.DataFrame()

    projections = w_dec[valid_feature_ids] @ components.T
    mass = np.square(projections)
    observed_mass = mass.sum(axis=1)
    normalized_mass = mass / np.clip(observed_mass[:, None], _EPS, None)

    component_numbers = np.arange(1, n + 1, dtype=np.float64)
    rows: list[dict] = []

    for row_idx, feature_id in enumerate(valid_feature_ids):
        raw = mass[row_idx]
        norm = normalized_mass[row_idx]
        observed = float(observed_mass[row_idx])
        effective_dim = _effective_dimension(norm)
        entropy = _entropy(norm)
        center = float((component_numbers * norm).sum())

        item: dict[str, float | int | str] = {
            "feature_id": int(feature_id),
            "pc_components_observed": int(n),
            "pc_mass_observed": observed,
            "pc_observed_mass": observed,  # friendly alias for reports
            "pc_mass_unobserved_tail": float(max(0.0, 1.0 - observed)),
            "pc_unobserved_mass": float(max(0.0, 1.0 - observed)),
            "effective_pc_dim": effective_dim,
            "pc_entropy": entropy,
            "pc_center_of_mass": center,
            "pc_explained_variance_top_1": float(explained_variance_ratio[:1].sum()),
            "pc_explained_variance_top_5": float(explained_variance_ratio[: min(5, n)].sum()),
            "pc_explained_variance_observed": float(explained_variance_ratio.sum()),
        }

        for k in top_components:
            kk = min(int(k), n)
            item[f"pc_mass_top_{k}"] = float(raw[:kk].sum())
            item[f"pc_norm_mass_top_{k}"] = float(norm[:kk].sum())

        norm_top1 = float(item.get("pc_norm_mass_top_1", 0.0))
        norm_top5 = float(item.get("pc_norm_mass_top_5", norm_top1))
        item["coverage_bucket"] = _coverage_bucket(
            observed_mass=observed,
            norm_top1=norm_top1,
            norm_top5=norm_top5,
            effective_dim=effective_dim,
            center_of_mass=center,
            n_components=n,
        )
        rows.append(item)

    return pd.DataFrame(rows)
