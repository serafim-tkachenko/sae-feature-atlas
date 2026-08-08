from __future__ import annotations

import numpy as np
import torch

from sae_feature_atlas.analysis.coverage import compute_decoder_residual_pc_alignment


class _SAE:
    W_dec = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )


def test_full_basis_normalized_mass_is_not_emitted(tmp_path) -> None:
    residual_path = tmp_path / "residual.npy"
    np.save(
        residual_path,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        ),
    )

    result = compute_decoder_residual_pc_alignment(
        _SAE(),
        residual_path,
        feature_ids=[0, 1],
        n_components=3,
        top_components=(1, 3, 20),
    )

    assert "pc_norm_mass_top_1" in result.columns
    assert "pc_mass_top_3" in result.columns
    assert "pc_norm_mass_top_3" not in result.columns
    assert "pc_norm_mass_top_20" not in result.columns
    assert "decoder_residual_pc_alignment_bucket" in result.columns
