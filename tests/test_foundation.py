import numpy as np
import pytest

from sae_feature_atlas.scientific.foundation import conjunction_table


def test_conjunction_requires_both_sources_and_keeps_support_failures():
    rows = [
        {"feature_id": 1, "source": "a", "status": "ok", "token_identity_p": 0.001},
        {"feature_id": 1, "source": "b", "status": "ok", "token_identity_p": 0.2},
        {"feature_id": 2, "source": "a", "status": "ok", "token_identity_p": 0.001},
        {"feature_id": 3, "source": "a", "status": "insufficient_regime_support"},
        {"feature_id": 3, "source": "b", "status": "ok", "token_identity_p": 0.001},
    ]
    result = conjunction_table(rows, [1, 2, 3], ["a", "b"])
    np.testing.assert_allclose(result.conjunction_p, [0.2, 1.0, 1.0])
    assert (result.q_by >= result.conjunction_p).all()


def test_conjunction_rejects_duplicate_source_results():
    row = {"feature_id": 1, "source": "a", "status": "ok", "token_identity_p": 0.01}
    with pytest.raises(ValueError, match="Duplicate"):
        conjunction_table([row, row], [1], ["a", "b"])


def test_source_confirmation_keeps_one_observation_per_duplicate_group(tmp_path):
    import pandas as pd
    from types import SimpleNamespace
    from sae_feature_atlas.scientific.foundation import confirm_sources
    from sae_feature_atlas.scientific.regimes import RegimeConfig, run_regimes

    docs = np.repeat(np.arange(100), 4)
    tokens = pd.DataFrame(
        {
            "text_id": docs,
            "token_pos": np.tile(np.arange(4), 100),
            "token_id": 1,
            "duplicate_group": docs,
            "source": np.where(docs % 2 == 0, "fineweb-edu-sample", "wikimedia-en"),
            "split": np.where(docs < 50, "discovery", "evaluation"),
        }
    )
    support = tokens.assign(n_positive_features=2)
    acts = tokens[["text_id", "token_pos"]].assign(
        feature_id=0,
        activation=np.expm1(
            np.tile([1.0, 3.0], 200) + np.random.default_rng(1).normal(0, 0.05, 400)
        ),
    )
    partners = tokens[["text_id", "token_pos"]].assign(
        feature_id=np.tile([1, 2], 200), activation=1.0
    )
    acts = pd.concat([acts, partners], ignore_index=True)
    rcfg = RegimeConfig(
        screen_features=1,
        min_discovery_support=150,
        min_discovery_documents=10,
        min_regime_support=3,
        min_regime_documents=3,
        min_partner_support=1,
        permutations=9,
        bootstraps=10,
        gmm_n_init=2,
    )
    run_regimes(acts, tokens, support, rcfg, tmp_path)
    path = tmp_path / "acts.parquet"
    acts.to_parquet(path, index=False)
    result = confirm_sources(
        SimpleNamespace(run_data_dir=tmp_path, sae_activations_path=path), rcfg
    )
    assert len(result) == 1
    for path in (tmp_path / "source_confirmation").glob("*.sample.parquet"):
        sample = pd.read_parquet(path)
        assert sample.duplicate_group.is_unique
        assert (sample.text_id >= 50).all()
