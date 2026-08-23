import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from sae_feature_atlas.scientific.regimes import (
    RegimeConfig,
    analyze_feature,
    assign_regimes,
    bh_adjust,
    js_divergence,
    match_controls,
    neighborhood_metrics,
    permute_labels,
    run_regimes,
)


def test_posterior_threshold_and_ambiguous():
    fit = dict(
        log_mean_low=1.0,
        log_mean_high=3.0,
        log_variance_low=0.25,
        log_variance_high=0.25,
        component_weight_low=0.5,
        component_weight_high=0.5,
    )
    labels, p = assign_regimes(np.expm1([1.0, 2.0, 3.0]), fit, 0.9)
    assert labels.tolist() == [0, -1, 1]
    assert np.allclose(p.sum(1), 1)
    with pytest.raises(ValueError):
        assign_regimes([0], fit)
    with pytest.raises(ValueError):
        RegimeConfig(posterior_threshold=0.5)


def test_divergence_sparse_zeros_and_ranks():
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1)
    assert js_divergence([1, 2], [2, 4]) == pytest.approx(0)
    assert np.isnan(js_divergence([0, 0], [1, 2]))
    m = neighborhood_metrics([10, 0, 0], [0, 10, 0], 10, 10, 20)
    assert m["neighbor_jaccard"] == 0
    assert m["cosine"] == 0
    assert m["conditional_l1"] == 2


def test_permutation_exact_stratum_and_regime_sizes():
    labels = np.array([0, 0, 1, 1, 1, 0, 0])
    strata = [np.arange(4), np.arange(4, 7)]
    a = permute_labels(labels, strata, np.random.default_rng(7))
    b = permute_labels(labels, strata, np.random.default_rng(7))
    assert np.array_equal(a, b)
    for idx in strata:
        assert a[idx].sum() == labels[idx].sum()


def test_bh_known_values_and_invalid():
    assert np.allclose(bh_adjust([0.01, 0.04, 0.03, 1]), [0.04, 0.053333333, 0.053333333, 1])
    assert len(bh_adjust([])) == 0
    with pytest.raises(ValueError):
        bh_adjust([np.nan])


def test_control_matching_caliper_and_no_replacement():
    fits = pd.DataFrame(
        dict(
            feature_id=[0, 1, 2, 3],
            fit_status=["ok"] * 4,
            delta_bic=[50, 40, 1, 2],
            n_points=[100, 101, 102, 1000],
            n_documents=[50, 50, 50, 100],
        )
    )
    m = match_controls(fits, fits.iloc[:2], 0.2)
    assert m.control_id.dropna().tolist() == [2]
    assert m.match_status.tolist() == ["matched", "no_control_within_caliper"]


def test_signal_null_bootstrap_orientation_and_determinism():
    n = 120
    label = np.tile([0, 1], n // 2)
    frame = pd.DataFrame(
        dict(
            text_id=np.repeat(np.arange(30), 4),
            token_pos=np.tile(np.arange(4), 30),
            token_index=np.arange(n),
            token_id=5,
            n_positive_features=2,
            regime=label,
        )
    )
    x = sparse.csr_matrix(np.column_stack([1 - label, label, np.ones(n)]))
    cfg = RegimeConfig(
        permutations=39,
        bootstraps=20,
        min_partner_support=2,
        min_regime_support=20,
        min_regime_documents=10,
    )
    a, edges, nulls, assigned = analyze_feature(frame, x, 2, cfg)
    b, _, nulls2, _ = analyze_feature(frame, x, 2, cfg)
    assert a["js_bits"] == 1
    assert a["document_p"] == 1 / 40
    assert a["document_movable_fraction"] == 1
    assert a["js_ci_low"] == 1
    pd.testing.assert_series_equal(pd.Series(a), pd.Series(b))
    assert nulls == nulls2
    assert all(e["feature_i"] < e["feature_j"] for e in edges)
    assert sum(e["edge_status"] == "observed_zero" for e in edges) == 2
    assert assigned.used_for_comparison.all()


def test_frozen_strata_cannot_generate_significance():
    frame = pd.DataFrame(
        dict(
            text_id=np.arange(20),
            token_pos=1,
            token_index=np.arange(20),
            token_id=np.arange(20),
            n_positive_features=2,
            regime=np.tile([0, 1], 10),
        )
    )
    x = sparse.csr_matrix(np.column_stack([1 - frame.regime, frame.regime, np.ones(20)]))
    cfg = RegimeConfig(
        permutations=9,
        bootstraps=10,
        min_regime_support=5,
        min_regime_documents=5,
        min_partner_support=1,
    )
    a, _, _, _ = analyze_feature(frame, x, 2, cfg)
    assert a["document_p"] == 1 and a["document_movable_fraction"] == 0


def test_end_to_end_split_schema_and_determinism(tmp_path):
    rng = np.random.default_rng(1)
    n = 400
    token = pd.DataFrame(
        dict(text_id=np.repeat(np.arange(100), 4), token_pos=np.tile(np.arange(4), 100), token_id=1)
    )
    support = token.assign(n_positive_features=2)
    acts = token[["text_id", "token_pos"]].assign(
        feature_id=0, activation=np.expm1(np.tile([1.0, 3.0], n // 2) + rng.normal(0, 0.05, n))
    )
    partners = token[["text_id", "token_pos"]].assign(
        feature_id=np.tile([1, 2], n // 2), activation=1.0
    )
    acts = pd.concat([acts, partners], ignore_index=True)
    cfg = RegimeConfig(
        screen_features=1,
        min_discovery_support=150,
        min_discovery_documents=10,
        min_regime_support=10,
        min_regime_documents=5,
        permutations=9,
        bootstraps=10,
        gmm_n_init=2,
    )
    info = run_regimes(acts, token, support, cfg, tmp_path)
    assert info["selected_candidates"] == 1
    population = pd.read_parquet(tmp_path / "regime_token_population.parquet")
    assert population.groupby("text_id").split.nunique().max() == 1
    comp = pd.read_parquet(tmp_path / "regime_neighborhood_comparison.parquet")
    assert comp.loc[0, "document_q_bh"] == 0.1
    edges = pd.read_parquet(tmp_path / "regime_coactivation.parquet")
    assert {"feature_i", "feature_j", "conditional_probability", "edge_status"} <= set(edges)
    assignments = pd.read_parquet(tmp_path / "regime_assignments.parquet")
    assert not set(assignments.text_id) & set(population[population.split == "discovery"].text_id)
    from dataclasses import replace
    from sae_feature_atlas.scientific.regimes import require_regime_config

    require_regime_config(tmp_path, cfg)
    with pytest.raises(ValueError, match="configuration mismatch"):
        require_regime_config(tmp_path, replace(cfg, posterior_threshold=0.95))
    empty_dir = tmp_path / "no_candidates"
    info = run_regimes(acts, token, support, replace(cfg, delta_bic_threshold=1e12), empty_dir)
    assert info["selected_candidates"] == 0
    assert pd.read_parquet(empty_dir / "regime_neighborhood_comparison.parquet").empty
    assert "regime" in pd.read_parquet(empty_dir / "regime_assignments.parquet")
