import numpy as np
import torch

from sae_feature_atlas.scientific.geometry_analysis import participation, permutation_shift


def test_participation_basis_and_scale():
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    pr, entropy, n90 = participation(x)
    np.testing.assert_allclose(pr, [1, 4])
    np.testing.assert_allclose(entropy, [1, 4])
    np.testing.assert_array_equal(n90, [1, 4])
    np.testing.assert_allclose(participation(x * 9)[0], pr)


def test_orthogonal_test_ignores_own_axis_and_is_rotation_invariant():
    rng = np.random.default_rng(42)
    x = torch.tensor(rng.normal(size=(40, 4)), dtype=torch.float64)
    labels = np.repeat([0, 1], 20)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    strata = [np.arange(40)]
    a, _, na = permutation_shift(x, labels, direction, strata, seed=9, permutations=99)
    changed = x.clone()
    changed[:, 0] += torch.tensor(labels * 1000)
    b, _, nb = permutation_shift(changed, labels, direction, strata, seed=9, permutations=99)
    np.testing.assert_allclose(na, nb, atol=1e-10)
    assert a["p"] == b["p"]
    q, _ = torch.linalg.qr(torch.tensor(rng.normal(size=(4, 4)), dtype=torch.float64))
    c, _, nc = permutation_shift(x @ q, labels, direction @ q, strata, seed=9, permutations=99)
    np.testing.assert_allclose(na, nc, atol=1e-10)
    np.testing.assert_allclose(a["orthogonal_squared"], c["orthogonal_squared"])


def test_frozen_strata_cannot_produce_evidence():
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 100.0], [0.0, 101.0]], dtype=torch.float64)
    result, _, _ = permutation_shift(
        x,
        np.array([0, 0, 1, 1]),
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        [np.array([0, 1]), np.array([2, 3])],
        seed=1,
        permutations=99,
    )
    assert result["p"] == 1
    assert result["movable_fraction"] == 0


def test_grouped_permutations_preserve_counts_and_exchangeability():
    from sae_feature_atlas.scientific.geometry_primary import GroupedPermuter

    labels = np.array([0, 1, 0, 1, 1, 1])
    strata = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    permuter = GroupedPermuter()
    rng = np.random.default_rng(51)
    draws = np.stack([permuter(labels, strata, rng) for _ in range(4000)])
    for s in strata:
        assert (draws[:, s].sum(1) == labels[s].sum()).all()
    assert np.max(np.abs(draws[:, :4].mean(0) - 0.5)) < 0.04
    assert np.array_equal(labels, [0, 1, 0, 1, 1, 1])


def test_large_frozen_mean_shift_counts_numerical_ties_conservatively():
    rng = np.random.default_rng(14)
    labels = np.r_[np.zeros(3, dtype=int), np.ones(7, dtype=int)]
    x = torch.tensor(rng.normal(size=(10, 9)) * 1e6, dtype=torch.float64)
    u = torch.tensor(rng.normal(size=9), dtype=torch.float64)
    u = u / u.norm()
    result, _, _ = permutation_shift(
        x, labels, u, [np.arange(3), np.arange(3, 10)], seed=92, permutations=99
    )
    assert result["p"] == 1
