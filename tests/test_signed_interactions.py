"""Checks for sign orientation and orthogonal energy decomposition."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "signed_analysis",
    Path(__file__).resolve().parents[1] / "scripts/analyze_signed_interactions.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_parity_recovers_bilinear_response_and_even_curvature():
    coefficients = {
        (0, 0): np.array([2.0, 1.0]),
        (1, 0): np.array([3.0, -4.0]),
        (0, 1): np.array([5.0, 2.0]),
        (1, 1): np.array([-7.0, 6.0]),
    }
    grid = {
        (s, t): sum(s**a * t**b * v for (a, b), v in coefficients.items())
        for s in (-1, 1)
        for t in (-1, 1)
    }
    result = module.parity_components(grid)
    for key, expected in coefficients.items():
        np.testing.assert_allclose(result[key], expected)
    assert sum(v @ v for v in result.values()) == pytest.approx(
        np.mean([v @ v for v in grid.values()])
    )
    with pytest.raises(ValueError):
        module.parity_components({(1, 1): [1.0, 2.0]})


def test_mixed_coefficient_is_stable_for_bilinear_function():
    response = np.array([2.0, -3.0])
    for alpha, beta in [(0.1, 0.005), (0.1, 0.01)]:
        grid = {(s, t): s * t * alpha * beta * response for s in (-1, 1) for t in (-1, 1)}
        result = module.parity_components(grid)
        np.testing.assert_allclose(result[1, 1] / (alpha * beta), response)
        for key in [(0, 0), (0, 1), (1, 0)]:
            np.testing.assert_allclose(result[key], 0)
