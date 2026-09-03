"""Constrained native-state edits and development-only interaction estimators."""

import numpy as np


def null_projector(w, u):
    """Orthogonal complement of the encoder and decoder, including collinearity."""
    matrix = np.column_stack([w, u]).astype(np.float64)
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=0) == 0):
        raise ValueError("Nonfinite or zero constraint vector")
    matrix /= np.linalg.norm(matrix, axis=0)
    basis, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    basis = basis[:, singular > singular[0] * 1e-10]
    return np.eye(len(w)) - basis @ basis.T


def context_delta(h, c, q, length, sign=1):
    """Rotate in the constraint nullspace by a specified chord length.

    Calculate once at the baseline; reuse exactly in both factorial arms.
    Degenerate directions fail explicitly rather than receiving a fallback edit.
    """
    h, c = np.asarray(h, dtype=np.float64), np.asarray(c, dtype=np.float64)
    if sign not in (-1, 1) or not np.isfinite(length) or length <= 0:
        raise ValueError("Require positive finite length and sign +/-1")
    r = q @ h
    radius = np.linalg.norm(r)
    if radius < 1e-10 or length >= 2 * radius:
        raise ValueError("Degenerate nullspace radius or excessive chord")
    v = q @ c
    v -= r * (r @ v) / radius**2
    if np.linalg.norm(v) < 1e-10:
        raise ValueError("Context direction has no usable tangent component")
    v /= np.linalg.norm(v)
    theta = sign * 2 * np.arcsin(length / (2 * radius))
    return (np.cos(theta) - 1) * r + np.sin(theta) * radius * v


def matched_contrast(x, labels, strata):
    """Within-stratum high-minus-low contrast, overlap weighted.

    This adjusts only the supplied strata, not every contextual confound.
    """
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if len(x) != len(labels) or len(x) != len(strata) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid contrast inputs")
    groups = {}
    for i, key in enumerate(strata):
        groups.setdefault(key, []).append(i)
    total, weight_sum, support = np.zeros(x.shape[1]), 0.0, 0
    for indices in groups.values():
        indices = np.asarray(indices)
        lo, hi = indices[labels[indices] == 0], indices[labels[indices] == 1]
        if not len(lo) or not len(hi):
            continue
        weight = len(lo) * len(hi) / (len(lo) + len(hi))
        total += weight * (x[hi].mean(0) - x[lo].mean(0))
        weight_sum += weight
        support += len(indices)
    if weight_sum == 0:
        raise ValueError("No within-stratum overlap")
    return total / weight_sum, support


def interaction(cells):
    cells = np.asarray(cells)
    if cells.shape[0] != 4:
        raise ValueError("Cell order must be baseline, decoder, context, combined")
    return cells[3] - cells[2] - cells[1] + cells[0]


def paired_bootstrap(values, groups, seed=42, draws=2000):
    """Percentile interval for the equally weighted document-mean effect."""
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty/nonfinite bootstrap population")
    means = np.asarray([values[groups == g].mean() for g in np.unique(groups)])
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(means, len(means), replace=True).mean() for _ in range(draws)])
    return dict(
        mean=float(means.mean()),
        low=float(np.quantile(boot, 0.025)),
        high=float(np.quantile(boot, 0.975)),
        documents=len(means),
    )
