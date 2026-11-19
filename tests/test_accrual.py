"""Tests for the corridor-accrual law via layer-dependent killing (Section 8.2-8.4).

Uses the two-layer oscillating Brownian motion, for which Lamperti's closed-form
law of the occupation fraction is available (Section 11.3), as a known-answer
check on the implementation in src/accrual.py.
"""

import numpy as np
import pytest

from src.accrual import (
    accrual_cdf,
    accrual_call_price,
    accrual_edge_masses,
    corridor_layer_mask,
    lamperti_two_layer_cdf,
)
from src.config import (
    TWO_LAYER_BREAKPOINTS,
    TWO_LAYER_CORRIDOR,
    TWO_LAYER_SPOT,
    TWO_LAYER_VARSIGMA_MINUS,
    TWO_LAYER_VARSIGMA_PLUS,
)
from src.spectral_core import build_layered_model, scale_transform_point


def _two_layer_model():
    Y = np.array(TWO_LAYER_BREAKPOINTS, dtype=float)
    varsigma = np.array([TWO_LAYER_VARSIGMA_MINUS, TWO_LAYER_VARSIGMA_PLUS])
    return build_layered_model(Y, varsigma)


def test_corridor_layer_mask_selects_expected_layer():
    model = _two_layer_model()
    mask = corridor_layer_mask(model, TWO_LAYER_CORRIDOR)
    # The corridor (100, 180) covers exactly the second (upper) layer.
    assert mask.tolist() == [False, True]


def test_accrual_cdf_matches_lamperti_closed_form():
    """Xi-derived CDF should agree with Lamperti's arctan law (Section 11.3)."""
    model = _two_layer_model()
    mask = corridor_layer_mask(model, TWO_LAYER_CORRIDOR)
    x0 = scale_transform_point(model, TWO_LAYER_SPOT)
    T = 1.0
    a_grid = np.array([0.2, 0.4, 0.6, 0.8])
    lamperti = lamperti_two_layer_cdf(a_grid, TWO_LAYER_VARSIGMA_MINUS, TWO_LAYER_VARSIGMA_PLUS)
    # The plain cosine expansion (no endpoint-singularity removal, Section 8.4)
    # converges only algebraically near a=0,1, so the tolerance here is looser
    # than the ~2e-11 reached in exp_5 with endpoint-singularity removal.
    for a, expected in zip(a_grid, lamperti):
        spectral = accrual_cdf(model, mask, T, x0, float(a), K1=64)
        assert spectral == pytest.approx(expected, abs=2e-3)


def test_accrual_cdf_is_between_zero_and_one():
    model = _two_layer_model()
    mask = corridor_layer_mask(model, TWO_LAYER_CORRIDOR)
    x0 = scale_transform_point(model, TWO_LAYER_SPOT)
    cdf = accrual_cdf(model, mask, 1.0, x0, 0.5, K1=32)
    assert 0.0 <= cdf <= 1.0 + 1e-6


def test_accrual_cdf_is_monotone_nondecreasing_in_a():
    model = _two_layer_model()
    mask = corridor_layer_mask(model, TWO_LAYER_CORRIDOR)
    x0 = scale_transform_point(model, TWO_LAYER_SPOT)
    grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    values = [accrual_cdf(model, mask, 1.0, x0, a, K1=32) for a in grid]
    diffs = np.diff(values)
    # The plain cosine expansion is not monotone pointwise (Gibbs-type ripples);
    # require monotonicity up to a small tolerance rather than exactly.
    assert np.all(diffs > -1e-3)


def test_accrual_call_price_nonnegative_and_bounded_by_mean_minus_strike():
    model = _two_layer_model()
    mask = corridor_layer_mask(model, TWO_LAYER_CORRIDOR)
    x0 = scale_transform_point(model, TWO_LAYER_SPOT)
    kappa = 0.5
    price = accrual_call_price(model, mask, 1.0, x0, kappa, K1=32)
    # alpha in [0,1] so (alpha-kappa)^+ in [0, 1-kappa].
    assert -1e-6 <= price <= (1.0 - kappa) + 1e-3


def test_accrual_edge_masses_vanish_when_spot_is_at_corridor_edge():
    """Spot started exactly at the interface: both edge masses vanish (Prop. 14 remark)."""
    model = _two_layer_model()
    rho0, rho1 = accrual_edge_masses(
        model, TWO_LAYER_CORRIDOR, TWO_LAYER_SPOT, T=1.0, r=0.0, n_modes=20
    )
    assert rho0 == 0.0
    assert rho1 == 0.0


def test_accrual_edge_masses_rho1_positive_when_spot_strictly_inside_corridor():
    model = _two_layer_model()
    spot_inside = 0.5 * (TWO_LAYER_CORRIDOR[0] + TWO_LAYER_CORRIDOR[1])
    rho0, rho1 = accrual_edge_masses(
        model, TWO_LAYER_CORRIDOR, spot_inside, T=1.0, r=0.0, n_modes=20
    )
    assert rho0 == 0.0
    assert 0.0 < rho1 <= 1.0
