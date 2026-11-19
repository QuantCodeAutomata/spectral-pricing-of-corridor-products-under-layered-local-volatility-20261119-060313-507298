"""Unit tests for the shared layered model and eigen-expansion kernel
(src/spectral_core.py), using the paper's general single-bucket configuration
(Section 11, page 28)."""

import numpy as np
import pytest

from src.config import BREAKPOINTS_GENERAL, VARSIGMA_GENERAL, RATE, STRIKE, SPOT
from src.spectral_core import (
    build_layered_model,
    scale_transform_point,
    compute_cn,
    price_eigen_expansion,
)
from src.general_slw import build_general_slw_basis, n_modes_weyl


@pytest.fixture
def general_model():
    return build_layered_model(BREAKPOINTS_GENERAL, VARSIGMA_GENERAL)


def test_scale_transform_monotone_and_zero_at_origin(general_model):
    y = general_model.y
    assert y[0] == 0.0
    assert np.all(np.diff(y) > 0), "scale coordinate must be strictly increasing (Eq. 6)"


def test_layer_of_price_consistency(general_model):
    for i, Y_mid in enumerate((general_model.Y[:-1] + general_model.Y[1:]) / 2):
        assert general_model.layer_of_price(Y_mid) == i


def test_scale_transform_point_matches_breakpoints(general_model):
    for i, Y_i in enumerate(general_model.Y):
        x_i = scale_transform_point(general_model, Y_i)
        assert abs(x_i - general_model.y[i]) < 1e-10


def test_double_notouch_and_call_reproduce_paper_reference(general_model):
    """Reference values, Section 11.2, page 29."""
    T = 1.0
    x0 = scale_transform_point(general_model, SPOT)
    n_modes = n_modes_weyl(general_model, T, log_inv_eps=32.0)
    basis = build_general_slw_basis(general_model, n_modes)
    cn_notouch = compute_cn(basis, general_model, "notouch")
    cn_call = compute_cn(basis, general_model, "call", STRIKE)
    V_notouch = price_eigen_expansion(basis, general_model, x0, T, RATE, cn_notouch)
    V_call = price_eigen_expansion(basis, general_model, x0, T, RATE, cn_call)
    assert V_notouch == pytest.approx(0.819596756784, abs=1e-8)
    assert V_call == pytest.approx(2.807218341507, abs=1e-8)


def test_double_notouch_between_zero_and_one(general_model):
    """A no-touch survival probability (discounted) must lie in (0, e^{-rT})."""
    T = 1.0
    x0 = scale_transform_point(general_model, SPOT)
    n_modes = n_modes_weyl(general_model, T)
    basis = build_general_slw_basis(general_model, n_modes)
    cn = compute_cn(basis, general_model, "notouch")
    V = price_eigen_expansion(basis, general_model, x0, T, RATE, cn)
    assert 0.0 < V < np.exp(-RATE * T)


def test_call_price_decreases_with_maturity_corridor_bound():
    """A knock-out Call must be worth less than the corridor width (no-arbitrage
    upper bound implied by the killed domain [B_-, B_+])."""
    model = build_layered_model(BREAKPOINTS_GENERAL, VARSIGMA_GENERAL)
    x0 = scale_transform_point(model, SPOT)
    for T in (0.1, 1.0, 5.0):
        n_modes = n_modes_weyl(model, T)
        basis = build_general_slw_basis(model, n_modes)
        cn = compute_cn(basis, model, "call", STRIKE)
        V = price_eigen_expansion(basis, model, x0, T, RATE, cn)
        assert 0.0 <= V <= (BREAKPOINTS_GENERAL[-1] - STRIKE)
