"""Cross-validation of the equal-SLW (Section 5) and general-SLW (Section 6)
spectral methods on the equal-SLW test configuration (Section 11.2), plus
eigenvalue-counting checks (Theorem 9, Eq. 36)."""

import numpy as np
import pytest

from src.config import BREAKPOINTS_GENERAL, EQUAL_SLW_SCALE, EQUAL_SLW_OMEGA_EXPECTED, RATE, STRIKE
from src.spectral_core import build_layered_model, scale_transform_point, compute_cn, price_eigen_expansion
from src.equal_slw import build_equal_slw_basis, check_equal_slw
from src.general_slw import build_general_slw_basis, n_modes_weyl, eigenvalue_count, locate_eigenvalues


@pytest.fixture
def equal_slw_model():
    L = np.diff(BREAKPOINTS_GENERAL)
    varsigma = EQUAL_SLW_SCALE * L
    return build_layered_model(BREAKPOINTS_GENERAL, varsigma)


def test_equal_slw_omega_matches_expected(equal_slw_model):
    omega = check_equal_slw(equal_slw_model)
    assert omega == pytest.approx(EQUAL_SLW_OMEGA_EXPECTED, abs=1e-3)


def test_equal_slw_eigenvalues_lie_in_unit_interval(equal_slw_model):
    """zeta_i = cos(varphi_i) in (-1, 1), Proposition 6."""
    basis = build_equal_slw_basis(equal_slw_model)
    # mu_n = xi_n^2 = (varphi_n/omega)^2 >= 0 for every mode.
    assert np.all(basis.mu >= 0.0)
    assert np.all(np.diff(basis.mu) >= -1e-9), "modes must be sorted ascending in mu"


def test_equal_slw_matches_general_slw_closed_form_double_notouch(equal_slw_model):
    """Expected outcome (Section 11.2): the two spectral methods agree to ~4e-15
    when applied to the same equal-SLW configuration."""
    S_star = 101.0
    T = 1.0
    x0 = scale_transform_point(equal_slw_model, S_star)

    basis_eq = build_equal_slw_basis(equal_slw_model)
    cn_eq = compute_cn(basis_eq, equal_slw_model, "notouch")
    V_eq = price_eigen_expansion(basis_eq, equal_slw_model, x0, T, RATE, cn_eq)

    n_modes = n_modes_weyl(equal_slw_model, T, log_inv_eps=32.0)
    basis_gen = build_general_slw_basis(equal_slw_model, n_modes)
    cn_gen = compute_cn(basis_gen, equal_slw_model, "notouch")
    V_gen = price_eigen_expansion(basis_gen, equal_slw_model, x0, T, RATE, cn_gen)

    assert V_eq == pytest.approx(V_gen, abs=1e-10)


def test_eigenvalue_counting_formula_matches_closed_form_eigenvalues(equal_slw_model):
    """Expected outcome: the counting formula (Theorem 9, Eq. 36) must locate
    exactly the closed-form eigenvalues of equal_slw to within bisection tolerance."""
    basis_closed = build_equal_slw_basis(equal_slw_model)
    xi_closed = np.sqrt(basis_closed.mu)[:10]
    xi_closed.sort()
    xi_counted = locate_eigenvalues(equal_slw_model, n_modes=10, tol=1e-13)
    np.testing.assert_allclose(xi_counted, xi_closed, rtol=1e-8)


def test_eigenvalue_count_is_monotone_nondecreasing(equal_slw_model):
    xis = np.linspace(0.1, 50, 30)
    counts = [eigenvalue_count(equal_slw_model, xi) for xi in xis]
    assert np.all(np.diff(counts) >= 0)


def test_general_slw_reproduces_40_lowest_equal_slw_eigenvalues():
    """Expected outcome (Section 11.2): the general method reproduces the 40
    lowest closed-form eigenvalues of the equal-SLW configuration to relative
    ~3.8e-15; this test uses a looser floating-point tolerance."""
    L = np.diff(BREAKPOINTS_GENERAL)
    varsigma = EQUAL_SLW_SCALE * L
    model = build_layered_model(BREAKPOINTS_GENERAL, varsigma)
    basis_closed = build_equal_slw_basis(model)
    xi_closed = np.sort(np.sqrt(basis_closed.mu))[:40]
    xi_general = locate_eigenvalues(model, n_modes=40, tol=1e-13)
    rel_err = np.max(np.abs(xi_general - xi_closed) / xi_closed)
    assert rel_err < 1e-6
