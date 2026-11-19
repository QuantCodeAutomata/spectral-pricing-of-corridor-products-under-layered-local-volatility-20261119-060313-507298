"""Greeks and layer vegas (Section 9).

Delta, gamma and theta follow in closed form from term-by-term
differentiation of the eigen-series (Eq. 56), already implemented in
``spectral_core.greeks_eigen_expansion``. Layer vegas use the
Hellmann-Feynman formula for the eigenvalue sensitivity (Eq. 58) and
complex-step differentiation of the full price series (Eq. 59), continuing
each eigenvalue to the perturbed (complex) volatility vector by two Newton
steps on the shooting function used in Corollary 10, as described in
Section 9.1 / page 25.
"""

from __future__ import annotations

import numpy as np

from src.general_slw import eigenfunction_norm2, propagate_eigenfunction
from src.spectral_core import (
    EigenBasis,
    LayeredModel,
    build_layered_model,
    compute_cn,
    layer_phi_squared_integral,
    price_eigen_expansion_complex,
    scale_transform_point,
)


def hellmann_feynman_eigenvalue_sensitivity(basis: EigenBasis, model: LayeredModel) -> np.ndarray:
    """d mu_n / d varsigma_i = (2 mu_n / varsigma_i) * p_{n,i}, Eq. (58).

    p_{n,i} = ||phi_n||^2 restricted to layer i, divided by the total norm;
    sum_i p_{n,i} = 1 for every mode (the mode's probability mass by layer).
    """
    phi2 = layer_phi_squared_integral(basis, model)  # (n_modes, N)
    p = phi2 / basis.norm2[:, None]
    row_sum = p.sum(axis=1)
    assert np.allclose(row_sum, 1.0, atol=1e-6), "layer weights p_{n,i} must sum to 1"
    varsigma = np.real(model.varsigma)
    dmu = (2.0 * basis.mu[:, None] / varsigma[None, :]) * p
    return dmu


def _shooting_function(model_pert: LayeredModel, xi: complex) -> complex:
    """D(xi) = u(y_N) for the pair propagated from (0, 1) at y_0 (Corollary 10).

    A true eigenvalue of the (possibly complex-volatility-perturbed) problem
    satisfies D(xi) = 0; this is the same propagation used to build the
    general-SLW eigenfunctions (``general_slw.propagate_eigenfunction``).
    """
    _, _, _, pair_N = propagate_eigenfunction(model_pert, xi)
    return pair_N[0]


def _continue_root(model_pert: LayeredModel, xi0: float, n_newton: int = 2) -> complex:
    """Two Newton steps continuing a real root xi0 to the perturbed problem.

    The Newton derivative dD/dxi is taken with a small REAL step in xi,
    independent of the (much smaller) complex-step perturbation carried by
    model_pert's volatilities; this keeps the O(h) imaginary signal intact.
    """
    xi = complex(xi0)
    for _ in range(n_newton):
        dxi = 1e-6 * max(abs(xi), 1.0)
        D0 = _shooting_function(model_pert, xi)
        Dp = _shooting_function(model_pert, xi + dxi)
        Dm = _shooting_function(model_pert, xi - dxi)
        Dprime = (Dp - Dm) / (2.0 * dxi)
        xi = xi - D0 / Dprime
    return xi


def _perturbed_basis(model_pert: LayeredModel, xi_real: np.ndarray, n_newton: int = 2) -> EigenBasis:
    """Rebuild the eigenbasis at the perturbed (complex) volatility vector,
    continuing every real root xi_real[n] to its complex counterpart and
    re-propagating the eigenfunction (Corollary 10) with complex arithmetic.
    """
    n_modes = len(xi_real)
    N = model_pert.N
    A = np.empty((n_modes, N), dtype=complex)
    B = np.empty((n_modes, N), dtype=complex)
    K = np.empty((n_modes, N), dtype=complex)
    norm2 = np.empty(n_modes, dtype=complex)
    xis = np.empty(n_modes, dtype=complex)
    ell = model_pert.ell
    for n in range(n_modes):
        xi_c = _continue_root(model_pert, xi_real[n], n_newton=n_newton)
        xis[n] = xi_c
        a, b, k, _ = propagate_eigenfunction(model_pert, xi_c)
        A[n], B[n], K[n] = a, b, k
        norm2[n] = eigenfunction_norm2(a, b, k, ell)
    return EigenBasis(mu=xis**2, A=A, B=B, k=K, norm2=norm2)


def full_layer_vega_complex_step(basis: EigenBasis, model: LayeredModel, x0: complex, T: float,
                                  r: float, payoff: str, K: float | None, S_star: float,
                                  h: float = 1e-20, n_newton: int = 2) -> np.ndarray:
    """d V / d varsigma_i for every layer i, via complex-step differentiation
    of the full eigen-series (Eq. 59), continuing the real roots of ``basis``
    to the perturbed complex volatility (``implementation_notes``, Section 9.1).

    Mode labels and the real roots ``basis.mu`` are held fixed while the
    volatility of one layer is perturbed by i*h; only the eigenfunction
    coefficients, norms and price are re-evaluated in complex arithmetic.
    """
    N = model.N
    xi_real = np.sqrt(basis.mu.astype(complex))
    Y = model.Y
    varsigma0 = np.real(model.varsigma).astype(complex)
    vegas = np.empty(N)
    for i in range(N):
        varsigma_pert = varsigma0.copy()
        varsigma_pert[i] = varsigma_pert[i] + 1j * h
        model_pert = build_layered_model(Y, varsigma_pert)
        basis_pert = _perturbed_basis(model_pert, xi_real, n_newton=n_newton)
        cn_pert = compute_cn(basis_pert, model_pert, payoff, K)
        # x0 = Phi(S_*) depends on every layer volatility between Y_0 and
        # S_*, so it must be re-evaluated (in complex arithmetic) at the
        # perturbed model rather than held at its unperturbed value.
        x0_pert = scale_transform_point(model_pert, S_star)
        V_pert = price_eigen_expansion_complex(basis_pert, model_pert, x0_pert, T, r, cn_pert)
        vegas[i] = V_pert.imag / h
    return vegas


def gamma_interface_kink_ratio(model: LayeredModel, interface_index: int) -> float:
    """Expected ratio Gamma(Y_i+)/Gamma(Y_i-) = varsigma_i^2/varsigma_{i+1}^2
    at a volatility interface (Eq. 57): price-coordinate gamma scales with
    1/varsigma^2 while the eigen-series value and its second x-derivative at
    the interface are continuous, so the discontinuity in the layer
    volatility shows up entirely in this ratio.
    """
    varsigma = np.real(model.varsigma)
    return float((varsigma[interface_index - 1] ** 2) / (varsigma[interface_index] ** 2))
