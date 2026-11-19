"""Joining the time buckets (Section 10): tiled local volatility, piecewise
constant in both price and time.

Each bucket k = 1,...,K carries its own layered model and eigenbasis, built
with the general-SLW method (Theorem 9 / Corollary 10) applied in the price
coordinate of that bucket. Consecutive buckets are connected by a transfer
matrix Q^{(k)} (Eq. 60) that re-expands bucket k+1's eigenbasis in bucket
k's own speed measure m_k(dy) = 2 dy / varsigma_k(y)^2; the bucket recursion
(Eq. 61) then propagates terminal payoffs backward in time, and the running
reward recursion (Eq. 65) propagates accrual-type payoffs using the
bucket-level stationary (resolvent) solution.

The speed measure m_k is piecewise constant layer by layer (varsigma_k is
constant on each layer), so every m_k-weighted integral below reduces to a
plain Gauss-Legendre quadrature of the two eigenfunctions in the *price*
coordinate S, scaled by the constant layer weight 2/varsigma_{i,k}^2 -- the
number of quadrature points per layer is not specified in the paper; 32
points are used here, generous for the piecewise cos/sin integrands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.general_slw import build_general_slw_basis, n_modes_weyl
from src.spectral_core import (
    EigenBasis,
    LayeredModel,
    build_layered_model,
    compute_cn,
    eval_phi,
    scale_transform_point,
)

GL_NODES_DEFAULT = 32  # Gauss-Legendre points per layer for Q^{(k)} and psi^{(k)} (not
# specified in the paper; chosen generously for the trigonometric integrands, Section 10.1)


@dataclass
class Bucket:
    model: LayeredModel
    basis: EigenBasis
    dt: float  # Delta t_k


def build_bucket(Y: np.ndarray, varsigma_k: np.ndarray, dt: float, r: float,
                  log_inv_eps: float) -> Bucket:
    """One time bucket's layered model and general-SLW eigenbasis (Eq. 62)."""
    model = build_layered_model(Y, varsigma_k)
    n_modes = n_modes_weyl(model, dt, log_inv_eps=log_inv_eps)
    basis = build_general_slw_basis(model, n_modes)
    return Bucket(model=model, basis=basis, dt=dt)


def _layer_gl_nodes(Y_lo: float, Y_hi: float, n_quad: int):
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    S = 0.5 * (Y_hi - Y_lo) * nodes + 0.5 * (Y_hi + Y_lo)
    w = 0.5 * (Y_hi - Y_lo) * weights
    return S, w


def _eval_basis_on_layer(basis: EigenBasis, model: LayeredModel, m: int, S: np.ndarray) -> np.ndarray:
    """phi_n(Phi(S)) for every mode n, S an array of prices inside layer m.

    Vectorized form of ``spectral_core.eval_phi`` restricted to one layer.
    """
    x = model.y[m] + np.sqrt(2.0) * (S - model.Y[m]) / model.varsigma[m] ** 2
    s = x - model.y[m]
    k = basis.k[:, m][:, None]
    A = basis.A[:, m][:, None]
    B = basis.B[:, m][:, None]
    return np.real(A * np.cos(k * s) + B * np.sin(k * s))  # (n_modes, len(S))


def transfer_matrix(bucket_from: Bucket, bucket_to: Bucket, n_quad: int = GL_NODES_DEFAULT):
    """Q^{(k)}_{mn} = <phi_n^{(k+1)}, phi_m^{(k)}>_{m_k} / ||phi_m^{(k)}||_{m_k}^2, Eq. (60).

    Also returns the m_k-weighted norms ||phi_m^{(k)}||_{m_k}^2 (used by the
    running-reward recursion, Eq. 65).
    """
    model_from, model_to = bucket_from.model, bucket_to.model
    assert np.allclose(model_from.Y, model_to.Y), "buckets must share breakpoints"
    n_from, n_to = bucket_from.basis.n_modes, bucket_to.basis.n_modes
    numerator = np.zeros((n_from, n_to))
    denom = np.zeros(n_from)
    for m in range(model_from.N):
        Y_lo, Y_hi = model_from.Y[m], model_from.Y[m + 1]
        S, w = _layer_gl_nodes(Y_lo, Y_hi, n_quad)
        weight_factor = 2.0 / np.real(model_from.varsigma[m]) ** 2  # bucket k's own speed density
        phi_from = _eval_basis_on_layer(bucket_from.basis, model_from, m, S)  # (n_from, nq)
        phi_to = _eval_basis_on_layer(bucket_to.basis, model_to, m, S)  # (n_to, nq)
        ww = w * weight_factor
        numerator += (phi_from * ww) @ phi_to.T
        denom += np.sum(phi_from**2 * ww, axis=1)
    Q = numerator / denom[:, None]
    return Q, denom


def bucket_recursion_terminal(buckets: list[Bucket], r: float, payoff: str,
                               K: float | None, S_star: float,
                               n_quad: int = GL_NODES_DEFAULT) -> float:
    """Price a terminal payoff across K time buckets, Eq. (61).

    z^{(K)} = (c_m / ||phi_m^{(K)}||_x^2)_m ,  then backward
    z^{(k)} = E_k Q^{(k)} z^{(k+1)} ,  E_k = diag(e^{-(r+mu_m^{(k)}) Delta t_k}),
    V = sum_m z_m^{(1)} phi_m^{(1)}(S_star).
    """
    K_buckets = len(buckets)
    last = buckets[-1]
    cn = compute_cn(last.basis, last.model, payoff, K)
    z = cn / last.basis.norm2
    z = np.exp(-(r + last.basis.mu) * last.dt) * z  # E_K, Eq. (61)
    for k in range(K_buckets - 2, -1, -1):
        Q, _ = transfer_matrix(buckets[k], buckets[k + 1], n_quad=n_quad)
        z = Q @ z
        E = np.exp(-(r + buckets[k].basis.mu) * buckets[k].dt)
        z = E * z
    x0 = scale_transform_point(buckets[0].model, S_star)
    phi0 = eval_phi(buckets[0].basis, buckets[0].model, x0, order=0)
    return float(np.real(np.sum(z * phi0)))


def layer_mask_in_range(model: LayeredModel, lo: float, hi: float) -> np.ndarray:
    """Boolean mask of layers entirely inside the price interval [lo, hi]."""
    return (model.Y[:-1] >= lo - 1e-9) & (model.Y[1:] <= hi + 1e-9)


def running_reward_pieces(model: LayeredModel, rate_by_layer: np.ndarray):
    """Layer-constant running-reward rate g, expressed exactly like a payoff
    piece list (alpha=rate, beta=0) so that ``compute_cn``'s machinery can be
    reused to project g onto the bucket's own eigenbasis."""
    pieces = []
    for m in range(model.N):
        pieces.append([(0.0, model.ell[m], float(rate_by_layer[m]), 0.0)])
    return pieces


def _cn_from_pieces(basis: EigenBasis, model: LayeredModel, pieces) -> np.ndarray:
    from src.spectral_core import integral_poly_trig

    cn = np.zeros(basis.n_modes)
    for m in range(model.N):
        k_col, A_col, B_col = basis.k[:, m], basis.A[:, m], basis.B[:, m]
        for s_lo, s_hi, alpha, beta in pieces[m]:
            if s_hi <= s_lo:
                continue
            cn += np.real(integral_poly_trig(alpha, beta, k_col, A_col, B_col, s_lo, s_hi))
    return cn


def _rate_for_bucket(rate_by_layer, k: int) -> np.ndarray:
    """``rate_by_layer`` is either one array shared by every bucket (for a
    running-reward rate that does not depend on time, e.g. the corridor
    occupation indicator) or a list with one array per bucket (for a rate
    that follows the bucket's own time-varying volatility, e.g. the
    log-return quadratic-variation rate g_i = varsigma_{i,k}^2)."""
    if isinstance(rate_by_layer, (list, tuple)):
        return rate_by_layer[k]
    return rate_by_layer


def stationary_solution(bucket: Bucket, r: float, rate_by_layer: np.ndarray) -> np.ndarray:
    """Omega_k^infty expanded in bucket k's own eigenbasis: the resolvent
    solution of (A_k - r) Omega = -g with absorbing boundaries,

        Omega(x) = sum_n phi_n(x) * c_n^{(g)} / ((mu_n + r) * ||phi_n||^2),

    which generalizes the r=0 stationary solution of Eq. (64) to a bucket
    discount rate r >= 0 via the resolvent kernel 1/(mu_n+r) in place of the
    semigroup kernel e^{-mu_n Delta t}.
    """
    pieces = running_reward_pieces(bucket.model, rate_by_layer)
    cg = _cn_from_pieces(bucket.basis, bucket.model, pieces)
    return cg / ((bucket.basis.mu + r) * bucket.basis.norm2)


def _stationary_solutions(buckets: list[Bucket], r: float, rate_by_layer) -> list[np.ndarray]:
    return [stationary_solution(b, r, _rate_for_bucket(rate_by_layer, k))
            for k, b in enumerate(buckets)]


def bucket_recursion_running(buckets: list[Bucket], r: float, rate_by_layer: np.ndarray,
                              S_star: float, n_quad: int = GL_NODES_DEFAULT) -> float:
    """Expected discounted running reward across K buckets, Eq. (65).

    Writing the value on bucket k, relative to that bucket's own stationary
    (particular) solution, as V_k(x) = Omega_k^infty(x) + sum_n z_n^{(k)} phi_n^{(k)}(x)
    (variation of parameters: Omega_k^infty solves the inhomogeneous bucket
    equation exactly, so the remainder z^{(k)} is a pure homogeneous mode
    that decays with the bucket's own semigroup), continuity of value at
    t_k together with the re-expansion Q^{(k)} gives the stated recursion

        z^{(k)} = E_k * (Q^{(k)} z^{(k+1)} + psi^{(k)}),
        psi_m^{(k)} = <Omega_{k+1}^infty - Omega_k^infty, phi_m^{(k)}>_{m_k} / ||phi_m^{(k)}||_{m_k}^2,

    with terminal condition V_K(T^-, x) = 0 (nothing left to accrue after
    maturity), i.e. z^{(K)} = -E_K w_K where w_K are Omega_K^infty's own
    eigen-coefficients; the total price is then
    V(0, S_star) = Omega_1^infty(S_star) + sum_m z_m^{(1)} phi_m^{(1)}(S_star).
    """
    K_buckets = len(buckets)
    omegas = _stationary_solutions(buckets, r, rate_by_layer)
    E_last = np.exp(-(r + buckets[-1].basis.mu) * buckets[-1].dt)
    z = -E_last * omegas[-1]
    for k in range(K_buckets - 2, -1, -1):
        Q, denom_k = transfer_matrix(buckets[k], buckets[k + 1], n_quad=n_quad)
        cross = _weighted_inner_with_series(buckets[k], buckets[k + 1], omegas[k + 1], n_quad=n_quad)
        psi = cross / denom_k - omegas[k]
        z = Q @ z + psi
        E = np.exp(-(r + buckets[k].basis.mu) * buckets[k].dt)
        z = E * z
    x0 = scale_transform_point(buckets[0].model, S_star)
    phi0 = eval_phi(buckets[0].basis, buckets[0].model, x0, order=0)
    omega0 = float(np.real(np.sum(omegas[0] * phi0)))
    return omega0 + float(np.real(np.sum(z * phi0)))


def _weighted_inner_with_series(bucket_from: Bucket, bucket_to: Bucket, coeffs_to: np.ndarray,
                                 n_quad: int = GL_NODES_DEFAULT) -> np.ndarray:
    """<sum_n coeffs_to[n] phi_n^{(to)}, phi_m^{(from)}>_{m_from} for every m."""
    model_from, model_to = bucket_from.model, bucket_to.model
    n_from = bucket_from.basis.n_modes
    out = np.zeros(n_from)
    for m in range(model_from.N):
        Y_lo, Y_hi = model_from.Y[m], model_from.Y[m + 1]
        S, w = _layer_gl_nodes(Y_lo, Y_hi, n_quad)
        weight_factor = 2.0 / np.real(model_from.varsigma[m]) ** 2
        phi_from = _eval_basis_on_layer(bucket_from.basis, model_from, m, S)
        phi_to = _eval_basis_on_layer(bucket_to.basis, model_to, m, S)
        func_to = coeffs_to @ phi_to  # (nq,)
        out += np.sum(phi_from * func_to[None, :] * (w * weight_factor)[None, :], axis=1)
    return out


def corridor_rate_indicator(model: LayeredModel, corridor: tuple[float, float]) -> np.ndarray:
    """Layer-constant rate g_i = 1 for layers inside the corridor, else 0
    (running reward rate for the expected accrual fraction, Section 10.1)."""
    mask = layer_mask_in_range(model, corridor[0], corridor[1])
    return mask.astype(float)


def corridor_variance_rate(model: LayeredModel, corridor: tuple[float, float]) -> np.ndarray:
    """Layer-constant rate g_i = varsigma_i^2 for layers inside the corridor,
    else 0: the quadratic-variation accrual rate d<S>_t = varsigma(S_t)^2 dt,
    restricted to the corridor (standard log-return corridor variance,
    Section 10.1, expressed in price units since varsigma is the price
    diffusion coefficient dS = varsigma(S) dW)."""
    mask = layer_mask_in_range(model, corridor[0], corridor[1])
    return mask.astype(float) * np.real(model.varsigma) ** 2
