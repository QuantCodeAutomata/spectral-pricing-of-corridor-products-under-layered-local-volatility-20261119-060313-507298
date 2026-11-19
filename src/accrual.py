"""Law of the corridor accrual via layer-dependent killing (Section 8.2, 8.3).

Proposition 14 kills the process at a layer-dependent rate q_i = beta * 1_{i in I_C}
while it occupies the corridor layers I_C, which preserves the tridiagonal
interface structure of Eq. (27) with nu_i = sqrt(lambda + q_i). Setting
beta = -i*u and integrating the resulting Laplace-transformed density over
the whole price domain gives hat_Psi(lambda; u) of Eq. (52); inverting in
time (Talbot contour) yields the accrual characteristic function
Xi(u) = E[e^{iu A_T}; T < tau] of Eq. (51). A cosine expansion (Eq. 53) then
recovers the law of alpha = A_T/T and prices nonlinear accrual payoffs.

Simplification relative to the paper (recorded in results/RESULTS.md): the
paper inverts a dedicated rectangle contour (Section 8.3) for complex u and
performs endpoint-singularity removal (Section 8.4) to accelerate the cosine
expansion's convergence. This implementation reuses the already-validated
optimized Talbot contour (``src.laplace_method.talbot_nodes``) for every u,
and uses the plain cosine expansion without endpoint-singularity removal,
at the cost of needing more cosine modes for the same accuracy.
"""

import numpy as np

from src.spectral_core import LayeredModel
from src.laplace_method import (
    talbot_nodes, _gauss_legendre_integral, laplace_density,
    solve_tridiagonal_batch, interface_matrix_batch, source_layer_terms,
)


def corridor_layer_mask(model: LayeredModel, corridor: tuple[float, float]) -> np.ndarray:
    """Boolean mask, one entry per layer, True where the layer lies in C = (c_lo, c_hi)."""
    c_lo, c_hi = corridor
    mid = 0.5 * (model.Y[:-1] + model.Y[1:])
    return (mid >= c_lo) & (mid <= c_hi)


def integrated_density(model: LayeredModel, nu, x0: float, n_quad: int = 32) -> complex:
    """hat_Psi(lambda; u) = integral_{y_0}^{y_N} Psi(lambda, q; x, x0) dx, Eq. (52):
    the Laplace-transformed density (for the given, possibly layer-dependent, nu)
    integrated over the whole price domain, evaluated layer by layer."""
    total = 0.0 + 0.0j
    for m in range(model.N):
        y_lo, y_hi = model.y[m], model.y[m + 1]

        def integrand(x, m=m):
            return laplace_density(model, nu, x0, x)

        total += _gauss_legendre_integral(integrand, y_lo, y_hi, n_quad)
    return total


def integrated_density_batch(model: LayeredModel, nu_batch: np.ndarray, x0: float,
                              n_quad: int = 32) -> np.ndarray:
    """Batched counterpart of ``integrated_density``: nu_batch has shape (B, N),
    one set of per-layer nu_i values per contour node z_k (Eq. 52). The
    tridiagonal system is solved once for all B nodes simultaneously
    (``solve_tridiagonal_batch``) and every layer's Gauss-Legendre quadrature
    is evaluated for the whole batch at once, instead of solving the system
    and integrating node by node."""
    B, N = nu_batch.shape
    diag, off = interface_matrix_batch(model, nu_batch)
    j, gamma1, gamma2 = source_layer_terms(model, x0)
    rhs = np.zeros((B, N - 1), dtype=complex)
    nu_j = nu_batch[:, j]
    sinh_arg_j = np.sinh(model.omega[j] * nu_j)
    if j - 1 >= 0:
        rhs[:, j - 1] += np.sinh(model.omega[j] * nu_j * gamma1) / sinh_arg_j
    if j <= N - 2:
        rhs[:, j] += np.sinh(model.omega[j] * nu_j * gamma2) / sinh_arg_j
    g = solve_tridiagonal_batch(diag, off, rhs) if N > 1 else np.zeros((B, 0), dtype=complex)
    g_full = np.concatenate([np.zeros((B, 1), dtype=complex), g, np.zeros((B, 1), dtype=complex)], axis=1)

    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    total = np.zeros(B, dtype=complex)
    x0r = np.real(x0)
    for m in range(N):
        y_lo, y_hi = model.y[m], model.y[m + 1]
        xq = 0.5 * (y_hi - y_lo) * nodes + 0.5 * (y_hi + y_lo)
        wq = 0.5 * (y_hi - y_lo) * weights
        nu_m = nu_batch[:, m]
        k_m = nu_m / model.sigma[m]
        ell_m = model.ell[m]
        s_left = xq - y_lo
        s_right = y_hi - xq
        denom = np.sinh(k_m * ell_m)
        contrib = (g_full[:, m][:, None] * np.sinh(k_m[:, None] * s_right[None, :])
                   + g_full[:, m + 1][:, None] * np.sinh(k_m[:, None] * s_left[None, :])) / denom[:, None]
        if m == j:
            lo = np.minimum(xq, x0r)
            hi = np.maximum(xq, x0r)
            Hj = (np.sinh(k_m[:, None] * (lo[None, :] - y_lo)) * np.sinh(k_m[:, None] * (y_hi - hi[None, :]))
                  / (k_m[:, None] * model.sigma[m] ** 2 * denom[:, None]))
            contrib = contrib + Hj
        total += contrib @ wq
    return total


def rectangle_contour_nodes(u: float, T: float, n_gl: int = 16):
    """Open rectangle contour of Section 8.3 for Fourier-in-accrual Laplace inversion.

    With u >= 0 the Bromwich line is deformed, without crossing any singularity
    (Lemma 3, which places every singularity in the half-strip Re(lambda) <=
    -mu_1, 0 <= Im(lambda) <= u), into the three-edge path: the bottom edge
    Im(lambda) = -delta from Re(lambda) = -infinity to cbar; the right edge
    Re(lambda) = cbar from Im(lambda) = -delta to u + delta; the top edge
    Im(lambda) = u + delta back to -infinity. Here cbar = 1/T, delta = 2/T, the
    horizontal tails are truncated at Re(lambda) = cbar - 40/T, and 16-node
    Gauss-Legendre panels of width min(pi/T, 1) (right edge) and min(1/T, 1)
    (horizontal edges) discretize the three edges. Returns nodes z and weights
    w with f(T) = sum_k w_k * F(z_k), i.e. w_k already folds in e^{lambda T}/(2 pi i).
    """
    cbar = 1.0 / T
    delta = 2.0 / T
    re_lo = cbar - 40.0 / T
    nodes_g, weights_g = np.polynomial.legendre.leggauss(n_gl)

    def panel_nodes(a: float, b: float, panel_width: float):
        n_panels = max(1, int(np.ceil((b - a) / panel_width)))
        edges = np.linspace(a, b, n_panels + 1)
        t_all, w_all = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            t_all.append(0.5 * (hi - lo) * nodes_g + 0.5 * (hi + lo))
            w_all.append(0.5 * (hi - lo) * weights_g)
        return np.concatenate(t_all), np.concatenate(w_all)

    # Bottom edge: lambda = t - i*delta, t: re_lo -> cbar, dlambda/dt = 1.
    t_b, w_b = panel_nodes(re_lo, cbar, min(1.0 / T, 1.0))
    z_b = t_b - 1j * delta
    w_b = w_b * 1.0

    # Right edge: lambda = cbar + i*t, t: -delta -> u+delta, dlambda/dt = i.
    t_r, w_r = panel_nodes(-delta, u + delta, min(np.pi / T, 1.0))
    z_r = cbar + 1j * t_r
    w_r = w_r * 1j

    # Top edge: lambda = t + i*(u+delta), t: cbar -> re_lo (reversed), dlambda/dt = 1.
    t_t, w_t = panel_nodes(re_lo, cbar, min(1.0 / T, 1.0))
    z_t = t_t + 1j * (u + delta)
    w_t = -w_t * 1.0  # traversed from cbar down to re_lo

    z = np.concatenate([z_b, z_r, z_t])
    dlambda_w = np.concatenate([w_b, w_r, w_t])
    w = dlambda_w * np.exp(z * T) / (2j * np.pi)
    return z, w


def accrual_characteristic(model: LayeredModel, corridor_mask: np.ndarray, T: float,
                            u: float, x0: float, beta_scale: float = 1.0,
                            n_talbot: int = 48, n_quad: int = 32) -> complex:
    """Xi(u) = E[e^{iu A_T}; T < tau], Eq. (51), via the rectangle contour of
    Section 8.3 (``rectangle_contour_nodes``), required whenever the killing
    rate is complex (u != 0); for u = 0 the standard Talbot rule used elsewhere
    in this project (``src.laplace_method.talbot_nodes``) would also apply,
    since the singularities then sit on the negative real axis, but the
    rectangle contour is used uniformly here for a single code path.

    ``beta_scale`` multiplies the corridor killing rate q_i = beta_scale * 1_{i in
    I_C}; the Fourier-in-accrual characteristic function of Proposition 14 uses
    beta_scale = -i*u.
    """
    z, w = rectangle_contour_nodes(u, T)
    q = np.where(corridor_mask, -1j * u * beta_scale, 0.0)
    nu_batch = np.sqrt(z[:, None] + q[None, :])
    out = integrated_density_batch(model, nu_batch, x0, n_quad=n_quad)
    return np.sum(w * out)


def characteristic_array(model: LayeredModel, corridor_mask: np.ndarray, T: float,
                          x0: float, K1: int, n_talbot: int = 48,
                          n_quad: int = 32) -> np.ndarray:
    """Xi(k*pi/T) for k = 0,...,K1, Eq. (51), computed once so that every cosine
    coefficient G_k (one per payoff) and every truncation K1' <= K1 can reuse
    the same, expensive, characteristic-function evaluations (Eq. 53)."""
    Xi = np.empty(K1 + 1, dtype=complex)
    for k in range(K1 + 1):
        u_k = k * np.pi / T
        Xi[k] = accrual_characteristic(model, corridor_mask, T, u_k, x0,
                                        n_talbot=n_talbot, n_quad=n_quad)
    return Xi


def price_from_characteristic(Xi: np.ndarray, G: np.ndarray) -> float:
    """E[g_p(alpha); T<tau] = (1/2)*G_0*Xi(0) + sum_{k=1}^{K1} G_k*Re(Xi(k*pi/T)),
    Eq. (53), from a precomputed characteristic array and a payoff's cosine
    coefficients G (both truncated consistently at the same K1)."""
    K1 = len(G) - 1
    return 0.5 * G[0] * np.real(Xi[0]) + float(np.dot(G[1:K1 + 1], np.real(Xi[1:K1 + 1])))


def cosine_modes_cdf(a_target: float, K1: int) -> np.ndarray:
    """G_k for g_p(a) = 1_{[0, a_target]} (the accrual CDF at level a_target),
    k = 0,...,K1: G_0 = 2*a_target, G_k = 2*sin(k*pi*a_target)/(k*pi), k >= 1."""
    k = np.arange(1, K1 + 1)
    G = np.empty(K1 + 1)
    G[0] = 2.0 * a_target
    G[1:] = 2.0 * np.sin(k * np.pi * a_target) / (k * np.pi)
    return G


def cosine_modes_accrual_call(kappa: float, K1: int) -> np.ndarray:
    """G_k for the accrual Call payoff g_p(a) = (a - kappa)^+, k = 0,...,K1:
    G_0 = (1 - kappa)^2 (plain integral of the payoff over [0,1]),
    G_k = 2*(cos(k*pi) - cos(k*pi*kappa)) / (k*pi)^2, k >= 1."""
    k = np.arange(1, K1 + 1)
    G = np.empty(K1 + 1)
    G[0] = (1.0 - kappa) ** 2
    G[1:] = 2.0 * (np.cos(k * np.pi) - np.cos(k * np.pi * kappa)) / (k * np.pi) ** 2
    return G


def price_accrual_payoff(model: LayeredModel, corridor_mask: np.ndarray, T: float, x0: float,
                          G: np.ndarray, n_talbot: int = 48, n_quad: int = 32) -> float:
    """E[g_p(alpha); T < tau] = (1/2)*G_0*Xi(0) + sum_{k=1}^{K1} G_k*Re(Xi(k*pi/T)), Eq. (53)."""
    K1 = len(G) - 1
    total = 0.5 * G[0] * np.real(
        accrual_characteristic(model, corridor_mask, T, 0.0, x0, n_talbot=n_talbot, n_quad=n_quad)
    )
    for k in range(1, K1 + 1):
        u_k = k * np.pi / T
        xi = accrual_characteristic(model, corridor_mask, T, u_k, x0, n_talbot=n_talbot, n_quad=n_quad)
        total += G[k] * np.real(xi)
    return total


def accrual_cdf(model: LayeredModel, corridor_mask: np.ndarray, T: float, x0: float,
                 a_target: float, K1: int, n_talbot: int = 48, n_quad: int = 32) -> float:
    """P(alpha <= a_target, T < tau) via the plain cosine expansion, Eq. (53)."""
    G = cosine_modes_cdf(a_target, K1)
    return price_accrual_payoff(model, corridor_mask, T, x0, G, n_talbot=n_talbot, n_quad=n_quad)


def accrual_call_price(model: LayeredModel, corridor_mask: np.ndarray, T: float, x0: float,
                        kappa: float, K1: int, n_talbot: int = 48, n_quad: int = 32) -> float:
    """E[(alpha - kappa)^+; T < tau] via the plain cosine expansion, Eq. (53)."""
    G = cosine_modes_accrual_call(kappa, K1)
    return price_accrual_payoff(model, corridor_mask, T, x0, G, n_talbot=n_talbot, n_quad=n_quad)


def lamperti_two_layer_cdf(a: np.ndarray, varsigma_minus: float, varsigma_plus: float) -> np.ndarray:
    """Lamperti's closed-form law for the fraction of time an oscillating Brownian
    motion (started at the interface) spends on the varsigma_plus side, used here
    as the independent reference for the two-layer configuration (Section 11.3):

        P(alpha <= a) = (2/pi) * arctan[(varsigma_plus/varsigma_minus) * sqrt(a/(1-a))].
    """
    a = np.clip(np.asarray(a, dtype=float), 1e-300, 1 - 1e-15)
    ratio = varsigma_plus / varsigma_minus
    return (2.0 / np.pi) * np.arctan(ratio * np.sqrt(a / (1.0 - a)))
