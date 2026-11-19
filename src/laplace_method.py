"""Laplace-transform pricing via Talbot contour inversion (Section 4, Section 8.1).

The Laplace-transformed interface densities solve the symmetric tridiagonal
system of Eq. (27). The coefficients mirror the flux matrix of the general-SLW
eigenproblem (``src.general_slw.flux_matrix``) with the oscillatory kernels
cot/sin replaced by the hyperbolic coth/sinh kernels appropriate to a resolvent
equation, since both arise from the same flux-continuity construction applied
to, respectively, phi'' = -(mu/sigma^2) phi and u'' = (lambda/sigma^2) u.

The point source at x0 inside layer j contributes the two right-hand-side
terms of Eq. (27) (Proposition 3) at the interfaces bounding layer j, and the
layer solution on the source layer itself carries, in addition to the
interface-density interpolation shared with every other layer, the standard
zero-boundary Green's-function kernel H_j for a unit point source (used here
to reconstruct the full Laplace-transformed transition density of the killed
diffusion, and then to price by integrating it against the payoff).
"""

import numpy as np
from scipy.linalg import solve_banded

from src.spectral_core import LayeredModel


def _coth(z: np.ndarray) -> np.ndarray:
    """coth(z) = (1+e^{-2z})/(1-e^{-2z}), stable for Re(z) >= 0 via expm1."""
    e = np.expm1(-2.0 * z)
    return -(2.0 + e) / e


def _csch(z: np.ndarray) -> np.ndarray:
    """1/sinh(z) = -2*e^{-z}/(1-e^{-2z}) = 2*e^{-z}/(-expm1(-2z))."""
    return -2.0 * np.exp(-z) / np.expm1(-2.0 * z)


def interface_matrix(model: LayeredModel, nu: np.ndarray) -> np.ndarray:
    """Tridiagonal K(lambda), Eq. (27): layer-independent nu (homogeneous killing).

    ``nu`` may be a scalar (shared across all layers) or an array nu_i of the
    same convention as ``model.sigma`` (layer-dependent killing, Section 8.2).
    """
    N = model.N
    sigma = model.sigma
    omega = model.omega
    nu = np.broadcast_to(np.asarray(nu, dtype=complex), (N,)).copy()
    arg = omega * nu
    coth = _coth(arg)
    csch = _csch(arg)
    diag = np.zeros(N - 1, dtype=complex)
    off = np.zeros(max(N - 2, 0), dtype=complex)
    for i in range(N - 1):
        diag[i] = sigma[i] * nu[i] * coth[i] + sigma[i + 1] * nu[i + 1] * coth[i + 1]
    for i in range(N - 2):
        off[i] = -sigma[i + 1] * nu[i + 1] * csch[i + 1]
    return diag, off


def solve_tridiagonal(diag: np.ndarray, off: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a symmetric tridiagonal system via scipy banded solver."""
    n = len(diag)
    if n == 0:
        return np.zeros(0, dtype=rhs.dtype)
    if n == 1:
        return rhs / diag
    ab = np.zeros((3, n), dtype=complex)
    ab[0, 1:] = off
    ab[1, :] = diag
    ab[2, :-1] = off
    return solve_banded((1, 1), ab, rhs)


def solve_tridiagonal_batch(diag: np.ndarray, off: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Vectorized Thomas algorithm solving a batch of B independent symmetric
    tridiagonal systems simultaneously: one system per contour node z of a
    Laplace inversion, so that Eq. (27) is solved at every node of the
    rectangle contour (Section 8.3) with B-wide numpy operations rather than
    with a Python loop over nodes (the number of interior interfaces N-1 is
    small, so the loop below runs over N-1 steps, not over the batch).

    diag: (B, n) complex, off: (B, n-1) complex, rhs: (B, n) complex -> (B, n).
    """
    B, n = diag.shape
    if n == 0:
        return np.zeros((B, 0), dtype=complex)
    if n == 1:
        return rhs / diag
    c = np.zeros((B, n - 1), dtype=complex)
    d = np.zeros((B, n), dtype=complex)
    c[:, 0] = off[:, 0] / diag[:, 0]
    d[:, 0] = rhs[:, 0] / diag[:, 0]
    for i in range(1, n):
        m = diag[:, i] - off[:, i - 1] * c[:, i - 1]
        if i < n - 1:
            c[:, i] = off[:, i] / m
        d[:, i] = (rhs[:, i] - off[:, i - 1] * d[:, i - 1]) / m
    x = np.zeros((B, n), dtype=complex)
    x[:, -1] = d[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = d[:, i] - c[:, i] * x[:, i + 1]
    return x


def interface_matrix_batch(model: LayeredModel, nu_batch: np.ndarray):
    """Batched counterpart of ``interface_matrix``: nu_batch has shape (B, N),
    one (possibly layer-dependent) nu_i vector per contour node. Returns
    diag (B, N-1), off (B, N-2), Eq. (27) evaluated at every node at once."""
    sigma = model.sigma
    omega = model.omega
    arg = omega[None, :] * nu_batch
    coth = _coth(arg)
    csch = _csch(arg)
    diag = (sigma[:-1][None, :] * nu_batch[:, :-1] * coth[:, :-1]
            + sigma[1:][None, :] * nu_batch[:, 1:] * coth[:, 1:])
    N = model.N
    if N > 2:
        off = -sigma[1:-1][None, :] * nu_batch[:, 1:-1] * csch[:, 1:-1]
    else:
        off = np.zeros((nu_batch.shape[0], 0), dtype=complex)
    return diag, off


def source_layer_terms(model: LayeredModel, x0: float):
    """Layer index j (0-based) containing x0, plus gamma1=(y_j-x0)/ell_j and
    gamma2=(x0-y_{j-1})/ell_j (Proposition 3)."""
    j = model.layer_of(np.real(x0))
    gamma1 = (model.y[j + 1] - x0) / model.ell[j]
    gamma2 = (x0 - model.y[j]) / model.ell[j]
    return j, gamma1, gamma2


def laplace_density(model: LayeredModel, nu, x0: float, x: np.ndarray):
    """LU(lambda; x, x0): the Laplace-transformed transition density at x for a
    source at x0, reconstructed layer by layer from the interface densities g
    and the source-layer Green's-function kernel H_j (Section 4, Proposition 3).
    """
    N = model.N
    diag, off = interface_matrix(model, nu)
    j, gamma1, gamma2 = source_layer_terms(model, x0)
    rhs = np.zeros(N - 1, dtype=complex)
    if j - 1 >= 0:
        rhs[j - 1] += np.sinh(model.omega[j] * nu_at(model, nu, j) * gamma1) / np.sinh(
            model.omega[j] * nu_at(model, nu, j))
    if j <= N - 2:
        rhs[j] += np.sinh(model.omega[j] * nu_at(model, nu, j) * gamma2) / np.sinh(
            model.omega[j] * nu_at(model, nu, j))
    g = solve_tridiagonal(diag, off, rhs)
    g_full = np.concatenate(([0.0], g, [0.0]))  # g_0 = g_N = 0, absorbing boundaries

    x = np.atleast_1d(np.asarray(x, dtype=complex))
    out = np.zeros_like(x)
    for m in range(N):
        mask = (np.real(x) >= np.real(model.y[m]) - 1e-12) & (np.real(x) <= np.real(model.y[m + 1]) + 1e-12)
        if not np.any(mask):
            continue
        xv = x[mask]
        nu_m = nu_at(model, nu, m)
        k_m = nu_m / model.sigma[m]
        ell_m = model.ell[m]
        s_left = xv - model.y[m]
        s_right = model.y[m + 1] - xv
        denom = np.sinh(k_m * ell_m)
        interp = (g_full[m] * np.sinh(k_m * s_right) + g_full[m + 1] * np.sinh(k_m * s_left)) / denom
        if m == j:
            x0r = np.real(x0)
            lo = np.minimum(np.real(xv), x0r)
            hi = np.maximum(np.real(xv), x0r)
            Hj = (np.sinh(k_m * (lo - model.y[m])) * np.sinh(k_m * (model.y[m + 1] - hi))
                  / (k_m * model.sigma[m] ** 2 * denom))
            interp = interp + Hj
        out[mask] = interp
    return out


def nu_at(model: LayeredModel, nu, m: int):
    """nu value on layer m, whether ``nu`` is a scalar or a per-layer array."""
    arr = np.broadcast_to(np.asarray(nu, dtype=complex), (model.N,))
    return arr[m]


def talbot_nodes(n_nodes: int, T: float):
    """Optimized fixed Talbot contour (Weideman, SIAM J. Numer. Anal. 44 (2006)),
    as referenced in Section 11 for the Laplace inversion. Returns nodes
    lambda_k and quadrature weights w_k such that f(T) ~= sum_k w_k * F(lambda_k).
    """
    M = n_nodes
    k = np.arange(M)
    theta = -np.pi + (k + 0.5) * (2 * np.pi / M)
    cot = 1.0 / np.tan(0.6407 * theta)
    z = (M / T) * (0.5017 * theta * cot - 0.6122 + 0.2645j * theta)
    dz = (M / T) * (-0.5017 * 0.6407 * theta / np.sin(0.6407 * theta) ** 2
                     + 0.5017 * cot + 0.2645j)
    h = 2 * np.pi / M  # trapezoid step in theta
    w = (h / (2j * np.pi)) * np.exp(z * T) * dz
    return z, w


def _gauss_legendre_integral(func, a: float, b: float, n_points: int = 48):
    """Gauss-Legendre quadrature of ``func`` on [a, b] (exponential convergence
    for the analytic sinh/payoff integrands here, standing in for the paper's
    closed-form layer integrals)."""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    w = 0.5 * (b - a) * weights
    return np.sum(w * func(x))


def price_laplace(model: LayeredModel, x0: float, T: float, r: float, payoff: str,
                   K: float | None = None, n_talbot: int = 32, n_quad: int = 48) -> complex:
    """V = e^{-rT} sum_k w_k * integral P(Phi^{-1}(x)) LU(lambda_k, x) dx, Section 4/8.1."""
    from src.spectral_core import payoff_layer_pieces

    pieces = payoff_layer_pieces(model, payoff, K)
    z, w = talbot_nodes(n_talbot, T)
    total = np.zeros_like(z)
    for layer_idx in range(model.N):
        y_lo = model.y[layer_idx]
        for s_lo, s_hi, alpha, beta in pieces[layer_idx]:
            if s_hi <= s_lo:
                continue
            x_lo, x_hi = y_lo + s_lo, y_lo + s_hi
            for ik, lam in enumerate(z):
                nu_lam = np.sqrt(lam)  # nu_i = sqrt(lambda) (homogeneous killing, Section 4)
                def integrand(x, nu_lam=nu_lam, y_lo=y_lo, alpha=alpha, beta=beta):
                    density = laplace_density(model, nu_lam, x0, x)
                    return (alpha + beta * (x - y_lo)) * density
                total[ik] += _gauss_legendre_integral(integrand, x_lo, x_hi, n_quad)
    V = np.exp(-r * T) * np.sum(w * total)
    return V

