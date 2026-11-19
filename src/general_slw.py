"""Spectral inversion II: general scaled layer widths (Section 6).

Locates every eigenvalue xi_n of -A exactly via the counting formula
(Theorem 9, Eq. 36), a specialization of the Wittrick-Williams principle,
using bisection together with Sylvester's law of inertia applied to a
pivoted LDL^T factorization of the flux matrix K(xi) (Eq. 35). Eigenfunctions
are then propagated layer by layer with the transfer matrices T_m(xi)
(Eq. 34) starting from the pair (0, 1) at y_0 (Corollary 10), and norms use
Eq. (37).
"""

import numpy as np
from scipy.linalg import ldl

from src.spectral_core import EigenBasis, LayeredModel


def flux_matrix(model: LayeredModel, xi: float) -> np.ndarray:
    """K(xi), Eq. (35): K_ii = xi*(sigma_i cot(omega_i xi) + sigma_{i+1} cot(omega_{i+1} xi)),
    K_{i,i+1} = -xi*sigma_{i+1}/sin(omega_{i+1} xi)."""
    N = model.N
    sigma = np.real(model.sigma)
    omega = np.real(model.omega)
    cot = 1.0 / np.tan(omega * xi)
    K = np.zeros((N - 1, N - 1))
    for i in range(N - 1):  # i = 1..N-1 paper -> 0-indexed i
        K[i, i] = xi * (sigma[i] * cot[i] + sigma[i + 1] * cot[i + 1])
    for i in range(N - 2):
        K[i, i + 1] = K[i + 1, i] = -xi * sigma[i + 1] / np.sin(omega[i + 1] * xi)
    return K


def n_negative_eigenvalues(K: np.ndarray) -> int:
    """n_-(K) via Sylvester's law of inertia applied to a pivoted LDL^T factorization."""
    if K.shape[0] == 0:
        return 0
    _, D, _ = ldl(K)
    # D is block diagonal with 1x1 or 2x2 blocks; count negative eigenvalues block by block.
    n = D.shape[0]
    neg = 0
    i = 0
    while i < n:
        if i + 1 < n and D[i + 1, i] != 0.0:
            block = D[i:i + 2, i:i + 2]
            evals = np.linalg.eigvalsh(block)
            neg += int(np.sum(evals < 0))
            i += 2
        else:
            neg += int(D[i, i] < 0)
            i += 1
    return neg


def eigenvalue_count(model: LayeredModel, xi: float) -> int:
    """#{n : xi_n < xi} = sum_i floor(omega_i xi / pi) + n_-(K(xi)), Eq. (36)."""
    omega = np.real(model.omega)
    floor_term = int(np.sum(np.floor(omega * xi / np.pi)))
    # Avoid poles: nudge xi slightly if any omega_i*xi is too close to a multiple of pi.
    xi_eval = _avoid_poles(model, xi)
    K = flux_matrix(model, xi_eval)
    return floor_term + n_negative_eigenvalues(K)


def _avoid_poles(model: LayeredModel, xi: float, tol: float = 1e-9) -> float:
    omega = np.real(model.omega)
    residues = (omega * xi) % np.pi
    if np.any(np.minimum(residues, np.pi - residues) < tol):
        return xi * (1.0 + 10 * tol)
    return xi


def locate_eigenvalues(model: LayeredModel, n_modes: int, tol: float = 1e-13) -> np.ndarray:
    """Bisection on the counting formula Eq. (36) to isolate xi_1 < xi_2 < ... < xi_{n_modes}."""
    omega_sum = np.real(model.omega).sum()
    xi_hi_guess = np.pi * (n_modes + model.N) / max(np.real(model.omega).min(), 1e-12)
    xi_lo, xi_hi = 1e-8, xi_hi_guess
    while eigenvalue_count(model, xi_hi) < n_modes:
        xi_hi *= 2.0
    xis = np.empty(n_modes)
    for n in range(1, n_modes + 1):
        lo, hi = xi_lo, xi_hi
        # Narrow an initial bracket containing xi_n using the monotone count.
        while eigenvalue_count(model, lo) >= n:
            lo /= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if hi - lo < tol * max(1.0, mid):
                break
            if eigenvalue_count(model, mid) < n:
                lo = mid
            else:
                hi = mid
        xis[n - 1] = 0.5 * (lo + hi)
    return xis


def _transfer_matrix(sigma_m: float, omega_m: float, xi: float) -> np.ndarray:
    """T_m(xi), Eq. (34): maps the pair (u, sigma^2 u_x) across layer m."""
    c, s = np.cos(omega_m * xi), np.sin(omega_m * xi)
    return np.array([[c, s / (sigma_m * xi)],
                      [-sigma_m * xi * s, c]])


def propagate_eigenfunction(model: LayeredModel, xi: float):
    """Propagate (phi, sigma^2 phi_x) from (0, 1) at y_0 through every layer, Corollary 10.

    Returns per-layer (a_m, b_m, k_m) with phi(x) = a_m*cos(k_m*s) + b_m*sin(k_m*s),
    s = x - y_{m-1}, k_m = xi/sigma_m, where (a_m, sigma_m*xi*b_m) is the pair at y_{m-1}.
    """
    N = model.N
    complex_mode = np.iscomplexobj(model.sigma) or np.iscomplexobj(xi)
    dtype = complex if complex_mode else float
    sigma = model.sigma.astype(dtype) if complex_mode else np.real(model.sigma)
    omega = model.omega.astype(dtype) if complex_mode else np.real(model.omega)
    pair = np.array([0.0, 1.0], dtype=dtype)
    a = np.empty(N, dtype=dtype)
    b = np.empty(N, dtype=dtype)
    k = np.empty(N, dtype=dtype)
    for m in range(N):
        a[m] = pair[0]
        b[m] = pair[1] / (sigma[m] * xi)
        k[m] = xi / sigma[m]
        T = _transfer_matrix(sigma[m], omega[m], xi)
        pair = T @ pair
    return a, b, k, pair  # pair at y_N; pair[0] should be ~0 at a true eigenvalue


def eigenfunction_norm2(a: np.ndarray, b: np.ndarray, k: np.ndarray, ell: np.ndarray) -> float:
    """||phi_n||^2 via Eq. (37)."""
    term = (a**2 + b**2) / 2.0 * ell
    term = term + (a**2 - b**2) / (4.0 * k) * np.sin(2.0 * k * ell)
    term = term + (a * b) / (2.0 * k) * (1.0 - np.cos(2.0 * k * ell))
    total = np.sum(term)
    return total if np.iscomplexobj(term) else float(total)


def build_general_slw_basis(model: LayeredModel, n_modes: int, tol: float = 1e-13) -> EigenBasis:
    """Corollary 10: general eigen-expansion via counting-formula bisection."""
    xis = locate_eigenvalues(model, n_modes, tol=tol)
    ell = np.real(model.ell)
    A = np.empty((n_modes, model.N))
    B = np.empty((n_modes, model.N))
    K = np.empty((n_modes, model.N))
    norm2 = np.empty(n_modes)
    for n, xi in enumerate(xis):
        a, b, k, _ = propagate_eigenfunction(model, xi)
        A[n], B[n], K[n] = a, b, k
        norm2[n] = eigenfunction_norm2(a, b, k, ell)
    return EigenBasis(mu=xis**2, A=A, B=B, k=K, norm2=norm2)


def n_modes_weyl(model: LayeredModel, T: float, log_inv_eps: float = 32.0) -> int:
    """Weyl's-law mode-count estimate M ~= (sum_i omega_i/pi) * sqrt(log(1/eps)/T)."""
    omega_sum = np.real(model.omega).sum()
    return int(np.ceil((omega_sum / np.pi) * np.sqrt(log_inv_eps / T))) + 5
