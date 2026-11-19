"""Spectral inversion I: equal scaled layer widths (Section 5).

Builds the closed-form eigen-decomposition of Theorem 7 when
omega_1 = ... = omega_N =: omega (Eq. 29). The interface matrix B
(Eq. 30) is the transition matrix of the hitting-probability chain between
neighboring interfaces (Proposition 4); its symmetrized form
Pi^{1/2} B Pi^{-1/2} is diagonalized with scipy.linalg.eigh_tridiagonal,
which implements the symmetric tridiagonal eigensolver referenced in the
paper as LAPACK dstev (Section 11).
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal

from src.spectral_core import EigenBasis, LayeredModel


def check_equal_slw(model: LayeredModel, rtol: float = 1e-9) -> float:
    """Assert omega_i is constant across layers and return the common value."""
    omega = np.real(model.omega)
    omega_mean = omega.mean()
    assert np.allclose(omega, omega_mean, rtol=rtol), "layers do not have equal SLW"
    return omega_mean


def _hitting_probability_matrix(sigma: np.ndarray):
    """B_{i,i-1}=sigma_i/(sigma_i+sigma_{i+1}), B_{i,i+1}=sigma_{i+1}/(..), Eq. (30)."""
    N = len(sigma)
    denom = sigma[:-1] + sigma[1:]  # sigma_i + sigma_{i+1}, i=1..N-1 (0-indexed: 0..N-2)
    return denom


def build_equal_slw_basis(model: LayeredModel) -> EigenBasis:
    """Theorem 7: closed-form spectral data under equal SLW."""
    omega = check_equal_slw(model)
    sigma = np.real(model.sigma)
    N = model.N

    denom = _hitting_probability_matrix(sigma)  # length N-1, pi_i = sigma_i+sigma_{i+1}
    pi = denom  # Pi = diag(sigma_i + sigma_{i+1}), i=1..N-1

    # J = Pi^{1/2} B Pi^{-1/2} is symmetric tridiagonal with zero diagonal and
    # off-diagonal J_{i,i+1} = sigma_{i+1}/sqrt(pi_i * pi_{i+1}) (page 13).
    if N - 1 >= 2:
        off = sigma[1:-1] / np.sqrt(pi[:-1] * pi[1:])
    else:
        off = np.array([])
    diag = np.zeros(N - 1)
    zeta, W = eigh_tridiagonal(diag, off)  # zeta ascending; W columns are eigenvectors of J,
    # orthonormal: W^T W = I. Since J = Pi^{1/2} B Pi^{-1/2}, the eigenvectors of B are
    # v = Pi^{-1/2} w, which automatically satisfy v^T Pi v = w^T w = 1 (Theorem 7).
    V = W / np.sqrt(pi)[:, None]

    phi_angle = np.arccos(np.clip(zeta, -1.0, 1.0))  # varphi_i in (0, pi), Proposition 6

    # Assemble both families of eigenfunctions in layer-wise cos/sin form,
    # phi(x) = A[m]*cos(k*s) + B[m]*sin(k*s), s = x - y_{m-1}, k = xi/sigma_m.
    modes_mu, modes_A, modes_B, modes_k, modes_norm2 = [], [], [], [], []

    # --- Family (a): interface modes, varphi = varphi_i + 2*pi*k or 2*pi*(k+1)-varphi_i.
    max_k_interface = _n_interface_harmonics(model)
    v_ext = np.zeros((N - 1, N + 1))
    v_ext[:, 1:N] = V.T  # v_ext[i, m] = v_{i,m}, with v_{i,0}=v_{i,N}=0 already
    for i in range(N - 1):
        for branch, varphi in (("lo", phi_angle[i]), ("hi", 2 * np.pi - phi_angle[i])):
            for kk in range(max_k_interface + 1):
                varphi_n = varphi + 2 * np.pi * kk if branch == "lo" else 2 * np.pi * (kk + 1) - phi_angle[i]
                xi = varphi_n / omega
                mu = xi**2
                # On layer m: phi(x) = [v_{m-1} sin(varphi(y_m-x)/ell_m) + v_m sin(varphi(x-y_{m-1})/ell_m)]/sin(varphi)
                A_row = np.zeros(N)
                B_row = np.zeros(N)
                sin_phi = np.sin(varphi_n)
                for m in range(N):
                    ell_m = np.real(model.ell[m])
                    k_m = xi / np.real(model.sigma[m])
                    v_lo, v_hi = v_ext[i, m], v_ext[i, m + 1]
                    # sin(varphi*(y_m-x)/ell_m) = sin(varphi - k_m*s) = sin(varphi)cos(k_m s) - cos(varphi)sin(k_m s)
                    # sin(varphi*(x-y_{m-1})/ell_m) = sin(k_m*s)
                    coeff_cos = v_lo * sin_phi
                    coeff_sin = -v_lo * np.cos(varphi_n) + v_hi
                    A_row[m] = coeff_cos / sin_phi
                    B_row[m] = coeff_sin / sin_phi
                modes_mu.append(mu)
                modes_A.append(A_row)
                modes_B.append(B_row)
                modes_k.append(xi / sigma)
                modes_norm2.append(omega / 2.0)

    # --- Family (b): interface-blind modes, varphi = n*pi, n>=1.
    max_n_blind = _n_blind_harmonics(model)
    norm2_b = (omega / 2.0) * np.sum(1.0 / sigma)
    for n in range(1, max_n_blind + 1):
        xi = n * np.pi / omega
        mu = xi**2
        A_row = np.zeros(N)
        B_row = np.zeros(N)
        for m in range(N):
            sign = (-1.0) ** (n * m)  # (-1)^{n*(m)} using 0-indexed m = (m_paper-1)
            A_row[m] = 0.0
            B_row[m] = sign / np.real(model.sigma[m])
        modes_mu.append(mu)
        modes_A.append(A_row)
        modes_B.append(B_row)
        modes_k.append(xi / sigma)
        modes_norm2.append(norm2_b)

    mu = np.array(modes_mu)
    order = np.argsort(mu)
    basis = EigenBasis(
        mu=mu[order],
        A=np.array(modes_A)[order],
        B=np.array(modes_B)[order],
        k=np.array(modes_k)[order],
        norm2=np.array(modes_norm2)[order],
    )
    return basis


def _n_interface_harmonics(model: LayeredModel, eps: float = 1e-16, max_T: float = 1e-3) -> int:
    """How many harmonics k>=0 to keep per interface mode branch for convergence."""
    omega_tot = np.real(model.omega).sum()
    # Weyl's law based cutoff, generous bound shared with general_slw.n_modes_weyl.
    xi_max = (omega_tot / np.pi) * np.sqrt(90.0 / max_T) / max(1, model.N - 1)
    k = int(np.ceil(xi_max * np.real(model.omega).mean() / (2 * np.pi))) + 5
    return max(k, 5)


def _n_blind_harmonics(model: LayeredModel, max_T: float = 1e-3) -> int:
    omega_mean = np.real(model.omega).mean()
    n = int(np.ceil((omega_mean / np.pi) * np.sqrt(90.0 / max_T))) + 5
    return max(n, 5)
