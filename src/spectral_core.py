"""Shared layered local-volatility model and eigen-expansion pricing kernel.

Implements the scale transformation (Definition 1, Eq. 6-7), the generic
eigen-expansion pricing formula (Eq. 33/38) and the Greeks (Eq. 56-58) that
both the equal-SLW (Section 5) and general-SLW (Section 6) spectral methods
feed into. Every eigenfunction, whichever family or method produced it, is
represented layer by layer as

    phi_n(x) = A[n, m] * cos(k[n, m] * s) + B[n, m] * sin(k[n, m] * s),
    s = x - y_{m-1},  k[n, m] = xi_n / sigma_m,

which is the common form of Eq. (32) and of Corollary 10. This lets payoff
integrals, derivatives and Greeks be computed once for every eigenbasis.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LayeredModel:
    """A layered local-volatility surface on one time bucket (Section 2)."""

    Y: np.ndarray  # price breakpoints Y_0,...,Y_N (N+1,)
    varsigma: np.ndarray  # layer volatilities varsigma_1,...,varsigma_N (N,)
    y: np.ndarray  # scale coordinates y_i = Phi(Y_i) (N+1,), y[0] = 0
    ell: np.ndarray  # scaled layer widths ell_i = y_i - y_{i-1} (N,)
    sigma: np.ndarray  # sigma_i = 1/varsigma_i (N,)
    omega: np.ndarray  # omega_i = ell_i / sigma_i = sqrt(2) L_i / varsigma_i (N,)

    @property
    def N(self) -> int:
        return len(self.varsigma)

    def layer_of(self, y_val: float) -> int:
        """0-based layer index m-1 such that y_{m-1} <= y_val <= y_m."""
        idx = int(np.searchsorted(self.y, y_val, side="right") - 1)
        return min(max(idx, 0), self.N - 1)

    def layer_of_price(self, S: float) -> int:
        """0-based layer index m such that Y_m <= S <= Y_{m+1}, in price units."""
        idx = int(np.searchsorted(self.Y, S, side="right") - 1)
        return min(max(idx, 0), self.N - 1)


def scale_transform(Y: np.ndarray, varsigma: np.ndarray) -> np.ndarray:
    """Phi(Y_i) = sqrt(2) * integral_{Y_0}^{Y_i} du/varsigma^2(u), Eq. (6).

    Since varsigma is piecewise constant, the integral is a cumulative sum
    of sqrt(2) * L_i / varsigma_i^2 over the layers.
    """
    L = np.diff(Y)
    increments = np.sqrt(2.0) * L / varsigma**2
    y = np.concatenate(([0.0], np.cumsum(increments)))
    return y


def scale_transform_point(model: LayeredModel, S: float) -> complex | float:
    """Phi(S) for a single point S (not necessarily a breakpoint), Eq. (6).

    Used for layer vegas: the image x0 = Phi(S_*) of the fixed physical spot
    depends on every layer volatility between Y_0 and S_*, so perturbing a
    layer below (or containing) S_* moves x0 itself and must be reflected
    with the same complex arithmetic as the rest of the complex-step vega
    (Eq. 59), not just the eigenfunctions and eigenvalues.
    """
    m = model.layer_of_price(S)
    Y_lo = model.Y[m]
    L_partial = S - Y_lo
    return model.y[m] + np.sqrt(2.0) * L_partial / model.varsigma[m] ** 2


def build_layered_model(Y: np.ndarray, varsigma: np.ndarray) -> LayeredModel:
    """Construct a LayeredModel from breakpoints and layer volatilities, Eq. (7)."""
    Y = np.asarray(Y, dtype=float)
    varsigma = np.asarray(varsigma, dtype=complex if np.iscomplexobj(varsigma) else float)
    assert len(Y) == len(varsigma) + 1, "Y must have one more entry than varsigma"
    assert np.all(np.diff(Y).real > 0), "breakpoints Y must be strictly increasing"
    y = scale_transform(Y, varsigma)
    ell = np.diff(y)
    sigma = 1.0 / varsigma
    omega = ell / sigma
    return LayeredModel(Y=Y, varsigma=varsigma, y=y, ell=ell, sigma=sigma, omega=omega)


@dataclass
class EigenBasis:
    """Layer-wise representation of an eigen-expansion of -A, Theorem 7 / Corollary 10."""

    mu: np.ndarray  # eigenvalues mu_n = xi_n^2 (n_modes,)
    A: np.ndarray  # cosine amplitudes A[n, m] (n_modes, N)
    B: np.ndarray  # sine amplitudes B[n, m] (n_modes, N)
    k: np.ndarray  # wave numbers k[n, m] = xi_n / sigma_m (n_modes, N)
    norm2: np.ndarray  # ||phi_n||^2 (n_modes,)

    @property
    def n_modes(self) -> int:
        return len(self.mu)


def eval_phi(basis: EigenBasis, model: LayeredModel, x: float, order: int = 0) -> np.ndarray:
    """phi_n(x) (order=0), phi_n'(x) (order=1) or phi_n''(x) (order=2) for every n."""
    m = model.layer_of(np.real(x) if np.iscomplexobj(x) else x)
    s = x - model.y[m]
    k = basis.k[:, m]
    A, B = basis.A[:, m], basis.B[:, m]
    cos_ks, sin_ks = np.cos(k * s), np.sin(k * s)
    if order == 0:
        return A * cos_ks + B * sin_ks
    if order == 1:
        return k * (-A * sin_ks + B * cos_ks)
    if order == 2:
        return -(k**2) * (A * cos_ks + B * sin_ks)
    raise ValueError("order must be 0, 1 or 2")


def _antiderivative_poly_trig(alpha: float, beta: float, k: np.ndarray, A: np.ndarray,
                               B: np.ndarray, s: float) -> np.ndarray:
    """F(s) with F' = (alpha + beta*s) * (A*cos(ks) + B*sin(ks)), closed form.

    Uses int cos(ks) ds = sin(ks)/k, int s*cos(ks) ds = cos(ks)/k^2 + s*sin(ks)/k,
    int sin(ks) ds = -cos(ks)/k, int s*sin(ks) ds = sin(ks)/k^2 - s*cos(ks)/k.
    """
    cos_ks, sin_ks = np.cos(k * s), np.sin(k * s)
    cos_term = alpha * sin_ks / k + beta * (cos_ks / k**2 + s * sin_ks / k)
    sin_term = -alpha * cos_ks / k + beta * (sin_ks / k**2 - s * cos_ks / k)
    return A * cos_term + B * sin_term


def integral_poly_trig(alpha: float, beta: float, k: np.ndarray, A: np.ndarray, B: np.ndarray,
                        s_lo: float, s_hi: float) -> np.ndarray:
    """integral_{s_lo}^{s_hi} (alpha + beta*s) * (A*cos(ks) + B*sin(ks)) ds."""
    return (_antiderivative_poly_trig(alpha, beta, k, A, B, s_hi)
            - _antiderivative_poly_trig(alpha, beta, k, A, B, s_lo))


def payoff_layer_pieces(model: LayeredModel, payoff: str, K: float | None = None):
    """Piecewise-linear payoff P(Phi^{-1}(x)) on each layer, in x-coordinates.

    Returns a list, one entry per layer m=0,...,N-1, of lists of
    (s_lo, s_hi, alpha, beta) pieces such that on s in [s_lo, s_hi],
    P(Phi^{-1}(y_m + s)) = alpha + beta * s. Phi^{-1}(x) = Y_{m-1} +
    varsigma_m^2 * (x - y_{m-1}) / sqrt(2) is affine on each layer (Eq. 38).
    """
    pieces = []
    for m in range(model.N):
        # dY/ds slope on layer m; kept complex (not np.real) so that the
        # complex-step vega of Eq. (59) picks up the payoff term's own
        # dependence on the perturbed layer volatility, not only the
        # eigenvalues' and eigenfunctions' dependence.
        c_m = model.varsigma[m] ** 2 / np.sqrt(2.0)
        Y_lo = model.Y[m]
        ell_m = model.ell[m]
        if payoff == "notouch":
            pieces.append([(0.0, ell_m, 1.0, 0.0)])
        elif payoff in ("call", "put"):
            sign = 1.0 if payoff == "call" else -1.0
            # price level at s: Y(s) = Y_lo + c_m * s; payoff intrinsic = sign*(Y(s)-K)
            s_strike = (K - Y_lo) / c_m if c_m != 0 else np.inf
            Y_hi = model.Y[m + 1]
            if payoff == "call":
                if K <= Y_lo:
                    pieces.append([(0.0, ell_m, Y_lo - K, c_m)])
                elif K >= Y_hi:
                    pieces.append([(0.0, ell_m, 0.0, 0.0)])
                else:
                    pieces.append([(0.0, s_strike, 0.0, 0.0),
                                    (s_strike, ell_m, Y_lo - K, c_m)])
            else:
                if K >= Y_hi:
                    pieces.append([(0.0, ell_m, K - Y_lo, -c_m)])
                elif K <= Y_lo:
                    pieces.append([(0.0, ell_m, 0.0, 0.0)])
                else:
                    pieces.append([(0.0, s_strike, K - Y_lo, -c_m),
                                    (s_strike, ell_m, 0.0, 0.0)])
        else:
            raise ValueError(f"unknown payoff {payoff!r}")
    return pieces


def compute_cn(basis: EigenBasis, model: LayeredModel, payoff: str,
               K: float | None = None) -> np.ndarray:
    """c_n = integral_{y_0}^{y_N} P(Phi^{-1}(x)) phi_n(x) dx, Eq. (38)."""
    pieces = payoff_layer_pieces(model, payoff, K)
    cn = np.zeros(basis.n_modes, dtype=basis.A.dtype)
    for m in range(model.N):
        k_col, A_col, B_col = basis.k[:, m], basis.A[:, m], basis.B[:, m]
        for s_lo, s_hi, alpha, beta in pieces[m]:
            if s_hi <= s_lo:
                continue
            cn += integral_poly_trig(alpha, beta, k_col, A_col, B_col, s_lo, s_hi)
    return cn


def price_eigen_expansion(basis: EigenBasis, model: LayeredModel, x0: float, T: float,
                           r: float, cn: np.ndarray) -> float:
    """V = e^{-rT} sum_n e^{-mu_n T} phi_n(x0) c_n / ||phi_n||^2, Eq. (38)."""
    phi0 = eval_phi(basis, model, x0, order=0)
    terms = np.exp(-basis.mu * T) * phi0 * cn / basis.norm2
    return np.real(np.exp(-r * T) * np.sum(terms))


def price_eigen_expansion_complex(basis: EigenBasis, model: LayeredModel, x0,
                                   T: float, r: float, cn: np.ndarray):
    """Same as price_eigen_expansion but keeps the result complex (no np.real).

    Used by the complex-step layer-vega formula (Eq. 59), which recovers the
    derivative from the imaginary part of an infinitesimally perturbed price;
    truncating to the real part would destroy that signal.
    """
    phi0 = eval_phi(basis, model, x0, order=0)
    terms = np.exp(-basis.mu * T) * phi0 * cn / basis.norm2
    return np.exp(-r * T) * np.sum(terms)


def layer_phi_squared_integral(basis: EigenBasis, model: LayeredModel) -> np.ndarray:
    """integral_{y_{m-1}}^{y_m} phi_n(x)^2 dx for every mode n and layer m, Eq. (37)
    applied layer by layer (rather than summed), used for the Hellmann-Feynman
    layer-vega weights p_{n,i} = (this integral) / ||phi_n||^2 (Section 9.1)."""
    ell = np.real(model.ell)
    A, B, k = basis.A, basis.B, basis.k
    term = (A**2 + B**2) / 2.0 * ell[None, :]
    term = term + (A**2 - B**2) / (4.0 * k) * np.sin(2.0 * k * ell[None, :])
    term = term + (A * B) / (2.0 * k) * (1.0 - np.cos(2.0 * k * ell[None, :]))
    return np.real(term)


def greeks_eigen_expansion(basis: EigenBasis, model: LayeredModel, x0: float, T: float,
                            r: float, cn: np.ndarray, S_star: float):
    """Delta, Gamma, Theta from the eigen-series, Eq. (56).

    Delta = sqrt(2)/varsigma_j^2 * e^{-rT} sum_n e^{-mu_n T} phi_n'(x0) c_n / ||phi_n||^2
    Theta = r*V + e^{-rT} sum_n mu_n e^{-mu_n T} phi_n(x0) c_n / ||phi_n||^2
    Gamma = -2/varsigma_j^2 * e^{-rT} sum_n mu_n e^{-mu_n T} phi_n(x0) c_n / ||phi_n||^2
          = (1/2 varsigma_j^2)^{-1} * (r*V - Theta)   [cross-check identity]
    """
    j = model.layer_of(S_star)
    varsigma_j = np.real(model.varsigma[j])
    phi0 = eval_phi(basis, model, x0, order=0)
    dphi0 = eval_phi(basis, model, x0, order=1)
    weight = np.exp(-basis.mu * T) * cn / basis.norm2
    V = np.real(np.exp(-r * T) * np.sum(weight * phi0))
    Delta = np.real(np.sqrt(2.0) / varsigma_j**2 * np.exp(-r * T) * np.sum(weight * dphi0))
    Theta_series = np.real(np.exp(-r * T) * np.sum(basis.mu * weight * phi0))
    Theta = r * V + Theta_series
    Gamma = -2.0 / varsigma_j**2 * Theta_series
    return V, Delta, Gamma, Theta
