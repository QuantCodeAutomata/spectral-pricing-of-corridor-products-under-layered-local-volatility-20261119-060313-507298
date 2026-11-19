"""Crank-Nicolson finite-difference reference schemes (Section 11, page 28-30).

The driftless layered local-volatility model dS_t = varsigma(S_t) dW_t is
defined (Section 2, Definition 1) through the scale map

    x = Phi(S) = sqrt(2) * integral_{Y_0}^{S} du / varsigma(u)^2,

under which the killed generator becomes the divergence-form operator

    dV/dt + d/dx( p(x) dV/dx ) - r V = 0,   p(x) = 1 / varsigma(x)^2 = sigma(x)^2,

on the bounded scale-space domain [y_0, y_N] = [Phi(B_-), Phi(B_+)], with
absorbing (Dirichlet zero) boundaries. This is exactly the equation whose
transmission condition across a layer interface -- continuity of V and of the
flux p(x) dV/dx = sigma(x)^2 dV/dx -- reproduces the interface matrix of Eq.
(35): assembling the two one-sided Dirichlet-to-Neumann maps of
-u''(x) = mu * varsigma_m^2 * u(x) on adjacent layers through this flux gives
exactly K_ii = xi*(sigma_i*cot(omega_i*xi) + sigma_{i+1}*cot(omega_{i+1}*xi)).
The two FD schemes below therefore discretize in the scale coordinate x:

* "fitted": a node on every interface y_i = Phi(Y_i), uniform sub-intervals
  within each layer. Because every interface is exactly a grid node, the
  flux p(x) dV/dx is evaluated separately on each half-cell with its own
  layer's p_m = sigma_m^2 and matched at the node (finite-volume treatment of
  a discontinuous coefficient, Section 11) -- a cell-width-weighted harmonic
  averaging of p (equivalently, of varsigma^{-2}) across the interface.
* "naive": a uniform grid over the price domain [B_-, B_+] = [Y_0, Y_N]
  (Section 11, implementation_notes: "construct a uniform spatial grid over
  [B_-, B_+] with n intervals ... sample varsigma at the grid nodes"),
  discretizing the original driftless local-volatility PDE
  dV/dt + (1/2) varsigma(S)^2 V_SS - r V = 0 directly in S with a plain
  constant-step central difference, sampling varsigma(S) at each node's own
  layer and ignoring the exact interface location whenever a node does not
  coincide with a breakpoint.

Both march with Crank-Nicolson, preceded by ``FD_RANNACHER_HALF_STEPS`` half
steps of implicit Euler (Rannacher smoothing) to damp the initial payoff
kink, as specified in ``implementation_notes``. Prices and Greeks are read at
the physical spot S_* via its image x_0 = Phi(S_*) by linear interpolation of
the grid solution.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from src.spectral_core import LayeredModel


def _layer_of(breakpoints: np.ndarray, points: np.ndarray) -> np.ndarray:
    """0-based layer index m such that breakpoints[m] <= v <= breakpoints[m+1]."""
    idx = np.searchsorted(breakpoints, points, side="right") - 1
    return np.clip(idx, 0, len(breakpoints) - 2)


def _x_to_price(model: LayeredModel, x: np.ndarray) -> np.ndarray:
    """Phi^{-1}(x): the affine inverse scale map on each layer (Eq. 6-7)."""
    m = _layer_of(model.y, x)
    varsigma = np.real(model.varsigma)[m]
    return model.Y[m] + varsigma ** 2 * (x - model.y[m]) / np.sqrt(2.0)


def fitted_grid(model: LayeredModel, n_total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale-space grid with a node on every interface y_i and uniform
    sub-intervals per layer.

    ``n_total`` sub-intervals are distributed across layers in proportion to
    the scaled layer width ``ell_i`` (at least one per layer). Returns the
    scale coordinates ``x``, the price coordinates ``S = Phi^{-1}(x)`` (for
    payoff evaluation), and the divergence-form coefficient
    ``p_cell = sigma_m^2`` on each cell ``[x_i, x_{i+1})`` (Section 2;
    matches the interface matrix of Eq. 35).
    """
    ell = np.real(model.ell)
    total_ell = ell.sum()
    counts = np.maximum(1, np.round(n_total * ell / total_ell).astype(int))
    nodes = [np.array([model.y[0]])]
    p_half = []  # p = sigma_m^2 on the sub-interval to the right of each node
    for m in range(model.N):
        sub = np.linspace(model.y[m], model.y[m + 1], counts[m] + 1)[1:]
        nodes.append(sub)
        p_half.append(np.full(counts[m], np.real(model.sigma[m]) ** 2))
    x = np.concatenate(nodes)
    p_cell = np.concatenate(p_half)
    S = _x_to_price(model, x)
    return x, S, p_cell


def naive_grid(model: LayeredModel, n_total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform grid in the physical price domain [Y_0, Y_N] = [B_-, B_+]
    (Section 11, implementation_notes), varsigma sampled at each node's own
    layer. Returns ``S`` (the grid, used both as PDE coordinate and payoff
    coordinate here), ``S`` again, and ``varsigma_node``."""
    S = np.linspace(model.Y[0], model.Y[-1], n_total + 1)
    layer_at_node = _layer_of(model.Y, S)
    varsigma_node = np.real(model.varsigma)[layer_at_node]
    return S, S, varsigma_node


def _build_generator_fitted(x: np.ndarray, p_cell: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tridiagonal generator L (interior nodes only) for the fitted scheme,
    in the scale coordinate x where the operator is in divergence form
    d/dx(p dV/dx) with p = sigma^2 (Eq. 6-7; the flux p dV/dx is the quantity
    continuous across a layer interface, Eq. 35).

    Flux-conservative finite volume: flux across cell [x_i, x_{i+1}) is
    p_cell[i] * (V_{i+1}-V_i)/h_i; the control volume at interior node i has
    width (h_{i-1}+h_i)/2, so
        (L V)_i = (2/(h_{i-1}+h_i)) * [p_cell[i]*(V_{i+1}-V_i)/h_i
                                       - p_cell[i-1]*(V_i-V_{i-1})/h_{i-1}] - r*V_i.
    Returns (sub, diag, sup) diagonals of L restricted to interior nodes
    1,...,n-2 (Dirichlet V=0 enforced at the two boundary nodes).
    """
    h = np.diff(x)
    n = len(x)
    sub = np.zeros(n - 2)
    diag = np.zeros(n - 2)
    sup = np.zeros(n - 2)
    for i in range(1, n - 1):
        j = i - 1
        h_lo, h_hi = h[i - 1], h[i]
        cv = 0.5 * (h_lo + h_hi)
        a_lo = p_cell[i - 1] / (h_lo * cv)
        a_hi = p_cell[i] / (h_hi * cv)
        sub[j] = a_lo
        diag[j] = -(a_lo + a_hi) - r
        sup[j] = a_hi
    return sub, diag, sup


def _build_generator_naive(S: np.ndarray, varsigma_node: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain constant-step central difference of the original PDE
    dV/dt + (1/2) varsigma(S)^2 V_SS - r V = 0 directly in the physical price
    coordinate S, with varsigma sampled at each node's own layer
    (non-conservative: ignores the true interface location between nodes).
    The node's own coefficient p_i = (1/2) varsigma_i^2 is substituted
    directly into the constant-coefficient three-point Laplacian
    p_i*(V_{i+1}-2V_i+V_{i-1})/h^2, rather than the flux-conservative
    averaging of the fitted scheme; this is what "sampling varsigma at the
    nodes" means operationally and is what the paper contrasts with the
    fitted scheme's harmonic-mean interface treatment (Section 11)."""
    p = 0.5 * varsigma_node[1:-1] ** 2
    h = S[1] - S[0]
    sub = p / h**2
    sup = p / h**2
    diag = -2.0 * p / h**2 - r
    return sub, diag, sup


def _cn_step(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, V: np.ndarray,
             dt: float, theta: float) -> np.ndarray:
    """One step of theta-scheme backward in time: (I - theta dt L) V^new = (I + (1-theta) dt L) V.

    ``sub[j]``, ``diag[j]``, ``sup[j]`` are the coefficients of V[j-1], V[j],
    V[j+1] in row j of the generator L (``sub[0]`` and ``sup[-1]`` multiply
    the Dirichlet boundary value 0 and are unused).
    """
    n = len(diag)
    rhs = diag * V
    rhs[1:] += sub[1:] * V[:-1]
    rhs[:-1] += sup[:-1] * V[1:]
    rhs = V + (1 - theta) * dt * rhs

    ab = np.zeros((3, n))
    ab[0, 1:] = -theta * dt * sup[:-1]
    ab[1, :] = 1.0 - theta * dt * diag
    ab[2, :-1] = -theta * dt * sub[1:]
    return solve_banded((1, 1), ab, rhs)


def crank_nicolson_price(model: LayeredModel, S_star: float, T: float, r: float,
                          payoff: str, K: float | None, n_total: int, n_steps: int,
                          scheme: str = "fitted", n_rannacher: int = 4) -> float:
    """Price a barrier/no-touch Call/Put by backward time-stepping from T to 0.

    ``scheme`` is "fitted" (interface-fitted grid, Eq. implied by Section 2's
    divergence form) or "naive" (uniform grid, Section 11 benchmark).
    """
    from src.spectral_core import scale_transform_point

    if scheme == "fitted":
        x, S, p_cell = fitted_grid(model, n_total)
        sub, diag, sup = _build_generator_fitted(x, p_cell, r)
        pos, pos_star = x, float(np.real(scale_transform_point(model, S_star)))
    elif scheme == "naive":
        x, S, varsigma_node = naive_grid(model, n_total)
        sub, diag, sup = _build_generator_naive(x, varsigma_node, r)
        pos, pos_star = x, S_star  # naive solves directly in price space S
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    V = _terminal_payoff(S, payoff, K)
    V_int = V[1:-1].copy()
    dt = T / n_steps
    n_rannacher = min(n_rannacher, n_steps)
    for _ in range(n_rannacher):
        V_int = _cn_step(sub, diag, sup, V_int, dt, theta=1.0)
    for _ in range(n_steps - n_rannacher):
        V_int = _cn_step(sub, diag, sup, V_int, dt, theta=0.5)
    V_full = np.concatenate(([0.0], V_int, [0.0]))
    return float(np.interp(pos_star, pos, V_full))


def _terminal_payoff(S: np.ndarray, payoff: str, K: float | None) -> np.ndarray:
    if payoff == "notouch":
        return np.ones_like(S)
    if payoff == "call":
        return np.maximum(S - K, 0.0)
    if payoff == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"unknown payoff {payoff!r}")
