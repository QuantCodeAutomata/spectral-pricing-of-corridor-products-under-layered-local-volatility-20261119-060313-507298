"""Experiment 1: single-bucket pricing under equal scaled layer widths (SLW).

Reproduces Section 5's closed-form spectral method (Theorem 7) and the three
validation checks of Section 11.2:
  (a) a single layer (N=1) reproduces the classical image formula for a
      homogeneous-volatility strip;
  (b) splitting a homogeneous strip into N = 1, 2, 5 equal layers reproduces
      the closed-form double no-touch price;
  (c) the equal-SLW configuration varsigma_i = EQUAL_SLW_SCALE * L_i agrees
      with the general-SLW method (Section 6) for double no-touch, Call and
      Put at T = 0.1, 1, 5.

Run: python -m exp.exp1
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from src.config import (
    BREAKPOINTS_GENERAL, VARSIGMA_GENERAL, SPOT, RATE, STRIKE, CORRIDOR,
    MATURITIES, EQUAL_SLW_SCALE, EQUAL_SLW_OMEGA_EXPECTED, LOG_INV_EPS_DEFAULT,
)
from src.spectral_core import (
    build_layered_model, scale_transform_point, compute_cn, price_eigen_expansion,
)
from src.equal_slw import build_equal_slw_basis, check_equal_slw
from src.general_slw import build_general_slw_basis, n_modes_weyl


def image_formula_double_notouch(S0: float, B_lo: float, B_hi: float, varsigma: float,
                                  r: float, T: float, n_terms: int = 4001) -> float:
    """Classical sine-series (method-of-images) formula for a double no-touch
    option on a homogeneous-volatility strip (B_lo, B_hi), used as the N=1
    analytic check of Section 11.2(a). The scaled Brownian motion x = (S-B_lo)/
    (B_hi-B_lo) solves the heat equation on (0,1) with absorbing boundaries, so

        P(T < tau) = (4/pi) * sum_{n odd >= 1} (1/n) sin(n*pi*x0) exp(-mu_n T),
        mu_n = (n*pi/width)^2 * varsigma^2 / 2,

    which converges geometrically in T and is independent of the paper's own
    eigen-expansion machinery.
    """
    width = B_hi - B_lo
    x0 = (S0 - B_lo) / width
    val = 0.0
    for n in range(1, n_terms + 1, 2):
        mu_n = (n * np.pi / width) ** 2 * (varsigma**2 / 2.0)
        val += (4.0 / (n * np.pi)) * np.sin(n * np.pi * x0) * np.exp(-mu_n * T)
    return np.exp(-r * T) * val


def homogeneous_strip_reference(S0, B_lo, B_hi, varsigma, r, T):
    return image_formula_double_notouch(S0, B_lo, B_hi, varsigma, r, T, n_terms=4001)


def split_strip_equal_layers(B_lo: float, B_hi: float, n_layers: int, varsigma_flat: float):
    Y = np.linspace(B_lo, B_hi, n_layers + 1)
    varsigma = np.full(n_layers, varsigma_flat)
    return Y, varsigma


def price_equal_slw(model, basis, x0, T, r, payoff, K=None):
    cn = compute_cn(basis, model, payoff, K)
    return price_eigen_expansion(basis, model, x0, T, r, cn)


def price_general_slw(model, T, r, payoff, K=None, log_inv_eps=LOG_INV_EPS_DEFAULT):
    n_modes = n_modes_weyl(model, T, log_inv_eps=log_inv_eps)
    basis = build_general_slw_basis(model, n_modes)
    x0 = scale_transform_point(model, SPOT)
    cn = compute_cn(basis, model, payoff, K)
    return price_eigen_expansion(basis, model, x0, T, r, cn), n_modes


def main():
    results = {}

    # --- Check (a): N=1 homogeneous strip vs classical image formula -------
    B_lo, B_hi, varsigma_flat = 20.0, 180.0, 10.0
    S0 = 100.0
    row_a = []
    for T in MATURITIES:
        Y1, v1 = split_strip_equal_layers(B_lo, B_hi, 1, varsigma_flat)
        model1 = build_layered_model(Y1, v1)
        basis1 = build_equal_slw_basis(model1)
        x0 = scale_transform_point(model1, S0)
        spectral = price_equal_slw(model1, basis1, x0, T, 0.0, "notouch")
        reference = homogeneous_strip_reference(S0, B_lo, B_hi, varsigma_flat, 0.0, T)
        row_a.append({"T": T, "spectral_N1": spectral, "image_formula": reference,
                       "abs_diff": abs(spectral - reference)})
    results["check_a_image_formula"] = row_a

    # --- Check (b): homogeneous strip split into N=1,2,5 equal layers -------
    row_b = []
    T_b = 1.0
    ref_notouch = None
    for n_layers in (1, 2, 5):
        Y, v = split_strip_equal_layers(B_lo, B_hi, n_layers, varsigma_flat)
        model = build_layered_model(Y, v)
        basis = build_equal_slw_basis(model)
        x0 = scale_transform_point(model, S0)
        price = price_equal_slw(model, basis, x0, T_b, 0.0, "notouch")
        if ref_notouch is None:
            ref_notouch = price
        row_b.append({"n_layers": n_layers, "notouch": price, "diff_vs_N1": abs(price - ref_notouch)})
    results["check_b_homogeneous_split"] = row_b

    # --- Check (c): equal-SLW configuration vs general-SLW method ----------
    L = np.diff(BREAKPOINTS_GENERAL)
    varsigma_eq = EQUAL_SLW_SCALE * L
    model_eq = build_layered_model(BREAKPOINTS_GENERAL, varsigma_eq)
    omega_common = check_equal_slw(model_eq)
    x0_eq = scale_transform_point(model_eq, SPOT)

    row_c = []
    for T in MATURITIES:
        basis_eq = build_equal_slw_basis(model_eq)
        t0 = time.time()
        for payoff, K in (("notouch", None), ("call", STRIKE), ("put", STRIKE)):
            v_eq = price_equal_slw(model_eq, basis_eq, x0_eq, T, RATE, payoff, K)
            v_gen, n_modes = price_general_slw(model_eq, T, RATE, payoff, K)
            row_c.append({
                "T": T, "payoff": payoff,
                "equal_slw_price": v_eq, "general_slw_price": v_gen,
                "abs_diff": abs(v_eq - v_gen), "general_n_modes": n_modes,
            })
        row_c[-1]["time_equal_slw_basis_s"] = time.time() - t0
    results["check_c_equal_vs_general"] = row_c
    results["omega_common"] = omega_common
    results["omega_expected"] = EQUAL_SLW_OMEGA_EXPECTED

    with open("results/exp1_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # --- Plot: convergence of check (b) with number of layers --------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ns = [r["n_layers"] for r in row_b]
    vals = [r["notouch"] for r in row_b]
    ax.plot(ns, vals, "o-")
    ax.set_xlabel("Number of equal layers splitting the homogeneous strip")
    ax.set_ylabel("Double no-touch price")
    ax.set_title("Exp. 1(b): homogeneous strip split into N equal layers")
    fig.tight_layout()
    fig.savefig("results/exp1_homogeneous_split.png", dpi=120)
    plt.close(fig)

    print(json.dumps(results, indent=2, default=float))
    print(f"omega_common = {omega_common:.6f} (expected ~{EQUAL_SLW_OMEGA_EXPECTED})")


if __name__ == "__main__":
    main()
