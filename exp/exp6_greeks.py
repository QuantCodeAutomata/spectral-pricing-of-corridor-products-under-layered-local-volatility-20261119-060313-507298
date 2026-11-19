"""Experiment 6: Greeks and layer vegas (Section 9).

Reproduces the closed-form Delta/Gamma/Theta of the eigen-series (Eq. 56),
checks the Theta/Gamma cross-check identity

    0.5 * varsigma_j^2 * Gamma = r*V - Theta

implied by Eq. (56) at the layer j containing the spot, computes the ten
layer vegas d V / d varsigma_i via the Hellmann-Feynman eigenvalue
sensitivity (Eq. 58) and via complex-step differentiation of the full
price series (Eq. 59), cross-checks the latter against a central
finite-difference approximation, and verifies the gamma interface kink
ratio (Eq. 57)

    Gamma(Y_i+) / Gamma(Y_i-) = varsigma_i^2 / varsigma_{i+1}^2

at the interface Y_5 = 100 of the "general configuration" (Section 11).

Run: python -m exp.exp6_greeks
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    BREAKPOINTS_GENERAL, VARSIGMA_GENERAL, SPOT, RATE, STRIKE,
    GREEKS_T, COMPLEX_STEP_H, GAMMA_KINK_SPOT, LOG_INV_EPS_DEFAULT,
)
from src.spectral_core import (
    build_layered_model, scale_transform_point, compute_cn,
    price_eigen_expansion, greeks_eigen_expansion,
)
from src.general_slw import n_modes_weyl, build_general_slw_basis
from src.greeks import (
    hellmann_feynman_eigenvalue_sensitivity,
    full_layer_vega_complex_step,
    gamma_interface_kink_ratio,
)


def main() -> None:
    results = {}

    # --- Step 1: build model and eigenbasis --------------------------------
    model = build_layered_model(BREAKPOINTS_GENERAL, VARSIGMA_GENERAL)
    n_modes = n_modes_weyl(model, GREEKS_T, log_inv_eps=LOG_INV_EPS_DEFAULT)
    basis = build_general_slw_basis(model, n_modes)
    results["n_modes"] = n_modes

    # --- Step 2: scale transform and payoff coefficients -------------------
    x0 = scale_transform_point(model, SPOT)
    cn_call = compute_cn(basis, model, "call", STRIKE)

    # --- Step 3: closed-form Greeks (Eq. 56) and Theta/Gamma identity ------
    V, Delta, Gamma, Theta = greeks_eigen_expansion(basis, model, x0, GREEKS_T, RATE, cn_call, SPOT)
    j = model.layer_of_price(SPOT)
    varsigma_j = float(np.real(model.varsigma)[j])
    lhs = 0.5 * varsigma_j**2 * Gamma
    rhs = RATE * V - Theta
    identity_residual = abs(lhs - rhs) / max(abs(rhs), 1e-300)
    results["greeks"] = {
        "V": V, "Delta": Delta, "Gamma": Gamma, "Theta": Theta,
        "layer_of_spot": j, "varsigma_j": varsigma_j,
        "identity_lhs_half_varsigma2_gamma": lhs,
        "identity_rhs_rV_minus_theta": rhs,
        "identity_relative_residual": identity_residual,
    }

    # --- Step 4a: Hellmann-Feynman eigenvalue sensitivity (Eq. 58) ---------
    dmu = hellmann_feynman_eigenvalue_sensitivity(basis, model)
    dmu_all_finite = bool(np.all(np.isfinite(dmu)))
    results["hellmann_feynman"] = {
        "dmu_shape": list(dmu.shape),
        "dmu_all_finite": dmu_all_finite,
        "dmu_sample_row0": dmu[0].tolist(),
    }

    # --- Step 4b: spectral full-series layer vegas (Eq. 59, complex step) -
    t0 = time.time()
    vegas_spectral = full_layer_vega_complex_step(
        basis, model, x0, GREEKS_T, RATE, "call", STRIKE, SPOT, h=COMPLEX_STEP_H,
    )
    t_spectral = time.time() - t0

    # --- Step 4c: central finite-difference layer vegas --------------------
    t0 = time.time()
    vegas_fd = np.empty(model.N)
    for i in range(model.N):
        h_fd = 1e-4 * VARSIGMA_GENERAL[i]

        varsigma_plus = VARSIGMA_GENERAL.copy()
        varsigma_plus[i] += h_fd
        model_plus = build_layered_model(BREAKPOINTS_GENERAL, varsigma_plus)
        basis_plus = build_general_slw_basis(model_plus, n_modes)
        x0_plus = scale_transform_point(model_plus, SPOT)
        cn_plus = compute_cn(basis_plus, model_plus, "call", STRIKE)
        V_plus = price_eigen_expansion(basis_plus, model_plus, x0_plus, GREEKS_T, RATE, cn_plus)

        varsigma_minus = VARSIGMA_GENERAL.copy()
        varsigma_minus[i] -= h_fd
        model_minus = build_layered_model(BREAKPOINTS_GENERAL, varsigma_minus)
        basis_minus = build_general_slw_basis(model_minus, n_modes)
        x0_minus = scale_transform_point(model_minus, SPOT)
        cn_minus = compute_cn(basis_minus, model_minus, "call", STRIKE)
        V_minus = price_eigen_expansion(basis_minus, model_minus, x0_minus, GREEKS_T, RATE, cn_minus)

        vegas_fd[i] = (V_plus - V_minus) / (2.0 * h_fd)
    t_fd = time.time() - t0

    abs_diff = np.abs(vegas_spectral - vegas_fd)
    rel_diff = abs_diff / np.maximum(np.abs(vegas_fd), 1e-300)
    results["layer_vegas"] = {
        "vegas_spectral": vegas_spectral.tolist(),
        "vegas_fd": vegas_fd.tolist(),
        "max_abs_diff": float(abs_diff.max()),
        "max_rel_diff": float(rel_diff.max()),
        "time_spectral_seconds": t_spectral,
        "time_fd_seconds": t_fd,
    }

    # --- Step 5: gamma interface kink ratio at Y_5 = 100 (Eq. 57) ----------
    # The second spatial derivative at an interface converges more slowly in
    # the mode index than the price itself, so this check uses a richer
    # truncation (tighter epsilon in the Weyl mode-count estimate, Eq. 62)
    # than the one selected for the at-the-money price/vega checks above.
    n_modes_kink = n_modes_weyl(model, GREEKS_T, log_inv_eps=90.0)
    basis_kink = build_general_slw_basis(model, n_modes_kink)
    cn_call_kink = compute_cn(basis_kink, model, "call", STRIKE)
    eps = 1e-6
    x0_minus_kink = scale_transform_point(model, GAMMA_KINK_SPOT - eps)
    _, _, Gamma_minus, _ = greeks_eigen_expansion(
        basis_kink, model, x0_minus_kink, GREEKS_T, RATE, cn_call_kink, GAMMA_KINK_SPOT - eps,
    )
    x0_plus_kink = scale_transform_point(model, GAMMA_KINK_SPOT + eps)
    _, _, Gamma_plus, _ = greeks_eigen_expansion(
        basis_kink, model, x0_plus_kink, GREEKS_T, RATE, cn_call_kink, GAMMA_KINK_SPOT + eps,
    )
    ratio_computed = Gamma_plus / Gamma_minus
    ratio_expected = gamma_interface_kink_ratio(model, 5)
    ratio_rel_diff = abs(ratio_computed - ratio_expected) / abs(ratio_expected)
    results["gamma_kink"] = {
        "interface_index": 5,
        "Y_interface": float(BREAKPOINTS_GENERAL[5]),
        "Gamma_minus": Gamma_minus,
        "Gamma_plus": Gamma_plus,
        "ratio_computed": ratio_computed,
        "ratio_expected": ratio_expected,
        "ratio_relative_difference": ratio_rel_diff,
    }

    # --- Save results -------------------------------------------------------
    with open("results/exp6_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # --- Plot: spectral vs central-FD layer vegas ---------------------------
    layer_idx = np.arange(model.N)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(layer_idx - width / 2, vegas_spectral, width, label="Spectral (complex step, Eq. 59)")
    ax.bar(layer_idx + width / 2, vegas_fd, width, label="Central finite difference")
    ax.set_xlabel("Layer index i")
    ax.set_ylabel("Layer vega d V / d varsigma_i")
    ax.set_title("Layer vegas: spectral complex-step vs central finite difference")
    ax.set_xticks(layer_idx)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp6_layer_vegas.png", dpi=150)
    plt.close(fig)

    # --- Print summary --------------------------------------------------
    print(f"n_modes = {n_modes}")
    print(f"V={V:.10f} Delta={Delta:.10f} Gamma={Gamma:.10f} Theta={Theta:.10f}")
    print(f"Theta/Gamma identity relative residual: {identity_residual:.3e}")
    print(f"Hellmann-Feynman dmu finite: {dmu_all_finite}, shape={dmu.shape}")
    print(f"Spectral vega time: {t_spectral:.3f}s, FD vega time: {t_fd:.3f}s")
    print(f"Layer vegas (spectral): {np.array2string(vegas_spectral, precision=6)}")
    print(f"Layer vegas (FD):       {np.array2string(vegas_fd, precision=6)}")
    print(f"Max abs diff: {abs_diff.max():.3e}, max rel diff: {rel_diff.max():.3e}")
    print(f"Gamma kink: Gamma(Y_5-)={Gamma_minus:.10f}, Gamma(Y_5+)={Gamma_plus:.10f}")
    print(f"Ratio computed={ratio_computed:.10f}, expected={ratio_expected:.10f}, "
          f"relative diff={ratio_rel_diff:.3e}")


if __name__ == "__main__":
    main()
