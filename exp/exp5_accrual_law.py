"""Experiment 5: law of the corridor accrual via layer-dependent killing
(Section 8.2-8.4, Section 11.3).

Validates the characteristic-function / cosine-expansion machinery of
``src.accrual``, with the endpoint-singularity removal of Section 8.4, on two
configurations:

(1) the two-layer oscillating Brownian motion, started at the interface,
    against Lamperti's closed-form law of the occupation fraction;
(2) the ten-layer general configuration with r = 0, checking the internal
    convergence of the corrected cosine expansion as the number of retained
    modes K1 grows.

Run: python -m exp.exp5_accrual_law
"""

import argparse
import json
import time

import numpy as np
from scipy import integrate

from src.config import (
    TWO_LAYER_BREAKPOINTS, TWO_LAYER_VARSIGMA_MINUS, TWO_LAYER_VARSIGMA_PLUS,
    TWO_LAYER_SPOT, TWO_LAYER_CORRIDOR, ACCRUAL_T,
    BREAKPOINTS_GENERAL, VARSIGMA_GENERAL, ACCRUAL_GENERAL_BARRIERS,
    SPOT, CORRIDOR, ACCRUAL_RATE, ACCRUAL_CALL_STRIKE,
)
from src.spectral_core import build_layered_model, scale_transform_point
from src.accrual import (
    corridor_layer_mask, lamperti_two_layer_cdf,
    characteristic_array, price_from_characteristic,
    cosine_modes_cdf, cosine_modes_accrual_call,
    accrual_edge_masses, cdf_payoff, accrual_call_payoff,
    price_accrual_payoff_corrected,
)


def two_layer_model():
    Y = np.array(TWO_LAYER_BREAKPOINTS)
    varsigma = np.array([TWO_LAYER_VARSIGMA_MINUS, TWO_LAYER_VARSIGMA_PLUS])
    return build_layered_model(Y, varsigma)


def general_model_with_barriers():
    """General configuration with its interior breakpoints extended to the
    knock-out barriers (20, 180) used for the law-of-the-accrual test
    (Section 11.3)."""
    lo, hi = ACCRUAL_GENERAL_BARRIERS
    Y = np.concatenate(([lo], BREAKPOINTS_GENERAL, [hi]))
    varsigma = np.concatenate(([VARSIGMA_GENERAL[0]], VARSIGMA_GENERAL, [VARSIGMA_GENERAL[-1]]))
    return build_layered_model(Y, varsigma)


def main():
    parser = argparse.ArgumentParser(description="Exp. 5: law of the corridor accrual (Sec. 8.2-8.4, 11.3)")
    parser.add_argument("--full", action="store_true",
                         help="use the paper's K1=256 modes for the general configuration "
                              "instead of the runtime-reduced K1=64")
    args = parser.parse_args()
    results = {}

    # --- (1) Two-layer oscillating BM vs Lamperti's closed-form law ---
    model2 = two_layer_model()
    x0_2 = scale_transform_point(model2, TWO_LAYER_SPOT)
    mask2 = corridor_layer_mask(model2, TWO_LAYER_CORRIDOR)
    a_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    K1_two_layer = 64  # paper's Table 4 setting with endpoint-singularity removal
    M_two_layer = 4  # number of endpoint asymptotic terms removed (Sec. 8.4)

    # S_* = 100 is exactly the left edge of C = (100, 180): both edge masses
    # vanish for the nondegenerate diffusion (Section 8.4).
    rho0_2, rho1_2 = accrual_edge_masses(model2, TWO_LAYER_CORRIDOR, TWO_LAYER_SPOT, ACCRUAL_T)

    t0 = time.perf_counter()
    Xi2 = characteristic_array(model2, mask2, ACCRUAL_T, x0_2, K1_two_layer)
    t_two_layer = time.perf_counter() - t0

    spectral_cdf = np.array([
        price_accrual_payoff_corrected(
            model2, mask2, ACCRUAL_T, x0_2,
            cosine_modes_cdf(a, K1_two_layer), *cdf_payoff(a)[:3],
            rho0_2, rho1_2, M_two_layer, breakpoints=cdf_payoff(a)[3], Xi=Xi2,
        )
        for a in a_grid
    ])
    lamperti_cdf = lamperti_two_layer_cdf(a_grid, TWO_LAYER_VARSIGMA_MINUS, TWO_LAYER_VARSIGMA_PLUS)
    abs_diff = np.abs(spectral_cdf - lamperti_cdf)

    # Accrual Call (strike kappa = 0.5): reference by integration by parts of
    # Lamperti's CDF, E[(A-kappa)^+] = integral_kappa^1 (1 - CDF(a)) da, valid
    # since the occupation fraction A lies in [0, 1].
    kappa = 0.5
    ref_call, _ = integrate.quad(
        lambda a: 1.0 - lamperti_two_layer_cdf(np.array([a]), TWO_LAYER_VARSIGMA_MINUS,
                                                 TWO_LAYER_VARSIGMA_PLUS)[0],
        kappa, 1.0, limit=200,
    )
    g_p, gp0, gp1, bps = accrual_call_payoff(kappa)
    spectral_call = price_accrual_payoff_corrected(
        model2, mask2, ACCRUAL_T, x0_2, cosine_modes_accrual_call(kappa, K1_two_layer),
        g_p, gp0, gp1, rho0_2, rho1_2, M_two_layer, breakpoints=bps, Xi=Xi2,
    )

    results["two_layer_vs_lamperti"] = {
        "a_grid": a_grid.tolist(),
        "spectral_cdf": spectral_cdf.tolist(),
        "lamperti_cdf": lamperti_cdf.tolist(),
        "abs_diff": abs_diff.tolist(),
        "max_abs_diff": float(abs_diff.max()),
        "accrual_call_strike": kappa,
        "spectral_accrual_call": spectral_call,
        "lamperti_accrual_call": ref_call,
        "accrual_call_abs_diff": abs(spectral_call - ref_call),
        "K1": K1_two_layer,
        "M": M_two_layer,
        "rho0": rho0_2,
        "rho1": rho1_2,
        "time_s": t_two_layer,
        "expected_max_abs_diff_paper": 2.1e-11,
    }

    # --- (2) General configuration, r = 0: internal convergence of K1 ---
    model_g = general_model_with_barriers()
    x0_g = scale_transform_point(model_g, SPOT)
    mask_g = corridor_layer_mask(model_g, CORRIDOR)
    M_general = 4
    rho0_g, rho1_g = accrual_edge_masses(model_g, CORRIDOR, SPOT, ACCRUAL_T)
    # Paper tests K1 up to 256 with endpoint-singularity removal; each
    # Talbot-contour evaluation here costs more per node than the paper's
    # reference implementation, so the default run is capped at K1=64 (see
    # RESULTS.md for the runtime-driven reduction); --full restores K1=256.
    K1_max = 256 if args.full else 64
    t0 = time.perf_counter()
    Xi_g = characteristic_array(model_g, mask_g, ACCRUAL_T, x0_g, K1_max)
    t_general = time.perf_counter() - t0
    K1_list = [64, 128, 256] if args.full else [16, 32, 64]
    cdf_vals, call_vals = {}, {}
    for K1 in K1_list:
        g_cdf, gp0_cdf, gp1_cdf, bps_cdf = cdf_payoff(0.5)
        cdf_vals[K1] = price_accrual_payoff_corrected(
            model_g, mask_g, ACCRUAL_T, x0_g, cosine_modes_cdf(0.5, K1),
            g_cdf, gp0_cdf, gp1_cdf, rho0_g, rho1_g, M_general, breakpoints=bps_cdf,
            Xi=Xi_g[:K1 + 1],
        )
        g_call, gp0_call, gp1_call, bps_call = accrual_call_payoff(ACCRUAL_CALL_STRIKE)
        call_vals[K1] = price_accrual_payoff_corrected(
            model_g, mask_g, ACCRUAL_T, x0_g, cosine_modes_accrual_call(ACCRUAL_CALL_STRIKE, K1),
            g_call, gp0_call, gp1_call, rho0_g, rho1_g, M_general, breakpoints=bps_call,
            Xi=Xi_g[:K1 + 1],
        )
        results.setdefault("general_configuration", {})[f"K1={K1}"] = {
            "cdf_at_0.5": cdf_vals[K1],
            "accrual_call": call_vals[K1],
        }
    results["general_configuration"]["characteristic_array_time_s"] = t_general
    results["general_configuration"]["K1_max"] = K1_max
    results["general_configuration"]["K1_max_paper"] = 256
    results["general_configuration"]["M"] = M_general
    results["general_configuration"]["rho0"] = rho0_g
    results["general_configuration"]["rho1"] = rho1_g

    K1_prev, K1_last = K1_list[-2], K1_list[-1]
    diff_cdf = abs(cdf_vals[K1_last] - cdf_vals[K1_prev])
    diff_call = abs(call_vals[K1_last] - call_vals[K1_prev])
    results["general_configuration_convergence"] = {
        f"cdf_change_{K1_prev}_to_{K1_last}": diff_cdf,
        f"accrual_call_change_{K1_prev}_to_{K1_last}": diff_call,
        "expected_cdf_change_paper_128_to_256": 2.4e-7,
        "expected_call_change_paper_128_to_256": 4.0e-10,
    }

    with open("results/exp5_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
