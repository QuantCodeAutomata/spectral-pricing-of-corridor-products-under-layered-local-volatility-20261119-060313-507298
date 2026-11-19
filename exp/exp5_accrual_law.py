"""Experiment 5: law of the corridor accrual via layer-dependent killing
(Section 8.2-8.4, Section 11.3).

Validates the characteristic-function / cosine-expansion machinery of
``src.accrual`` on two configurations:

(1) the two-layer oscillating Brownian motion, started at the interface,
    against Lamperti's closed-form law of the occupation fraction;
(2) the ten-layer general configuration with r = 0, checking the internal
    convergence of the cosine expansion as the number of retained modes K1
    grows.

Run: python -m exp.exp5_accrual_law
"""

import json
import time

import numpy as np

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
    results = {}

    # --- (1) Two-layer oscillating BM vs Lamperti's closed-form law ---
    model2 = two_layer_model()
    x0_2 = scale_transform_point(model2, TWO_LAYER_SPOT)
    mask2 = corridor_layer_mask(model2, TWO_LAYER_CORRIDOR)
    a_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    K1_two_layer = 24  # paper uses K1=64 with endpoint-singularity removal (Sec. 8.4);
    # this implementation uses the plain cosine expansion (see module docstring),
    # whose cost grows with the rectangle-contour node count needed at large u,
    # so the run uses a smaller but still rapidly converging K1 (see RESULTS.md).
    t0 = time.perf_counter()
    Xi2 = characteristic_array(model2, mask2, ACCRUAL_T, x0_2, K1_two_layer)
    t_two_layer = time.perf_counter() - t0
    spectral_cdf = np.array([
        price_from_characteristic(Xi2, cosine_modes_cdf(a, K1_two_layer)) for a in a_grid
    ])
    lamperti_cdf = lamperti_two_layer_cdf(a_grid, TWO_LAYER_VARSIGMA_MINUS, TWO_LAYER_VARSIGMA_PLUS)
    abs_diff = np.abs(spectral_cdf - lamperti_cdf)

    results["two_layer_vs_lamperti"] = {
        "a_grid": a_grid.tolist(),
        "spectral_cdf": spectral_cdf.tolist(),
        "lamperti_cdf": lamperti_cdf.tolist(),
        "abs_diff": abs_diff.tolist(),
        "max_abs_diff": float(abs_diff.max()),
        "K1": K1_two_layer,
        "time_s": t_two_layer,
        "expected_max_abs_diff_paper": 2.1e-11,
    }

    # --- (2) General configuration, r = 0: internal convergence of K1 ---
    model_g = general_model_with_barriers()
    x0_g = scale_transform_point(model_g, SPOT)
    mask_g = corridor_layer_mask(model_g, CORRIDOR)
    K1_max = 32  # paper tests K1 up to 256 with endpoint-singularity removal;
    # the plain cosine expansion used here needs more modes for the same
    # accuracy and each mode is itself more expensive at large u, so the run
    # is capped at K1=32 (see RESULTS.md for the runtime-driven reduction).
    t0 = time.perf_counter()
    Xi_g = characteristic_array(model_g, mask_g, ACCRUAL_T, x0_g, K1_max)
    t_general = time.perf_counter() - t0
    K1_list = [8, 16, 32]
    cdf_vals, call_vals = {}, {}
    for K1 in K1_list:
        cdf_vals[K1] = price_from_characteristic(Xi_g, cosine_modes_cdf(0.5, K1))
        call_vals[K1] = price_from_characteristic(Xi_g, cosine_modes_accrual_call(ACCRUAL_CALL_STRIKE, K1))
        results.setdefault("general_configuration", {})[f"K1={K1}"] = {
            "cdf_at_0.5": cdf_vals[K1],
            "accrual_call": call_vals[K1],
        }
    results["general_configuration"]["characteristic_array_time_s"] = t_general
    results["general_configuration"]["K1_max"] = K1_max

    diff_cdf = abs(cdf_vals[32] - cdf_vals[16])
    diff_call = abs(call_vals[32] - call_vals[16])
    results["general_configuration_convergence"] = {
        "cdf_change_16_to_32": diff_cdf,
        "accrual_call_change_16_to_32": diff_call,
        "expected_cdf_change_paper_128_to_256": 2.4e-7,
        "expected_call_change_paper_128_to_256": 4.0e-10,
    }

    with open("results/exp5_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
