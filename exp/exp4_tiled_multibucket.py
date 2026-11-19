"""Experiment 4: tiled local volatility, multi-bucket pricing with eigenbasis
re-expansion (Section 10, Section 11.1).

Reproduces the bucket recursion (Eq. 61) for terminal payoffs (Call, double
no-touch) and the running-reward recursion (Eq. 65) for the expected corridor
accrual and the corridor price variance, on the tiled surface of twelve
monthly buckets over one year. Validates: (a) identical buckets reproduce the
time-homogeneous single-bucket price; (b) the internal-convergence check
between a modest and a generous per-bucket mode-count truncation.

Run: python -m exp.exp4_tiled_multibucket
"""

import json
import time

import numpy as np

from src.config import (
    BREAKPOINTS_GENERAL, VARSIGMA_GENERAL, SPOT, RATE, STRIKE, CORRIDOR,
    N_BUCKETS, BUCKET_T_TOTAL, LOG_INV_EPS_TILED_RUN, LOG_INV_EPS_TILED_FULL,
    GAUSS_LEGENDRE_POINTS_TRANSFER,
    EXPECTED_CALL_TILED, EXPECTED_NOTOUCH_TILED, EXPECTED_ACCRUAL_TILED,
    EXPECTED_CORRIDOR_VAR_TILED,
)
from src.spectral_core import build_layered_model
from src.multibucket import (
    build_bucket, bucket_recursion_terminal, bucket_recursion_running,
    corridor_rate_indicator, corridor_variance_rate,
)
from src.general_slw import build_general_slw_basis, n_modes_weyl


def tiled_volatility(t_bar_k: float) -> np.ndarray:
    """varsigma_{i,k} = varsigma_i^G * (1.2 - 0.4*t_bar_k) + 2*(t_bar_k-1/2)*(i-5.5)/4.5
    (Section 11.1), i = 1..10 (1-indexed layer number)."""
    i = np.arange(1, len(VARSIGMA_GENERAL) + 1)
    return VARSIGMA_GENERAL * (1.2 - 0.4 * t_bar_k) + 2.0 * (t_bar_k - 0.5) * (i - 5.5) / 4.5


def build_tiled_buckets(log_inv_eps: float, n_buckets: int = N_BUCKETS, T_total: float = BUCKET_T_TOTAL):
    dt = T_total / n_buckets
    buckets = []
    for k in range(n_buckets):
        t_bar_k = (k + 0.5) * dt  # bucket midpoint
        varsigma_k = tiled_volatility(t_bar_k)
        buckets.append(build_bucket(BREAKPOINTS_GENERAL, varsigma_k, dt, RATE, log_inv_eps))
    return buckets


def homogeneous_buckets(log_inv_eps: float, n_buckets: int, T_total: float):
    """All buckets share the general configuration's volatilities (time-homogeneous
    check (a))."""
    dt = T_total / n_buckets
    buckets = [build_bucket(BREAKPOINTS_GENERAL, VARSIGMA_GENERAL, dt, RATE, log_inv_eps)
               for _ in range(n_buckets)]
    return buckets


def main():
    results = {}

    t0 = time.perf_counter()
    buckets_run = build_tiled_buckets(LOG_INV_EPS_TILED_RUN)
    mode_counts = [b.basis.n_modes for b in buckets_run]
    call_run = bucket_recursion_terminal(buckets_run, RATE, "call", STRIKE, SPOT,
                                          n_quad=GAUSS_LEGENDRE_POINTS_TRANSFER)
    notouch_run = bucket_recursion_terminal(buckets_run, RATE, "notouch", None, SPOT,
                                             n_quad=GAUSS_LEGENDRE_POINTS_TRANSFER)
    t_terminal = time.perf_counter() - t0

    # The corridor occupation indicator does not depend on time (the layer
    # breakpoints are identical across buckets), so one array serves every
    # bucket; the quadratic-variation rate g_i = varsigma_{i,k}^2 follows
    # each bucket's own tiled volatility, so it is supplied per bucket.
    indicator = corridor_rate_indicator(buckets_run[0].model, CORRIDOR)
    accrual_run = bucket_recursion_running(buckets_run, RATE, indicator, SPOT,
                                            n_quad=GAUSS_LEGENDRE_POINTS_TRANSFER)
    var_rates = [corridor_variance_rate(b.model, CORRIDOR) for b in buckets_run]
    corridor_var_run = bucket_recursion_running(buckets_run, RATE, var_rates, SPOT,
                                                 n_quad=GAUSS_LEGENDRE_POINTS_TRANSFER)
    t_running = time.perf_counter() - t0 - t_terminal

    results["executed_configuration"] = {
        "n_buckets": N_BUCKETS,
        "log_inv_eps_run": LOG_INV_EPS_TILED_RUN,
        "mode_counts_per_bucket": mode_counts,
        "terminal_payoffs_time_s": t_terminal,
        "running_rewards_time_s": t_running,
    }
    results["tiled_surface"] = {
        "call_price": call_run,
        "notouch_price": notouch_run,
        "expected_accrual": accrual_run,
        "corridor_variance": corridor_var_run,
        "call_expected_paper": EXPECTED_CALL_TILED,
        "notouch_expected_paper": EXPECTED_NOTOUCH_TILED,
        "accrual_expected_paper": EXPECTED_ACCRUAL_TILED,
        "corridor_variance_expected_paper": EXPECTED_CORRIDOR_VAR_TILED,
        "call_abs_diff": abs(call_run - EXPECTED_CALL_TILED),
        "notouch_abs_diff": abs(notouch_run - EXPECTED_NOTOUCH_TILED),
        "accrual_abs_diff": abs(accrual_run - EXPECTED_ACCRUAL_TILED),
        "corridor_variance_abs_diff": abs(corridor_var_run - EXPECTED_CORRIDOR_VAR_TILED),
    }

    # Check (a): identical buckets reproduce the single-bucket price.
    model_single = build_layered_model(BREAKPOINTS_GENERAL, VARSIGMA_GENERAL)
    n_single = n_modes_weyl(model_single, BUCKET_T_TOTAL, log_inv_eps=LOG_INV_EPS_TILED_RUN)
    basis_single = build_general_slw_basis(model_single, n_single)
    from src.spectral_core import scale_transform_point, compute_cn, price_eigen_expansion
    x0 = scale_transform_point(model_single, SPOT)
    cn_call = compute_cn(basis_single, model_single, "call", STRIKE)
    call_single = price_eigen_expansion(basis_single, model_single, x0, BUCKET_T_TOTAL, RATE, cn_call)

    buckets_identical = homogeneous_buckets(LOG_INV_EPS_TILED_RUN, N_BUCKETS, BUCKET_T_TOTAL)
    call_identical = bucket_recursion_terminal(buckets_identical, RATE, "call", STRIKE, SPOT,
                                                n_quad=GAUSS_LEGENDRE_POINTS_TRANSFER)
    results["check_a_time_homogeneous"] = {
        "call_single_bucket": call_single,
        "call_identical_buckets": call_identical,
        "abs_diff": abs(call_single - call_identical),
        "expected_abs_diff_paper": 4.4e-15,
    }

    # Check (b): internal convergence between the run truncation and a higher one.
    buckets_full = build_tiled_buckets(LOG_INV_EPS_TILED_FULL)
    call_full = bucket_recursion_terminal(buckets_full, RATE, "call", STRIKE, SPOT,
                                           n_quad=GAUSS_LEGENDRE_POINTS_TRANSFER)
    results["check_b_internal_convergence"] = {
        "log_inv_eps_run": LOG_INV_EPS_TILED_RUN,
        "log_inv_eps_full": LOG_INV_EPS_TILED_FULL,
        "call_run": call_run,
        "call_full": call_full,
        "abs_diff": abs(call_run - call_full),
        "expected_abs_diff_paper": 9.3e-15,
    }

    with open("results/exp4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
