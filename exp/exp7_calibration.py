"""Experiment 7: calibration to vanilla quotes (Section 11.5).

Ten layer volatilities varsigma_1,...,varsigma_10 of the "general
configuration" (Section 11) are recovered by Newton's method from ten
synthetic out-of-the-money vanilla quotes (one strike per layer), starting
from a flat initial guess (``CALIBRATION_START_VARSIGMA``).

The outermost two breakpoints of the general configuration are widened to
the cut-offs ``CALIBRATION_CUTOFFS`` (Section 11.5: "continuing" the
outermost layer volatilities out to 20 and 180) while every interior layer
width and volatility is kept as in ``BREAKPOINTS_GENERAL``. The ten interior
volatilities ``VARSIGMA_GENERAL`` generate the synthetic quotes and are the
Newton iteration's recovery target; they are never used inside the Newton
loop itself.

Each Newton step solves J * delta = (quotes - V(varsigma)) for the update
delta, where J is the 10x10 spectral vega Jacobian J[k, i] = d V_k /
d varsigma_i (Eq. 59), assembled one row (one quote) at a time by
``greeks.full_layer_vega_complex_step`` via complex-step differentiation of
the full eigen-series price (Section 9.1). The paper reports quadratic
convergence in about 5-6 iterations, recovering the generating volatilities
to about 1e-11 relative accuracy.

Run: python -m exp.exp7_calibration
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    BREAKPOINTS_GENERAL, VARSIGMA_GENERAL, CALIBRATION_CUTOFFS, CALIBRATION_T,
    CALIBRATION_SPOT, CALIBRATION_RATE, CALIBRATION_STRIKES,
    CALIBRATION_START_VARSIGMA, CALIBRATION_MAX_ITER, CALIBRATION_TOL,
    LOG_INV_EPS_DEFAULT,
)
from src.spectral_core import (
    build_layered_model, scale_transform_point, compute_cn, price_eigen_expansion,
)
from src.general_slw import n_modes_weyl, build_general_slw_basis
from src.greeks import full_layer_vega_complex_step


def build_calibration_breakpoints() -> np.ndarray:
    """Y_calib: BREAKPOINTS_GENERAL with the outermost two breakpoints
    replaced by CALIBRATION_CUTOFFS (Section 11.5)."""
    Y_calib = BREAKPOINTS_GENERAL.copy()
    Y_calib[0] = CALIBRATION_CUTOFFS[0]
    Y_calib[-1] = CALIBRATION_CUTOFFS[1]
    return Y_calib


def payoff_for_strike(K: float) -> str:
    """OTM payoff type: Put below spot, Call above spot (Section 11.5)."""
    return "put" if K < CALIBRATION_SPOT else "call"


def price_quotes(Y_calib: np.ndarray, varsigma: np.ndarray, strikes: np.ndarray,
                  payoffs: list) -> tuple:
    """Build the model/basis at ``varsigma`` and price every (payoff, strike)
    quote with the eigen-expansion formula (Eq. 38). Returns (V, n_modes)."""
    model = build_layered_model(Y_calib, varsigma)
    n_modes = n_modes_weyl(model, CALIBRATION_T, log_inv_eps=LOG_INV_EPS_DEFAULT)
    basis = build_general_slw_basis(model, n_modes)
    x0 = scale_transform_point(model, CALIBRATION_SPOT)
    V = np.empty(len(strikes))
    for idx, (payoff, K) in enumerate(zip(payoffs, strikes)):
        cn = compute_cn(basis, model, payoff, K)
        V[idx] = price_eigen_expansion(basis, model, x0, CALIBRATION_T, CALIBRATION_RATE, cn)
    return V, model, basis, x0, n_modes


def main() -> None:
    t_start = time.time()
    Y_calib = build_calibration_breakpoints()
    strikes = CALIBRATION_STRIKES
    payoffs = [payoff_for_strike(K) for K in strikes]

    # --- Step 1: generating model and synthetic quotes ----------------------
    generating_model = build_layered_model(Y_calib, VARSIGMA_GENERAL)
    n_modes_gen = n_modes_weyl(generating_model, CALIBRATION_T, log_inv_eps=LOG_INV_EPS_DEFAULT)
    basis_gen = build_general_slw_basis(generating_model, n_modes_gen)
    x0_gen = scale_transform_point(generating_model, CALIBRATION_SPOT)

    quotes = np.empty(len(strikes))
    for idx, (payoff, K) in enumerate(zip(payoffs, strikes)):
        cn = compute_cn(basis_gen, generating_model, payoff, K)
        quotes[idx] = price_eigen_expansion(
            basis_gen, generating_model, x0_gen, CALIBRATION_T, CALIBRATION_RATE, cn,
        )

    # --- Step 2: Newton's method on the 10 layer volatilities ---------------
    varsigma = np.full(10, CALIBRATION_START_VARSIGMA)
    convergence_log = []
    n_iter_used = 0
    converged = False

    for iteration in range(CALIBRATION_MAX_ITER):
        model = build_layered_model(Y_calib, varsigma)
        n_modes_iter = n_modes_weyl(model, CALIBRATION_T, log_inv_eps=LOG_INV_EPS_DEFAULT)
        basis = build_general_slw_basis(model, n_modes_iter)
        x0 = scale_transform_point(model, CALIBRATION_SPOT)

        V = np.empty(10)
        for k, (payoff, K) in enumerate(zip(payoffs, strikes)):
            cn_k = compute_cn(basis, model, payoff, K)
            V[k] = price_eigen_expansion(basis, model, x0, CALIBRATION_T, CALIBRATION_RATE, cn_k)

        error = quotes - V
        max_abs_error = float(np.max(np.abs(error)))
        convergence_log.append({"iteration": iteration, "max_abs_error": max_abs_error})
        n_iter_used = iteration + 1

        if max_abs_error < CALIBRATION_TOL:
            converged = True
            break

        # --- Spectral vega Jacobian (Eq. 59), one row per quote -------------
        J = np.empty((10, 10))
        for k, (payoff, K) in enumerate(zip(payoffs, strikes)):
            J[k, :] = full_layer_vega_complex_step(
                basis, model, x0, CALIBRATION_T, CALIBRATION_RATE, payoff, K, CALIBRATION_SPOT,
            )

        delta = np.linalg.solve(J, error)
        varsigma = np.maximum(varsigma + delta, 0.5)

    t_total = time.time() - t_start

    max_rel_error = float(np.max(np.abs(varsigma - VARSIGMA_GENERAL) / VARSIGMA_GENERAL))

    # --- Save results ---------------------------------------------------------
    results = {
        "strikes": strikes.tolist(),
        "payoffs": payoffs,
        "quotes": quotes.tolist(),
        "varsigma_generating": VARSIGMA_GENERAL.tolist(),
        "varsigma_start": CALIBRATION_START_VARSIGMA,
        "varsigma_recovered": varsigma.tolist(),
        "max_relative_error": max_rel_error,
        "converged": converged,
        "n_iterations": n_iter_used,
        "convergence_log": convergence_log,
        "total_wall_clock_seconds": t_total,
    }
    with open("results/exp7_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # --- Plot: Newton convergence --------------------------------------------
    iters = [entry["iteration"] for entry in convergence_log]
    errs = [entry["max_abs_error"] for entry in convergence_log]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(iters, errs, marker="o")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("max_abs_error (price units)")
    ax.set_title("Calibration convergence: spectral-vega Newton method (Section 11.5)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/exp7_convergence.png", dpi=150)
    plt.close(fig)

    # --- Print summary --------------------------------------------------------
    print(f"Converged: {converged} in {n_iter_used} iterations")
    print(f"Recovered varsigma: {np.array2string(varsigma, precision=6)}")
    print(f"Generating varsigma: {np.array2string(VARSIGMA_GENERAL, precision=6)}")
    print(f"Max relative error vs generating volatilities: {max_rel_error:.3e}")
    print("Convergence log (iteration, max_abs_error):")
    for entry in convergence_log:
        print(f"  {entry['iteration']}: {entry['max_abs_error']:.6e}")
    print(f"Total wall-clock time: {t_total:.3f}s")


if __name__ == "__main__":
    main()
