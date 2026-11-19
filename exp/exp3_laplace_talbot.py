"""Experiment 3: Laplace-transform pricing via Talbot contour inversion.

Reproduces Section 4 (tridiagonal interface matrix) and Section 8.1 (Talbot
inversion), Section 11's comparison of the Laplace-transform method against
the two spectral methods of Sections 5-6 for double no-touch and Call prices
at T = 0.1, 1, 5 on the general configuration, together with per-price timing.

Run: python -m exp.exp3_laplace_talbot
"""

import json
import time

from src.config import (
    BREAKPOINTS_GENERAL, VARSIGMA_GENERAL, SPOT, RATE, STRIKE, MATURITIES,
    TALBOT_NODES, LOG_INV_EPS_DEFAULT,
)
from src.spectral_core import build_layered_model, scale_transform_point, compute_cn, price_eigen_expansion
from src.general_slw import build_general_slw_basis, n_modes_weyl
from src.laplace_method import price_laplace


def main():
    results = {}
    model = build_layered_model(BREAKPOINTS_GENERAL, VARSIGMA_GENERAL)
    x0 = scale_transform_point(model, SPOT)

    rows = []
    for T in MATURITIES:
        n_modes = n_modes_weyl(model, T, log_inv_eps=LOG_INV_EPS_DEFAULT)
        basis = build_general_slw_basis(model, n_modes)
        row = {"T": T, "n_talbot_nodes": TALBOT_NODES, "spectral_n_modes": n_modes}
        for payoff, K in (("notouch", None), ("call", STRIKE)):
            cn = compute_cn(basis, model, payoff, K)
            v_spectral = price_eigen_expansion(basis, model, x0, T, RATE, cn)

            t0 = time.perf_counter()
            v_laplace = price_laplace(model, x0, T, RATE, payoff, K, n_talbot=TALBOT_NODES)
            elapsed = time.perf_counter() - t0

            row[f"{payoff}_spectral"] = v_spectral
            row[f"{payoff}_laplace"] = abs(v_laplace.real) if hasattr(v_laplace, "real") else v_laplace
            row[f"{payoff}_abs_diff"] = abs(v_laplace.real - v_spectral)
            row[f"{payoff}_time_s"] = elapsed
        rows.append(row)
    results["laplace_vs_spectral"] = rows
    results["expected_agreement"] = 6e-13
    results["expected_time_ms_per_price_paper"] = 2.0
    results["note_on_timing"] = (
        "This Python implementation integrates each layer's payoff piece against "
        "the Laplace-transformed density with Gauss-Legendre quadrature rather "
        "than the paper's closed-form elementary antiderivatives (Section 4), so "
        "measured timings are slower than the paper's ~2 ms per price; see "
        "results/RESULTS.md."
    )

    with open("results/exp3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
