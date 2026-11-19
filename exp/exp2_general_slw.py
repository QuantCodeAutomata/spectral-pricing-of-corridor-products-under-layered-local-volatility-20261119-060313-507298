"""Experiment 2: single-bucket pricing under general scaled layer widths (SLW).

Reproduces Section 6's counting-formula / bisection spectral method
(Theorem 9, Wittrick-Williams principle) and the validation checks of
Section 11.2:
  (a) the general method reproduces the 40 lowest closed-form eigenvalues of
      the equal-SLW configuration (Section 5) to relative ~3.8e-15;
  (b) on the genuinely non-equal-SLW "general configuration" it prices
      double no-touch and Call, compared with the paper's spectral reference
      values (double no-touch = 0.819596756784, Call = 2.807218341507) and
      with the equal-SLW method on the equal-SLW configuration.

Run: python -m exp.exp2_general_slw
"""

import json

import numpy as np

from src.config import (
    BREAKPOINTS_GENERAL, VARSIGMA_GENERAL, SPOT, RATE, STRIKE,
    MATURITIES, EQUAL_SLW_SCALE, LOG_INV_EPS_DEFAULT,
)
from src.spectral_core import build_layered_model, scale_transform_point, compute_cn, price_eigen_expansion
from src.equal_slw import build_equal_slw_basis
from src.general_slw import build_general_slw_basis, n_modes_weyl, locate_eigenvalues


def main():
    results = {}

    # --- Check (a): eigenvalue counting vs closed-form equal-SLW spectrum --
    L = np.diff(BREAKPOINTS_GENERAL)
    varsigma_eq = EQUAL_SLW_SCALE * L
    model_eq = build_layered_model(BREAKPOINTS_GENERAL, varsigma_eq)
    basis_closed = build_equal_slw_basis(model_eq)
    n_check = 40
    mu_closed = basis_closed.mu[:n_check]

    xi_general = locate_eigenvalues(model_eq, n_check)
    mu_general = xi_general**2
    rel_err = np.abs(mu_general - mu_closed) / np.abs(mu_closed)
    results["check_a_eigenvalue_counting"] = {
        "n_modes_checked": n_check,
        "max_relative_error": float(rel_err.max()),
        "mean_relative_error": float(rel_err.mean()),
        "expected_max_relative_error": 3.8e-15,
    }

    # --- Check (b): general configuration pricing, non-equal SLW -----------
    model_gen = build_layered_model(BREAKPOINTS_GENERAL, VARSIGMA_GENERAL)
    x0_gen = scale_transform_point(model_gen, SPOT)
    row_b = []
    for T in MATURITIES:
        n_modes = n_modes_weyl(model_gen, T, log_inv_eps=LOG_INV_EPS_DEFAULT)
        basis_gen = build_general_slw_basis(model_gen, n_modes)
        prices = {}
        for payoff, K in (("notouch", None), ("call", STRIKE), ("put", STRIKE)):
            cn = compute_cn(basis_gen, model_gen, payoff, K)
            prices[payoff] = price_eigen_expansion(basis_gen, model_gen, x0_gen, T, RATE, cn)
        row_b.append({"T": T, "n_modes": n_modes, **prices})
    results["check_b_general_configuration"] = row_b

    t1_row = next(r for r in row_b if r["T"] == 1.0)
    results["reference_comparison_T1"] = {
        "notouch_computed": t1_row["notouch"],
        "notouch_expected": 0.819596756784,
        "call_computed": t1_row["call"],
        "call_expected": 2.807218341507,
    }

    # --- Check (c): general method vs equal-SLW method on equal-SLW config -
    row_c = []
    for T in MATURITIES:
        basis_eq_closed = build_equal_slw_basis(model_eq)
        n_modes = n_modes_weyl(model_eq, T, log_inv_eps=LOG_INV_EPS_DEFAULT)
        basis_eq_general = build_general_slw_basis(model_eq, n_modes)
        for payoff, K in (("notouch", None), ("call", STRIKE)):
            cn_closed = compute_cn(basis_eq_closed, model_eq, payoff, K)
            v_closed = price_eigen_expansion(basis_eq_closed, model_eq, x0_gen, T, RATE, cn_closed)
            cn_general = compute_cn(basis_eq_general, model_eq, payoff, K)
            v_general = price_eigen_expansion(basis_eq_general, model_eq, x0_gen, T, RATE, cn_general)
            row_c.append({
                "T": T, "payoff": payoff,
                "equal_slw_closed_form": v_closed, "general_counting_method": v_general,
                "abs_diff": abs(v_closed - v_general),
            })
    results["check_c_general_vs_equal_slw"] = row_c

    with open("results/exp2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
