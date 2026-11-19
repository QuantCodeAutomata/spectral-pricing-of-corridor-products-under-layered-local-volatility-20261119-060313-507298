"""Experiment 8: Crank-Nicolson finite-difference benchmark (Section 11,
Section 11.2).

Benchmarks the two Crank-Nicolson finite-difference reference schemes of
``src/fd_benchmark.py`` against the spectral method on the single-bucket
"general configuration" of Section 11:

* "fitted": a grid node on every volatility interface Y_i; the scheme is
  second-order convergent in the number of grid points (Section 11.2).
* "naive": a uniform grid that samples volatility at nodes without regard
  to interface location; the scheme drops to first order (or shows
  irregular, non-monotone convergence) near volatility interfaces
  (Section 11.2).

For both schemes and both payoffs (double no-touch, Call), prices are
computed over a range of grid resolutions ``n`` (space sub-intervals, with
``n_steps = n`` Crank-Nicolson time steps, i.e. "n space intervals with n
time steps" as in the paper), compared against the spectral reference
prices, and the empirical convergence order is estimated between
consecutive resolutions. Richardson extrapolation of the two finest
fitted-scheme runs is also reported.

Run: python -m exp.exp8_fd_benchmark [--full]
"""

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    BREAKPOINTS_GENERAL, VARSIGMA_GENERAL, SPOT, RATE, STRIKE,
    FD_RESOLUTIONS_PAPER, FD_RESOLUTIONS_RUN, FD_RANNACHER_HALF_STEPS,
    FD_SPECTRAL_NOTOUCH_REF, FD_SPECTRAL_CALL_REF,
)
from src.spectral_core import build_layered_model
from src.fd_benchmark import crank_nicolson_price

T_SINGLE_BUCKET = 1.0  # single maturity T=1 used in the Section 11.2 FD benchmark table

SCHEMES = ("fitted", "naive")
PAYOFFS = (("notouch", None), ("call", STRIKE))


def _reference_price(payoff: str) -> float:
    """Spectral reference price for a given payoff (Section 11.2)."""
    return FD_SPECTRAL_NOTOUCH_REF if payoff == "notouch" else FD_SPECTRAL_CALL_REF


def run_sweep(model, resolutions: tuple) -> tuple[list, dict]:
    """Run the FD benchmark over every (resolution, scheme, payoff) combination.

    Returns the flat results table (list of dicts) and a dict of elapsed
    wall-clock seconds per (scheme, payoff) full resolution sweep.
    """
    table = []
    timings = {}
    for scheme in SCHEMES:
        for payoff, K in PAYOFFS:
            key = f"{scheme}_{payoff}"
            t0 = time.time()
            for n in resolutions:
                price = crank_nicolson_price(
                    model, SPOT, T_SINGLE_BUCKET, RATE, payoff, K,
                    n_total=n, n_steps=n, scheme=scheme,
                    n_rannacher=FD_RANNACHER_HALF_STEPS,
                )
                error = price - _reference_price(payoff)
                table.append({
                    "resolution": n, "scheme": scheme, "payoff": payoff,
                    "price": price, "error": error,
                })
            timings[key] = time.time() - t0
    return table, timings


def convergence_orders(table: list, resolutions: tuple) -> list:
    """Empirical convergence order log2(error[n]/error[2n]) for each
    consecutive pair of resolutions, per scheme and payoff (Section 11.2:
    fitted ~2.0, naive ~1.0 or irregular near interfaces)."""
    orders = []
    by_key = {(r["scheme"], r["payoff"], r["resolution"]): r["error"] for r in table}
    for scheme in SCHEMES:
        for payoff, _ in PAYOFFS:
            for n_lo, n_hi in zip(resolutions[:-1], resolutions[1:]):
                e_lo = by_key[(scheme, payoff, n_lo)]
                e_hi = by_key[(scheme, payoff, n_hi)]
                if e_lo == 0.0 or e_hi == 0.0 or (e_lo > 0) != (e_hi > 0):
                    order = float("nan")
                else:
                    order = float(np.log2(abs(e_lo) / abs(e_hi)))
                orders.append({
                    "scheme": scheme, "payoff": payoff,
                    "n_from": n_lo, "n_to": n_hi, "order": order,
                })
    return orders


def richardson_extrapolation(table: list, resolutions: tuple) -> dict:
    """Richardson extrapolation of the two finest fitted-scheme runs,
    assuming second-order convergence: V_extrap = (4*V_finest - V_second)/3
    (Section 11.2)."""
    n_second, n_finest = resolutions[-2], resolutions[-1]
    result = {}
    for payoff, _ in PAYOFFS:
        prices = {r["resolution"]: r["price"] for r in table
                   if r["scheme"] == "fitted" and r["payoff"] == payoff}
        v_second, v_finest = prices[n_second], prices[n_finest]
        v_extrap = (4.0 * v_finest - v_second) / 3.0
        ref = _reference_price(payoff)
        result[payoff] = {
            "n_second_finest": n_second, "n_finest": n_finest,
            "v_second_finest": v_second, "v_finest": v_finest,
            "v_extrapolated": v_extrap, "spectral_ref": ref,
            "diff_vs_ref": v_extrap - ref,
        }
    return result


def make_plot(table: list, path: str) -> None:
    """Log-log plot of |error| vs resolution n, one line per
    (scheme, payoff) combination."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for scheme in SCHEMES:
        for payoff, _ in PAYOFFS:
            rows = sorted(
                (r for r in table if r["scheme"] == scheme and r["payoff"] == payoff),
                key=lambda r: r["resolution"],
            )
            ns = [r["resolution"] for r in rows]
            errs = [abs(r["error"]) for r in rows]
            ax.loglog(ns, errs, "o-", label=f"{scheme} / {payoff}")
    ax.set_xlabel("grid resolution n")
    ax.set_ylabel("absolute error vs spectral reference")
    ax.set_title("Exp. 8: FD convergence vs spectral reference (Section 11.2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_summary(table: list, orders: list, richardson: dict, timings: dict) -> None:
    print("\n=== Exp. 8: FD benchmark summary (Section 11.2) ===")
    print(f"{'scheme':8s} {'payoff':8s} {'n':6s} {'price':>14s} {'error':>14s}")
    for r in sorted(table, key=lambda r: (r["scheme"], r["payoff"], r["resolution"])):
        print(f"{r['scheme']:8s} {r['payoff']:8s} {r['resolution']:6d} "
              f"{r['price']:14.9f} {r['error']:14.3e}")

    print("\n--- empirical convergence order log2(err_n / err_2n) ---")
    for o in orders:
        print(f"{o['scheme']:8s} {o['payoff']:8s} n {o['n_from']:5d} -> {o['n_to']:5d} : "
              f"order = {o['order']:.3f}")

    print("\n--- Richardson extrapolation (fitted scheme, two finest n) ---")
    for payoff, r in richardson.items():
        print(f"{payoff:8s} n={r['n_second_finest']},{r['n_finest']} -> "
              f"V_extrap = {r['v_extrapolated']:.10f}, "
              f"diff vs spectral ref = {r['diff_vs_ref']:.3e}")

    print("\n--- timings (s per scheme/payoff resolution sweep) ---")
    for key, secs in timings.items():
        print(f"{key:20s} {secs:8.2f} s")


def main() -> None:
    """Entry point: run the FD benchmark sweep, compute convergence orders
    and Richardson extrapolation, save results and plot (Section 11.2)."""
    parser = argparse.ArgumentParser(description="Exp. 8: FD benchmark (Section 11.2)")
    parser.add_argument("--full", action="store_true",
                         help="use FD_RESOLUTIONS_PAPER (adds n=3200; slower, "
                              "quadratic in n per Crank-Nicolson step count)")
    args = parser.parse_args()
    resolutions = FD_RESOLUTIONS_PAPER if args.full else FD_RESOLUTIONS_RUN

    model = build_layered_model(BREAKPOINTS_GENERAL, VARSIGMA_GENERAL)

    table, timings = run_sweep(model, resolutions)
    orders = convergence_orders(table, resolutions)
    richardson = richardson_extrapolation(table, resolutions)

    results = {
        "resolutions": list(resolutions),
        "T": T_SINGLE_BUCKET,
        "table": table,
        "convergence_orders": orders,
        "richardson_extrapolation": richardson,
        "timings_seconds": timings,
    }
    with open("results/exp8_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    make_plot(table, "results/exp8_convergence.png")
    print_summary(table, orders, richardson, timings)


if __name__ == "__main__":
    main()
