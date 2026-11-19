# Spectral Pricing of Corridor Products under Layered Local Volatility

Reproduction of selected experiments from the paper **"Spectral Pricing of
Corridor Products under Layered Local Volatility"**.

This repository implements the spectral pricing framework for barrier and
no-touch options, corridor accruals and corridor variance under a local
volatility model that is piecewise constant in price (layered) and,
separately, piecewise constant in both price and time (tiled). Pricing uses
closed-form eigen-decompositions (equal scaled layer widths), eigenvalue
counting with bisection and transfer-matrix propagation (general scaled layer
widths), and Laplace-transform inversion on an optimized Talbot contour. All
configurations (breakpoints, layer volatilities, spot, rate, strikes,
maturities) are taken from the paper's numerical section and live in
`src/config.py`; no market data is used anywhere in this repository.

## Layout

```
src/        shared library: layered model, eigen-expansion kernel, both
            spectral methods, Laplace method, multi-bucket recursion,
            law of the accrual, Greeks, Crank-Nicolson FD benchmark, config
exp/        one script per experiment, calling only the src/ API
data/       data/generate_synthetic.py builds the paper's layered-volatility
            configurations from src/config.py (no market data is used)
tests/      pytest tests of src/ components and of the experiment scripts
results/    tables, metrics and plots written by the experiment scripts,
            plus results/RESULTS.md with the comparison to the paper
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest -m "not market_data"
```

(there is no market-data path in this repository, so this is the same as
`python -m pytest`; the marker is kept for consistency with the standard
test-running convention).

## Experiments

Each experiment is run as a module from the repository root and writes its
artifacts directly into `results/`:

```bash
python -m exp.exp1_equal_slw
python -m exp.exp2_general_slw
python -m exp.exp3_laplace
python -m exp.exp4_tiled
python -m exp.exp5_accrual
python -m exp.exp6_greeks
python -m exp.exp7_calibration
python -m exp.exp8_fd_benchmark
```

`reproduce.sh` runs the full test suite followed by every experiment in
sequence.

## Experiments reproduced

1. **exp1** — single-bucket pricing under equal scaled layer widths (SLW):
   closed-form eigen-decomposition (Section 5).
2. **exp2** — single-bucket pricing under general SLW: eigenvalue counting
   and bisection, transfer-matrix eigenfunctions (Section 6).
3. **exp3** — Laplace-transform pricing via Talbot contour inversion
   (Section 4, Section 8.1).
4. **exp4** — tiled local volatility: multi-bucket pricing with eigenbasis
   re-expansion (Section 10).
5. **exp5** — law of the corridor accrual via layer-dependent killing
   (Section 8.2-8.4).
6. **exp6** — Greeks and layer vegas via complex-step differentiation
   (Section 9).
7. **exp7** — calibration of layer volatilities to vanilla quotes
   (Section 11.5).
8. **exp8** — Crank-Nicolson finite-difference benchmark, fitted vs. naive
   schemes (Section 11.2).

See `results/RESULTS.md` for the numerical comparison against the paper,
including every deviation and any result not reproduced.
