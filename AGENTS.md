# Repository guide for coding agents

This repository reproduces selected experiments from the paper **"Spectral
Pricing of Corridor Products under Layered Local Volatility"**: spectral and
Laplace-transform pricing of barrier, no-touch and corridor-accrual options
under a local-volatility surface that is piecewise constant in price
(layered) and, for the multi-bucket experiment, also piecewise constant in
time (tiled).

## Layout

```
src/        shared library, flat modules: config.py (paper-derived
            parameters), spectral_core.py (layered model, scale transform,
            eigen-expansion kernel, payoff integrals, Greeks), equal_slw.py
            (closed-form equal-scaled-layer-width spectral method),
            general_slw.py (eigenvalue counting/bisection and transfer-matrix
            propagation for general scaled layer widths), laplace_method.py
            (Talbot-contour Laplace inversion), multibucket.py (bucket
            recursion and transfer matrices between bucket eigenbases),
            accrual.py (law of the corridor accrual via layer-dependent
            killing), greeks.py (complex-step layer vegas), fd_benchmark.py
            (Crank-Nicolson fitted and naive finite-difference schemes)
exp/        one script per experiment (exp1_equal_slw.py ... exp8_fd_benchmark.py),
            calling only the src/ API with the signatures as written
data/       generate_synthetic.py builds the paper's layered- and
            tiled-volatility configurations from src/config.py; no market
            data is used by any experiment in this repository
tests/      pytest tests of src/ components (synthetic/fixture inputs) and
            of the experiment scripts
results/    tables, metrics (JSON) and plots (PNG) written directly by the
            experiment scripts, plus results/RESULTS.md with the comparison
            against the paper's reported values
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or `.env` file are needed: every experiment is driven entirely by
the closed-form model configurations in `src/config.py`.

## Tests

```bash
python -m pytest -m "not market_data"
```

There are no `market_data`-marked tests in this repository (no experiment
loads market data), so a plain `python -m pytest` is equivalent.

## Running the experiments

```bash
python -m exp.exp1_equal_slw
python -m exp.exp2_general_slw
python -m exp.exp3_laplace_talbot
python -m exp.exp4_tiled_multibucket
python -m exp.exp5_accrual_law
python -m exp.exp6_greeks
python -m exp.exp7_calibration
python -m exp.exp8_fd_benchmark
```

`reproduce.sh` runs the test suite followed by all eight experiments in this
order. Every run of an experiment script overwrites its files in `results/`;
after a rerun, check the numbers and verdicts in `results/RESULTS.md` against
the new `results/exp*_results.json` / `.png` files and update them by hand if
they changed.

## Conventions

* All experiment parameters (breakpoints, layer volatilities, spot, rate,
  strikes, corridor bounds, maturities, mode-count truncations, quadrature
  and bisection tolerances) live only in `src/config.py` as module-level
  `UPPER_CASE` variables, each with a comment naming its meaning, unit and
  the paper section it comes from. No experiment script defines its own
  copy of a parameter.
* `data/generate_synthetic.py` is the only place that assembles the paper's
  layered- and tiled-volatility configurations into `LayeredModel`/bucket
  objects; `src/` and `exp/` call it rather than re-deriving breakpoints or
  volatilities inline.
* Library code is flat under `src/` (no sub-packages); experiment scripts
  live only under `exp/` and import only from `src/` and `data/`; all pytest
  files live under `tests/`.
* Experiments run as `python -m exp.<name>` and tests as `python -m pytest`
  from the repository root; no script edits `sys.path`.
* Deviations from the paper (accuracy gaps, timing gaps, methodology
  substitutions) are recorded only in `results/RESULTS.md`, never as code
  comments or silent defaults in `src/config.py`.
