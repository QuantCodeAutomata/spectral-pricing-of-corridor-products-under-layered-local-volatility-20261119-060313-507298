#!/usr/bin/env bash
# Runs the test suite and every experiment from the repository root.
set -uo pipefail

python -m pytest -m "not market_data" || exit 1

python -m exp.exp1_equal_slw
python -m exp.exp2_general_slw
python -m exp.exp3_laplace
python -m exp.exp4_tiled
python -m exp.exp5_accrual
python -m exp.exp6_greeks
python -m exp.exp7_calibration
python -m exp.exp8_fd_benchmark
