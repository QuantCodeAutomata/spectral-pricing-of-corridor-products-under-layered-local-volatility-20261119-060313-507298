"""Paper-derived parameters for every experiment (Section 11, Numerical implementation
and benchmarks). No classes or dicts: every variable is a module-level constant with a
comment stating its meaning, unit and source.

Paper: "Spectral Pricing of Corridor Products under Layered Local Volatility".
"""

import numpy as np

# ---------------------------------------------------------------------------
# Shared single-bucket "general configuration" (Section 11, page 28)
# ---------------------------------------------------------------------------
BREAKPOINTS_GENERAL = np.array(
    [80, 85, 89.5, 93.5, 97, 100, 103, 106.5, 110.5, 115, 120], dtype=float
)  # price breakpoints Y_0,...,Y_10, in price units (Section 11)
VARSIGMA_GENERAL = np.array(
    [18, 15, 12.5, 11, 10, 9.5, 10, 11, 12.5, 14], dtype=float
)  # layer volatilities varsigma^G_1,...,varsigma^G_10, skewed smile (Section 11)

SPOT = 101.0  # spot S_* in price units (Section 11), inside layer 6
RATE = 0.02  # risk-free rate r, annualized (Section 11)
STRIKE = 100.0  # Call/Put strike K (Section 11)
CORRIDOR = (97.0, 103.0)  # corridor C = (97, 103), layers 5 and 6 (Section 11)
MATURITIES = (0.1, 1.0, 5.0)  # test maturities T in years (Section 11.2)

# ---------------------------------------------------------------------------
# exp_1: equal scaled layer widths (SLW) configuration (Section 11.2)
# ---------------------------------------------------------------------------
EQUAL_SLW_SCALE = 10.0 / 3.0  # varsigma_i = EQUAL_SLW_SCALE * L_i so that omega_i const
EQUAL_SLW_OMEGA_EXPECTED = 0.4243  # expected common SLW value omega_i (Section 11.2)

# ---------------------------------------------------------------------------
# Spectral truncation (Weyl's law mode count, Section 6 / Theorem 9)
# ---------------------------------------------------------------------------
LOG_INV_EPS_DEFAULT = 32.0  # log(1/epsilon) truncation level used in Section 11 examples
BISECTION_TOL = 1e-13  # relative bisection tolerance for eigenvalue location (Theorem 9)

# ---------------------------------------------------------------------------
# exp_3: Laplace-transform (Talbot) inversion
# ---------------------------------------------------------------------------
TALBOT_NODES = 32  # number of optimized Talbot contour nodes (Weideman 2006; Section 11)

# ---------------------------------------------------------------------------
# exp_4: tiled local volatility, multi-bucket configuration (Section 11.1)
# ---------------------------------------------------------------------------
N_BUCKETS = 12  # number of monthly time buckets over T=1 year (Section 11.1)
BUCKET_T_TOTAL = 1.0  # total maturity spanned by the buckets, years (Section 11.1)
LOG_INV_EPS_TILED_RUN = 20.0  # executed per-bucket truncation level log(1/eps) (Section 11.1)
LOG_INV_EPS_TILED_FULL = 90.0  # paper's high-accuracy internal convergence check (Section 11.1)
GAUSS_LEGENDRE_POINTS_TRANSFER = 32  # quadrature points per layer for transfer-matrix
# integrals Q^{(k)} (Eq. 60); not explicitly specified in the paper, chosen generously
# for the piecewise-trigonometric integrands.

EXPECTED_CALL_TILED = 2.7457782018  # paper's reference Call price on the tiled surface
EXPECTED_NOTOUCH_TILED = 0.8159150345  # paper's reference double no-touch on the tiled surface
EXPECTED_ACCRUAL_TILED = 0.3875968500  # paper's reference expected corridor accrual
EXPECTED_CORRIDOR_VAR_TILED = 39.69713630  # paper's reference discounted corridor price variance
EXPECTED_DELTA_TILED = 0.1631869452  # paper's reference Call delta on the tiled surface
EXPECTED_GAMMA_TILED = -0.0157540053  # paper's reference Call gamma on the tiled surface

# ---------------------------------------------------------------------------
# exp_5: law of the corridor accrual (Section 8, Section 11.3)
# ---------------------------------------------------------------------------
TWO_LAYER_VARSIGMA_MINUS = 12.0  # varsigma on (20,100) for the oscillating-BM test
TWO_LAYER_VARSIGMA_PLUS = 8.0  # varsigma on (100,180) for the oscillating-BM test
TWO_LAYER_BREAKPOINTS = (20.0, 100.0, 180.0)  # knock-out barriers and interface
TWO_LAYER_SPOT = 100.0  # spot started exactly at the interface
TWO_LAYER_CORRIDOR = (100.0, 180.0)  # corridor for the two-layer oscillating BM test
ACCRUAL_T = 1.0  # maturity for the law-of-the-accrual tests (Section 11.3)

ACCRUAL_GENERAL_BARRIERS = (20.0, 180.0)  # outer knock-out barriers, general config
ACCRUAL_RATE = 0.0  # risk-free rate used for the law-of-the-accrual tests (r=0, Section 11.3)
RECTANGLE_C_BAR = 1.0  # bar_c = 1/T multiplier for the rectangle Laplace contour (Section 8.3)
RECTANGLE_DELTA = 2.0  # delta = 2/T for the rectangle Laplace contour (Section 8.3)
RECTANGLE_TRUNCATION = 40.0  # horizontal edge truncated at Re(lambda)=bar_c-40/T (Section 8.3)
RECTANGLE_GL_NODES = 16  # Gauss-Legendre nodes per contour edge (Section 8.3)
ENDPOINT_REMOVAL_M = 4  # number of asymptotic terms M for endpoint singularity removal
ENDPOINT_REMOVAL_K1 = 64  # cosine-mode cutoff K_1 used with endpoint removal (Section 11.3)
ACCRUAL_CALL_STRIKE = 0.5  # strike kappa for the accrual Call payoff test; not specified
# numerically in the paper beyond "accrual Call", a representative at-the-money strike is used

# ---------------------------------------------------------------------------
# exp_6: Greeks and layer vegas (Section 9)
# ---------------------------------------------------------------------------
GREEKS_T = 1.0  # maturity for the Greeks/vega tests (Section 11.2)
COMPLEX_STEP_H = 1e-20  # complex-step differentiation step size (Section 9.1)
GAMMA_KINK_SPOT = 100.0  # spot placed exactly at interface Y_5=100 for the gamma-kink test

# ---------------------------------------------------------------------------
# exp_7: calibration to vanilla quotes (Section 11.5)
# ---------------------------------------------------------------------------
CALIBRATION_CUTOFFS = (20.0, 180.0)  # outer price cut-offs replacing B_-, B_+ (Section 11.5)
CALIBRATION_T = 1.0  # maturity of the calibration quotes (Section 11.5)
CALIBRATION_SPOT = 100.0  # spot for the calibration quotes (Section 11.5)
CALIBRATION_RATE = 0.0  # risk-free rate for the calibration test (Section 11.5)
CALIBRATION_STRIKES = np.array(
    [83, 87, 91.5, 95, 98.5, 101.5, 104.5, 108.5, 112.5, 117], dtype=float
)  # one OTM strike per layer (Section 11.5); strikes below spot are Puts, above are Calls
CALIBRATION_START_VARSIGMA = 12.0  # flat initial volatility guess for Newton's method
CALIBRATION_MAX_ITER = 15  # maximum Newton iterations (paper reports 5-6, Section 11.5)
CALIBRATION_TOL = 1e-10  # convergence tolerance on the largest pricing error

# ---------------------------------------------------------------------------
# exp_8: Crank-Nicolson finite-difference benchmark (Section 11.2)
# ---------------------------------------------------------------------------
FD_RESOLUTIONS_PAPER = (100, 200, 400, 800, 1600, 3200)  # paper's tested grid resolutions
FD_RESOLUTIONS_RUN = (100, 200, 400, 800, 1600)  # executed resolutions (3200 kept available
# via --full; it only sharpens the already-established second- vs first-order convergence
# rates and was dropped to shorten the benchmark's run time)
FD_RANNACHER_HALF_STEPS = 4  # half-steps of implicit Euler at the start (Rannacher smoothing)
FD_SPECTRAL_NOTOUCH_REF = 0.819596756784  # spectral reference, double no-touch (Section 11.2)
FD_SPECTRAL_CALL_REF = 2.807218341507  # spectral reference, Call (Section 11.2)
