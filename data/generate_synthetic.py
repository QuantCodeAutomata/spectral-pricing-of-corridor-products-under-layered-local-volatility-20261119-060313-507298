"""Synthetic layered local-volatility configurations used by every experiment.

The paper's numerical experiments (Section 11) are run on fully specified
layered volatility surfaces (breakpoints and layer volatilities), not on
observed market data: this module turns the parameters of ``src/config.py``
into the ``LayeredModel`` objects (single bucket) or lists of
``LayeredModel`` objects (tiled, multi-bucket) that the experiments price.
"""

from __future__ import annotations

import numpy as np

from src import config
from src.spectral_core import LayeredModel, build_layered_model


def general_configuration_model() -> LayeredModel:
    """The skewed-smile 'general configuration', Section 11, page 28."""
    return build_layered_model(config.BREAKPOINTS_GENERAL, config.VARSIGMA_GENERAL)


def equal_slw_configuration_model() -> LayeredModel:
    """Equal-SLW configuration varsigma_i = EQUAL_SLW_SCALE * L_i (Section 11.2),
    built on the same breakpoints as the general configuration."""
    L = np.diff(config.BREAKPOINTS_GENERAL)
    varsigma = config.EQUAL_SLW_SCALE * L
    return build_layered_model(config.BREAKPOINTS_GENERAL, varsigma)


def two_layer_oscillating_bm_model() -> LayeredModel:
    """Two-layer oscillating Brownian motion (Section 11.3): varsigma_- on
    (20, 100), varsigma_+ on (100, 180), used for the Lamperti closed-form
    benchmark of the law of the corridor accrual."""
    Y = np.array(config.TWO_LAYER_BREAKPOINTS, dtype=float)
    varsigma = np.array(
        [config.TWO_LAYER_VARSIGMA_MINUS, config.TWO_LAYER_VARSIGMA_PLUS], dtype=float
    )
    return build_layered_model(Y, varsigma)


def accrual_general_configuration_model() -> LayeredModel:
    """General configuration with outer knock-out barriers at 20 and 180
    (Section 11.3), used for the law-of-the-accrual test on the skewed smile."""
    Y = np.concatenate((
        [config.ACCRUAL_GENERAL_BARRIERS[0]],
        config.BREAKPOINTS_GENERAL,
        [config.ACCRUAL_GENERAL_BARRIERS[1]],
    ))
    varsigma = np.concatenate((
        [config.VARSIGMA_GENERAL[0]], config.VARSIGMA_GENERAL, [config.VARSIGMA_GENERAL[-1]]
    ))
    return build_layered_model(Y, varsigma)


def calibration_configuration_model(varsigma: np.ndarray | None = None) -> LayeredModel:
    """Calibration configuration (Section 11.5): generating/current layer
    volatilities on the interior breakpoints, continued flat beyond the
    cut-offs 20 and 180 (the paper states the outer volatilities are
    'continued' from the interior without giving a specific rule; continuing
    the nearest interior layer value is used here)."""
    v = config.VARSIGMA_GENERAL if varsigma is None else np.asarray(varsigma, dtype=float)
    Y = np.concatenate(([config.CALIBRATION_CUTOFFS[0]], config.BREAKPOINTS_GENERAL,
                        [config.CALIBRATION_CUTOFFS[1]]))
    varsigma_full = np.concatenate(([v[0]], v, [v[-1]]))
    return build_layered_model(Y, varsigma_full)


def tiled_surface_models() -> list[LayeredModel]:
    """Volatility piecewise constant in price and in time (Section 11.1):
    varsigma_{i,k} = varsigma_i^G * (1.2 - 0.4*tbar_k) + 2*(tbar_k - 1/2)*(i-5.5)/4.5,
    one LayeredModel per monthly bucket k=1,...,12, i=1,...,10 (1-indexed in the
    paper's formula, 0-indexed internally as i_idx = i-1)."""
    K = config.N_BUCKETS
    dt = config.BUCKET_T_TOTAL / K
    models = []
    i_idx = np.arange(1, len(config.VARSIGMA_GENERAL) + 1, dtype=float)  # i = 1,...,10
    for k in range(1, K + 1):
        tbar_k = (k - 0.5) * dt  # midpoint of bucket k, Section 11.1
        varsigma_k = (config.VARSIGMA_GENERAL * (1.2 - 0.4 * tbar_k)
                      + 2.0 * (tbar_k - 0.5) * (i_idx - 5.5) / 4.5)
        models.append(build_layered_model(config.BREAKPOINTS_GENERAL, varsigma_k))
    return models


def bucket_boundaries() -> np.ndarray:
    """Bucket boundaries t_0=0 < t_1 < ... < t_K=T, Section 11.1."""
    K = config.N_BUCKETS
    return np.linspace(0.0, config.BUCKET_T_TOTAL, K + 1)
