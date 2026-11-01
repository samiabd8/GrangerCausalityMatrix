"""grangercm: Granger Causality Matrices for multivariate time series.

Compute and visualize pairwise Granger causality across many time series
as a labeled matrix / heatmap, analogous to a correlation table.

Basic usage
-----------
>>> import pandas as pd
>>> from grangercm import make_stationary, granger_matrix
>>> stat_df, orders = make_stationary(df, method="log_diff")
>>> result = granger_matrix(stat_df, maxlag=5, correction="fdr_bh")
>>> result.plot()
>>> result.edges(alpha=0.05)
"""

from .core import GCMatrixResult, granger_matrix
from .nonlinear import nonlinear_granger_matrix
from .plot import plot_gcm
from .prep import adf_report, kpss_report, make_stationary

__all__ = [
    "granger_matrix",
    "nonlinear_granger_matrix",
    "GCMatrixResult",
    "make_stationary",
    "adf_report",
    "kpss_report",
    "plot_gcm",
]

__version__ = "0.1.0"
