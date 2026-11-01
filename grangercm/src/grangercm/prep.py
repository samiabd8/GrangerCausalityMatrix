"""Stationarity testing and differencing helpers.

These wrap the manual ADF / first-difference workflow that is usually done
by hand before a Granger causality analysis, since Granger tests assume
(weakly) stationary inputs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def adf_report(data: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Run the Augmented Dickey-Fuller test on every numeric column.

    Returns a table with the ADF statistic, p-value and a stationarity flag.
    A small p-value means the series is (trend-)stationary.
    """
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for col in data.select_dtypes(include=[np.number]).columns:
        series = data[col].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval = adfuller(series)[:2]
        rows.append({"variable": col, "adf_stat": stat,
                     "p_value": pval, "stationary": pval < alpha})
    return pd.DataFrame(rows).set_index("variable")


def kpss_report(data: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Run the KPSS test on every numeric column.

    Note KPSS reverses the hypotheses relative to ADF: a *large* p-value
    means the series is stationary. Using ADF and KPSS together is the
    recommended robust check.
    """
    from statsmodels.tsa.stattools import kpss

    rows = []
    for col in data.select_dtypes(include=[np.number]).columns:
        series = data[col].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval = kpss(series, nlags="auto")[:2]
        rows.append({"variable": col, "kpss_stat": stat,
                     "p_value": pval, "stationary": pval > alpha})
    return pd.DataFrame(rows).set_index("variable")


def make_stationary(
    data: pd.DataFrame,
    method: str = "diff",
    max_diff: int = 2,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """Transform every column until the ADF test reports stationarity.

    Parameters
    ----------
    data : pandas.DataFrame
        Wide-format input series.
    method : {'diff', 'log_diff'}
        ``'diff'`` applies successive differencing; ``'log_diff'`` takes the
        log first (the log-returns transform used in the original R code) and
        then differences.
    max_diff : int
        Maximum number of differencing rounds before giving up.
    alpha : float
        Significance level for the ADF stationarity decision.

    Returns
    -------
    (transformed, info) : tuple
        ``transformed`` is the stationary DataFrame; ``info`` maps each
        column to the differencing order applied.
    """
    from statsmodels.tsa.stattools import adfuller

    if method not in ("diff", "log_diff"):
        raise ValueError("method must be 'diff' or 'log_diff'")

    work = data.select_dtypes(include=[np.number]).copy()
    if method == "log_diff":
        if (work <= 0).any().any():
            raise ValueError("log_diff requires strictly positive values.")
        work = np.log(work)

    info: dict[str, int] = {}
    for col in work.columns:
        series = work[col]
        order = 0
        while order < max_diff:
            s = series.dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pval = adfuller(s)[1]
            if pval < alpha:
                break
            series = series.diff()
            order += 1
        work[col] = series
        info[col] = order

    return work.dropna(), info
