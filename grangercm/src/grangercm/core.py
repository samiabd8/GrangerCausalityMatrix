"""Core Granger Causality Matrix computation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import permutations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

_VALID_TESTS = ("ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest")
_VALID_AGG = ("min", "mean", "median", "max")
_VALID_CORR = (None, "bonferroni", "holm", "fdr_bh", "fdr_by")


@dataclass
class GCMatrixResult:
    """Result of a Granger causality matrix computation.

    The matrix is directional: entry [effect, cause] is the p-value for the
    hypothesis that ``cause`` Granger-causes ``effect``. Rows are effects
    (the predicted series), columns are causes (the predictor series).
    """

    pvalues: pd.DataFrame
    statistics: pd.DataFrame
    lags: pd.DataFrame
    test: str
    maxlag: int | Sequence[int]
    aggregate: str
    correction: str | None
    pvalues_raw: pd.DataFrame = field(repr=False, default=None)

    @property
    def variables(self) -> list[str]:
        return list(self.pvalues.columns)

    def significant(self, alpha: float = 0.05) -> pd.DataFrame:
        """Boolean matrix of significant directional links at level ``alpha``."""
        mask = (self.pvalues < alpha).copy()
        arr = mask.values.copy()
        np.fill_diagonal(arr, False)
        return pd.DataFrame(arr, index=mask.index, columns=mask.columns)

    def edges(self, alpha: float = 0.05) -> pd.DataFrame:
        """Long-format table of significant cause -> effect edges."""
        sig = self.significant(alpha)
        rows = []
        for effect in sig.index:
            for cause in sig.columns:
                if sig.loc[effect, cause]:
                    rows.append(
                        {
                            "cause": cause,
                            "effect": effect,
                            "p_value": self.pvalues.loc[effect, cause],
                            "statistic": self.statistics.loc[effect, cause],
                            "lag": self.lags.loc[effect, cause],
                        }
                    )
        return pd.DataFrame(rows, columns=["cause", "effect", "p_value",
                                           "statistic", "lag"])

    def to_frame(self) -> pd.DataFrame:
        """Long-format table of every directional pair (including diagonal=NaN)."""
        recs = []
        for effect in self.pvalues.index:
            for cause in self.pvalues.columns:
                recs.append(
                    {
                        "cause": cause,
                        "effect": effect,
                        "p_value": self.pvalues.loc[effect, cause],
                        "statistic": self.statistics.loc[effect, cause],
                        "lag": self.lags.loc[effect, cause],
                    }
                )
        return pd.DataFrame(recs)

    def plot(self, **kwargs):
        """Convenience wrapper around :func:`grangercm.plot.plot_gcm`."""
        from .plot import plot_gcm

        return plot_gcm(self, **kwargs)


def _correct(pvals: np.ndarray, method: str) -> np.ndarray:
    """Apply multiple-testing correction to off-diagonal p-values only."""
    from statsmodels.stats.multitest import multipletests

    flat = pvals.flatten()
    finite = np.isfinite(flat)
    corrected = flat.copy()
    if finite.sum() > 0:
        corrected[finite] = multipletests(flat[finite], method=method)[1]
    return corrected.reshape(pvals.shape)


def granger_matrix(
    data: pd.DataFrame,
    variables: Iterable[str] | None = None,
    maxlag: int | Sequence[int] = 5,
    test: str = "ssr_ftest",
    aggregate: str = "min",
    correction: str | None = None,
    verbose: bool = False,
) -> GCMatrixResult:
    """Compute the pairwise Granger Causality Matrix for a set of time series.

    Parameters
    ----------
    data : pandas.DataFrame
        Wide-format data; each column is one (already stationary) time series.
        Use :func:`grangercm.prep.make_stationary` first if needed.
    variables : iterable of str, optional
        Subset of columns to test. Defaults to all numeric columns.
    maxlag : int or sequence of int
        Lags passed to the underlying test. If an int, lags ``1..maxlag`` are
        tested and aggregated; if a sequence, exactly those lags are tested.
    test : {'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'}
        Which statistic from statsmodels' grangercausalitytests to extract.
    aggregate : {'min', 'mean', 'median', 'max'}
        How to collapse the per-lag p-values into a single matrix entry.
        ``'min'`` matches the most common convention in the literature.
    correction : {None, 'bonferroni', 'holm', 'fdr_bh', 'fdr_by'}
        Multiple-testing correction applied across all off-diagonal cells.
    verbose : bool
        Print progress for each tested pair.

    Returns
    -------
    GCMatrixResult
        Holds the p-value matrix, statistic matrix, selected-lag matrix and
        metadata. Rows are effects, columns are causes.
    """
    if test not in _VALID_TESTS:
        raise ValueError(f"test must be one of {_VALID_TESTS}, got {test!r}")
    if aggregate not in _VALID_AGG:
        raise ValueError(f"aggregate must be one of {_VALID_AGG}, got {aggregate!r}")
    if correction not in _VALID_CORR:
        raise ValueError(f"correction must be one of {_VALID_CORR}, got {correction!r}")

    from statsmodels.tsa.stattools import grangercausalitytests

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
    variables = list(variables)
    if len(variables) < 2:
        raise ValueError("Need at least 2 variables to build a matrix.")

    lags = list(range(1, maxlag + 1)) if isinstance(maxlag, int) else list(maxlag)
    agg_fn = {"min": np.min, "mean": np.mean,
              "median": np.median, "max": np.max}[aggregate]

    n = len(variables)
    pv = pd.DataFrame(np.full((n, n), np.nan), index=variables, columns=variables)
    st = pd.DataFrame(np.full((n, n), np.nan), index=variables, columns=variables)
    lg = pd.DataFrame(np.full((n, n), np.nan), index=variables, columns=variables)

    for effect, cause in permutations(variables, 2):
        # statsmodels: column 0 is the series being predicted (effect),
        # column 1 is the candidate causal series (cause).
        pair = data[[effect, cause]].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(pair, maxlag=lags, verbose=False)
        per_lag_p = np.array([res[L][0][test][1] for L in lags])
        per_lag_s = np.array([res[L][0][test][0] for L in lags])
        idx = int(np.argmin(per_lag_p))  # lag driving the aggregate decision
        pv.loc[effect, cause] = agg_fn(per_lag_p)
        st.loc[effect, cause] = per_lag_s[idx]
        lg.loc[effect, cause] = lags[idx]
        if verbose:
            print(f"{cause} -> {effect}: p={pv.loc[effect, cause]:.4g} "
                  f"(lag {lags[idx]})")

    pv_raw = pv.copy()
    if correction is not None:
        pv = pd.DataFrame(_correct(pv.values, correction),
                          index=variables, columns=variables)

    return GCMatrixResult(
        pvalues=pv, statistics=st, lags=lg, test=test, maxlag=maxlag,
        aggregate=aggregate, correction=correction, pvalues_raw=pv_raw,
    )
