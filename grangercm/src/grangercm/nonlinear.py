"""Nonlinear Granger causality backend.

The linear Granger test in :mod:`grangercm.core` asks whether a linear
autoregression of ``effect`` improves when lagged ``cause`` terms are added.
A nonlinear relationship (``cause`` influences ``effect`` through a curve, a
threshold, an interaction) can be invisible to that test.

This module provides a model-agnostic nonlinear analogue. For each ordered
pair it fits two predictors of ``effect``:

* a *restricted* model using only lagged ``effect`` terms;
* an *unrestricted* model using lagged ``effect`` **and** lagged ``cause``.

If the unrestricted model predicts out-of-sample significantly better, we say
``cause`` nonlinearly Granger-causes ``effect``. Significance is assessed with
a one-sided Diebold-Mariano-style test on the per-observation squared errors,
so no distributional assumption on the estimator is needed.

The default estimator is a small scikit-learn ``MLPRegressor``; any regressor
exposing ``fit`` / ``predict`` can be supplied via ``estimator``.
"""

from __future__ import annotations

import warnings
from itertools import permutations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .core import _VALID_AGG, _VALID_CORR, GCMatrixResult, _correct


def _lag_matrix(series: np.ndarray, lag: int) -> np.ndarray:
    """Stack columns [t-1, t-2, ..., t-lag] for a 1-D series."""
    return np.column_stack([np.roll(series, k) for k in range(1, lag + 1)])


def _build_design(effect: np.ndarray, cause: np.ndarray, lag: int):
    """Return (y, X_restricted, X_unrestricted) aligned and trimmed of warmup."""
    y = effect[lag:]
    e_lags = _lag_matrix(effect, lag)[lag:]
    c_lags = _lag_matrix(cause, lag)[lag:]
    return y, e_lags, np.hstack([e_lags, c_lags])


def _dm_statistic(err_restricted: np.ndarray, err_unrestricted: np.ndarray) -> float:
    """Mean loss differential: how much the unrestricted model reduces MSE.

    Positive means the model that also saw lagged ``cause`` predicts better.
    Used as the test statistic for the permutation test below.
    """
    return float(np.mean(err_restricted**2 - err_unrestricted**2))


def _default_estimator():
    from sklearn.neural_network import MLPRegressor

    return MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        random_state=0,
    )


def _blocked_splits(n: int, n_folds: int):
    """Yield (train_idx, test_idx) for contiguous-block CV.

    Each fold's test set is one contiguous block; the model trains on all
    other observations. Blocks (rather than random folds) respect the
    temporal ordering of the lagged design.
    """
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    for k in range(n_folds):
        lo, hi = edges[k], edges[k + 1]
        test = np.arange(lo, hi)
        train = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        if len(train) >= 8 and len(test) >= 4:
            yield train, test


def nonlinear_granger_matrix(
    data: pd.DataFrame,
    variables: Iterable[str] | None = None,
    maxlag: int | Sequence[int] = 5,
    estimator=None,
    aggregate: str = "min",
    correction: str | None = None,
    n_folds: int = 5,
    n_permutations: int = 200,
    block_size: int | None = None,
    random_state: int = 0,
    verbose: bool = False,
) -> GCMatrixResult:
    """Compute a nonlinear Granger Causality Matrix.

    Drop-in analogue of :func:`grangercm.granger_matrix` that replaces the
    linear autoregression with an arbitrary nonlinear estimator and assesses
    significance with a block-permutation test instead of an in-sample
    F-test.

    For each ordered pair the function compares two cross-validated
    predictors of ``effect``: a *restricted* model using only lagged
    ``effect`` and an *unrestricted* model that also sees lagged ``cause``.
    The test statistic is the out-of-sample MSE reduction. Its null
    distribution is built by repeatedly **block-shuffling** the ``cause``
    series and recomputing the statistic. Block shuffling preserves the
    autocorrelation of ``cause`` while destroying its temporal alignment with
    ``effect``, so the null is "``cause`` carries no predictive information
    about ``effect`` beyond chance." A permutation null is used rather than an
    analytic Diebold-Mariano variance because a re-estimated flexible learner
    adds estimation variance that the analytic test does not capture -- which
    otherwise produces spurious causality when the effect series is close to
    noise.

    Parameters
    ----------
    data : pandas.DataFrame
        Wide-format stationary time series, one per column.
    variables : iterable of str, optional
        Subset of columns; defaults to all numeric columns.
    maxlag : int or sequence of int
        Lag orders to test (aggregated, as in the linear version).
    estimator : scikit-learn-style regressor, optional
        Any object with ``fit(X, y)`` / ``predict(X)``. A fresh clone is fit
        per (pair, lag, fold, model). Defaults to a small ``MLPRegressor``.
    aggregate : {'min', 'mean', 'median', 'max'}
        How to collapse per-lag p-values.
    correction : {None, 'bonferroni', 'holm', 'fdr_bh', 'fdr_by'}
        Multiple-testing correction across off-diagonal cells.
    n_folds : int
        Number of contiguous CV blocks for the out-of-sample comparison.
    n_permutations : int
        Number of block-shuffled replicates forming the null distribution.
        The smallest attainable p-value is ``1 / (n_permutations + 1)``.
    block_size : int, optional
        Length of the contiguous blocks used when shuffling ``cause``.
        Defaults to ``max(10, n // 20)``.
    random_state : int
        Seed for both the estimators and the permutation shuffling.
    verbose : bool
        Print progress per pair.

    Returns
    -------
    GCMatrixResult
        Same container as the linear version; ``test`` is
        ``"nonlinear_perm"`` and ``statistics`` holds the observed MSE
        reduction.

    Notes
    -----
    Cost is roughly ``n_folds * (n_permutations + 1)`` model fits per pair per
    lag. For wide panels or expensive estimators, lower ``n_permutations``
    (100 is often enough for screening), restrict ``variables``/``maxlag``,
    or pass a cheaper ``estimator`` (e.g. ``sklearn.linear_model.Ridge``).
    """
    if aggregate not in _VALID_AGG:
        raise ValueError(f"aggregate must be one of {_VALID_AGG}")
    if correction not in _VALID_CORR:
        raise ValueError(f"correction must be one of {_VALID_CORR}")

    try:
        from sklearn.base import clone
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "nonlinear_granger_matrix requires scikit-learn. "
            "Install with: pip install 'grangercm[nonlinear]'"
        ) from exc

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
    variables = list(variables)
    if len(variables) < 2:
        raise ValueError("Need at least 2 variables to build a matrix.")

    lags = list(range(1, maxlag + 1)) if isinstance(maxlag, int) else list(maxlag)
    agg_fn = {"min": np.min, "mean": np.mean,
              "median": np.median, "max": np.max}[aggregate]
    base = estimator if estimator is not None else _default_estimator()
    rng = np.random.default_rng(random_state)

    n = len(variables)
    pv = pd.DataFrame(np.full((n, n), np.nan), index=variables, columns=variables)
    st = pd.DataFrame(np.full((n, n), np.nan), index=variables, columns=variables)
    lg = pd.DataFrame(np.full((n, n), np.nan), index=variables, columns=variables)

    def _cv_errors(X, y):
        """Held-out residuals for every observation via blocked CV."""
        preds = np.full(len(y), np.nan)
        for train, test in _blocked_splits(len(y), n_folds):
            model = clone(base)
            if hasattr(model, "random_state"):
                model.set_params(random_state=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X[train], y[train])
                preds[test] = model.predict(X[test])
        mask = ~np.isnan(preds)
        return mask, preds

    def _block_shuffle(series: np.ndarray, blk: int) -> np.ndarray:
        """Permute a series in contiguous blocks, preserving local structure."""
        n_blk = int(np.ceil(len(series) / blk))
        order = rng.permutation(n_blk)
        chunks = [series[i * blk:(i + 1) * blk] for i in order]
        return np.concatenate(chunks)[: len(series)]

    for effect, cause in permutations(variables, 2):
        e = data[effect].to_numpy(dtype=float)
        c = data[cause].to_numpy(dtype=float)
        blk = block_size or max(10, len(e) // 20)
        per_p, per_s = [], []
        for L in lags:
            y, e_lags, _ = _build_design(e, c, L)
            if len(y) < 4 * n_folds:
                per_p.append(np.nan)
                per_s.append(np.nan)
                continue

            # Restricted residuals depend only on lagged effect -> compute once.
            mask_r, pred_r = _cv_errors(e_lags, y)

            def _stat_for(cause_series):
                c_lags = _lag_matrix(cause_series, L)[L:]
                Xu = np.hstack([e_lags, c_lags])
                mask_u, pred_u = _cv_errors(Xu, y)
                m = mask_r & mask_u
                err_r = (y - pred_r)[m]
                err_u = (y - pred_u)[m]
                return _dm_statistic(err_r, err_u)

            observed = _stat_for(c)
            null = np.array([_stat_for(_block_shuffle(c, blk))
                             for _ in range(n_permutations)])
            # One-sided permutation p-value with the +1 correction.
            p = (1 + np.sum(null >= observed)) / (n_permutations + 1)
            per_p.append(p)
            per_s.append(observed)

        per_p = np.array(per_p, dtype=float)
        per_s = np.array(per_s, dtype=float)
        if np.all(np.isnan(per_p)):
            continue
        idx = int(np.nanargmin(per_p))
        pv.loc[effect, cause] = agg_fn(per_p[~np.isnan(per_p)])
        st.loc[effect, cause] = per_s[idx]
        lg.loc[effect, cause] = lags[idx]
        if verbose:
            print(f"{cause} -> {effect}: p={pv.loc[effect, cause]:.4g} "
                  f"(lag {lags[idx]})")

    pv_raw = pv.copy()
    if correction is not None:
        pv = pd.DataFrame(_correct(pv.values, correction),
                          index=variables, columns=variables)

    return GCMatrixResult(
        pvalues=pv, statistics=st, lags=lg, test="nonlinear_perm",
        maxlag=maxlag, aggregate=aggregate, correction=correction,
        pvalues_raw=pv_raw,
    )
