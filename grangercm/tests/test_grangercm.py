"""Tests for grangercm."""

import numpy as np
import pandas as pd
import pytest

from grangercm import adf_report, granger_matrix, make_stationary


def _causal_data(n=400, seed=0):
    """x drives y at lag 1; z is independent noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.6 * x[t - 1] + 0.2 * y[t - 1] + rng.normal(scale=0.3)
    z = rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def test_matrix_shape_and_diagonal():
    res = granger_matrix(_causal_data(), maxlag=3)
    assert res.pvalues.shape == (3, 3)
    assert np.all(np.isnan(np.diag(res.pvalues.values)))  # diagonal is NaN


def test_detects_known_direction():
    res = granger_matrix(_causal_data(), maxlag=3)
    # x -> y should be significant; y -> x should not
    assert res.pvalues.loc["y", "x"] < 0.05
    assert res.pvalues.loc["x", "y"] > 0.05


def test_correction_increases_pvalues():
    raw = granger_matrix(_causal_data(), maxlag=3, correction=None)
    corr = granger_matrix(_causal_data(), maxlag=3, correction="bonferroni")
    finite = np.isfinite(raw.pvalues.values)
    assert np.all(corr.pvalues.values[finite] >= raw.pvalues.values[finite] - 1e-9)


def test_edges_table():
    res = granger_matrix(_causal_data(), maxlag=3)
    edges = res.edges(alpha=0.05)
    assert set(edges.columns) == {"cause", "effect", "p_value", "statistic", "lag"}
    assert (("x", "y") in set(zip(edges.cause, edges.effect)))


def test_invalid_args():
    df = _causal_data()
    with pytest.raises(ValueError):
        granger_matrix(df, test="bogus")
    with pytest.raises(ValueError):
        granger_matrix(df, aggregate="bogus")
    with pytest.raises(ValueError):
        granger_matrix(df[["x"]])  # too few variables


def test_make_stationary():
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(size=300))
    df = pd.DataFrame({"a": walk, "b": np.cumsum(rng.normal(size=300))})
    stat, orders = make_stationary(df, method="diff")
    assert all(o >= 1 for o in orders.values())
    rep = adf_report(stat)
    assert rep["stationary"].all()


def test_explicit_lag_list():
    res = granger_matrix(_causal_data(), maxlag=[2, 4])
    vals = res.lags.values.flatten()
    vals = vals[np.isfinite(vals)]
    assert set(vals).issubset({2.0, 4.0})


# --- nonlinear backend -------------------------------------------------------

def _nonlinear_data(n=300, seed=0):
    """x drives y through a nonlinear (tanh) lag-1 channel; z is noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.8 * np.tanh(2 * x[t - 1]) + 0.2 * y[t - 1] + rng.normal(scale=0.3)
    z = rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def _fast_kwargs():
    """Cheap settings so the test suite stays quick."""
    from sklearn.tree import DecisionTreeRegressor
    return dict(
        maxlag=[1],
        estimator=DecisionTreeRegressor(max_depth=4, random_state=0),
        n_permutations=49,
        n_folds=4,
    )


def test_nonlinear_matrix_shape():
    from grangercm import nonlinear_granger_matrix
    res = nonlinear_granger_matrix(_nonlinear_data(), **_fast_kwargs())
    assert res.pvalues.shape == (3, 3)
    assert res.test == "nonlinear_perm"
    assert np.all(np.isnan(np.diag(res.pvalues.values)))


def test_nonlinear_pvalues_in_unit_interval():
    from grangercm import nonlinear_granger_matrix
    res = nonlinear_granger_matrix(_nonlinear_data(), **_fast_kwargs())
    vals = res.pvalues.values.flatten()
    vals = vals[np.isfinite(vals)]
    assert np.all((vals > 0) & (vals <= 1))


def test_nonlinear_detects_direction():
    from sklearn.ensemble import RandomForestRegressor

    from grangercm import nonlinear_granger_matrix
    res = nonlinear_granger_matrix(
        _nonlinear_data(n=400),
        maxlag=[1],
        estimator=RandomForestRegressor(n_estimators=40, random_state=0),
        n_permutations=99,
        n_folds=4,
    )
    # x -> y is the true nonlinear channel; y -> x must not flag.
    assert res.pvalues.loc["y", "x"] < 0.10
    assert res.pvalues.loc["x", "y"] > 0.10


def test_nonlinear_shares_result_api():
    from grangercm import nonlinear_granger_matrix
    res = nonlinear_granger_matrix(_nonlinear_data(), **_fast_kwargs())
    edges = res.edges(alpha=0.10)
    assert set(edges.columns) == {"cause", "effect", "p_value", "statistic", "lag"}
