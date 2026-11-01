# grangercm

**Granger Causality Matrices and heatmaps for multivariate time series.**

`grangercm` computes pairwise Granger causality across every pair of series in
a dataset and returns it as a labeled matrix — a *Granger Causality Matrix*
(GCM) — that reads like a correlation table. It also draws the matrix as a
p-value heatmap.

## Why

`statsmodels` gives you a bivariate Granger test. Turning that into an
all-pairs matrix, handling stationarity, picking lags, and correcting for the
N²−N simultaneous tests is left to copy-pasted notebook snippets. `grangercm`
packages the whole workflow.

## Install

```bash
pip install grangercm          # core
pip install "grangercm[plot]"  # with heatmap support
```

## Quickstart

```python
import pandas as pd
from grangercm import make_stationary, adf_report, granger_matrix

df = pd.read_csv("prices.csv", index_col=0)

# 1. Make the series stationary (log-returns + differencing)
stat_df, orders = make_stationary(df, method="log_diff")
print(adf_report(stat_df))

# 2. Build the Granger Causality Matrix
result = granger_matrix(
    stat_df,
    maxlag=5,            # test lags 1..5, aggregate per pair
    test="ssr_ftest",
    aggregate="min",     # collapse lags by min p-value
    correction="fdr_bh", # Benjamini-Hochberg across all cells
)

# 3. Inspect
print(result.pvalues)            # the matrix: rows=effect, cols=cause
print(result.edges(alpha=0.05))  # significant cause -> effect links

# 4. Visualize
result.plot()                    # heatmap
```

## Reading the matrix

Entry `[effect, cause]` is the p-value for "**cause** Granger-causes
**effect**". Rows are effects (predicted series), columns are causes
(predictors). The diagonal is undefined (`NaN`). A low p-value means the
column series helps predict the row series beyond the row's own past.

## Nonlinear Granger causality

The linear test can miss relationships that act through a curve, threshold,
or interaction. `nonlinear_granger_matrix` replaces the autoregression with
any scikit-learn-style estimator and assesses significance with a
block-permutation test:

```python
from grangercm import nonlinear_granger_matrix
from sklearn.ensemble import RandomForestRegressor

nl = nonlinear_granger_matrix(
    stat_df,
    maxlag=[2],
    estimator=RandomForestRegressor(n_estimators=100, random_state=0),
    n_permutations=200,
    n_folds=5,
)
nl.plot()
```

It returns the same `GCMatrixResult` object, so `.edges()`, `.plot()` and the
rest of the API work unchanged. Install the extra with
`pip install "grangercm[nonlinear]"`.

## Notes & caveats

- Granger causality is predictive, not structural causality.
- Inputs must be (weakly) stationary — use `make_stationary` first.
- `aggregate="min"` is the common literature convention but is anti-conservative;
  pair it with `correction` for honest inference.
- The nonlinear backend is much more expensive
  (`n_folds * (n_permutations + 1)` model fits per pair per lag); restrict
  `variables`, `maxlag`, or `n_permutations` for wide panels.

## Documentation

A full worked walkthrough is in `docs/getting_started.ipynb` (rebuild it with
`python docs/build_vignette.py`).

## License

MIT
