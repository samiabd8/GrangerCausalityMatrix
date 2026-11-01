# gcmatrix

**Granger Causality Matrices for multivariate time series in R.**

`gcmatrix` computes pairwise Granger causality across every pair of series in
a dataset and returns it as a labeled matrix — a *Granger Causality Matrix* —
that reads like a correlation table, with a built-in p-value heatmap.

## Why

`lmtest::grangertest()` is bivariate. `vars::causality()` works off a single
fitted VAR. Neither gives you a directional all-pairs p-value matrix with
stationarity prep, lag selection, and multiple-testing correction. `gcmatrix`
packages that workflow.

## Install

```r
# install.packages("remotes")
remotes::install_github("yourname/gcmatrix")
```

## Quickstart

```r
library(gcmatrix)

df <- read.csv("prices.csv")[-1]   # drop a date column

# 1. Make the series stationary (log-returns + differencing)
prep <- make_stationary(df, method = "log_diff")
adf_report(prep$data)

# 2. Build the Granger Causality Matrix
gcm <- granger_matrix(
  prep$data,
  maxlag     = 5,       # test lags 1:5, aggregate per pair
  aggregate  = "min",   # collapse lags by min p-value
  correction = "BH"     # Benjamini-Hochberg across all cells
)

# 3. Inspect
print(gcm)              # the matrix
summary(gcm)            # significant links
gcm_edges(gcm, 0.05)    # data frame of cause -> effect edges

# 4. Visualize
plot(gcm)               # p-value heatmap
```

## Reading the matrix

Entry `[effect, cause]` is the p-value for "**cause** Granger-causes
**effect**". Rows are effects (predicted series); columns are causes
(predictors). The diagonal is `NA`.

## Nonlinear Granger causality

The linear test can miss relationships acting through a curve, threshold, or
interaction. `nonlinear_granger_matrix()` replaces the autoregression with any
`learner` (a closure returning a fitted predictor) and assesses significance
with a block-permutation test:

```r
nl <- nonlinear_granger_matrix(
  prep$data,
  maxlag         = 2,
  n_permutations = 200,
  n_folds        = 5
)
plot(nl)
```

It returns the same `"gcmatrix"` object, so `summary()`, `plot()`,
`gcm_edges()` all work unchanged. The default learner is a regression tree
(`rpart`); pass your own `learner` for random forests, neural nets, splines,
etc. See `?nonlinear_granger_matrix` for the learner interface.

## Caveats

- Granger causality is predictive, not structural causality.
- Inputs must be (weakly) stationary — use `make_stationary()` first.
- `aggregate = "min"` is conventional but anti-conservative; pair it with a
  `correction`.
- The nonlinear backend is much more expensive
  (`n_folds * (n_permutations + 1)` model fits per pair per lag); restrict
  `variables`, `maxlag`, or `n_permutations` for wide panels.

## Documentation

See the *Getting started* vignette: `vignette("getting-started", "gcmatrix")`.

## License

MIT
