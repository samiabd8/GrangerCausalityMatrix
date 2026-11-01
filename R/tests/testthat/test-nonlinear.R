poly_learner <- function(X, y) {
  X <- as.matrix(X)
  feats <- cbind(X, X^2)
  fit <- lm.fit(cbind(1, feats), y)
  beta <- fit$coefficients
  beta[is.na(beta)] <- 0
  function(newX) {
    nX <- as.matrix(newX)
    as.numeric(cbind(1, cbind(nX, nX^2)) %*% beta)
  }
}

make_nl_data <- function(n = 350, seed = 1) {
  set.seed(seed)
  x <- rnorm(n)
  y <- numeric(n)
  for (t in 2:n) {
    y[t] <- 0.8 * tanh(2 * x[t - 1]) + 0.2 * y[t - 1] + rnorm(1, sd = 0.3)
  }
  data.frame(x = x, y = y, z = rnorm(n))
}

test_that("nonlinear matrix has correct shape and class", {
  gcm <- nonlinear_granger_matrix(make_nl_data(), maxlag = 1,
                                  learner = poly_learner,
                                  n_permutations = 49, n_folds = 4)
  expect_s3_class(gcm, "gcmatrix")
  expect_equal(dim(gcm$pvalues), c(3, 3))
  expect_equal(gcm$test, "nonlinear_perm")
  expect_true(all(is.na(diag(gcm$pvalues))))
})

test_that("nonlinear p-values lie in (0, 1]", {
  gcm <- nonlinear_granger_matrix(make_nl_data(), maxlag = 1,
                                  learner = poly_learner,
                                  n_permutations = 49, n_folds = 4)
  p <- gcm$pvalues[!is.na(gcm$pvalues)]
  expect_true(all(p > 0 & p <= 1))
})

test_that("nonlinear test detects the known causal direction", {
  gcm <- nonlinear_granger_matrix(make_nl_data(n = 400), maxlag = 1,
                                  learner = poly_learner,
                                  n_permutations = 99, n_folds = 4)
  expect_lt(gcm$pvalues["y", "x"], 0.10)   # x -> y picked up
  expect_gt(gcm$pvalues["x", "y"], 0.10)   # y -> x not flagged
})

test_that("nonlinear result works with shared helpers", {
  gcm <- nonlinear_granger_matrix(make_nl_data(), maxlag = 1,
                                  learner = poly_learner,
                                  n_permutations = 49, n_folds = 4)
  e <- gcm_edges(gcm, alpha = 0.10)
  expect_named(e, c("cause", "effect", "p_value", "statistic", "lag"))
})
