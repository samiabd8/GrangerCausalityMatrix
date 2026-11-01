test_that("matrix has correct shape and NA diagonal", {
  set.seed(1)
  df <- data.frame(a = rnorm(200), b = rnorm(200), c = rnorm(200))
  gcm <- granger_matrix(df, maxlag = 3)
  expect_s3_class(gcm, "gcmatrix")
  expect_equal(dim(gcm$pvalues), c(3, 3))
  expect_true(all(is.na(diag(gcm$pvalues))))
})

test_that("known causal direction is detected", {
  set.seed(42)
  n <- 400
  x <- rnorm(n)
  y <- numeric(n)
  for (t in 2:n) y[t] <- 0.6 * x[t - 1] + 0.2 * y[t - 1] + rnorm(1, sd = 0.3)
  z <- rnorm(n)
  df <- data.frame(x = x, y = y, z = z)
  gcm <- granger_matrix(df, maxlag = 3)
  expect_lt(gcm$pvalues["y", "x"], 0.05)   # x -> y significant
  expect_gt(gcm$pvalues["x", "y"], 0.05)   # y -> x not significant
})

test_that("correction does not decrease p-values", {
  set.seed(7)
  df <- data.frame(a = rnorm(300), b = rnorm(300), c = rnorm(300))
  raw  <- granger_matrix(df, maxlag = 3, correction = "none")
  corr <- granger_matrix(df, maxlag = 3, correction = "bonferroni")
  off <- !is.na(raw$pvalues)
  expect_true(all(corr$pvalues[off] >= raw$pvalues[off] - 1e-9))
})

test_that("gcm_edges returns expected columns", {
  set.seed(1)
  df <- data.frame(a = rnorm(200), b = rnorm(200), c = rnorm(200))
  e <- gcm_edges(granger_matrix(df, maxlag = 3))
  expect_named(e, c("cause", "effect", "p_value", "statistic", "lag"))
})

test_that("invalid input errors", {
  df <- data.frame(a = rnorm(100))
  expect_error(granger_matrix(df), "at least 2")
})

test_that("explicit lag vector is respected", {
  set.seed(3)
  df <- data.frame(a = rnorm(250), b = rnorm(250), c = rnorm(250))
  gcm <- granger_matrix(df, maxlag = c(2L, 4L))
  used <- gcm$lags[!is.na(gcm$lags)]
  expect_true(all(used %in% c(2, 4)))
})
