#' Compute a nonlinear Granger Causality Matrix
#'
#' Nonlinear analogue of \code{\link{granger_matrix}}. For every ordered pair
#' it compares two cross-validated predictors of \code{effect}: a
#' \emph{restricted} model using only lagged \code{effect}, and an
#' \emph{unrestricted} model that also sees lagged \code{cause}. The test
#' statistic is the out-of-sample mean-squared-error reduction; its null
#' distribution is built by block-shuffling the \code{cause} series, which
#' preserves its autocorrelation while destroying temporal alignment with
#' \code{effect}.
#'
#' A permutation null is used (rather than an analytic test) because a
#' re-estimated flexible learner adds estimation variance that an analytic
#' variance does not capture; without it, near-noise effect series can show
#' spurious causality.
#'
#' @param data A data frame or matrix of stationary numeric time series.
#' @param variables Optional character vector of columns to test. Defaults to
#'   all numeric columns.
#' @param maxlag Single integer (lags \code{1:maxlag}) or an integer vector of
#'   specific lags.
#' @param learner A function \code{function(X, y)} returning a fitted model
#'   that supports \code{predict(model, newdata)}. Defaults to a regression
#'   tree via \code{\link[stats]{loess}}-free base R (see Details). Supply
#'   your own to use random forests, neural nets, splines, etc.
#' @param aggregate How to collapse per-lag p-values: \code{"min"},
#'   \code{"mean"}, \code{"median"} or \code{"max"}.
#' @param correction Multiple-testing correction across off-diagonal cells;
#'   passed to \code{\link[stats]{p.adjust}}.
#' @param n_folds Number of contiguous cross-validation blocks.
#' @param n_permutations Number of block-shuffled replicates in the null.
#'   The smallest attainable p-value is \code{1 / (n_permutations + 1)}.
#' @param block_size Length of the contiguous blocks used when shuffling
#'   \code{cause}. Defaults to \code{max(10, n \%/\% 20)}.
#' @param seed Integer seed for reproducible folds and permutations.
#' @param verbose Logical; print progress per pair.
#'
#' @return A \code{"gcmatrix"} object (see \code{\link{granger_matrix}}),
#'   with \code{test = "nonlinear_perm"} and \code{statistics} holding the
#'   observed MSE reduction.
#'
#' @details
#' The default \code{learner} is a shallow regression tree built with
#' \code{rpart} if available, otherwise a multiple linear model (which makes
#' the result a slow linear test -- install \pkg{rpart} or pass your own
#' nonlinear \code{learner} for genuine nonlinear detection).
#'
#' A learner is any closure of the form
#' \preformatted{
#'   my_learner <- function(X, y) {
#'     fit <- some_model(y ~ ., data = data.frame(y = y, X))
#'     function(newX) predict(fit, newdata = as.data.frame(newX))
#'   }
#' }
#'
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 300
#' x <- rnorm(n)
#' y <- numeric(n)
#' for (t in 2:n) y[t] <- 0.8 * tanh(2 * x[t - 1]) + 0.2 * y[t - 1] +
#'                        rnorm(1, sd = 0.3)
#' df <- data.frame(x = x, y = y, z = rnorm(n))
#' gcm <- nonlinear_granger_matrix(df, maxlag = 1, n_permutations = 99)
#' print(gcm)
#' }
#'
#' @seealso \code{\link{granger_matrix}} for the linear version.
#' @importFrom stats p.adjust median na.omit predict lm
#' @export
nonlinear_granger_matrix <- function(data,
                                     variables = NULL,
                                     maxlag = 5,
                                     learner = NULL,
                                     aggregate = c("min", "mean",
                                                   "median", "max"),
                                     correction = c("none", "bonferroni",
                                                    "holm", "BH", "BY"),
                                     n_folds = 5L,
                                     n_permutations = 200L,
                                     block_size = NULL,
                                     seed = 1L,
                                     verbose = FALSE) {
  aggregate <- match.arg(aggregate)
  correction <- match.arg(correction)
  set.seed(seed)

  data <- as.data.frame(data)
  num_cols <- names(data)[vapply(data, is.numeric, logical(1))]
  if (is.null(variables)) variables <- num_cols
  variables <- intersect(variables, num_cols)
  if (length(variables) < 2L) {
    stop("Need at least 2 numeric variables to build a matrix.", call. = FALSE)
  }

  lags <- if (length(maxlag) == 1L) seq_len(maxlag) else as.integer(maxlag)
  agg_fn <- switch(aggregate, min = min, mean = mean,
                   median = stats::median, max = max)
  if (is.null(learner)) learner <- .default_learner()

  n <- length(variables)
  pv <- st <- lg <- matrix(NA_real_, n, n,
                           dimnames = list(variables, variables))

  for (effect in variables) {
    for (cause in variables) {
      if (effect == cause) next
      e <- data[[effect]]
      c <- data[[cause]]
      blk <- if (is.null(block_size)) max(10L, length(e) %/% 20L) else block_size
      per_p <- per_s <- numeric(length(lags))

      for (li in seq_along(lags)) {
        L <- lags[li]
        design <- .build_design(e, c, L)
        y <- design$y
        if (length(y) < 4L * n_folds) {
          per_p[li] <- NA_real_; per_s[li] <- NA_real_; next
        }
        # Restricted predictions reuse only lagged effect.
        pred_r <- .cv_predict(design$e_lags, y, learner, n_folds)

        stat_for <- function(cause_series) {
          c_lags <- .lag_matrix(cause_series, L)
          c_lags <- c_lags[(L + 1):nrow(c_lags), , drop = FALSE]
          Xu <- cbind(design$e_lags, c_lags)
          pred_u <- .cv_predict(Xu, y, learner, n_folds)
          ok <- !is.na(pred_r) & !is.na(pred_u)
          mean((y[ok] - pred_r[ok])^2 - (y[ok] - pred_u[ok])^2)
        }

        observed <- stat_for(c)
        null <- vapply(seq_len(n_permutations),
                       function(i) stat_for(.block_shuffle(c, blk)),
                       numeric(1))
        per_p[li] <- (1 + sum(null >= observed)) / (n_permutations + 1)
        per_s[li] <- observed
      }

      if (all(is.na(per_p))) next
      best <- which.min(per_p)
      pv[effect, cause] <- agg_fn(per_p[!is.na(per_p)])
      st[effect, cause] <- per_s[best]
      lg[effect, cause] <- lags[best]
      if (verbose) {
        message(sprintf("%s -> %s: p=%.4g (lag %d)",
                        cause, effect, pv[effect, cause], lags[best]))
      }
    }
  }

  pv_raw <- pv
  if (correction != "none") {
    off <- !diag(TRUE, n)
    pv[off] <- stats::p.adjust(pv[off], method = correction)
  }

  structure(
    list(pvalues = pv, statistics = st, lags = lg, pvalues_raw = pv_raw,
         variables = variables, maxlag = maxlag, aggregate = aggregate,
         correction = correction, test = "nonlinear_perm"),
    class = "gcmatrix"
  )
}

# --- internal helpers --------------------------------------------------------

# Column-bind [t-1, ..., t-lag] for a numeric vector.
.lag_matrix <- function(series, lag) {
  n <- length(series)
  out <- matrix(NA_real_, n, lag)
  for (k in seq_len(lag)) out[(k + 1):n, k] <- series[1:(n - k)]
  out
}

# Aligned response and lagged-effect design, warmup rows dropped.
.build_design <- function(effect, cause, lag) {
  e_lags <- .lag_matrix(effect, lag)
  keep <- (lag + 1):length(effect)
  list(y = effect[keep], e_lags = e_lags[keep, , drop = FALSE])
}

# Held-out predictions via contiguous-block cross-validation.
.cv_predict <- function(X, y, learner, n_folds) {
  n <- length(y)
  edges <- floor(seq(0, n, length.out = n_folds + 1))
  preds <- rep(NA_real_, n)
  for (k in seq_len(n_folds)) {
    test <- (edges[k] + 1):edges[k + 1]
    train <- setdiff(seq_len(n), test)
    if (length(train) < 8L || length(test) < 4L) next
    predict_fn <- learner(X[train, , drop = FALSE], y[train])
    preds[test] <- predict_fn(X[test, , drop = FALSE])
  }
  preds
}

# Permute a vector in contiguous blocks (preserves local autocorrelation).
.block_shuffle <- function(series, blk) {
  n <- length(series)
  n_blk <- ceiling(n / blk)
  ord <- sample(n_blk)
  chunks <- lapply(ord, function(i) {
    idx <- ((i - 1) * blk + 1):min(i * blk, n)
    series[idx]
  })
  unlist(chunks, use.names = FALSE)[seq_len(n)]
}

# Default learner: shallow regression tree (rpart) or linear fallback.
.default_learner <- function() {
  if (requireNamespace("rpart", quietly = TRUE)) {
    function(X, y) {
      df <- data.frame(y = y, X)
      fit <- rpart::rpart(y ~ ., data = df,
                          control = rpart::rpart.control(maxdepth = 5L))
      function(newX) {
        nd <- as.data.frame(newX)
        names(nd) <- names(df)[-1]
        as.numeric(stats::predict(fit, newdata = nd))
      }
    }
  } else {
    warning("Package 'rpart' not installed; falling back to a linear learner. ",
            "Install 'rpart' or pass a nonlinear 'learner' for genuine ",
            "nonlinear detection.", call. = FALSE)
    function(X, y) {
      df <- data.frame(y = y, X)
      fit <- stats::lm(y ~ ., data = df)
      function(newX) {
        nd <- as.data.frame(newX)
        names(nd) <- names(df)[-1]
        as.numeric(stats::predict(fit, newdata = nd))
      }
    }
  }
}
