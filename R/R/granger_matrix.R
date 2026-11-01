#' Compute a Granger Causality Matrix
#'
#' Computes pairwise Granger causality across every ordered pair of columns in
#' a multivariate time series and returns the result as a labeled matrix. The
#' matrix is directional: entry \code{[effect, cause]} is the p-value for the
#' hypothesis that \code{cause} Granger-causes \code{effect}. Rows are effects
#' (the predicted series); columns are causes (the predictors).
#'
#' @param data A data frame or matrix of (already stationary) numeric time
#'   series, one series per column. Use \code{\link{make_stationary}} first if
#'   the series are not stationary.
#' @param variables Optional character vector of column names to test. Defaults
#'   to all numeric columns.
#' @param maxlag Either a single integer (lags \code{1:maxlag} are tested and
#'   aggregated) or an integer vector of specific lags to test.
#' @param aggregate How to collapse per-lag p-values into one matrix entry.
#'   One of \code{"min"}, \code{"mean"}, \code{"median"}, \code{"max"}.
#'   \code{"min"} matches the most common convention in the literature.
#' @param correction Multiple-testing correction applied across all
#'   off-diagonal cells. Passed to \code{\link[stats]{p.adjust}}; one of
#'   \code{"none"}, \code{"bonferroni"}, \code{"holm"}, \code{"BH"}, \code{"BY"}.
#' @param verbose Logical; if \code{TRUE}, print progress per pair.
#'
#' @return An object of class \code{"gcmatrix"}: a list with elements
#'   \code{pvalues}, \code{statistics}, \code{lags} (all square matrices),
#'   \code{pvalues_raw} (uncorrected), and metadata. Has \code{print},
#'   \code{summary} and \code{plot} methods.
#'
#' @examples
#' set.seed(1)
#' n <- 300
#' x <- rnorm(n)
#' y <- numeric(n)
#' for (t in 2:n) y[t] <- 0.6 * x[t - 1] + 0.2 * y[t - 1] + rnorm(1, sd = 0.3)
#' z <- rnorm(n)
#' df <- data.frame(x = x, y = y, z = z)
#' gcm <- granger_matrix(df, maxlag = 4, correction = "BH")
#' print(gcm)
#'
#' @seealso \code{\link{make_stationary}}, \code{\link{adf_report}}
#' @importFrom lmtest grangertest
#' @importFrom stats p.adjust median na.omit
#' @export
granger_matrix <- function(data,
                           variables = NULL,
                           maxlag = 5,
                           aggregate = c("min", "mean", "median", "max"),
                           correction = c("none", "bonferroni", "holm",
                                          "BH", "BY"),
                           verbose = FALSE) {
  aggregate <- match.arg(aggregate)
  correction <- match.arg(correction)

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

  n <- length(variables)
  pv <- st <- lg <- matrix(NA_real_, n, n,
                           dimnames = list(variables, variables))

  for (effect in variables) {
    for (cause in variables) {
      if (effect == cause) next
      pair <- stats::na.omit(data[, c(effect, cause)])
      per_p <- numeric(length(lags))
      per_s <- numeric(length(lags))
      for (li in seq_along(lags)) {
        # grangertest(effect ~ cause): tests whether `cause` improves
        # prediction of `effect`.
        gt <- lmtest::grangertest(pair[[effect]], pair[[cause]],
                                  order = lags[li])
        per_s[li] <- gt$F[2]
        per_p[li] <- gt$`Pr(>F)`[2]
      }
      best <- which.min(per_p)
      pv[effect, cause] <- agg_fn(per_p)
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
         correction = correction),
    class = "gcmatrix"
  )
}

#' Extract significant causal edges from a Granger Causality Matrix
#'
#' @param object A \code{"gcmatrix"} object.
#' @param alpha Significance level.
#' @return A data frame with one row per significant \code{cause -> effect}
#'   link: columns \code{cause}, \code{effect}, \code{p_value},
#'   \code{statistic}, \code{lag}.
#' @export
gcm_edges <- function(object, alpha = 0.05) {
  stopifnot(inherits(object, "gcmatrix"))
  pv <- object$pvalues
  out <- data.frame(cause = character(), effect = character(),
                    p_value = numeric(), statistic = numeric(),
                    lag = numeric(), stringsAsFactors = FALSE)
  for (effect in rownames(pv)) {
    for (cause in colnames(pv)) {
      if (effect == cause) next
      p <- pv[effect, cause]
      if (!is.na(p) && p < alpha) {
        out <- rbind(out, data.frame(
          cause = cause, effect = effect, p_value = p,
          statistic = object$statistics[effect, cause],
          lag = object$lags[effect, cause],
          stringsAsFactors = FALSE))
      }
    }
  }
  out[order(out$p_value), ]
}

#' @export
print.gcmatrix <- function(x, ...) {
  cat("Granger Causality Matrix\n")
  cat(sprintf("  variables : %s\n", paste(x$variables, collapse = ", ")))
  cat(sprintf("  aggregate : %s    correction: %s\n",
              x$aggregate, x$correction))
  cat("  p-values (rows = effect, cols = cause):\n\n")
  print(round(x$pvalues, 4))
  invisible(x)
}

#' @export
summary.gcmatrix <- function(object, alpha = 0.05, ...) {
  edges <- gcm_edges(object, alpha = alpha)
  cat(sprintf("Granger Causality Matrix: %d variables, %d significant links ",
              length(object$variables), nrow(edges)))
  cat(sprintf("at alpha = %.3g\n\n", alpha))
  if (nrow(edges) > 0) print(edges, row.names = FALSE)
  invisible(edges)
}
