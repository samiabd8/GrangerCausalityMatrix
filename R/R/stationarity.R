#' Augmented Dickey-Fuller stationarity report
#'
#' Runs the ADF test on every numeric column and returns a tidy report. A
#' small p-value indicates the series is stationary.
#'
#' @param data A data frame or matrix of numeric time series.
#' @param alpha Significance level for the stationarity flag.
#' @return A data frame with columns \code{variable}, \code{adf_stat},
#'   \code{p_value} and \code{stationary}.
#' @examples
#' if (requireNamespace("tseries", quietly = TRUE)) {
#'   df <- data.frame(a = cumsum(rnorm(200)), b = rnorm(200))
#'   adf_report(df)
#' }
#' @export
adf_report <- function(data, alpha = 0.05) {
  if (!requireNamespace("tseries", quietly = TRUE)) {
    stop("Package 'tseries' is required for adf_report().", call. = FALSE)
  }
  data <- as.data.frame(data)
  num_cols <- names(data)[vapply(data, is.numeric, logical(1))]
  rows <- lapply(num_cols, function(col) {
    s <- stats::na.omit(data[[col]])
    res <- suppressWarnings(tseries::adf.test(s))
    data.frame(variable = col, adf_stat = unname(res$statistic),
               p_value = res$p.value, stationary = res$p.value < alpha,
               stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}

#' Make time series stationary by differencing
#'
#' Successively differences each column until the ADF test reports
#' stationarity (or \code{max_diff} is reached). Optionally takes logs first,
#' reproducing the common log-returns transform.
#'
#' @param data A data frame or matrix of numeric time series.
#' @param method Either \code{"diff"} (plain differencing) or \code{"log_diff"}
#'   (log-transform then difference; requires strictly positive values).
#' @param max_diff Maximum number of differencing rounds.
#' @param alpha Significance level for the ADF stationarity decision.
#' @return A list with \code{data} (the stationary data frame) and
#'   \code{orders} (named integer vector of differencing orders applied).
#' @examples
#' if (requireNamespace("tseries", quietly = TRUE)) {
#'   df <- data.frame(a = cumsum(rnorm(200)), b = cumsum(rnorm(200)))
#'   out <- make_stationary(df)
#'   out$orders
#' }
#' @export
make_stationary <- function(data, method = c("diff", "log_diff"),
                            max_diff = 2L, alpha = 0.05) {
  method <- match.arg(method)
  if (!requireNamespace("tseries", quietly = TRUE)) {
    stop("Package 'tseries' is required for make_stationary().", call. = FALSE)
  }
  data <- as.data.frame(data)
  num_cols <- names(data)[vapply(data, is.numeric, logical(1))]
  work <- data[num_cols]

  if (method == "log_diff") {
    if (any(work <= 0, na.rm = TRUE)) {
      stop("log_diff requires strictly positive values.", call. = FALSE)
    }
    work <- as.data.frame(lapply(work, log))
  }

  orders <- integer(length(num_cols))
  names(orders) <- num_cols
  for (col in num_cols) {
    series <- work[[col]]
    order <- 0L
    while (order < max_diff) {
      s <- stats::na.omit(series)
      pval <- suppressWarnings(tseries::adf.test(s)$p.value)
      if (pval < alpha) break
      series <- c(NA, diff(series))
      order <- order + 1L
    }
    work[[col]] <- series
    orders[col] <- order
  }

  list(data = stats::na.omit(work), orders = orders)
}
