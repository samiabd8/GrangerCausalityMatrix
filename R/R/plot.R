#' Plot a Granger Causality Matrix as a heatmap
#'
#' Draws the p-value matrix as a heatmap using base graphics, so the package
#' has no plotting dependencies. Rows are effects, columns are causes; low
#' p-values (significant causality) are highlighted.
#'
#' @param x A \code{"gcmatrix"} object.
#' @param alpha Significance threshold used for the colour midpoint.
#' @param annot Logical; write the numeric p-value in each cell.
#' @param col A vector of colours forming the gradient (low p -> high p).
#' @param main Optional plot title.
#' @param ... Currently ignored.
#' @return Invisibly returns \code{x}.
#' @examples
#' set.seed(1)
#' df <- data.frame(a = rnorm(200), b = rnorm(200), c = rnorm(200))
#' gcm <- granger_matrix(df, maxlag = 3)
#' plot(gcm)
#' @importFrom graphics image axis text mtext box
#' @importFrom grDevices colorRampPalette
#' @export
plot.gcmatrix <- function(x, alpha = 0.05, annot = TRUE,
                          col = c("#1a9850", "#ffffbf", "#d73027"),
                          main = NULL, ...) {
  pv <- x$pvalues
  n <- nrow(pv)
  vars <- rownames(pv)

  # Map p-values to a [0,1] colour scale with alpha at the midpoint, so the
  # gradient is visually centered on the significance threshold. The diagonal
  # is NA and is left NA (image() renders NA cells transparent).
  scaled <- pv
  finite <- !is.na(pv)
  lo <- finite & pv <= alpha
  hi <- finite & pv > alpha
  scaled[lo] <- 0.5 * (pv[lo] / alpha)
  scaled[hi] <- 0.5 + 0.5 * ((pv[hi] - alpha) / (1 - alpha))

  pal <- grDevices::colorRampPalette(col)(100)

  # image() plots column-major and bottom-up; transpose and flip rows so the
  # matrix reads top-left to bottom-right like a printed table.
  m <- t(scaled[n:1, , drop = FALSE])
  graphics::image(seq_len(n), seq_len(n), m, zlim = c(0, 1),
                  col = pal, axes = FALSE, xlab = "", ylab = "")
  graphics::axis(3, at = seq_len(n), labels = vars, tick = FALSE, las = 2)
  graphics::axis(2, at = seq_len(n), labels = rev(vars), tick = FALSE,
                 las = 2)
  graphics::mtext("Cause (predictor)", side = 3, line = 2.5)
  graphics::mtext("Effect (predicted)", side = 2, line = 2.5)
  graphics::mtext(main %||% sprintf("Granger Causality Matrix (correction: %s)",
                                    x$correction),
                  side = 1, line = 1, font = 2)
  graphics::box()

  if (annot) {
    for (i in seq_len(n)) {       # column index = cause
      for (j in seq_len(n)) {     # row index from bottom = effect (reversed)
        p <- pv[n - j + 1, i]
        if (!is.na(p)) {
          graphics::text(i, j, formatC(p, format = "f", digits = 3),
                         cex = 0.8)
        }
      }
    }
  }
  invisible(x)
}

`%||%` <- function(a, b) if (is.null(a)) b else a
