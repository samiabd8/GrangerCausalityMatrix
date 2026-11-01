rm(list = ls())
set.seed(1996)
library(MASS)
library(lmtest)
library(NlinTS)
library(neuralnet)
library(tidyverse)
library(comprehenr)

data_path <- './datayobit1h.csv'

log_returns <-
  read_csv(data_path,
           col_select = -c('Date'),
           show_col_types = FALSE) %>%
  log() %>%
  as.matrix() %>%
  diff(lag = 1) %>%
  as_tibble() %>%
  mutate_all(replace_na, 0) %>%
  slice(-c(6055, 6353)) %>%
  as_tibble()

#find optimal linear lag

K <- ncol(log_returns) # Number of columns
J <- seq(1:12)
L <- seq(1:(K - 1))

caus_matrix <- function(k) {
  y <- log_returns[, k] %>% rep(times = K) %>% as.data.frame()
  X <- log_returns[, -k]
  
  gtest <- function(test_order) {
    metrics <-
      to_list(for (i in L)
        grangertest(y[, i] ~ as.matrix(X[, i]), order = test_order))
    return (data.frame(F = as.matrix(to_list(for (metric in metrics)
      metric$F[2])),
      P = as.matrix(to_list(
        for (metric in metrics)
          metric$`Pr(>F)`[2]
      ))))
  }
  
  Glist <- to_list(for (j in J)
    gtest(j))
  
  # store pvalues
  output <- matrix(unlist(Glist), ncol = (K - 1), byrow = TRUE)
  I = seq(from = 2, to = 24, by = 2)
  pvals = matrix(NA, ncol = 1, nrow = 12)
  pvals = output[I, ]
}

pvals_btc <- caus_matrix(k = 1)
colnames(pvals_btc) <- colnames(log_returns[, -1])
t(rowMeans(pvals_btc))
plot(rowMeans(pvals_btc))
#optlag in lin case is 5h

#nonlinear granger

nlg_matrix <- matrix(NA, nrow = 12, ncol = 14)
for (k in 1:12) {
  y <- log_returns[, k] %>% rep(times = K) %>% as.data.frame()
  X <- log_returns[, -k]
  
  nlg <- list()
  pv_nlg <- list()
  
  for (i in 1:14) {
    nlg[[i]] <-
      nlin_causality.test(y[, i], X[, i], lag = k, c(5, 3), c(5, 3), 50, TRUE)
    nlg_matrix[k, i] <- nlg[[i]]$pvalue
  }
}

plot(rowMeans(nlg_matrix))
#optlag in nl case is 3h
rm(i, J, k, K, L)
rm(X, y)
