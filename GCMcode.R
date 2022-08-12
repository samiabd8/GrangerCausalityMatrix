rm(list=ls())
set.seed(1996)
library(MASS)
library(lmtest)
library(NlinTS)
library(neuralnet)

t0 <- Sys.time()

setwd('/home/dmnsh/Projects/GrangerCausalityMatrixPackage')

par(mfrow=c(1,1))
data = as.data.frame(read.csv("./datayobit1h.csv", header = TRUE,sep = ","))

prices <- as.matrix(data[,-1])
log_returns <- as.data.frame(diff(log(prices), lag=1))
rm(prices)

log_returns[is.na(log_returns)] <- 0
log_returns <- as.matrix(log_returns)
log_returns <- as.data.frame(log_returns[-c(6055,6353),])
rm(data)

#find optimal linear lag

K <- ncol(log_returns)
J <- seq(1:12)
L <- seq(1:(K-1))
caus_matrix <- function(k){
  y <- log_returns[,k]
  y <- cbind(y,y,y,y,y,y,y,y,y,y,y,y,y,y)
  X <- log_returns[,-k]
  G <- function(j) {
    g <- list()
    for (i in L) {
      g[[i]]  <- grangertest(y[,i] ~ X[,i], order=j) }
    
    foo <- function(c) {
      F <- c$F[2]
      P <- c$`Pr(>F)`[2]
      data.frame(F = F, P = P) }
    do.call(rbind, lapply(g, foo)) }
  
  Glist <- list()
  for (j in J) {
    Gval = G(j)
    Glist[[length(Glist)+1]]=Gval }
  
  
  #store pvalues
  output <- matrix(unlist(Glist),ncol=(K-1),byrow=TRUE)
  I = seq(from=2,to=24,by=2)
  pvals = matrix(NA,ncol=1,nrow=12)
  pvals = output[I,]
}
print(Sys.time() - t0)

pvals_btc <- caus_matrix(k=1)
colnames(pvals_btc) <- colnames(log_returns[,-1])
t(rowMeans(pvals_btc))
plot(rowMeans(pvals_btc))
#optlag in lin case is 5h

#nonlinear granger

nlg_matrix <- matrix(NA,nrow=12,ncol=14)
for (k in 1:12){
  y <- log_returns[,k]
  y <- cbind(y,y,y,y,y,y,y,y,y,y,y,y,y,y)
  X <- log_returns[,-k]
  nlg <- list()
  pv_nlg <- list()
  for (i in 1:14){
    nlg[[i]] <- nlin_causality.test(y[,i],X[,i],lag=k,c(5,3),c(5,3),50,TRUE)
    nlg_matrix[k,i] <- nlg[[i]]$pvalue}}
plot(rowMeans(nlg_matrix))
#optlag in nl case is 3h
rm(i,J,k,K,L)
rm(X,y)


