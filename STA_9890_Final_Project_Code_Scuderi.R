rm(list = ls())    #delete objects
cat("\014")
library(glmnet)
library(tidyr)
library(randomForest)
library(ggplot2)
library(gridExtra)

set.seed(1)

# Initial code to capture timestamps for running time calculation (12 measure points)
RunningTime = matrix(0, nrow = 11, ncol = 1)
RunningTime = rep(0,11)
rt = 1

Timestamp1 = Sys.time()
# Load the file, and remove columns not relevant for the study
source.data = read.csv("BTC-USD_Source.csv")
summary(source.data)
source.data = source.data [,-c(3,4,5,6,7)]

# Clean up N/As using preceding and following values
source.data = fill(source.data, c(3,4,5,6,7,8,9,10), .direction = c("up"))
source.data = fill(source.data, c(3,4,5,6,7,8,9,10), .direction = c("down"))

#Load y and X matrices
p_hist = 7                                    # we analyze 7 days history for every predictor
p = 56                                        # total amount of predictors = 56
n = nrow(source.data) - p_hist - 1
X = matrix(0, nrow = n, ncol = p_hist)
X_final = matrix(0, nrow = n, ncol = 1)
y = matrix(0, nrow = n, ncol = 1)

for (c in 3:(ncol(source.data))) {
  time_series1 = source.data[, c]
  time_series2 = source.data[, 2]
  k=1
  for (i in (p_hist+1):(n+p_hist)) {
    X[k,]  = time_series1[(i-1) : (i-p_hist)] # first row of X is time_series[p_hist:1]
    y[k,1] = time_series2[i]
    k      = k + 1
  }
  X_final = cbind(X_final,X)
}
X = X_final[,-c(1)]
#y = log(y)

# standardize X values
mu       =   as.vector(apply(X, 2, 'mean'))
sd       =   as.vector(apply(X, 2, 'sd'))
X.orig   =   X
for (i in c(1:n)){
  X[i,]   =    (X[i,] - mu)/sd
}

Timestamp2 = Sys.time()
RunningTime[rt] = Timestamp2 - Timestamp1 
rt = 1 + rt

# Split data into 2 mutually exclusive sets and assign to X.train/X.test + y.train/y.test
# Define vector variables to store the results of each loop

n.test  =   floor(0.2 * n)
n.train =   n  - n.test

M = 100

mse.test.las = rep(0,M)
mse.train.las = rep(0,M)
r2.test.las = rep(0,M)
r2.train.las = rep(0,M)

mse.test.en = rep(0,M)
mse.train.en = rep(0,M)
r2.test.en = rep(0,M)
r2.train.en = rep(0,M)

mse.test.rid = rep(0,M)
mse.train.rid = rep(0,M)
r2.test.rid = rep(0,M)
r2.train.rid = rep(0,M)

mse.test.rf = rep(0,M)
mse.train.rf = rep(0,M)
r2.test.rf = rep(0,M)
r2.train.rf = rep(0,M)

# Shuffle the samples and load X.train, y.train, X.test, y.test. 
# Start loop L(1:100), load results in variables.
# Calculate R2 vector for each loop, for R2-Train and R2-Test

for (m in 1:M) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]

  #cat("Analysis Loop #",m,"for Lasso, EL, Ridge, Random Forests","\n")
  
# Cross Validation for Lasso, alpha = 1, folds = 10
  cv.lasso                         =     cv.glmnet(X.train, y.train, alpha = 1, intercept = T, type.measure = "mae", nfolds = 10)
  lasso.fit                        =     glmnet(X.train, y.train, alpha = 1, intercept = T, lambda = cv.lasso$lambda.min)
  y.train.hat.las                  =     predict(lasso.fit, newx = X.train, type = "response") 
  y.test.hat.las                   =     predict(lasso.fit, newx = X.test, type = "response")  
  y.train.hat.las                  =     as.vector(y.train.hat.las)
  y.test.hat.las                   =     as.vector(y.test.hat.las)

  mse.test.las[m]                  =     mean((y.test - y.test.hat.las)^2)
  mse.train.las[m]                 =     mean((y.train - y.train.hat.las)^2)

  r2.test.las[m]                   =     1-mean((y.test - y.test.hat.las)^2)/mean((y - mean(y))^2)
  r2.train.las[m]                  =     1-mean((y.train - y.train.hat.las)^2)/mean((y - mean(y))^2)

# Cross Validation for Elastic Net, alpha = 0.5, folds = 10
  cv.en                            =     cv.glmnet(X.train, y.train, alpha = 0.5, intercept = T, type.measure = "mae", nfolds = 10)
  en.fit                           =     glmnet(X.train, y.train, alpha = 0.5, intercept = T, lambda = cv.en$lambda.min)
  y.train.hat.en                   =     predict(en.fit, newx = X.train, type = "response")  
  y.test.hat.en                    =     predict(en.fit, newx = X.test, type = "response") 
  y.train.hat.en                   =     as.vector(y.train.hat.en)
  y.test.hat.en                    =     as.vector(y.test.hat.en)
 
  mse.test.en[m]                   =     mean((y.test - y.test.hat.en)^2)
  mse.train.en[m]                  =     mean((y.train - y.train.hat.en)^2)
 
  r2.test.en[m]                    =     1-mean((y.test - y.test.hat.en)^2)/mean((y - mean(y))^2)
  r2.train.en[m]                   =     1-mean((y.train - y.train.hat.en)^2)/mean((y - mean(y))^2)
  
# Cross Validation for Ridge, alpha = 0, folds = 10
  cv.rid                           =     cv.glmnet(X.train, y.train, alpha = 0, intercept = T, type.measure = "mae", nfolds = 10)
  rid.fit                          =     glmnet(X.train, y.train, alpha = 0, intercept = T, lambda = cv.rid$lambda.min)
  y.train.hat.rid                  =     predict(rid.fit, newx = X.train, type = "response")  
  y.test.hat.rid                   =     predict(rid.fit, newx = X.test, type = "response")  
  y.train.hat.rid                  =     as.vector(y.train.hat.rid)
  y.test.hat.rid                   =     as.vector(y.test.hat.rid)

  mse.test.rid[m]                  =     mean((y.test - y.test.hat.rid)^2)
  mse.train.rid[m]                 =     mean((y.train - y.train.hat.rid)^2)
  r2.test.rid[m]                   =     1-mean((y.test - y.test.hat.rid)^2)/mean((y - mean(y))^2)
  r2.train.rid[m]                  =     1-mean((y.train - y.train.hat.rid)^2)/mean((y - mean(y))^2)
  
  # fit Random Forest and calculate and record the train and test R squares 
  
  rf                               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat.rf                    =     predict(rf, X.test)
  y.train.hat.rf                   =     predict(rf, X.train)
  
  mse.test.rf[m]                   =     mean((y.test - y.test.hat.rf)^2)
  mse.train.rf[m]                  =     mean((y.train - y.train.hat.rf)^2)
  r2.test.rf[m]                    =     1-mean((y.test - y.test.hat.rf)^2)/mean((y - mean(y))^2)
  r2.train.rf[m]                   =     1-mean((y.train - y.train.hat.rf)^2)/mean((y - mean(y))^2) 
  
  cat(sprintf("m=%3.f| r2.test.las=%.2f, r2.test.en=%.2f, r2.test.rid=%.2f, r2.test.rf=%.2f | r2.train.las=%.2f, r2.train.en=%.2f, r2.train.rid=%.2f, r2.train.rf=%.2f| \n", m,  r2.test.las[m], r2.test.en[m], r2.test.rid[m],r2.test.rf[m], r2.train.las[m], r2.train.en[m], r2.train.rid[m], r2.train.rf[m]))
  
}

Timestamp3 = Sys.time()
RunningTime[rt] = Timestamp3 - Timestamp2
rt = 1 + rt

# Question 4.b: Boxplots for R2 Train / R2 Test
r2.test = cbind(r2.test.las, r2.test.en, r2.test.rid, r2.test.rf)
r2.train = cbind(r2.train.las, r2.train.en, r2.train.rid, r2.train.rf)
boxplot(r2.test, main ='R2 - Test - 100 Runs', names = c('Lasso', 'Elastic Net', 'Ridge', 'RF'), col= c('blue', 'green', 'yellow', 'red'))
boxplot(r2.train, main ='R2 - Train - 100 Runs', names = c('Lasso', 'Elastic Net', 'Ridge', 'RF'), col= c('blue', 'green', 'yellow', 'red'))

r2.las2 = cbind(r2.test.las, r2.train.las)
r2.en2 = cbind(r2.test.en, r2.train.en)
r2.rid2 = cbind(r2.test.rid, r2.train.rid)
r2.rf2 = cbind(r2.test.rf, r2.train.rf)
boxplot(r2.las2, main ='R2 Lasso - 100 Runs', names = c('Test', 'Train'), col= c('blue', 'green'), ylim = c(0.6,1))
boxplot(r2.en2, main ='R2 Elastic Net - 100 Runs', names = c('Test', 'Train'), col= c('blue', 'green'), ylim = c(0.6,1))
boxplot(r2.rid2, main ='R2 Ridge - 100 Runs', names = c('Test', 'Train'), col= c('blue', 'green'), ylim = c(0.6,1))
boxplot(r2.rf2, main ='R2 RF - 100 Runs', names = c('Test', 'Train'), col= c('blue', 'green'), ylim = c(0.6,1))

Timestamp4 = Sys.time()
RunningTime[rt] = Timestamp4 - Timestamp3 
rt = 1 + rt

# Question 4.c: 10-Fold CV Curves for Lasso, Elastic, and Ridge
cv.lasso                        =     cv.glmnet(X.train, y.train, alpha = 1, intercept = T, type.measure = "mae", nfolds = 10)
cv.en                           =     cv.glmnet(X.train, y.train, alpha = 0.5, intercept = T, type.measure = "mae", nfolds = 10)
cv.rid                          =     cv.glmnet(X.train, y.train, alpha = 0, intercept = T, type.measure = "mae", nfolds = 10)

plot(cv.lasso, main = "10-fold CV curve - Lasso")
plot(cv.en, main = "10-fold CV curve - Elastic Net")
plot(cv.rid, main = "10-fold CV curve - Ridge")

Timestamp5 = Sys.time()
RunningTime[rt] = Timestamp5 - Timestamp4 
rt = 1 + rt

# Question 4.d: side-by-side boxplots of train/test residuals
lasso.fit                        =     glmnet(X.train, y.train, alpha = 1, intercept = T, lambda = cv.lasso$lambda.min)
y.train.hat.las                  =     X.train %*% lasso.fit$beta + lasso.fit$a0  
y.test.hat.las                   =     X.test %*% lasso.fit$beta  + lasso.fit$a0  
y.train.hat.las                  =     as.vector(y.train.hat.las)
y.test.hat.las                   =     as.vector(y.test.hat.las)

en.fit                          =     glmnet(X.train, y.train, alpha = 0.5, intercept = T, lambda = cv.en$lambda.min)
y.train.hat.en                  =     X.train %*% en.fit$beta + en.fit$a0  
y.test.hat.en                   =     X.test %*% en.fit$beta  + en.fit$a0  
y.train.hat.en                  =     as.vector(y.train.hat.en)
y.test.hat.en                   =     as.vector(y.test.hat.en)

rid.fit                          =     glmnet(X.train, y.train, alpha = 0, intercept = T, lambda = cv.rid$lambda.min)
y.train.hat.rid                  =     X.train %*% rid.fit$beta + rid.fit$a0  
y.test.hat.rid                   =     X.test %*% rid.fit$beta  + rid.fit$a0  
y.train.hat.rid                  =     as.vector(y.train.hat.rid)
y.test.hat.rid                   =     as.vector(y.test.hat.rid)

rf                               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
y.test.hat.rf                    =     predict(rf, X.test)
y.train.hat.rf                   =     predict(rf, X.train)

res.test.las = matrix(0, nrow = n.test, ncol = 1)
res.train.las = matrix(0, nrow = n.train, ncol = 1)
res.test.en = matrix(0, nrow = n.test, ncol = 1)
res.train.en = matrix(0, nrow = n.train, ncol = 1)
res.test.rid = matrix(0, nrow = n.test, ncol = 1)
res.train.rid = matrix(0, nrow = n.train, ncol = 1)
res.test.rf = matrix(0, nrow = n.test, ncol = 1)
res.train.rf = matrix(0, nrow = n.train, ncol = 1)

res.train.las = y.train - y.train.hat.las
res.test.las = y.test - y.test.hat.las
boxplot(res.train.las, res.test.las, main ='Lasso Residuals - Train/Test - 1 Sample', names = c('Train', 'Test'), col= c('blue', 'green'))

res.train.en = y.train - y.train.hat.en
res.test.en = y.test - y.test.hat.en
boxplot(res.train.en, res.test.en, main ='Elastic Net Residuals - Train/Test - 1 Sample', names = c('Train', 'Test'), col= c('blue', 'green'))

res.train.rid = y.train - y.train.hat.rid
res.test.rid = y.test - y.test.hat.rid
boxplot(res.train.rid, res.test.rid, main ='Ridge Residuals - Train/Test - 1 Sample', names = c('Train', 'Test'), col= c('blue', 'green'))

res.train.rf = y.train - y.train.hat.rf
res.test.rf = y.test - y.test.hat.rf
boxplot(res.train.rf, res.test.rf, main ='RF Residuals - Train/Test - 1 Sample', names = c('Train', 'Test'), col= c('blue', 'green'))

Timestamp6 = Sys.time()
RunningTime[rt] = Timestamp6 - Timestamp5 
rt = 1 + rt

# Question 4.e: bar-plots (with bootstrapped error bars) of estimated coeficients and importance of parameters

bootstrapSamples =     100
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.las.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.rid.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit BS to Random Forests
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  
  # fit BS to Lasso
  a                =     1 # Alpha Lasso = 1
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.las.bs[,m]  =     as.vector(fit$beta)
  
  # fit BS to Elastic Net
  a                =     0.5 # Alpha Elastic-Net = 0.5
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit$beta)
  
  # fit BS to Ridge
  a                =     0 # Alpha Ridge = 0
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.rid.bs[,m]  =     as.vector(fit$beta)
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
}
# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd     = apply(beta.rf.bs, 1, "sd")
las.bs.sd    = apply(beta.las.bs, 1, "sd")
en.bs.sd     = apply(beta.en.bs, 1, "sd")
rid.bs.sd    = apply(beta.rid.bs, 1, "sd")

Timestamp7 = Sys.time()
RunningTime[rt] = Timestamp7 - Timestamp6 
rt = 1 + rt

# fit Random Forest to the whole data
rf               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

Timestamp8 = Sys.time()
RunningTime[rt] = Timestamp8 - Timestamp7 
rt = 1 + rt

# fit Lasso to the whole data
a=1 # Alpha for Lasso = 1
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
las.fit          =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

Timestamp9 = Sys.time()
RunningTime[rt] = Timestamp9 - Timestamp8 
rt = 1 + rt

# fit Elastic Net to the whole data
a=0.5 # Alpha for Elastic-Net = 0.5
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
en.fit           =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

Timestamp10 = Sys.time()
RunningTime[rt] = Timestamp10 - Timestamp9 
rt = 1 + rt

# fit Ridge to the whole data
a=0 # Alpha for Ridge = 0
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
rid.fit          =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

Timestamp11 = Sys.time()
RunningTime[rt] = Timestamp11 - Timestamp10 
rt = 1 + rt

betaS.rf               =     data.frame(c(1:p), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

betaS.las              =     data.frame(c(1:p), as.vector(las.fit$beta), 2*en.bs.sd)
colnames(betaS.las)     =     c( "feature", "value", "err")

betaS.en               =     data.frame(c(1:p), as.vector(en.fit$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

betaS.rid              =     data.frame(c(1:p), as.vector(rid.fit$beta), 2*en.bs.sd)
colnames(betaS.rid)     =     c( "feature", "value", "err")

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) + ggtitle("RF Coeficients") + 
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) 

lasPlot =  ggplot(betaS.las, aes(x=feature, y=value)) + ggtitle("Lasso Coeficients") + 
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) + ggtitle("EN Coeficients") + 
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) + ggtitle("Ridge Coeficients") + 
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

grid.arrange(rfPlot, lasPlot, enPlot, ridPlot, nrow = 4)

# we need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.las$feature    =  factor(betaS.las$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rid$feature    =  factor(betaS.rid$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) + ggtitle("RF Coeficients") + 
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

lasPlot =  ggplot(betaS.las, aes(x=feature, y=value)) + ggtitle("Lasso Coeficients") + 
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) + ggtitle("EN Coeficients") + 
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) + ggtitle("Ridge Coeficients") + 
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

grid.arrange(rfPlot, lasPlot, enPlot, ridPlot, nrow = 4)

Timestamp12 = Sys.time()
RunningTime[rt] = Timestamp12 - Timestamp11
RunningTime[2] = RunningTime[2]*60
RunningTime[6] = RunningTime[6]*60

# 4.f)Running time calculation
barplot(RunningTime, main="Code Running Time", ylab = "Seconds", 
        names.arg=c("D(2-1)", "D(3-2)", "D(4-3)", "D(5-4)", "D(6-5)", "D(7-6)", "D(8-7)", "D(9-8)", "D(10-9)", "D(11-10)", "D(12-11)")) 
