rm(list = ls())    #delete objects
cat("\014")
library(glmnet)
library(tidyr)
setwd("/Users/cfscu/Documents/Carlos' Folders/MBA/0.Courses/2020/2.STA_9890_Statistical_Learning/Project")

# Load the file, and remove columns not relevant for the study
source.data = read.csv("BTC-USD_Source.csv")
summary(source.data)
source.data = source.data [,-c(3,4,5,6,7)]

# Clean up N/As using preceding and following values
source.data = fill(source.data, c(3,4,5,6,7,8,9,10), .direction = c("up"))
source.data = fill(source.data, c(3,4,5,6,7,8,9,10), .direction = c("down"))

# standardize data using formula 6.6 ISRL
colsd = rep(0,10)
colsd = apply(source.data[,2:10],2,sd)
for (k in 2:10) {
  source.data[,k] = source.data[,k] / colsd[(k-1)]
}

#standardize the data so that all standardized variables are given a mean of zero and a standard deviation of one.
p = 7
n = nrow(source.data) - p - 1
X = matrix(0, nrow = n, ncol = p)
X_final = matrix(0, nrow = n, ncol = 1)
y = matrix(0, nrow = n, ncol = 1)

for (c in 3:(ncol(source.data))) {
  time_series1 = source.data[, c]
  time_series2 = source.data[, 2]
  k=1
  for (i in (p+1):(n+p)) {
    X[k,]  = time_series1[(i-1) : (i-p)] # first row of X is time_series[p:1]
    y[k,1] = time_series2[i]
    k      = k + 1
  }
  X_final = cbind(X_final,X)
}
X = X_final[,-c(1)]

# Split data into 2 mutually exclusive sets and assign to X.train/X.test + y.train/y.test
# Define vector variables to store the results of each loop

n.test  =   floor(0.2 * n)
n.train =   n  - n.test

n.loops = 100

mse.test.las = matrix(0, nrow = n.loops, ncol = 1)
mse.train.las = matrix(0, nrow = n.loops, ncol = 1)
r2.test.las = matrix(0, nrow = n.loops, ncol = 1)
r2.train.las = matrix(0, nrow = n.loops, ncol = 1)

mse.test.el = matrix(0, nrow = n.loops, ncol = 1)
mse.train.el = matrix(0, nrow = n.loops, ncol = 1)
r2.test.el = matrix(0, nrow = n.loops, ncol = 1)
r2.train.el = matrix(0, nrow = n.loops, ncol = 1)

mse.test.rid = matrix(0, nrow = n.loops, ncol = 1)
mse.train.rid = matrix(0, nrow = n.loops, ncol = 1)
r2.test.rid = matrix(0, nrow = n.loops, ncol = 1)
r2.train.rid = matrix(0, nrow = n.loops, ncol = 1)

# Shuffle the samples and load X.train, y.train, X.test, y.test. 
# Start loop L(1:100), load results in variables.
# Calculate R2 vector for each loop, for R2-Train and R2-Test

for (L in 1:n.loops) {
  i.mix = sample(1:n)
  X = X[i.mix,]
  y = y[i.mix]
  X.train = X[1:n.train, ]
  X.test  = X[-(1:n.train), ]
  y.train = y[1:n.train]
  y.test  = y[-(1:n.train)]

  cat("Analysis Loop #",L,"for Lasso, EL, Ridge, Random Forests","\n")
  
# Cross Validation for Lasso, alpha = 1, folds = 10
  cv.lasso                         =     cv.glmnet(X.train, y.train, alpha = 1, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
  lasso.fit                        =     glmnet(X.train, y.train, alpha = 1, intercept = T, lambda = cv.lasso$lambda.min)
  y.train.hat.las                  =     X.train %*% lasso.fit$beta + lasso.fit$a0  
  y.test.hat.las                   =     X.test %*% lasso.fit$beta  + lasso.fit$a0  
  y.train.hat.las                  =     as.vector(y.train.hat.las)
  y.test.hat.las                   =     as.vector(y.test.hat.las)

  mse.test.las[L,1] = sum((y.test - y.test.hat.las)^2)/length(y.test)
  mse.train.las[L,1] = sum((y.train - y.train.hat.las)^2)/length(y.train)
  ssy = sum((y - mean(y))^2)/length(y)
  r2.test.las[L,1] = 1 - mse.test.las[L,1] / ssy
  r2.train.las[L,1] = 1 - mse.train.las[L,1] / ssy

# Cross Validation for Elastic Net, alpha = 0.5, folds = 10
  cv.el                           =     cv.glmnet(X.train, y.train, alpha = 0.5, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
  el.fit                          =     glmnet(X.train, y.train, alpha = 0.5, intercept = T, lambda = cv.el$lambda.min)
  y.train.hat.el                  =     X.train %*% el.fit$beta + el.fit$a0  
  y.test.hat.el                   =     X.test %*% el.fit$beta  + el.fit$a0  
  y.train.hat.el                  =     as.vector(y.train.hat.el)
  y.test.hat.el                   =     as.vector(y.test.hat.el)

  mse.test.el[L,1] = sum((y.test - y.test.hat.el)^2)/length(y.test)
  mse.train.el[L,1] = sum((y.train - y.train.hat.el)^2)/length(y.train)
  r2.test.el[L,] = 1 - mse.test.el[L,1] / ssy
  r2.train.el[L,1] = 1 - mse.train.el[L,1] / ssy
  
# Cross Validation for Ridge, alpha = 0, folds = 10
  cv.rid                           =     cv.glmnet(X.train, y.train, alpha = 0, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
  rid.fit                          =     glmnet(X.train, y.train, alpha = 0, intercept = T, lambda = cv.rid$lambda.min)
  y.train.hat.rid                  =     X.train %*% rid.fit$beta + rid.fit$a0  
  y.test.hat.rid                   =     X.test %*% rid.fit$beta  + rid.fit$a0  
  y.train.hat.rid                  =     as.vector(y.train.hat.rid)
  y.test.hat.rid                   =     as.vector(y.test.hat.rid)

  mse.test.rid[L,1] = sum((y.test - y.test.hat.rid)^2)/length(y.test)
  mse.train.rid[L,1] = sum((y.train - y.train.hat.rid)^2)/length(y.train)
  r2.test.rid[L,] = 1 - mse.test.rid[L,1] / ssy
  r2.train.rid[L,1] = 1 - mse.train.rid[L,1] / ssy
}

# Question 4.b: Boxplots for R2 Train / R2 Test
r2.test = cbind(r2.test.las, r2.test.el, r2.test.rid)
r2.train = cbind(r2.train.las, r2.train.el, r2.train.rid)
boxplot(r2.test, main ='R2 - Test - 100 Runs', names = c('Lasso', 'Elastic Net', 'Ridge'), col= c('blue', 'green', 'yellow'))
boxplot(r2.train, main ='R2 - Train - 100 Runs', names = c('Lasso', 'Elastic Net', 'Ridge'), col= c('blue', 'green', 'yellow'))

# Question 4.c: 10-Fold CV Curves for Lasso and Ridge
cv.lasso                        =     cv.glmnet(X.train, y.train, alpha = 1, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
cv.el                           =     cv.glmnet(X.train, y.train, alpha = 0.5, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)
cv.rid                          =     cv.glmnet(X.train, y.train, alpha = 0, family = "gaussian",intercept = T, type.measure = "mae", nfolds = 10)

plot(cv.lasso, main = "10-fold CV curve - Lasso")
plot(cv.el, main = "10-fold CV curve - Elastic Net")
plot(cv.rid, main = "10-fold CV curve - Ridge")

# Question 4.d: side-by-side boxplots of train/test residuals
lasso.fit                        =     glmnet(X.train, y.train, alpha = 1, intercept = T, lambda = cv.lasso$lambda.min)
y.train.hat.las                  =     X.train %*% lasso.fit$beta + lasso.fit$a0  
y.test.hat.las                   =     X.test %*% lasso.fit$beta  + lasso.fit$a0  
y.train.hat.las                  =     as.vector(y.train.hat.las)
y.test.hat.las                   =     as.vector(y.test.hat.las)

el.fit                          =     glmnet(X.train, y.train, alpha = 0.5, intercept = T, lambda = cv.el$lambda.min)
y.train.hat.el                  =     X.train %*% el.fit$beta + el.fit$a0  
y.test.hat.el                   =     X.test %*% el.fit$beta  + el.fit$a0  
y.train.hat.el                  =     as.vector(y.train.hat.el)
y.test.hat.el                   =     as.vector(y.test.hat.el)

rid.fit                          =     glmnet(X.train, y.train, alpha = 0, intercept = T, lambda = cv.rid$lambda.min)
y.train.hat.rid                  =     X.train %*% rid.fit$beta + rid.fit$a0  
y.test.hat.rid                   =     X.test %*% rid.fit$beta  + rid.fit$a0  
y.train.hat.rid                  =     as.vector(y.train.hat.rid)
y.test.hat.rid                   =     as.vector(y.test.hat.rid)

res.test.las = matrix(0, nrow = n.test, ncol = 1)
res.train.las = matrix(0, nrow = n.train, ncol = 1)
res.test.el = matrix(0, nrow = n.test, ncol = 1)
res.train.el = matrix(0, nrow = n.train, ncol = 1)
res.test.rid = matrix(0, nrow = n.test, ncol = 1)
res.train.rid = matrix(0, nrow = n.train, ncol = 1)

res.train.las = y.train - y.train.hat.las
res.test.las = y.test - y.test.hat.las
boxplot(res.train.las, res.test.las, main ='Lasso Residuals - Train/Test - 1 Sample', names = c('Train', 'Test'), col= c('blue', 'green'))

res.train.el = y.train - y.train.hat.el
res.test.el = y.test - y.test.hat.el
boxplot(res.train.el, res.test.el, main ='Elastic Net Residuals - Train/Test - 1 Sample', names = c('Train', 'Test'), col= c('blue', 'green'))

res.train.rid = y.train - y.train.hat.rid
res.test.rid = y.test - y.test.hat.rid
boxplot(res.train.rid, res.test.rid, main ='Ridge Residuals - Train/Test - 1 Sample', names = c('Train', 'Test'), col= c('blue', 'green'))
