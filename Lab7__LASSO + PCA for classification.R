# Lab_XI_ October 17th - Regularization for Classification ####

## Colon dataset ####
load("colon_data.RData")

library(glmnet)
y <- as.factor(train$y)
x <- as.matrix(train[,-1]) #remove the response (y)
test.m <-as.matrix(test[,-1]) #remove the response (y)

#estimation 
grid <- 10^seq(10,-2,length.out=100)
lasso.fit <- glmnet(x,y,family="binomial",alpha=1,lambda=grid) #classification
plot(lasso.fit)

#RNGkind(): check which version of function RStudio use
RNGkind(sample.kind = "Rejection")
set.seed(1234)
cv.out<-cv.glmnet(x,y,alpha=1,family="binomial")
plot(cv.out)

# LASSO - Test error estimation with the best lambda ####
best.lambda <- cv.out$lambda.min
yhat <- predict(lasso.fit,s=best.lambda,newx=test.m,
                type="class")
table(test$y,yhat)
mean(yhat!=test$y)

# Compute coefficient estimates on the full dataset, 
# using the best optimal lambda by cross validation
lasso.coeff <- predict(lasso.fit, type="coefficient", s=best.lambda)
sum(lasso.coeff!=0)

which(lasso.coeff!=0)

#Dimension reduction by PCA and logistic regression ####
x <- as.matrix(train[,-1])
xx <- scale(x,T,T) #standardlized 

#5-fold CV 
k <- 5
set.seed(1234)
folds <- sample(1:k,nrow(x),replace=T)

err <- matrix(NA,k,25)

for(i in 1:k){
  xx.train<- xx[folds!=i,]
  xx.test <- xx[folds==i,]
  ytest<- factor(train$y[folds==i], level=0:1)
  #y.test is transformed into a factor so as to ensure that 
  #both levels will appear when doing misc, in case that 
  #validation set has units from one class only 
  
  ytrain<- train$y[folds!=i]
  
  svd.xx <- svd(xx.train)
  xx.pcs <- svd.xx$v
  
  for(j in 1:25){
    xx.train <- as.matrix(xx.train)%*%xx.pcs[,1:j]
    xx.test <- as.matrix(xx.test)%*% xx.pcs[,1:j]
    
    data.svd <- data.frame(y=c(ytrain,ytest),rbind(xx.train,xx.test))
    out.pcs <- glm(y~.,data.svd,subset=1:nrow(xx.train),family="binomial",maxit=100)
    p.hat <- predict(out.pcs,newdata=data.svd[(nrow(xx.train)+1):nrow(data.svd),],type="response")
    y.hat <- factor(ifelse(p.hat>0.5,1,0),level=0:1)
    err[i,j] <- mean(y.hat!=ytest)
  }
}

col(err)

best_no <- which.min(colMeans(err))
best_no 

##PCR ####
svd.xx <- svd(xx)
xx.pcs <- svd.xx$v

test.xx <- test[,-1]
for(i in 1:ncol(test.xx)){
  test.xx[,i] <- (test.xx[,i]-mean(x[,i]))/sd(x[,i])
}

xx.train <- xx%*%xx.pcs[,1:best_no]
xx.test <- as.matrix(test.xx)%*%xx.pcs[,1:best_no]

data.svd <- data.frame(y=c(train$y,test$y),rbind(xx.train,xx.test))
out.pcs <- glm(y~,data.svd,subset=1:nrow(xx.train),family="binomial")
pp.hat <- predict(out.pcs, newdata = data.svd[(nrow(xx.train)+1):nrow(data.svd),],type="response")
yhat <- factor(ifelse(pp.hat>0.5,1,0),levels=0:1)

table(yhat,test$y)

mean(yhat!=test$y)



#SIS=
#get rid of some redundant variables when the variables are too much than the obs 