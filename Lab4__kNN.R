# Lab VI - October 3rd - kNN

## Simulated example ####
#install.packages('mvtnorm')
library(mvtnorm)

# The populations' parameters
mu0 <- c(1,2)
mu1 <-c(2,1.5)
sigma_0<- matrix(c(2,0.5,0.5,2),2,2)
sigma_1 <- matrix(c(1.5,0,0,1.5),2,2)

set.seed(12345)
x0 <- rmvnorm(6,mu0,sigma_0)
x1 <- rmvnorm(6,mu1,sigma_1)
x <-rbind(x0,x1)
y <- rep(c(0,1),c(6,6))

# Plot the observations
plot(x,col=ifelse(y==1,"cornflowerblue","coral"),
     lwd=1.5,xlab="X1",ylab="X2",
     main="Simulated Example")

# The grid of points 
x1.news <- seq(from=0,to=4,length.out=100) 
x2.news <- seq(from=0,to=5,length.out=90)
x.new <- expand.grid(x=x1.news, y=x2.news)

library(class)  #the library that contains  the knn()function

# 1-NN ####
#with prob =T the proportion of the votes for the winning class are returned as attribute
out.1 = knn(train=x,test=x.new,cl=y,k=1,prob=T)

prob.1NN <- attributes(out.1)$prob #all of them are equal to 1
prob.1NN <- ifelse(out.1==0,prob.1NN,1-prob.1NN)

prob.1NN.m<-matrix(prob.1NN,length(x1.news),length(x2.news))

#3-NN ####
out.3 <- knn(train=x,test=x.new,cl=y,k=3,prob=T)

prob.3NN <- attributes(out.3)$prob #all of them are > 0.5
prob.3NN <- ifelse(out.3==0,prob.3NN,1-prob.3NN) #getBackToCheck

prob.3NN.m<-matrix(prob.3NN,length(x1.news),length(x2.news))

# 5-NN ####
out.5 <- knn(train=x,test=x.new,cl=y,k=5,prob=T)

prob.5NN <- attributes(out.5)$prob # all of them are >0.5
prob.5NN <- ifelse(out.5==0,prob.5NN, 1-prob.5NN)

prob.5NN.m<-matrix(prob.5NN,length(x1.news),length(x2.news))

#Plot the results####
par(mfrow = c(1,3))

# Plot no.1 +++ 1-NN
contour(x1.news,x2.news,prob.1NN.m,levels=0.5,labels="",xlab="x1",ylab="x2",
        main="1-NN decision boundary")
#To color the allocation areas
points(x.new,pch=".",cex=1.2,col=ifelse(prob.1NN>0.5,"coral","cornflowerblue"))

#To add the original points
points(x,col=ifelse(y==1,"cornflowerblue","coral"),lwd=1.5)

# Plot no2. +++ 3-NN 
contour(x1.news, x2.news, prob.3NN.m,levels=0.5,labels="",
        xlab="x1",ylab="x2",main="3-NN decision boundary")

#To color the allocation areas 
points(x.new, pch=".",cex=1.2,col=ifelse(prob.3NN>0.5,"coral","cornflowerblue"))

# To add the original points
points(x,col=ifelse(y==1,"cornflowerblue","coral"),lwd=1.5)


#Plot no.3 ++++ 5-NN 
contour(x1.news, x2.news, prob.5NN.m, levels=0.5, labels="",xlab="x1",ylab="x2",
        main="5-NN decision boundary")

# To color the allocation areas
points(x.new, pch=".",cex=1.2, col=ifelse(prob.5NN>0.5,"coral","cornflowerblue"))

# To add the original points 
points(x,col=ifelse(y==1,"cornflowerblue","coral"),lwd=1.5)

# Comparison between k-NN and Bayes Classfier ####
# The probability of each observation points 
# according to the two population 
dx.new.0 <- dmvnorm(x.new, mean=mu0,sigma=sigma_0)
dx.new.1 <- dmvnorm(x.new, mean=mu1,sigma=sigma_1)

#P(Y=0|X_0)
posterior.x.new0 <- (0.5*dx.new.0)/(0.5*dx.new.0+0.5*dx.new.1)

# The posterior probability of belonging to population 0
# for each point 
p.xnew0 <- matrix(posterior.x.new0,length(x1.news),length(x2.news))


# Let's build the plot 
contour(x1.news, x2.news, p.xnew0, levels=0.5, labels=""
        ,xlab="x1",ylab="x2",
        main="Bayese decision boundary")
points(x.new, pch=".",cex=1.2, col=ifelse(posterior.x.new0>0.5,"coral","cornflowerblue"))

## 2. Example: South Africa Heartdata ####
library(ElemStatLearn)
data("SAheart")
summary(SAheart)



n <- nrow(SAheart)
x <- SAheart[,-c(5,10)] #remove categorical variable and response
y <- SAheart[,10] #extract the response

# Train + Validation Set
library(class)
set.seed(17)
index <- sample(1:n,ceiling(n/2),replace=F)

train <- x[index,]
train_y <- y[index]
train_std<- scale(train,T,T) #Standardizes 
ntrain <- nrow(train)

# 5-NN cv 
K <-5 
set.seed(1234)
folds <- sample(1:K,ntrain,replace=T)
k<- c(1,3,5,15,25,75) # vector containing the no of neighbors
err_cv <- matrix(NA,5,6,dimnames=list(NULL,paste("K=",c(1,3,5,15,25,75))))

for (i in 1: K){
  x.val <- train_std[folds==i,]
  x.train <-train_std[folds!=i,]
  y.val <- train_y[folds==i]
  y.train <- train_y[folds!=i]
  
  for(j in 1:length(k)){
    y.hat <- knn(train=x.train,test=x.val,cl=y.train,k=k[j])
    err_cv[i,j] <-mean(y.hat!=y.val)
  }
}                 
apply(err_cv,2,mean)

which.min(apply(err_cv,2,mean))
# Before estimating the test error on the validation set 
# we need to standardize the latter using information of the training set.

mean_x <- apply(train,2,mean) # compute the mean of each column 
sd_x <- apply(train,2,sd)

test <- x[-index,]
test_y<- y[-index]

# Via for-cycle
test_std <- test
for(j in 1:ncol(test)){
  test_std[,j] <- (test[,j]-mean_x[j])/sd_x[j] #use the training set stat to standardlize 
}

# Alternatively, via matrix-product 
mean.m <- matrix(mean_x,nrow(test),ncol(test),
                 byrow=T)
sd.m <- matrix(sd_x,nrow(test),ncol(test),
               byrow=T)
test_std2 <- (test-mean.m)/sd.m

# Prediction 
y.hat <- knn(train=train_std,test=test_std,cl=train_y,k=5)
misc(y.hat,test_y)
