# Lab X - October 15th - Dimension Reduction methods ####

## Prostate dataset ####
library(ElemStatLearn)
data("prostate")

install.packages('pls')
library(pls)

set.seed(1234)
pcr.fit<-pcr(lpsa~.,data=prostate[,-ncol(prostate)],
             scale=TRUE,validation="CV")
summary(pcr.fit) #return the root MSE 


validationplot(pcr.fit,val.type = "MSEP",
               legendpos="top")
#The smallest MSEP is at M=8
#This amount is to perform least square bc when all of the components are used in PCR 
# no dimension reduction occurs \


### Select the optimal no. of PCs ####
#select #  of components based on Heuristics 
# choose the model with leastes components which is still one std from the best model 
ncomp.onesigma<-selectNcomp(pcr.fit,
                            method="onesigma",
                            plot=T)
#select # of components via randomization 
#permutation approach , to test whetehr adding a new component is beneficial (backward method)
ncomp.permut<-selectNcomp(pcr.fit,
                          method="randomization",
                          plot=T)

summary(pcr.fit)

### PCR on training set & evaluate test set performance ####
set.seed(1234)
x<-prostate[,-ncol(prostate)]
y<-prostate$lpsa
n<-nrow(x)

train<-sample(1:n,ceiling(n/2))
y.test<-y[-train]

set.seed(1234)
pcr.fit<-pcr(lpsa~.,data=x, subset=train,
            scale=T,validation="CV")
validationplot(pcr.fit, val.type="MSEP",legendpos="top")

pcr.pred<-predict(pcr.fit,x[-train,1:8],ncomp = 5)
mean((pcr.pred-y.test)^2)

## Alternatively, with eigen() ####
#eigenvectorsofthecorrelationmatrixofX
x.pcs<-eigen(cor(prostate[train,1:8]))$vectors #loading of PC
x.train<-scale(prostate[train,1:8],T,T)%*%x.pcs[,1:5]
# remember to scale the data 
test.x <-x[-train,1:8]
for (i in 1:8){ #standarlize the test as well, no variance of test set, use the one from train
  test.x[,i]<-(test.x[,i]-mean(prostate[train,i]))/sd(prostate[train,i])
} 
x.test <-as.matrix(test.x)%*%x.pcs[,1:5] #test PCs 
y.train<-prostate[train,]$lpsa
y.test<-prostate[-train,]$lpsa

data.pcs<-data.frame(y=c(y.train,y.test),rbind(x.train,x.test))
out.pcs<-lm(y~.,data=data.pcs,subset=1:length(train))
# run LR to see how well PC explains?
yhat<-predict(out.pcs,
              newdata = data.pcs[(length(train)+1):nrow(data.pcs),-1])
mean((yhat-y.test)^2)


## Alternatively, from svd() ####
#to find the right singular vectors of matrixX:
xx<-scale(prostate[train,1:8],T,T)
svd.xx<-svd(xx)
xx.pcs<-svd.xx$v[,1:5]

xx.train<-xx%*%xx.pcs
test.xx <- prostate[-train,1:8]
for (i in 1:8){
  test.xx[,i] <- (test.xx[,i]-mean(prostate[train,i]))/sd(prostate[train,i])
}
xx.test<-as.matrix(test.xx)%*%xx.pcs[,1:5]
y.train<-prostate[train,]$lpsa
y.test<-prostate[-train,]$lpsa

data.svd<-data.frame(y=c(y.train,y.test),
                     rbind(xx.train,xx.test))
out.pcs.svd<-lm(y~.,data=data.svd,subset=1:length(train)) #similarly for svd()
yy.hat.svd<-predict(out.pcs.svd,
                    newdata=data.svd[(length(train)+1):nrow(data.svd),-1])
mean((yy.hat.svd-y.test)^2)



# final PCR model is hard to intepretate
# because it doesn't provide variables selections 
# use the full data with the optimal components M=5, 
# to do cross validation 
pcr.fit<-pcr(lpsa~.,data=prostate[,-ncol(prostate)],scale=TRUE,ncomp=5)
summary(pcr.fit)



