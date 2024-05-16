install.packages("e1071")
library(e1071)
library(MASS)
library(ggplot2)
library(tidyverse)
library(corrplot)
library(car)
library(ROCR)
library(caret)



setwd('/Users/nupoor/Documents/MSDA/DA Arka Roy')
bb_train <- read.csv("/Users/nupoor/Documents/MSDA/DA Arka Roy/BBBC-Train.csv", sep=',')
bb_test <- read.csv("/Users/nupoor/Documents/MSDA/DA Arka Roy/BBBC-Test.csv", sep=',')

head(bb_train)
str(bb_train)

table(bb_train$Choice)
table(bb_train$Gender)
table(bb_train$Amount_purchased)
table(bb_train$First_purchased)

#as.factor from integer as we need to do it if its int or chr
bb_train$Choice = as.factor(bb_train$Choice)
bb_train$Gender = as.factor(bb_train$Gender)

bb_test$Choice = as.factor(bb_test$Choice)
bb_test$Gender = as.factor(bb_test$Gender)

#Correlation
bb_num <- dplyr::select_if(bb_train, is.numeric)                                                    
M = cor(bb_num)
corrplot(M, method = c("number"))


#remove Last_pruchase from both as train and test
bb_train = dplyr::select(bb_train, -Last_purchase)
bb_test = dplyr::select(bb_test, -Last_purchase)

bb_train = dplyr::select(bb_train, -P_Child)


bb_train = dplyr::select(bb_train, -P_Art)


bb_train = dplyr::select(bb_train, -P_DIY)


bb_train = dplyr::select(bb_train, -First_purchase)



bb_num <- dplyr::select_if(bb_train, is.numeric)                                                    
M = cor(bb_num)
corrplot(M, method = c("number"))


ggplot(data = bb_train) +
  geom_bar(mapping = aes(x=Choice, fill = Gender), position = "dodge")

ggplot(data = bb_train) +
  geom_bar(mapping = aes(x=Choice, fill = First_purchase), position = "dodge")

##set.seed and confusion matrix

set.seed(1)


m1.log = glm(Choice ~ ., data = bb_train, family = binomial, control=list(maxit=50))
summary(m1.log)
vif(m1.log)

predprob_log <- predict.glm(m1.log, bb_test, type = "response")  
predclass_log = ifelse(predprob_log >= 0.5, "1", "0")

caret::confusionMatrix(as.factor(predclass_log), bb_test$Choice, positive = "1")

#balancing the data

bb_train_buy= bb_train %>% filter(Choice == "1")
bb_train_no = bb_train %>% filter(Choice == "0")
set.seed(1)
sample_bb = sample_n(bb_train_no, nrow(bb_train_buy))
bb_bal = rbind(bb_train_buy,sample_bb)


m1.log_bal = glm(Choice ~ ., data = bb_bal, family = binomial)
summary(m1.log_bal) ## look at results
vif(m1.log_bal)

predprob_log_bal <- predict.glm(m1.log_bal, bb_test, type = "response")  ## for logit
predclass_log_bal = ifelse(predprob_log_bal >= 0.5, "1", "0")

caret::confusionMatrix(as.factor(predclass_log_bal), bb_test$Choice, positive = "1")


##optimal cutoff

#ROC Curve and AUC
pred <- prediction(predprob_log_bal, bb_train$Choice)
#Predicted Probability and True Classification

# area under curve
auc <- round(as.numeric(performance(pred, measure = "auc")@y.values),3)

#plotting the ROC curve and computing AUC
perf <- performance(pred, "tpr","fpr")
plot(perf,colorize = T, main = "ROC Curve")
text(0.5,0.5, paste("AUC:", auc))

# computing threshold for cutoff to best trade off sensitivity and specificity
#first sensitivity
plot(unlist(performance(pred, "sens")@x.values), unlist(performance(pred, "sens")@y.values), 
     type="l", lwd=2, 
     ylab="Sensitivity", xlab="Cutoff", main = paste("Maximized Cutoff\n","AUC: ",auc))

par(new=TRUE) # plot another line in same plot

#second specificity
plot(unlist(performance(pred, "spec")@x.values), unlist(performance(pred, "spec")@y.values), 
     type="l", lwd=2, col='red', ylab="", xlab="")
axis(4, at=seq(0,1,0.2)) #specificity axis labels
mtext("Specificity",side=4, col='red')

#find where the lines intersect
min.diff <-which.min(abs(unlist(performance(pred, "sens")@y.values) - unlist(performance(pred, "spec")@y.values)))
min.x<-unlist(performance(pred, "sens")@x.values)[min.diff]
min.y<-unlist(performance(pred, "spec")@y.values)[min.diff]
optimal <-min.x #this is the optimal points to best trade off sensitivity and specificity

abline(h = min.y, lty = 3)
abline(v = min.x, lty = 3)
text(min.x,0,paste("optimal threshold=",round(optimal,2)), pos = 3)

## SVM

form1 <- Choice ~ .

tuned = tune.svm(form1, data = bb_train,
                 gamma = seq(0.01, .1, by = 0.01),
                 cost = seq(.1, 1, by = 0.1))

tuned$best.parameters

mysvm= svm(formula=form1, data=bb_train, gamma=tuned$best.parameters$gamma,
           cost=tuned$best.parameters$cost )
summary(mysvm)

svmpredict=predict(mysvm, bb_test, type = "response")
table(pred=svmpredict, true=bb_test$Choice)

caret::confusionMatrix(as.factor(svmpredict), bb_test$Choice, positive = "1")
