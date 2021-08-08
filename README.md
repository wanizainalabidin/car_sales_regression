# Regression Models Predicting Price of Car
## Welcome to my project on Car Sales Regression

### Created by Wani Zainal Abidin


#Regression Models to Predict Price of Cars
dir_name <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dir_name)


#get dataset
pwc <- read.csv("car_price.csv")

#loading tidyverse package to drop any NAs
library(tidyverse)
install.packages("tidyverse")
pwc <- pwc %>% drop_na()

#Data Prep 1: converting categorical variables into factors
pwc$CarName = as.factor(pwc$CarName)
pwc$fueltype = as.factor(pwc$fueltype)
pwc$aspiration = as.factor(pwc$aspiration)
pwc$doornumber = as.factor(pwc$doornumber)
pwc$carbody = as.factor(pwc$carbody)
pwc$drivewheel = as.factor(pwc$drivewheel)
pwc$enginelocation = as.factor(pwc$enginelocation)
pwc$enginetype = as.factor(pwc$enginetype)
pwc$cylindernumber = as.factor(pwc$cylindernumber)
pwc$fuelsystem = as.factor(pwc$fuelsystem)

#Data Prep 2: converting factors into numeric for regression
pwc$CarName = as.numeric(pwc$CarName)
pwc$fueltype = as.numeric(pwc$fueltype)
pwc$aspiration = as.numeric(pwc$aspiration)
pwc$doornumber = as.numeric(pwc$doornumber)
pwc$carbody = as.numeric(pwc$carbody)
pwc$drivewheel = as.numeric(pwc$drivewheel)
pwc$enginelocation = as.numeric(pwc$enginelocation)
pwc$enginetype = as.numeric(pwc$enginetype)
pwc$cylindernumber = as.numeric(pwc$cylindernumber)
pwc$fuelsystem = as.numeric(pwc$fuelsystem)

# Data Prep 3: removing the car_id column since it is a primary key not necessary for the analysis
pwc$car_ID <- NULL

# Data Prep 4: finding highly correlated values to drop one variable (for consideration)
cor.pwc <- cor(pwc, use="complete.obs")

#strongly correlated values removed
pwc1 <- pwc
pwc1$curbweight <- NULL

#install packages to split
install.packages("caTools")
library(caTools)

# Data Prep 5: split the test set without k-Fold Cross Validation
set.seed(123)
split=sample.split(pwc$price, SplitRatio = 0.9)
training.set<- subset(pwc, split==TRUE)
test.set <- subset(pwc, split==FALSE)

#Train Model 1.1: linear regression of training set
training.lin.reg <- lm(formula= price ~., data=training.set)
summary(training.lin.reg)

#Metric for Train set: RMSE
sqrt(mean((training.set$price - predict(training.lin.reg, training.set)) ^ 2))


#Test Model 1.1: predicting test set results
test.car.price <- predict(training.lin.reg, test.set)

#Evaluate Model 1.1: using R2 and RMSE value to determine predictability of model 
  ## This evaluation will be improved through CV which is the final RMSE value
R2(test.car.price, test.set$price)
RMSE(test.car.price, test.set$price)

#installing packages for k-Fold Cross Validation
install.packages("caret") 
library(caret)

# Evaluate Model 1.1: using function for k-Fold Cross validation and mean of R2
cv.fold <- createFolds(training.set$price, k=10)

cross.v <- lapply(cv.fold, function(x){
  training.with.fold <- training.set[-x,]
  test.with.fold <- training.set[x,]
  lin.reg <- lm(formula= price ~., data = training.with.fold)
  prediction <- predict(training.lin.reg, newdata = test.with.fold)
  model.validation <- RMSE(prediction, test.with.fold$price)
  return(model.validation)
})

#calculating the mean of RMSE of Model 1.2 
avg.accuracy <- mean(as.numeric(cross.v))

#Train Model 1.2: Decision Tree 
library(caret)
library(rpart)
library(rpart.plot)
DT <- rpart(formula= price ~., data=training.set)

#Test Model 1.2 : Predicting Test Set 
DT.pred <- predict(DT, test.set)

#Metric for Model 1.2: RMSE
sqrt(mean((training.set$price - predict(DT, training.set)) ^ 2))


#Evaluating Model 1.2: RMSE and R2 Without Cross Validation
  ## This evaluation will be improved through CV which is the final RMSE value
R2(DT.pred, test.set$price)
RMSE(DT.pred, test.set$price)

# Visualising the Decision Tree Regression
plot(DT, uniform = TRUE, main = "MPG Decision Tree using Regression")
text(DT, use.n=TRUE, cex=0.6)
rpart.plot(DT)


#Evaluate Model 1.2: K-Cross Validation with Decision Trees
dt.fold <- createFolds(training.set$price, k=10)
dt.cv <- lapply(dt.fold, function(x){
  train.dtfold <- training.set[-x,]
  test.dtfold <- training.set[x,]
  DT.mod<- rpart(formula= price ~., data=train.dtfold)
  DT.prediction <- predict(DT, test.dtfold)
  DT.eval<- RMSE(DT.prediction, test.dtfold$price)
  return(DT.eval)
})

# Calculating Accuracy of Model 1.2 
dt.accuracy <- mean(as.numeric(dt.cv))

#Train Model 1.3: Random forest regression 
install.packages("randomForest")
library(randomForest)
random.regressor <- randomForest(x=training.set[1:24],
                                 y= training.set$price,ntree = 500, mtry=13, importance = TRUE)
#Test Model 1.3: Predict random forest
random.predict <- predict(random.regressor, test.set)

#Metric for Model 1.3: RMSE
sqrt(mean((training.set$price - predict(random.regressor, training.set)) ^ 2))


#Evaluation of Model 1.3: Calculating RMSE and R2
## This evaluation will be improved through CV which is the final RMSE value
RMSE(random.predict, test.set$price)
R2(random.predict, test.set$price)

#Evaluate Model 1.3: k-Fold Cross Validation 
rf.fold <- createFolds(training.set$price, k=10)
rf.cv <- lapply(rf.fold, function(x){
  train.rf.fold <- training.set[-x,]
  test.rf.fold <- training.set[x,]
  RF.mod<- randomForest(x=training.set[1:24],
                        y= training.set$price,ntree = 500)
  RF.prediction <- predict(RF.mod,test.rf.fold)
  RF.eval<- RMSE(RF.prediction, test.rf.fold$price)
  return(RF.eval)
})

#Calculate Accuracy of Model 1.3: Average of RMSE
rf.accuracy <- mean(as.numeric(rf.cv))

#Grid Search to Find optimal parameters
train(form = price~., data=training.set, method="rf")

install.packages("varImp")
library(varImp)

#Feature Selection of Indepedent Variables
importance(random.regressor, type=1)


#Data Prep: Split data after removing curbweight attribute
split = sample.split(pwc1$price, SplitRatio = 0.9)
curb.train <- subset(pwc1, split==TRUE)
curb.test <- subset(pwc1, split==FALSE)


# Train Model 2.1: lin reg of train set after removing curbweight attribute
results.train <-lm(formula= price~., data=curb.train)
summary(results.train)
#Metric for Model 2.1: RMSE
sqrt(mean((curb.train$price - predict(results.train, curb.train)) ^ 2))


# Test Model 2.1: predicting results and finding R2 after removing curbweight
test.price <- predict(results.train, newdata = curb.test)

# Evaluate Model 2.1: using R2 and RMSE value to determine predictability of model 
## This evaluation will be improved through CV which is the final RMSE value
R2(test.price, curb.test$price)
RMSE(test.price, curb.test$price)

# Evaluate Model 2.1: K-Folds Cross Validation after removing curbweight attribute
folds <- createFolds(curb.train$price, k=10)
cv <- lapply(folds, function(fold){
  training_fold<-curb.train[-fold,]
  test_fold<-curb.train[fold,]
  linear <- lm(formula= price~., data=training_fold)
  y_price <- predict(results.train, newdata = test_fold)
  accuracy <- RMSE(y_price, test_fold$price)
  return(accuracy)
})

#calculating the mean of R2 for Model 2.1
mean.accuracy <- mean(as.numeric(cv))

#Train Model 2.2: Decision Tree 
library(caret)
library(rpart)
library(rpart.plot)
DT.curb <- rpart(formula= price ~., data=curb.train)

#Test Model 2.2 : Predicting Test Set 
DT.pred.curb <- predict(DT, curb.test)

#Metric for Model 2.2: RMSE
sqrt(mean((curb.train$price - predict(DT, curb.train)) ^ 2))


#Evaluating Model 2.2: RMSE and R2 Without Cross Validation
## This evaluation will be improved through CV which is the final RMSE value
R2(DT.pred.curb, curb.test$price)
RMSE(DT.pred.curb, curb.test$price)

# Visualising the Decision Tree Regression
plot(DT.curb, uniform = TRUE, main = "MPG Decision Tree using Regression")
text(DT.curb, use.n=TRUE, cex=0.6)
rpart.plot(DT.curb)


#Evaluate Model 2.2: K-Cross Validation with Decision Trees
dt.fold.curb <- createFolds(curb.train$price, k=10)
dt.cv.curb <- lapply(dt.fold.curb, function(x){
  train.dtfold <- curb.train[-x,]
  test.dtfold <- curb.train[x,]
  DT.mod<- rpart(formula= price ~., data=train.dtfold)
  DT.prediction <- predict(DT.curb, test.dtfold)
  DT.eval<- RMSE(DT.prediction, test.dtfold$price)
  return(DT.eval)
})

# Calculating Accuracy of Model 2.2
dt.accuracy.curb <- mean(as.numeric(dt.cv.curb))

#Train Model 2.3: Random forest regression 
install.packages("randomForest")
library(randomForest)
random.regressor.curb <- randomForest(x=curb.train[1:23],
                                      y= curb.train$price,ntree = 500, mtry = 12, importance = TRUE)
#Test Model 2.3: Predict random forest
random.predict.curb <- predict(random.regressor.curb, curb.test)

#Metric for Model 2.3: RMSE
sqrt(mean((curb.train$price - predict(random.regressor.curb, curb.train)) ^ 2))


#Evaluation of Model 2.3: Calculating RMSE and R2
## This evaluation will be improved through CV which is the final RMSE value
RMSE(random.predict.curb, curb.test$price)
R2(random.predict.curb, curb.test$price)

#Evaluate Model 2.3: k-Fold Cross Validation 
rf.fold.curb <- createFolds(curb.train$price, k=10)
rf.cv.curb <- lapply(rf.fold.curb, function(x){
  train.rf.fold <- curb.train[-x,]
  test.rf.fold <- curb.train[x,]
  RF.mod<- randomForest(x=curb.train[1:23],
                        y= curb.train$price,ntree = 500)
  RF.prediction <- predict(RF.mod,test.rf.fold)
  RF.eval<- RMSE(RF.prediction, test.rf.fold$price)
  return(RF.eval)
})

#Grid search to find optimal parameters
train(form = price~., data=curb.train, method="rf")

#Calculate Accuracy of Model 2.3: Average of RMSE
rf.accuracy.curb <- mean(as.numeric(rf.cv.curb))

#Feature selection for Model 2.3
importance(random.regressor.curb, type=1)





