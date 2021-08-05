#assignment
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
split=sample.split(pwc$price, SplitRatio = 0.5)
training.set<- subset(pwc, split==TRUE)
test.set <- subset(pwc, split==FALSE)

#Train Model: linear regression of training set
training.lin.reg <- lm(formula= price ~., data=training.set)
summary(training.lin.reg)

#Test Model: predicting test set results
test.car.price <- predict(training.lin.reg, test.set)

#Evaluate Model 1: using R2 and RMSE value to determine predictability of model 
R2(test.car.price, test.set$price)
RMSE(test.car.price, test.set$price)


#installing packages for k-Fold Cross Validation
install.packages("caret") 
library(caret)

# Evaluate Model 2: using function for k-Fold Cross validation and mean of R2
cv.fold <- createFolds(training.set$price, k=10)

cross.v <- lapply(cv.fold, function(x){
  training.with.fold <- training.set[-x,]
  test.with.fold <- training.set[x,]
  lin.reg <- lm(formula= price ~., data = training.with.fold)
  prediction <- predict(training.lin.reg, newdata = test.with.fold)
  model.validation <- R2(prediction, test.with.fold$price)
  return(model.validation)
})

#calculating the mean of R2 
avg.accuracy <- mean(as.numeric(cross.v))

#Data Prep: Split data after removing curbweight attribute
split = sample.split(pwc1$price, SplitRatio = 0.5)
curb.train <- subset(pwc1, split==TRUE)
curb.test <- subset(pwc1, split==FALSE)


# Train Model: lin reg of train set after removing curbweight attribute
results.train <-lm(formula= price~., data=curb.train)
summary(results.train)

# Test Model: predicting results and finding R2 after removing curbweight
test.price <- predict(results.train, newdata = curb.test)

# Evaluate Model: using R2 and RMSE value to determine predictability of model 
R2(test.price, curb.test$price)
RMSE(test.price, curb.test$price)

# Evaluate Model 2: K-Folds Cross Validation after removing curbweight attribute
folds <- createFolds(curb.train$price, k=10)
cv <- lapply(folds, function(fold){
  training_fold<-curb.train[-fold,]
  test_fold<-curb.train[fold,]
  linear <- lm(formula= price~., data=training_fold)
  y_price <- predict(results.train, newdata = test_fold)
  accuracy <- R2(y_price, test_fold$price)
  return(accuracy)
})

#calculating the mean of R2
mean.accuracy <- mean(as.numeric(cv))
