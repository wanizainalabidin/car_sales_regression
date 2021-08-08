# Welcome to my project on Car Prices 
### In this project, we are trying to predict the price of cars based on several independent variables. This stems from the company's (Geely Auto) intention of entering into the American market. 

In order to do this, we will first do the following to set our working directory

```
dir_name <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dir_name)
```

After, we will load our data (in csv format) by running the following code: 
```
pwc <- read.csv("car_price.csv")
```

Next, we will remove the NA values by loading the tidyverse package. Alternatively, we can also use the function na.omit(): 
```
pwc <- pwc %>% drop_na()
```
Since our data contains both categorical and numerical values, we have to convert the categorical variables into factors:
```
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
```
Now that R recognised these variables as factors, we will convert them into numeric values for our machine learning model. 
```
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
```

To check if the independent variables have been properly converted, we can use the following function: 
```
str(pwc)
``` 
Nice! Now that everything is in order, we have an extra variable that is not needed in the analysis - Car ID. Since CarID is a primary key, we can remove this variable by running the following code:
```
pwc$car_ID <- NULL
```
Next, we can use the following code to find highly correlated values. By doing so, we can identify if our model performs better with or without the highly correlated values

```
cor.pwc <- cor(pwc, use="complete.obs")
```
Seems like curbweight is highly correlated with other variables such as car length, car width and wheelbase. We will run the regression model with and without curb weight
```
pwc1 <- pwc
pwc1$curbweight <- NULL

```
Now that our data has been pre-processed, we can split the data using the following package and code
```
install.packages("caTools")
library(caTools)
set.seed(123)
split=sample.split(pwc$price, SplitRatio = 0.9)
training.set<- subset(pwc, split==TRUE)
test.set <- subset(pwc, split==FALSE)
```
After splitting, we can test our linear regression model
```
training.lin.reg <- lm(formula= price ~., data=training.set)
summary(training.lin.reg)
```
Output running the above will give us the list of variables with the standard error, p-values and estimate score. 
![alt text](https://github.com/wanizainalabidin/car_sales_regression/blob/main/ML%20Images/Model%201%20Linear%20Regression.png)

To understand if our model has performed well, we first need to compute the RMSE score of the train set
```
sqrt(mean((training.set$price - predict(training.lin.reg, training.set)) ^ 2))

```
Now, we can test our model through the following code:

```
test.car.price <- predict(training.lin.reg, test.set)

```

After testing the model, we need to measure the performance of our test model. This is done by calculating the R square and RMSE scores
```
R2(test.car.price, test.set$price)
RMSE(test.car.price, test.set$price)

```

Now that we get the scores, can we ensure that our model performs even better? We will use k-fold cross validation for this.
```
cv.fold <- createFolds(training.set$price, k=10)

cross.v <- lapply(cv.fold, function(x){
  training.with.fold <- training.set[-x,]
  test.with.fold <- training.set[x,]
  lin.reg <- lm(formula= price ~., data = training.with.fold)
  prediction <- predict(training.lin.reg, newdata = test.with.fold)
  model.validation <- RMSE(prediction, test.with.fold$price)
  return(model.validation)
})
```

Let's calculate the score of the RMSE for our model after k-fold cross validation

```
avg.accuracy <- mean(as.numeric(cross.v))
```

Since we have done our linear regression model, we can now proceed with the second regression model - decision tree. We will first install and/or load the caret and rpart package

```
library(caret)
library(rpart)
library(rpart.plot)
```

Now that we have downloaded the packages, we can train our model using the rpart function

```
DT <- rpart(formula= price ~., data=training.set)
```

We can also proceed with testing our test set
 ```
 DT.pred <- predict(DT, test.set)

```

Let's calculate the RMSE score of the train model for comparison with the test model and later the k-fold cross validation
```
sqrt(mean((training.set$price - predict(DT, training.set)) ^ 2))
```

We will also calculate the RMSE value of the test set
```
RMSE(DT.pred, test.set$price)

```

To visualise the model, we can use the rpart.plot function 
```
rpart.plot(DT)

```

The decision tree below describes the terminal and root node which corresponds to significant variables that predict the prices of cars.
![alt text](https://github.com/wanizainalabidin/car_sales_regression/blob/main/ML%20Images/Model%201%20DT.png)

After calculating the RMSE scores, we should evaluate the performance of our model through the k-fold cross validation
```
dt.fold <- createFolds(training.set$price, k=10)
dt.cv <- lapply(dt.fold, function(x){
  train.dtfold <- training.set[-x,]
  test.dtfold <- training.set[x,]
  DT.mod<- rpart(formula= price ~., data=train.dtfold)
  DT.prediction <- predict(DT, test.dtfold)
  DT.eval<- RMSE(DT.prediction, test.dtfold$price)
  return(DT.eval)
})

```

We can futher compare the model's performance by calculating the RMSE score after k-fold cross validation by running the following code: 
```
dt.accuracy <- mean(as.numeric(dt.cv))

```

The last regression model that we will use for this case (with the curb weight variable) will be random forest. To do so, we will have to install the following packages
```
install.packages("randomForest")
library(randomForest)
```

Now that we have loaded the packages, we can train our model

```
random.regressor <- randomForest(x=training.set[1:24],
                                 y= training.set$price,ntree = 500, mtry=13, importance = TRUE)
```

Since we have trained the model, we can also test the model to see how well the model has performed
```
random.predict <- predict(random.regressor, test.set)

```

To measure performance, we will calculate the RMSE score of training set which will be compared to with the test set
```
sqrt(mean((training.set$price - predict(random.regressor, training.set)) ^ 2))

```
The RMSE value of the test set will be gathered through the following code:
```
RMSE(random.predict, test.set$price)

```

We can evaluate the model by doing k-fold cross validation
```
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
```

To calculate the mean score of the RMSE, we will run the following code
```
train(form = price~., data=training.set, method="rf")

```

As peformed above, we have specified the parameter, mtry=13. To get this magic number, we run the following code which allows R provide us with the optimal value for the parameter.
```
train(form = price~., data=training.set, method="rf")

```

To determine independent variables that are significant in predicting the prices of cars, feature selection can be done by calling the importance function
```
importance(random.regressor, type=1)

```

This concludes the regression models on the first dataset - with the highly correlated value, curb weight. We will now move ahead to look at how model performance changes when we exclude curb weight.

To use linear regression on the second dataset, we will start by splitting the dataset with 90-10 proportion.
```
split = sample.split(pwc1$price, SplitRatio = 0.9)
curb.train <- subset(pwc1, split==TRUE)
curb.test <- subset(pwc1, split==FALSE)
```

Now, we can train the model by running the lm function
```
results.train <-lm(formula= price~., data=curb.train)
summary(results.train)

```

The results by calling the summary function is as follows:
![alt text](https://github.com/wanizainalabidin/car_sales_regression/blob/main/ML%20Images/Model%202%20Linear%20Regression.png)

We will now calculate the RMSE score of the train set without the k-fold cross validation.

```
sqrt(mean((curb.train$price - predict(results.train, curb.train)) ^ 2))

```

Afterwards, we can test the model on the test set

```
test.price <- predict(results.train, newdata = curb.test)

```

To compare model's performance, we will calculate the value of the RMSE score of the test set

```
RMSE(test.price, curb.test$price)

```

Now, we evaluate the model using the k-fold cross validation
```
folds <- createFolds(curb.train$price, k=10)
cv <- lapply(folds, function(fold){
  training_fold<-curb.train[-fold,]
  test_fold<-curb.train[fold,]
  linear <- lm(formula= price~., data=training_fold)
  y_price <- predict(results.train, newdata = test_fold)
  accuracy <- RMSE(y_price, test_fold$price)
  return(accuracy)
})
```

Since 10 RMSE scores were returned, we can find the mean of the RMSE score for comparison

```
mean.accuracy <- mean(as.numeric(cv))

```

The second model we will use will be the decision tree. We will build the model using the following code

```
DT.curb <- rpart(formula= price ~., data=curb.train)

```

We can now predict our test set using the model that was built above

```
DT.pred.curb <- predict(DT, curb.test)

```

Calculate the RMSE score of the train set using the following code:

```
sqrt(mean((curb.train$price - predict(DT, curb.train)) ^ 2))

```

Calculate the RMSE score of the test set using the following code:

```
RMSE(DT.pred.curb, curb.test$price)
```

We can visualise the decision tree by running the following code:
```
rpart.plot(DT.curb)
```

Let's now visualise the deicision tree to identify significant variables through the root and terminal nodes.
![alt text](https://github.com/wanizainalabidin/car_sales_regression/blob/main/ML%20Images/Model%202%20DT.png)


Evaluate the model performance through the k-fold cross validation
```
dt.fold.curb <- createFolds(curb.train$price, k=10)
dt.cv.curb <- lapply(dt.fold.curb, function(x){
  train.dtfold <- curb.train[-x,]
  test.dtfold <- curb.train[x,]
  DT.mod<- rpart(formula= price ~., data=train.dtfold)
  DT.prediction <- predict(DT.curb, test.dtfold)
  DT.eval<- RMSE(DT.prediction, test.dtfold$price)
  return(DT.eval)
})
```

Since the RMSE scores of 10 folds were created, we can use the following code to find the average
```
dt.accuracy.curb <- mean(as.numeric(dt.cv.curb))

```

The last model that will be trained using the dataset will be the random forest

We can start building the machine learning model throug the following code:
 
 ```
 random.regressor.curb <- randomForest(x=curb.train[1:23],
                                      y= curb.train$price,ntree = 500, mtry = 12, importance = TRUE)
 ```
 
 We can then test out model on the test set through the following code:
 
 ```
 random.predict.curb <- predict(random.regressor.curb, curb.test)

```

Now we can calculate the RMSE score of the train set 

```
sqrt(mean((curb.train$price - predict(random.regressor.curb, curb.train)) ^ 2))

```

For comparison, let's calculate the RMSE score of the test set

```
RMSE(random.predict.curb, curb.test$price)

```

We can now evaluate the model performance through the k-fold cross validation as show in the following code:

```
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
```

The following code will provide us with the average of the RMSE scores

```
rf.accuracy.curb <- mean(as.numeric(rf.cv.curb))

```

As show above, the mtry was pegged at 12. This value is derived through hyperparameter optimisation by running the following code: 

```
train(form = price~., data=curb.train, method="rf")

```

Lastly, to identify significant variables, we will call the importance function. Type 1 refers to the %IncMSE

```
importance(random.regressor.curb, type=1)

```


 











