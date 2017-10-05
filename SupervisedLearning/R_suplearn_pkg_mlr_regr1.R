#Supervised Machine Learning Techniques
#Regression
#Model building and prediction using multiple linear regression and random forest
##There are shorter ways of fitting these models but this is just to show how they are done with the mlr package
#Imo, mlr package provides a very nice structured framework for data science and machine learning

library(datasets)
library(mlr)
library(lmtest) ##for bptest

#Load data
data(iris)
iris

#Explore data
summary(iris)
str(iris)
head(iris)
var_class = sapply(iris,class)

#Randomly split data into 70% train and 30% test
set.seed(123)
n=nrow(iris)
train=sample(n,size=floor(0.7*n))
test=setdiff(1:n,train)

#List all learners available for regression task
regr.learners.list = listLearners(obj='regr')

#Regression task would be to predict Sepal.Length based on the other variables

#Define task 
iris.task = makeRegrTask(data=iris, target='Sepal.Length')
getTaskDescription(iris.task)

#-------------Multiple Linear Regression-------------#
##There are shorter ways of fitting multiple linear reg models, but this is just to show how the can be done with the mlr package


#Define multiple linear regression learner
mlr.lrn = makeLearner('regr.lm')

#Fit model
model.mlr = train(mlr.lrn,iris.task,subset=train)

#Model summary and diagnostics -> check Gauss Markov Assumptions
summary(model.mlr$learner.model)
qplot(model.mlr$learner.model$residuals)
plot(model.mlr$learner.model$residuals)
mean(model.mlr$learner.model$residuals)
shapiro.test(model.mlr$learner.model$residuals)
bptest(model.mlr$learner.model)

#Predict values
model.mlr.pred = predict(model.mlr,iris.task,subset=test)

#Evaluate performance of the learner
performance(model.mlr.pred,measure = list(rmse,mae))

#//-------------Multiple Linear Regression-------------#


#-------------Random Forest-------------#
#Define learner
rf.lrn = makeLearner('regr.randomForest')

#Fit model
model.rf = train(rf.lrn,iris.task,subset=train)

#Model summary 
summary(model.rf$learner.model)

#Predict values
model.rf.pred = predict(model.rf,iris.task,subset=test)

#Evaluate performance of the learner
performance(model.rf.pred,measure = list(rmse,mae))

#//-------------Random Forest-------------#