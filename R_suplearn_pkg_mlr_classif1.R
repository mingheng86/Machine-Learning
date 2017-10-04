#Supervised Machine Learning Techniques
#Classification
#Model building and prediction using multinomial reg, KNN, and LDA
#And benchmark experiments..
##There are shorter ways of fitting these models but this is just to show how they are done with the mlr package
#Imo, mlr package provides a very nice structured framework for data science and machine learning

#Load libraries
library(datasets)
library(mlr)

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

#List all learners available for classification task
regr.learners.list = listLearners(obj='classif')

#Classification task would be to predict Species based on the other variables

#Define task 
iris.task = makeRegrTask(data=iris, target='Species')
getTaskDescription(iris.task)

#-------------Multinomial Regression-------------#

#Define learner
mnr.lrn = makeLearner('classif.multinom')

#Fit model
model.mnr = train(mnr.lrn,iris.task,subset=train)

#Model summary 
summary(model.mnr$learner.model)

#Predict values
model.mnr.pred = predict(model.mnr,iris.task,subset=test)

#Evaluate performance of the learner
performance(model.mnr.pred,measure = list(acc,mmce))
calculateConfusionMatrix(model.mnr.pred)

#//-------------Multiple Linear Regression-------------#


#-------------kNN-------------#
#Define learner
getParamSet('classif.knn')
knn.lrn = makeLearner('classif.knn', par.vals=list(k=5))

#Fit model
model.knn = train(knn.lrn,iris.task,subset=train)

#Model summary
summary(model.knn$learner.model)
model.knn$learner.model$importance

#Predict values
model.knn.pred = predict(model.knn,iris.task)

#Evaluate performance
performance(model.knn.pred,measure=list(acc,mmce))
calculateConfusionMatrix(model.knn.pred)


#//-------------kNN-------------#


#-------------Linear Discrimnant Analysis (LDA)-------------#
#Define learner
getParamSet('classif.lda')
lda.lrn = makeLearner('classif.lda')

#Fit model
model.lda = train(knn.lrn,iris.task,subset=train)

#Model summary
summary(model.lda$learner.model)
model.lda$learner.model$importance

#Predict values
model.lda.pred = predict(model.lda,iris.task)

#Evaluate performance
performance(model.lda.pred,measure=list(acc,mmce))
calculateConfusionMatrix(model.lda.pred)


#//-------------kNN-------------#


#-------------Benchmark Experiments- comparing multiple learners-------------#
#Learners to be compared
lrns = list(mnr.lrn, knn.lrn, lda.lrn)

#Choose resample strat, 3 fold CV
rdesc = makeResampleDesc('CV',iters= 3)

#Choose benchmark measures
meas = list(mmce,acc)

#Conduct benchmark experiments
bmr = benchmark(lrns, iris.task, rdesc, measures=meas)

#Get benchmark performances
bmr
getBMRPerformances(bmr)
getBMRAggrPerformances(bmr)

#Visualise benchmark performances
plotBMRBoxplots(bmr, measure=mmce)
plotBMRSummary(bmr)
convertBMRToRankMatrix(bmr,mmce)
plotBMRRanksAsBarChart(bmr,mmce)
