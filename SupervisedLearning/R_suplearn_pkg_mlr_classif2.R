#Supervised Machine Learning Techniques
#Classification 
#Model building and prediction using xgboost

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


#-------------XGBOOST-------------#
getParamSet('classif.xgboost')

#Define learner
xgb.lrn = makeLearner('classif.xgboost')


#Hyper parameter tuning of paramaters
xgb.hyparam = makeParamSet(
  makeIntegerParam("nrounds",lower=200,upper=600),
  makeIntegerParam("max_depth",lower=3,upper=20),
  makeNumericParam('eta', lower = -3, upper = 3, trafo = function(x) pnorm(10^x)),
  makeNumericParam('gamma', lower = 0.5, upper = 1),
  makeNumericParam('lambda', lower = 0.5, upper = 0.6),
  makeNumericParam('alpha', lower = 0, upper = 0.5)
  )

xgb.ctrl = makeTuneControlRandom(maxit = 10)
xgb.rsamp = makeResampleDesc('CV',iters= 3)
xgb.tuned = tuneParams('classif.xgboost', task=iris.task, resampling = xgb.rsamp,measures=acc,
                       par.set = xgb.hyparam, control = xgb.ctrl)

#Examine learning curves
xgb.hypar = generateHyperParsEffectData(xgb.tuned,partial.dep = TRUE)
reg.lrn = makeLearner('regr.lm')
plotHyperParsEffect(xgb.hypar, x='iteration', y='acc.test.mean', plot.type='line', partial.dep.learn = reg.lrn)

#Set the learner based on tuned results
xgb.lrn = setHyperPars(makeLearner('classif.xgboost'),par.vals=xgb.tuned$x)

#Fit model using tuned results
model.xgb = train(xgb.lrn,iris.task,subset=train)

#Model summary 
summary(model.xgb$learner.model)

#Predict values
model.xgb.pred = predict(model.xgb,iris.task,subset=test)

#Evaluate performance of the learner
performance(model.xgb.pred,measure = list(acc,mmce))
calculateConfusionMatrix(model.xgb.pred)

#//-------------XGBOOST-------------#
