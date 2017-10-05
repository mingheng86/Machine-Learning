#Time Series Forecasting with VECM model
#Fit VECM model to time series data and generate forecasts

#load libraries
library(vars)
library(urca)

set.seed(123)

#simulate VECM time series data
nobs = 1000

e1 = rnorm(nobs,0,0.5)
e2 = rnorm(nobs,0,0.5)
e3 = rnorm(nobs,0,0.5)

u1.ar1 = arima.sim(model = list(ar=0.7),innov = e1, n = nobs)
u2.ar2 = arima.sim(model = list(ar=0.3),innov = e2, n = nobs)
y3 = cumsum(e3)

#how the series are related
y1 = 0.7*y3 + u1.ar1
y2 = -0.3*y3 + u2.ar2

Y = data.frame(y1,y2,y3)

#estimate the VECM using Johansen approach
vecm = ca.jo(Y, type = 'trace', spec = 'transitory')

#estimate the VECM eqns - no. of cointegrating r/s is 2
cajorls(vecm,r=2)

#convert VECM to VAR representation
vecm.level = vec2var(vecm,r=2) #r is no. of coint r/s

#diagnostic checks
normality.test(vecm.level)
arch.test(vecm.level)
serial.test(vecm.level)

#predict using VECM
vecm.pred = predict(vecm.level,n.ahead=10)
fanchart(vecm.pred)

#forecast error variance decomposition
plot(fevd(vecm.level),col=(1:3))
