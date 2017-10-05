#Time Series Forecasting with ARIMA model
#2 types of model fit - auto.arima() and manual/empirical fit based on acf and pacf plots

#load libraries
library(forecast)
library(lmtest)
library(tseries)
library(TSA)

set.seed(123)

#Generate simulated ARIMA(1,1,1) data
ts_sim = arima.sim(list(order =c(1,1,1), ar=0.7,ma=0.3),n=200)

#Plot ts
plot.ts(ts_sim)

###<<----AUTO ARIMA-------->>###

#auto fit an arima model
model_auto_arima = auto.arima(ts_sim)

#model summary
summary(model_auto_arima)

##--model diagnostics
#check residuals
plot(model_auto_arima$residuals)

#test for normality
shapiro.test(model_auto_arima$residuals)

#test for autocorr
acf(model_auto_arima$residuals)
Box.test(model_auto_arima$residuals,lag=20,type='Ljung-Box')

#test for heteroskedascity
McLeod.Li.test(model_auto_arima)

##//--model diagnostics

#forecast 10 steps ahead
model_auto_arima_forecast = forecast(model_auto_arima,h=10)

###//<<----AUTO ARIMA-------->>###

###<<----Manual ARIMA-------->>###

#check unit root
adf.test(ts_sim)

#check unit root of first diff
adf.test(diff(ts_sim))

#acf and pacf plots
par(mfrow=c(2,1))
acf(diff(ts_sim))
pacf(diff(ts_sim))

#fit ARIMA model with AR and MA lag lengths identified from acf and pacf plots
model_manual_arima = arima(ts_sim,c(1,1,1))


##--model diagnostics
#check residuals
plot(model_manual_arima$residuals)

#test for normality
shapiro.test(model_manual_arima$residuals)

#test for autocorr
acf(model_manual_arima$residuals)
Box.test(model_manual_arima$residuals,lag=20,type='Ljung-Box')

#test for heteroskedascity
McLeod.Li.test(model_manual_arima)

##//--model diagnostics

#forecast 10 steps ahead
model_manual_arima_forecast = predict(model_manual_arima, n.ahead=10)

###//<<----Manual ARIMA-------->>###


