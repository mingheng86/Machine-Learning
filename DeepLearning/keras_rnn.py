'''
Keras - Recurrent Neural Network - learn from 20% training, and fit the 80%)
******** Treated CO(GT) as the Y. Rest as predictors.

Data from https://archive.ics.uci.edu/ml/datasets/Air+Quality

Data Set Information:

The dataset contains 9358 instances of hourly averaged responses from an array of 
5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. 
The device was located on the field in a significantly polluted area, at road level,
within an Italian city. Data were recorded from March 2004 to February 2005 (one year)
representing the longest freely available recordings of on field deployed air quality 
chemical sensor devices responses. Ground Truth hourly averaged concentrations for CO, 
Non Metanic Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx) and Nitrogen Dioxide (NO2) 
and were provided by a co-located reference certified analyzer. 
Evidences of cross-sensitivities as well as both concept and sensor drifts 
are present as described in De Vito et al., Sens. And Act. B, Vol. 129,2,2008 (citation required) 
eventually affecting sensors concentration estimation capabilities. 
Missing values are tagged with -200 value. 

This dataset can be used exclusively for research purposes. Commercial purposes are fully excluded. 

Adapted from https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
'''
#Set working directory
#import os
#os.chdir(r'~')

#import packages
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load data 
dataset = pd.read_csv('AirQualityUCI.csv',  parse_dates = [['Date', 'Time']], index_col=0)

#replace the missing values encoded as -200 with NaN
dataset = dataset.replace(to_replace = -200, value = np.NaN)

#Interpolate the missing values 
dataset = dataset.interpolate(method='linear')

# ensure all data is float
values = dataset.values
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[list(range(14, 26))], axis=1, inplace=True)
print(reframed.head())


# split into train and test sets (20% train, 80% test)
values = reframed.values
nobs_train = int(values.shape[0]*0.2)
train = values[:nobs_train, :]
test = values[nobs_train:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


n_epochs = 15
n_batch_size = 100

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)