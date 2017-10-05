import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.datasets import mnist
 
#Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
#Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Normalise the data from [0,256] to [0,1]
X_train /= 255
X_test /= 255
 
#Preprocess class labels - one-hot encoding
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

 
#Set model architecture 
model = Sequential()
model.add(Dense(1280, activation ='relu',input_shape=(784,)))
model.add(Dense(640, activation = 'relu'))
model.add(Dense(1280, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
#Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=1000, epochs=10, verbose=1)
 
#Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)



