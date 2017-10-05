import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K
from keras.datasets import mnist

#Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 

#setup
nb_batch = 100
nb_epochs = 10
num_classes = 10
 
#input image dimensions
img_rows, img_cols  = 28,28

#Preprocess input data
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


#X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
#X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Normalise the data from [0,256] to [0,1]
X_train /= 255
X_test /= 255
 
#Preprocess class labels - one-hot encoding
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)


#Set model architecture 
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
#Fit model on training data
model.fit(X_train, Y_train, 
          batch_size=nb_batch, epochs=nb_epochs, verbose=1)
 
#Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

#verify
#y_pred = model.predict(X_test)
#Y_pred = np.argmax(y_pred,axis=1)
#Y_test = np.argmax(Y_test,axis=1)
#accuracy = (len(Y_test) - np.count_nonzero(Y_pred - Y_test) + 0.0)/len(Y_test)
#print(accuracy)