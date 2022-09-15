import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

X_train=X_train/255
X_test=X_test/255

X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)

from keras.layers.convolutional import Conv2D
cnn=models.Sequential([
                       layers.Conv2D(filters=30,activation='relu',kernel_size =(3,3),input_shape=(28,28,1)),
                       layers.MaxPooling2D((2,2)),

                       layers.Conv2D(filters=64,activation='relu',kernel_size=(3,3)),
                       layers.MaxPooling2D((2,2)),

                       layers.Flatten(),
                       layers.Dense(100,activation='relu'),
                       layers.Dense(10,activation='sigmoid')

])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=5)
cnn.evaluate(X_test,y_test)
