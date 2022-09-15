#simple artificial neural network

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

print(X_train.shape)
#(50000, 32, 32, 3)
#image, shape of image, rgb

print(X_test.shape)
print(y_train.shape)
X_train[0]

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
X_train=X_train/255
X_test=X_test/255

#y_train(10000,32,32)
y_train.reshape(-1,)#will keep 1 st  argument same and rest will reshape it
y_train[0]


y_test.reshape(-1,)
def plot_sample(index):
  plt.figure(figsize=(15,2))
  plt.imshow(X_train[index])
  #print(classes[y_train[index]])

plot_sample(3)

ann=models.Sequential([
                       layers.Flatten(input_shape=(32,32,3)),
                       layers.Dense(3000,activation='relu'),
                       layers.Dense(1000,activation='relu'),
                       layers.Dense(100,activation='relu'),
                       layers.Dense(10,activation='sigmoid'),

])
ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)
