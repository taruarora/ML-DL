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

cnn=models.Sequential([
    layers.Conv2D(filters=32, kerne_size=(3,3), activation='relu', input=(32,32,3))
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters=64, kerne_size=(3,3), activation='relu')
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)
cnn.evaluate(X_test,y_test)

y_pred = cnn.predict(X_test)
y_pred[:5]

y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]
y_test[:5]
plot_sample(X_test, y_test,3)
classes[y_classes[3]]
