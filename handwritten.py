import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train),(X_test,y_test)=keras.datasets.mnist.load_data()
#print("len(X_train) : "+str(len(X_train)))
#print("(X_train[0].shape) : "+str(X_train[0].shape))
#print("X_train[0] ")
#print(X_train[0])

plt.matshow(X_train[0])
plt.show()
#print(X_train.shape)#(6000, 28,28)=(no. of samples, matrix_rows, matrix_cols)
#print(X_train[1][1])
X_train_flat = X_train.reshape(len(X_train), 28*28)
X_test_flat = X_test.reshape(len(X_test), 28*28)

X_train_flat =X_train_flat/255
X_test_flat =X_test_flat /255 #converting thhem in the range of 0 to 1
#X_train_flat[0]->1 is image ka matrix

model=keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')])
model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_flat[:300], y_train[:300], epochs=2)
model.evaluate(X_test_flat, y_test)

y_predict=model.predict(X_test_flat)
print(y_predict[0])

np.argmax(y_predict[0])

#confusion matrix
y_predicted_labels = [np.argmax(i) for i in y_predict]
y_predicted_labels[:5]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
print(cm)


#adding hidden layer activation func->relu/tanh could work
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='tanh'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flat, y_train, epochs=5)

