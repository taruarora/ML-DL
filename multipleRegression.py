import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse

diabetes=datasets.load_diabetes()
print(diabetes.keys())

#['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
diabetes_x=diabetes.data
x_train=diabetes_x[:30]
x_test=diabetes_x[:-30]

y_train=diabetes.target[:30]
y_test=diabetes.target[:-30]

model=linear_model.LinearRegression()
model.fit(x_train,y_train)#line formed

y_predict=model.predict(x_test)

print("Mean Squared Error",mse(y_test,y_predict))

slope=model.coef_
c=model.intercept_
print(slope)

print(diabetes.DESCR)


#plot map not in multiple regression
#plt.scatter(x_test,y_test)
#plt.plot(x_test,y_predict)
#plt.show()
