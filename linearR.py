import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

diabetes=datasets.load_diabetes()
print(diabetes.keys())

#['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
diabetes_x=diabetes.data[:,2]
X_train=diabetes[:,:30]
X_test=diabetes_x[:-30]


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train=diabetes.target[:30]
#sc_y = StandardScaler()
y_train = sc_X.fit_transform(y_train)


y_test=diabetes.target[:-30]

model=linear_model.LinearRegression()
model.fit(X_train,y_train)#line formed

y_predict=model.predict(X_test)

print("Mean Squared Error",mse(y_test,y_predict))

slope=model.coef_
c=model.intercept_
print(slope)


#plot map
plt.scatter(x_test,y_test)
plt.xlabel('sepal_width')
plt.ylabel('flower')
plt.plot(x_test,y_predict)
plt.show()



