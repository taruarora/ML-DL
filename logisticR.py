from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


iris=datasets.load_iris()


print(x)
#print(x.shape)
#y=(iris.target==2).astype(int)

#print(y)
#tx,vx,ty,vy=train_test_split(x,y,random_state=1)
#clf=LogisticRegression(random_state=1)
#clf.fit(tx,ty)
#y_prob=clf.predict_proba(vx)

#print(y_prob)

#to plot the graph
#plt.plot(vx,y_prob[:,1], "g-",label="virginica")
#plt.show()
