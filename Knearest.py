import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.neighbors import KNeighborsClassifier

#loading iris
iris=datasets.load_iris()

#print(iris.data[0:4])
#print("done")
#print(iris.target[51])

print(iris.DESCR)

features=iris.data
labels=iris.target

#displaying it as a dataframe

df=pd.DataFrame(data=iris.data,index=iris.target)
print(df.head(150))

#training the classifier
clf=KNeighborsClassifier()
clf.fit(features,labels)

#predict
preds=clf.predict([[31,1,1,1]])
print(preds)
