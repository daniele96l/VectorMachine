#Based on the feature of the flower we want to know the type

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import  svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

x = iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolous', 'Iris Virginica']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = svm.SVC()
model.fit(x_train,y_train)

predictions = model.predict(x_test)
acc = accuracy_score(y_test,predictions)

print(acc)