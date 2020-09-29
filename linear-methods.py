import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  accuracy_score
from sklearn.metrics import classification_report

#trainX = pd.read_csv("data/perceptron-train.csv")
#testX =pd.read_csv("data/perceptron-test.csv")

Xtrain = pd.read_csv('data/perceptron-train.csv', header=None, usecols=np.arange(1,3))
ytrain = pd.read_csv('data/perceptron-train.csv', header=None, usecols=[0])

Xtest = pd.read_csv('data/perceptron-test.csv', header=None, usecols=np.arange(1,3))
ytest = pd.read_csv('data/perceptron-test.csv', header=None, usecols=[0])

clf = Perceptron(random_state=241)
clf.fit(Xtrain, ytrain.values.ravel())

predictions = clf.predict(Xtest)

accuracy = accuracy_score(ytest,predictions)

print(accuracy)

print(classification_report(clf.predict(Xtest),ytest))

"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(Xtrain)
X_test_scaled = scaler.transform(Xtest)
clf.fit(X_train_scaled, ytrain.values.ravel())
predictions_scaled = clf.predict(X_test_scaled)
accuracy_scaled = accuracy_score(ytest,predictions_scaled)

print(accuracy_scaled)
print(round((accuracy_scaled-accuracy),3))

"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(Xtrain)
X_test_scaled = scaler.transform(Xtest)

clf.fit(X_train_scaled, ytrain)
acc2 = accuracy_score(ytest,clf.predict(X_test_scaled))

print(acc2)