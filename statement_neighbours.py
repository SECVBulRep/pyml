import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

data = pd.read_csv('wine.data', header=None)

#print(data)

X = pd.read_csv('wine.data', header=None, usecols=np.arange(1,14))
y = pd.read_csv('wine.data', header=None, usecols=[0])

print(X)
print(y)

X = pd.read_csv('wine.data', header=None, usecols=list(range(1,14)))
y = pd.read_csv('wine.data', header=None, usecols=[0]).values.reshape(len(X),)

print(X)
print(y)

 
kf = KFold(shuffle=True,random_state=42,n_splits=5)

means = []

for k in range(1, 51):
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(X, y)
    array = cross_val_score(estimator=kn, X=X, y=y, cv=kf, scoring='accuracy')
    m = array.mean()
    means.append(m)


print(max(means))
print(np.argmax(means))

means2 = []

for k in range(1, 51):
    kn = KNeighborsClassifier(n_neighbors=k)
    scaler = preprocessing.StandardScaler().fit(X)
    X_train_transformed = scaler.transform(X)
    kn.fit(X_train_transformed, y)
    array = cross_val_score(estimator=kn, X=X_train_transformed, y=y, cv=kf, scoring='accuracy')
    m = array.mean()
    means2.append(m)

print(max(means2))
print(np.argmax(means2))