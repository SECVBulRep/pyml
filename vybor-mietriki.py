import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
import sklearn
from sklearn.model_selection import cross_val_score

print('The scikit-learn version is {}.'.format(sklearn.__version__))

boston = load_boston()

print(boston.data) 

scaler = preprocessing.StandardScaler().fit(boston.data)

X_train_transformed = scaler.transform(boston.data)
y = boston.target
print(X_train_transformed)

p_list = np.linspace(1, 10, num=200 )

print(p_list)

kf = KFold(shuffle=True,random_state=42,n_splits=5)
means = []

for k in range(1, 201):
    p = p_list[k-1]
    kn = KNeighborsRegressor(n_neighbors=5,weights='distance',p=p,metric='minkowski')
    kn.fit(X_train_transformed, y)
    array = cross_val_score(estimator=kn, X=X_train_transformed, y=y, cv=kf, scoring='neg_mean_squared_error')
    m = array.mean()
    means.append(m)

print (means)

print(max(means))
print(np.argmax(means))
print(p_list[np.argmax(means)])
 #kn = KNeighborsRegressor()