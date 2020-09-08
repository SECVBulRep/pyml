import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics

data = pd.read_csv('wine.data', header=None)

#print(data)

X = pd.read_csv('wine.data', header=None, usecols=np.arange(1,14))
y = pd.read_csv('wine.data', header=None, usecols=[0])

#X = pd.read_csv('wine.data', header=None, usecols=list(xrange(1,14)))
#y = pd.read_csv('wine.data', header=None, usecols=[0]).values.reshape(len(X),)
 
#kf = KFold(n=len(X), n_folds=5, shuffle=True, random_state=42)
