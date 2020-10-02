import numpy as np
import pandas as pd
from  sklearn.svm import SVC

Xtrain = pd.read_csv('data/svm-data.csv', header=None, usecols=np.arange(1,3))
ytrain = pd.read_csv('data/svm-data.csv', header=None, usecols=[0])



clf = SVC(kernel='linear', C=100000 , random_state=241)


clf.fit(Xtrain, ytrain)

print(clf)