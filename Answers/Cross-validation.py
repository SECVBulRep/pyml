import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.model_selection import KFold


"""
X, y = datasets.load_iris(return_X_y=True)


#print (X, y)
print (X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#обучили  - првоерили
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
score = clf.score(X_test, y_test)

print(score)


# делаем проверку 
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)

print(scores.shape)
print(scores)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#делаем выборочную проверку 
scores2 = cross_val_score(
    clf, X, y, cv=5, scoring='f1_macro')

print(scores2.shape)
print(scores2)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

n_samples = X.shape[0]
print(n_samples)

#scaling 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)

print (X_train)

X_train_transformed = scaler.transform(X_train)

print(X_train_transformed)

clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
score = clf.score(X_test_transformed, y_test)

print(score)
"""

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s %s" % (train, test))




