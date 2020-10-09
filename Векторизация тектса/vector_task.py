from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn import svm

newsgroups = datasets.fetch_20newsgroups(
    subset='all', categories=['alt.atheism', 'sci.space'])

y = newsgroups.target

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
print(vectorizer.get_feature_names())

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
#gs.fit(X, y)
# 0.9932804406678872
# 1.0

clf = svm.SVC(kernel='linear', random_state=241,C=1)
clf.fit(X,y)


weights = np.absolute(clf.coef_.toarray())

max_weights = sorted(zip(weights[0], vectorizer.get_feature_names()))[-10:]
max_weights.sort(key=lambda x: x[1])
print(max_weights)


f = open('submission.txt', 'w')
for w, c in max_weights[:-1]:
    f.write(c)
    f.write(',')
f.write(max_weights[-1][1])
f.close()