import pandas
import re
import sys 
sys.path.append("..")
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz 

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

main_data_frame = pandas.DataFrame(data=data, columns=['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])
main_data_frame = main_data_frame[["Pclass", "Fare", "Age", "Sex", 'Survived']].dropna().replace("female",0).replace("male",1)


X = main_data_frame[["Pclass", "Fare", "Age", "Sex"]]
Y = main_data_frame[["Survived"]]

print(X)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, Y)

print(clf.feature_importances_)
tree.plot_tree(clf) 

#plt.show()

dot_data = tree.export_graphviz(clf, out_file=None,                       
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("iris") 