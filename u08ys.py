import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')

#print(data[:10])
#print(data.head())
#print(data['Pclass'])

#print(data['Survived'].value_counts())

#print(100*data['Survived'].value_counts()/data['Survived'].count())

#print(100*data['Pclass'].value_counts()/data['Pclass'].count())


print(data['Age'].mean())

print(data['Age'].median())