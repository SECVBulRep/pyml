import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
import re
import sys
 
sys.path.append("..")


#print(data[:10])
#print(data.head())
#print(data['Pclass'])

print(data['Survived'].value_counts())

print(100*data['Survived'].value_counts()/data['Survived'].count())

#print(100*data['Pclass'].value_counts()/data['Pclass'].count())


#print(data['Age'].mean())
#print(data['Age'].median())

#corr = data['SibSp'].corr(data['Parch'])

#print(corr)

def clean_name(name):
 
    # Первое слово до запятой - фамилия
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)
 
    # Если есть скобки - то имя пассажира в них
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)
    # Удаляем обращения
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)
    # Берем первое оставшееся слово и удаляем кавычки
    name = name.split(' ')[0].replace('"', '')
    return name


names = data[data['Sex'] == 'female']['Name'].map(clean_name)
name_counts = names.value_counts()
 
print(6, name_counts.head(1).index.values[0])