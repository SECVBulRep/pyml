
import numpy as np
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
data = pd.read_csv(url, header=None, na_values='?')

# даем наименования столбцам
data.columns = ['A' + str(i) for i in range(1, 16)] + ['class']

#  печать начала и конца
print(data.head)
print(data.tail)

#  опиасние столбцов (только числовые)
print(data.describe())


# Выделим числовые и категориальные признаки:
categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print (categorical_columns)
print (numerical_columns)

# Теперь мы можем получить некоторую общую информацию по категориальным признакам:
print (data[categorical_columns].describe())

# Определить полный перечень значений категориальных признаков можно, например, так:
for c in categorical_columns:
    print (data[c].unique())

print('done')
