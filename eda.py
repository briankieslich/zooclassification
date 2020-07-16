#!/home/brian/miniconda3/bin/python3.7
# encoding: utf-8
"""
Read the docs, obey PEP 8 and PEP 20 (Zen of Python, import this)

Build on:    Spyder
Python ver: 3.7.3

Created on Thu Oct 17 21:14:04 2019

@author: brian
"""

# %% modules:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)


#%% data eda

df = pd.read_csv('data/zoo.csv')
animal_class = pd.read_csv('data/class.csv')


# No missing values
df.isnull().sum()

df.loc[df.animal_name=='frog', :]

# remove one 'frog'
# df = df.loc[~df.duplicated(subset='animal_name', keep='first'), :].reset_index(drop=True)

df.class_type.hist()

# remove small groups
# df.legs.value_counts()
# df.loc[df.legs==5, :]
# Starfish
# df.loc[df.legs==8, :]
# octopus, scorpion
# df = df.loc[(df.legs != 5) & (df.legs != 8)].reset_index()

# Does seasnakes breathe? YES
df.loc[76, 'breathes'] = 1



#%% classification with DecisionTreeClassifier

y = df['class_type']
X = df.drop(columns=['animal_name', 'class_type']).values

dtc = DecisionTreeClassifier(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42,
                                                    stratify=y)

dtc.fit(X_train, y_train)
print(f'Train score: {dtc.score(X_train, y_train)}')
print(f'Test score : {dtc.score(X_test, y_test)}')


y_pred = dtc.predict(X_test)
dd = pd.DataFrame({'test': y_test, 'pred': y_pred})
dd.loc[dd.test != dd.pred, :]


df.iloc[73:77, :]
df.loc[df.class_type==5, :].head()

fig = plt.figure(figsize=(20, 20))
plot_tree(dtc, rounded=True, precision=2, fontsize=12);

# nr 75 seasnake is the only one of the reptiles, that is aquatic,
# hence it is classified as a bug. See bottom left: x[5] <= 0.5 split into type 3 or 5.


#%% classification with KNeighborsClassifier

y = df.class_type
X = df.drop(columns=['animal_name', 'class_type']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

knc = KNeighborsClassifier(n_jobs=15, n_neighbors=3, weights='distance', p=1)
knc.fit(X_train, y_train)
print(f'Train_score: {knc.score(X_train, y_train)}')
print(f'Test_score: {knc.score(X_test, y_test)}')

y_pred = knc.predict(X_test)
sum(y_pred != y_test)






#%% something

rfc = RandomForestClassifier()
df.class_type.value_counts()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

params = {'n_estimators': [10, 15, 20, 25, 30, 35, 38, 40, 41, 42, 45, 100, 110, 150],
          'criterion': ['gini'],
          'max_depth': [3, 4, 5, 8, 10, 13, 15, 17, 21, 30],
          'random_state': [0, 42, 101]
          }

grid = GridSearchCV(estimator=rfc,
                    param_grid=params,
                    cv=10,
                    refit=True,
                    n_jobs=15)

grid.fit(X_train, y_train)
print(f'Train score: {grid.score(X_train, y_train)}')
print(f'Test score : {grid.score(X_test, y_test)}')

print(grid.best_params_)
print(grid.best_score_)

