# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:47:00 2021

@author: aishw
"""

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
import pickle


#importing dataset
data = pd.read_csv('Churn_Modelling.csv')

#Checking for null value
x=data.isnull()



#Deleting the RowNumber column
data=data.drop(["RowNumber"],axis=1)


#List of continuous and categorical variables/features

continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary']
categorical_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']

#Separating the train and test data using a 80%-20% split

data_train = data.sample(frac=0.8, random_state=100)
data_test = data.drop(data_train.index)





data_train = data_train[['Exited'] + continuous_vars + categorical_vars]

#turning 0 values of numerical categorical features into -1
#to introduce negative relation in the calculations

data_train.loc[data_train.HasCrCard == 0, 'HasCrCard'] = -1
data_train.loc[data_train.IsActiveMember == 0, 'IsActiveMember'] = -1

#list of categorical variables

var_list = ['Geography', 'Gender']

#Turning the categorical variables into one-hot vectors
for var in var_list:
  for val in data_train[var].unique():
    data_train[var + '_' + val] = np.where(data_train[var] == val, 1, -1)
data_train = data_train.drop(var_list, axis=1)
k=data_train.head()

#Normalize the continuous variables from 0 to 1
min_values = data_train[continuous_vars].min()
max_values = data_train[continuous_vars].max()

data_train[continuous_vars] = (data_train[continuous_vars] - min_values) / (max_values - min_values)
s=data_train.head()

RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_impurity_split=None,min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
RF.fit(data_train.loc[:, data_train.columns != 'Exited'],data_train.Exited)
pickle.dump(RF,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
