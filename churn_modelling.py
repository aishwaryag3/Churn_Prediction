# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#importing dataset
data = pd.read_csv('Churn_Modelling.csv')
#check sample of data
a=data.head()
#Checking for null value
x=data.isnull()
print(data.isnull().sum())
print(data.isnull().values.any())
#Deleting the RowNumber column
data=data.drop(["RowNumber"],axis=1)
#Statistical description of dataset
y = data.describe()

#To display the column details
data.info()
#Creating a pie chart for churn or not churned
labels=["Not Churned","Churned"]
myexplode=[0.2,0]
mycolors=['r','y']
plt.pie(data['Exited'].value_counts(),labels=labels,autopct='%1.1f%%',radius=1,explode=myexplode,shadow=True,colors=mycolors,startangle=90)
plt.title("Customers Churned and Not Churned")
plt.show()
#Creating a pie chart for reason for churning
sizes = [data.Reason[data['Reason'] =="High Service Charges/Rate of Interest" ].count(), data.Reason[data['Reason'] == "Long Response Times"].count(),data.Reason[data['Reason']=="Inexperienced Staff / Bad customer service "].count(),data.Reason[data['Reason'] =="Excess Documents Required" ].count()]
labels=["High Service Charges","Long Response Time","Inexperienced Staff","Exccess document required"]
mycolors=['r','y','b','c']
plt.pie(sizes,labels=labels,autopct='%1.1f%%',colors=mycolors,startangle=90)
plt.title("Reason for churning")
plt.show()
#creating bar chart for categorical values
ax=sns.catplot(x="Gender",hue="Exited",col="Exited",data=data,kind="count",height=4,aspect=.7,palette="Set1")
plt.subplots_adjust(top=0.7)
ax.fig.suptitle("Bar Chart to find dependency between Exited and Gender Coulumns")
bx=sns.catplot(x="Geography",hue="Exited",col="Exited",data=data,kind="count",height=4,aspect=.7,palette="Set1")
plt.subplots_adjust(top=0.7)
bx.fig.suptitle("Bar Chart to find dependency between Exited and Geography Coulumns")
cx=sns.catplot(x="HasCrCard",hue="Exited",col="Exited",data=data,kind="count",height=4,aspect=.7,palette="Set1")
plt.subplots_adjust(top=0.7)
cx.fig.suptitle("Bar Chart to find dependency between Exited and HasCreditCard Coulumns")
dx=sns.catplot(x="IsActiveMember",hue="Exited",col="Exited",data=data,kind="count",height=4,aspect=.7,palette="Set1")
plt.subplots_adjust(top=0.7)
dx.fig.suptitle("Bar Chart to find dependency between Exited and IsActiveMember Coulumns")
ex=sns.catplot(x="NumOfProducts",hue="Exited",col="Exited",data=data,kind="count",height=4,aspect=.7,palette="Set1")
plt.subplots_adjust(top=0.7)
ex.fig.suptitle("Bar Chart to find dependency between Exited and NumOfProducts Coulumns")
plt.show()
#Creating Box Plot for Numerical/continuous values

sns.boxplot(x="Exited",y="Age",hue="Exited",data=data,palette="Set1").set_title("Box Plot for mapping dependency between Exited and Age")
plt.show()
sns.boxplot(x="Exited",y="CreditScore",hue="Exited",data=data,palette="Set1").set_title("Box Plot for mapping dependency between Exited and CreditScore")
plt.show()
sns.boxplot(x="Exited",y="Balance",hue="Exited",data=data,palette="Set1").set_title("Box Plot for mapping dependency between Exited and Balance")
plt.show()
sns.boxplot(x="Exited",y="EstimatedSalary",hue="Exited",data=data,palette="Set1").set_title("Box Plot for mapping dependency between Exited and EstimatedSalary")
plt.show()
sns.boxplot(x="Exited",y="NumOfProducts",hue="Exited",data=data,palette="Set1").set_title("Box Plot for mapping dependency between Exited and NumOfProducts")
plt.show()
sns.boxplot(x="Exited",y="Tenure",hue="Exited",data=data,palette="Set1").set_title("Box Plot for mapping dependency between Exited and Tenure")
plt.show()
#List of continuous and categorical variables/features

continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary']
categorical_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender','Reason']

#Separating the train and test data using a 80%-20% split

data_train = data.sample(frac=0.8, random_state=100)
data_test = data.drop(data_train.index)

#Check the number of rows in each data set for verification

print('Number of rows in train data: ', len(data_train))
print('Number of rows in test data: ', len(data_test))

print()

data_train = data_train[['Exited'] + continuous_vars + categorical_vars]
print(data_train.head())
# turning 0 values of numerical categorical features into -1
# to introduce negative relation in the calculations

data_train.loc[data_train.HasCrCard == 0, 'HasCrCard'] = -1
data_train.loc[data_train.IsActiveMember == 0, 'IsActiveMember'] = -1

print(data_train.head())
# list of categorical variables

var_list = ['Geography', 'Gender','Reason']

# turning the categorical variables into one-hot vectors

for var in var_list:
  for val in data_train[var].unique():
    data_train[var + '_' + val] = np.where(data_train[var] == val, 1, -1)

data_train = data_train.drop(var_list, axis=1)

k=data_train.head()
#normalize the continuous variables from 0 to 1
min_values = data_train[continuous_vars].min()
max_values = data_train[continuous_vars].max()

data_train[continuous_vars] = (data_train[continuous_vars] - min_values) / (max_values - min_values)
s=data_train.head()


#sns.kdeplot("NumOfProducts","Balance")
#plt.show()

facet = sns.FacetGrid(data, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"Age",shade= True)
facet.set(xlim=(0, data["Age"].max()))
facet.add_legend()
plt.show()

facet = sns.FacetGrid(data, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"Balance",shade= True)
facet.set(xlim=(0, data["Balance"].max()))
facet.add_legend()
plt.show()

facet = sns.FacetGrid(data, hue="Exited",aspect=3)
facet.map(sns.kdeplot,"CreditScore",shade= True)
facet.set(xlim=(0, data["CreditScore"].max()))
facet.add_legend()
plt.show()

facet = sns.FacetGrid(data,hue="Exited",aspect=3)
facet.map(sns.kdeplot,"NumOfProducts","Balance",shade= True)
facet.set(xlim=(0, data["NumOfProducts"].max()))
facet.add_legend()
plt.show()

sns.stripplot(x="Exited",y="Balance")
