import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
import seaborn as sns

train=pd.read_csv(r'C:\Users\manis\Downloads\3f488f10aa3d11ea\Dataset\Train.csv')
test=pd.read_csv(r'C:\Users\manis\Downloads\3f488f10aa3d11ea\Dataset\Test.csv')
print(train.columns)

#checking null values
print("Null Percentage of Columns")
print((train.isnull().sum()*100)/train.isnull().count())
train=train.interpolate(method='from_derivatives')
print("After using Interpolate")
print((train.isnull().sum()*100)/train.isnull().count())


print("Null Percentage of Columns in test")
print((test.isnull().sum()*100)/test.isnull().count())
test=test.interpolate(method='from_derivatives')
print("After using Interpolate")
print((test.isnull().sum()*100)/test.isnull().count())

#fill_list=['Age','Time_of_service','Pay_Scale','Work_Life_balance','VAR2','VAR4']
#for emp in train['Employee_ID']:
   #print(train.loc[train['Employee_ID'] == emp,fill_list])

#outliers
columns_dict={'Age':1,'Education_Level':2,
              'Time_of_service':3, 'Time_since_promotion':4, 'growth_rate':5, 'Travel_Rate':6,
              'Post_Level':7, 'Pay_Scale':8,
              'Work_Life_balance':9, 'VAR1':10, 'VAR2':11, 'VAR3':12, 'VAR4':13, 'VAR5':14, 'VAR6':15,
              'VAR7':16, 'Attrition_rate':17}
columns_dict_test={'Age':1,'Education_Level':2,
              'Time_of_service':3, 'Time_since_promotion':4, 'growth_rate':5, 'Travel_Rate':6,
              'Post_Level':7, 'Pay_Scale':8,
              'Work_Life_balance':9, 'VAR1':10, 'VAR2':11, 'VAR3':12, 'VAR4':13, 'VAR5':14, 'VAR6':15,
              'VAR7':16,}
# Detect outliers in each variable using box plots.
plt.figure(figsize=(20,30))
for variable,i in columns_dict.items():
                     plt.subplot(5,4,i)
                     plt.boxplot(train[variable],whis=1.5)
                     plt.title(variable)
#plt.show()

plt.figure(figsize=(20,30))
for variable,i in columns_dict_test.items():
                     plt.subplot(5,4,i)
                     plt.boxplot(test[variable],whis=1.5)
                     plt.title(variable)
#plt.show()


plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(train['VAR4'])
plt.title("before")
train['VAR4']=winsorize(train['VAR4'],limits=[0.20,0.20])
train['VAR1']=winsorize(train['VAR1'],limits=[0.05,0.05])
plt.subplot(1,2,2)
plt.hist(train['VAR4'])
plt.title("after")
#plt.show()
plt.figure(figsize=(15,15))
sns.heatmap(train.corr(), square=True, annot=True, linewidths=.5)
#plt.show()

x = pd.DataFrame(train).to_csv('trainml.csv', header=True, index=None)

test['VAR4']=winsorize(test['VAR4'],limits=[0.20,0.20])
test['VAR!']=winsorize(test['VAR1'],limits=[0.05,0.05])
y = pd.DataFrame(test).to_csv('testml.csv', header=True, index=None)