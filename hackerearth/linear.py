import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as seabornInstance
train=pd.read_csv(r'C:\Users\manis\PycharmProjects\hackerearth\hackerearth\trainml.csv')
test=pd.read_csv(r'C:\Users\manis\PycharmProjects\hackerearth\hackerearth\testml.csv')

X=train[[
              'Time_of_service', 'Time_since_promotion', 'growth_rate',

              'Work_Life_balance', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6',
              'VAR7']].values
y=train['Attrition_rate'].values

xvalues_test=test[[
              'Time_of_service', 'Time_since_promotion', 'growth_rate',

              'Work_Life_balance', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6',
              'VAR7']].values

plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(train['Attrition_rate'])
#plt.show()
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.27)
lm = LinearRegression(normalize=True)
model = lm.fit(x_train,y_train)
y_pred = lm.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
rms=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rms)
print(1-rms)

print(metrics.r2_score(y_test,y_pred))

#test
test['Attrition_rate']=lm.predict(xvalues_test)
print(test['Attrition_rate'].head(10))
#submission
submission=test[['Employee_ID','Attrition_rate']]
print(submission.head())
sub= pd.DataFrame(submission).to_csv('submission.csv', header=True, index=None)