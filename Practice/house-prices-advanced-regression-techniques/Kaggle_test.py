import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv('train.csv')
print(train)
print()
test = pd.read_csv('test.csv')
print(test)
for i in train.columns:
  null  =  train[i].isnull().sum()
  print(i,null)
for i in test.columns:
    null_test = test[i].isnull().sum()
    print(i, null_test)
train.drop(columns=['Alley','PoolQC','Fence','MiscFeature'],inplace = True)
print(train)
test.drop(columns=['Alley','PoolQC','Fence','MiscFeature'],inplace = True)
print(test)
fill_col= ['LotFrontage']
for i in train.columns:
  train[i].fillna(train[i].mode()[0],inplace=True)
for i in test.columns:
  test[i].fillna(train[i].mode()[0],inplace=True)
'''Feature Engineering '''
'''train['HasPool'] = (train['PoolArea'] > 0).astype(int) #pool or no?
test['HasPool'] = (test['PoolArea'] > 0).astype(int)
train['TotalSqFt'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF'] # Total square feetage
test['TotalSqFt'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
train['TotalBathrooms'] = train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']) #number of Bathrooms
test['TotalBathrooms'] = test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath'])
train['HasFireplace'] = (train['Fireplaces'] > 0).astype(int)# fireplace
test['HasFireplace'] = (test['Fireplaces'] > 0).astype(int)'''
from sklearn.preprocessing import LabelEncoder
categorical_columns = train.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le
from sklearn.preprocessing import LabelEncoder
categorical_columns = test.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])
    label_encoders[col] = le

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import xgboost as xgb
import sklearn
from sklearn.preprocessing import LabelEncoder
x = train.iloc[:,:-1].values
y =  train.iloc[:,-1]# all rows except -1 index
rfmodel =RandomForestRegressor(n_estimators=300,max_features=10,max_depth=12)
model = XGBRegressor(n_estimators=400,learning_rate=0.05, max_depth=5)
model.fit(x,y)
rfmodel.fit(x,y)
y_test_pred = model.predict(test)
y_testrf_pred = rfmodel.predict(test)
avg = (y_testrf_pred+y_test_pred)/2

submission_df = pd.DataFrame({'Id': test['Id'], 'SalePrice':avg})
submission_df.to_csv('submission_rf.csv', index=False)
submission_df.head(10)




