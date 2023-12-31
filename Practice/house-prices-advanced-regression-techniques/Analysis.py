import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

raw_df = pd.read_csv('train.csv')
#print(raw_df)
missing_var =raw_df.isna().sum()
#print (missing_var)
#missing_var.to_excel('missing_variable.xlsx')
target_col=['SalePrice']
removal_col =['Alley','MasVnrType','FireplaceQu','PoolQC','Fence','MiscFeature']
impute_col = ['LotFrontage']
row_removal_col = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']
'''Dropping columbs with high missing values'''
"remove columb,imputecol,row removal"
col_drop_df = raw_df.drop(columns=removal_col)
#print (col_drop_df)
from sklearn.impute import SimpleImputer
#col_drop_df.describe().to_excel('important parameters.xlsx')
"Checking the distribution of LotFrontage columb"
#sns.boxplot(col_drop_df,x="LotFrontage")
#sns.histplot(col_drop_df['LotFrontage'], kde=True,color='blue')
#plt.show()
'''Cheking Mean, Median and mode '''
#print (col_drop_df['LotFrontage'].mode())
'''Mean is selected for LotFrontage'''
''''''
#sns.histplot(col_drop_df['MasVnrArea'], kde=True,color='blue')
#plt.show()
simimp=SimpleImputer(strategy='mean')
imp_df =col_drop_df
imp_df[impute_col] = simimp.fit_transform(col_drop_df[impute_col])
imp_df.dropna(axis=0,how='any',inplace=True)
imp_df.drop('Id',axis=1)
numerical_col = imp_df.select_dtypes(include=['number']).columns
categorial_col = imp_df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(drop='first',sparse_output=False)
X_train, X_test, y_train, y_test = train_test_split(imp_df,imp_df['SalePrice'],test_size=0.25,random_state=42)
# Encode Xtrain and Xtest
X_enc_train = encoder.fit_transform()










