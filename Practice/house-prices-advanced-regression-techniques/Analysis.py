import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

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
numerical_col = imp_df.select_dtypes(include=['number']).columns.tolist()
categorial_col = imp_df.select_dtypes(include=['object']).columns.tolist()

numerical_col.remove('SalePrice')
#print (numerical_col)
encoder = OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)
imp_df = imp_df.drop('Id',axis=1)
X_train, X_test, y_train, y_test = train_test_split(imp_df,imp_df['SalePrice'],test_size=0.30,random_state=42)
# minmax scaling of the traing and testing input Xtrain and Xtest
X_test=X_test.drop('SalePrice',axis=1)
X_train=X_train.drop('SalePrice',axis=1)
#print(X_test.select_dtypes(include=['number']).columns.tolist())
#print(X_test.select_dtypes(include=['object']).columns.tolist())
cat_col=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
minmax_col =['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
#scaling and encoding
scaler = MinMaxScaler().fit(X_train[minmax_col])
X_train[minmax_col]=scaler.transform(X_train[minmax_col])
X_test[minmax_col] = scaler.transform(X_test[minmax_col])
encoder.fit(X_train[cat_col])
encoded_col = list(encoder.get_feature_names_out(cat_col))
X_train[encoded_col] = encoder.transform(X_train[cat_col])
X_test[encoded_col] =encoder.transform(X_test[cat_col])
X = X_train[encoded_col+minmax_col]
x_test =X_test[encoded_col+minmax_col]
'''Training a XGB model'''


#defining a repitative function for MSR calculation.

def rmse (a,b):
    return mean_squared_error(a,b,squared=False)

def train_and_evaluate(X_train, train_targets, X_val, val_targets, **params):
    model = XGBRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, train_targets)
    train_rmse = rmse(model.predict(X_train), train_targets)
    val_rmse = rmse(model.predict(X_val), val_targets)
    return model, train_rmse, val_rmse



#print(train_and_evaluate(X,y_train,x_test,y_test, n_estimators=40, max_depth=5))

'''Writing a function to take in training data and Target and do a gride search on a list of parameters '''
def optimize_xgb_hyper (X_train,y_train,X_test,y_test):
    xgb_model = XGBRegressor()

    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': [35,100,200,300,400,500,550,600,700],
        'learning_rate': [0.2,0.22],
        'n_jobs':[-1],
        'gamma':[0.1,0.2,0.3],
        'max_depth': [4,5],
        'subsample': [0.1,0.5,0.8],
        'colsample_bytree': [0.6,0.7,0.8,0.9]
    }

    # Create the Grid Search Cross-Validation object
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

    # Fit the model to the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Use the best hyperparameters to train the final model
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = final_model.predict(X_test)

    # Calculate the Mean Squared Error on the test set
    mse = mean_squared_error(y_test, y_pred,squared=False)


    return best_params, final_model, mse

print(optimize_xgb_hyper(X,y_train,x_test,y_test))





