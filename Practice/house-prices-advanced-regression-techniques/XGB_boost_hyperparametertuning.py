import numpy as np
import pandas as pd
from xgboost import XGBRegressor
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
def optimize_xgb_hyper (X_train,y_train,X_test,y_test):
    xgb_model = XGBRegressor()

    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': [300,400,500,550,600,700],
        'learning_rate': [0.1,0.2,0.22],
        'n_jobs':[None,-1],
        'gamma':[0.02,0.05,0.1],
        'max_depth': [4,5,8,10,12],
        'subsample': [0.1,0.5,0.8],
        'colsample_bytree': [0.6,0.7,0.8]
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