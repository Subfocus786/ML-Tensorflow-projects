from XGB_boost_hyperparametertuning import optimize_xgb_hyper

from Analysis import X_test,X_train,y_train,y_test

best_params, final_model, mse =optimize_xgb_hyper(X_train,y_train,X_test,y_test)

print ('the best params',best_params)
print('MSE= ',mse)