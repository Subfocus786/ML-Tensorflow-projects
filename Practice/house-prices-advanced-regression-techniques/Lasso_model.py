from Kaggle_test import  train
from Kaggle_test import  test
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,LassoCV
import pandas as pd


'''creating a scaler object'''
scaler = StandardScaler()

iter=100000
lasso_model = Lasso(max_iter=iter,alpha=0.1)
x = train.iloc[:,:-1].values
y =  train.iloc[:,-1]
#lasso_model.fit(x,y)

'''making predictions'''
lassocv_model = LassoCV(max_iter=1000000,alphas=[0.01,0.1,0.2,1,10],cv=7)
lassocv_model.fit(x,y)

prediction = lassocv_model.predict(test)
submission_df = pd.DataFrame({'Id': test['Id'], 'SalePrice': prediction*1000000})
submission_df.to_csv('submission_lassocv.csv', index=False)




