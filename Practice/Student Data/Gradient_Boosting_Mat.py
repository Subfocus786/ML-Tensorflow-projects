import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import X
mat_raw_df = pd.read_csv('mat2.csv')
X = mat_raw_df
def pass_fail(score): #defining a pass fail function
    threshold=10
    return 'pass' if score > threshold else 'fail'  # 1= pass , 0=fail

grade_cols=['G1','G2','G3']
for col in grade_cols:
    X[col] = X[col].apply(pass_fail) # using apply function to  rows G1 ,G2 , G3
X = X.drop('Id',axis=1)
fail_col=['failures']
def failure(n): # converting past faliurs to a catogorical Dtype
    return 'acc' if n<3 else 'nacc'
for col in fail_col:
    X[col] = X[col].apply(failure)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(X, test_size=0.25, random_state=42) # intitialising validation/testing data frames, specific size and always a stable seed (random number genrator)
input_cols = list(train_df.columns [1:-1]) # defining input columns , removing target columbs
target_col = 'G3' #target columb
# making new data frame for training inputs and targets using speficic columbs and .copy()
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
''' making a validation input and data frame'''
''' making a test input and data frame'''
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()
''' to find numeric and categorical columbs'''
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist() #in train_inputs.select_dtypes parameter (includes).columbs .tolist() functionwith a range
categorical_cols = train_inputs.select_dtypes(object).columns.tolist()
#print(numeric_cols)
#print (categorical_cols)
'''Bring in a scalar object from preprossesing'''
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler() #creating the scalar object
scalar.fit(X[numeric_cols]) # Fitting for computation
# transforming the data with the scalar (0 to 1)
train_inputs[numeric_cols] = scalar.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols]= scalar.transform(test_inputs[numeric_cols])
'''Encoder for Categorical values'''
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore') # Getting a encoder and making a encoder object
raw2_data_df = X[categorical_cols].fillna('unknown') #creating a dataframe with NaN value = 'unknown'
encoder.fit(raw2_data_df[categorical_cols]) #fitting the encoder to all categorical columbs
#print(encoder.categories_) # checking the encoder categories
'''Generate encoded columb names'''
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#print(encoded_cols) # Testing out new encoded columbs
'''Transforming  and  '''
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

'''Removning orignal categorical columbs and fitting in encoded ones '''
X_train = train_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

from xgboost import XGBClassifier
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train,train_targets)












