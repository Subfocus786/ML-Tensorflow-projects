import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
raw_data_df = pd.read_csv('weatherAUS.csv')# loading the data frame
#print (raw_data_df)# checking if data is outputted property in DF
raw_data_df.dropna(subset=['RainToday','RainTomorrow'], inplace = True) #dropna function is used to remove rows with null values
train_val_df, test_df = train_test_split(raw_data_df, test_size=0.2, random_state=42) # intitialising validation/testing data frames, specific size and always a stable seed (random number genrator)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42) # making a training and validation data train
input_cols = list(train_df.columns [1:-1]) # defining input columns , removing target columbs
target_col = 'RainTomorrow' #target columb
# making new data frame for training inputs and targets using speficic columbs and .copy()
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()
''' making a validation input and data frame'''
val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()
''' making a test input and data frame'''
test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()
''' to find numeric and categorical columbs'''
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist() #in train_inputs.select_dtypes parameter (includes).columbs .tolist() functionwith a range
categorical_cols = train_inputs.select_dtypes(object).columns.tolist()
''' bring in a simple imputer to fill missing data frame '''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean') # creating a simple imputer object which works on filling with means
''' checking the number of missing '''
#Num_missing=raw_data_df[numeric_cols].isna().sum()
'''Fitting the imputer to the raw file  '''
imputer.fit(raw_data_df[numeric_cols]) # only towards the numeric columns of the raw data frame and averages are calculated
'''Filled in to training /validation and testing data'''
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols]= imputer.transform(test_inputs[numeric_cols])
'''Bring in a scalar object from preprossesing'''
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler() #creating the scalar object
scalar.fit(raw_data_df[numeric_cols]) # Fitting for computation
# transforming the data with the scalar (0 to 1)
train_inputs[numeric_cols] = scalar.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scalar.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols]= scalar.transform(test_inputs[numeric_cols])
'''Encoder for Categorical values'''
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore') # Getting a encoder and making a encoder object
raw2_data_df = raw_data_df[categorical_cols].fillna('unknown') #creating a dataframe with NaN value = 'unknown'
encoder.fit(raw2_data_df[categorical_cols]) #fitting the encoder to all categorical columbs
#print(encoder.categories_) # checking the encoder categories
'''Generate encoded columb names'''
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#print(encoded_cols) # Testing out new encoded columbs
'''Transforming  and  '''
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])
#print (train_inputs)
'''Pre processing the data completed ''' # saving processed files
import pyarrow
'''train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')

pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')'''

'''Training a regression model'''
from sklearn.linear_model import LogisticRegression
LGsolver = LogisticRegression(tol=0.0001,solver='liblinear',max_iter=1000,) #creating the solver object
LGsolver.fit(train_inputs[numeric_cols + encoded_cols], train_targets)
'''Predictiction'''
'''X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]
train_preds = LGsolver.predict(X_train)'''

'''We can can also get probablistic predictions '''
train_probs = LGsolver.predict_proba(X_train)

'''Acurracy of the model'''
from sklearn.metrics import accuracy_score
print(accuracy_score(train_targets,train_preds))
''' Confusion matrix'''

from sklearn.metrics import confusion_matrix
#print(confusion_matrix(train_targets, train_preds, normalize='true'))
'''Checking against random and all no data '''
def random_guess(inputs):
    return np.random.choice(["No", "Yes"], len(inputs))
def all_no(inputs):
    return np.full(len(inputs), "No")
def get_model():
    return LGsolver











