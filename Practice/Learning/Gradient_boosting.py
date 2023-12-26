import opendatasets as op


import opendatasets #for handling data from kaggel
raw_df = pd.read_csv('weatherAUS.csv')
'''Null values must be removed '''
#print (raw_df)
raw_df.dropna(subset=['RainTomorrow'],inplace=True) #null values in Rain tomorrow
'''creating train test validation split based on the seperation of years'''
year = pd.to_datetime(raw_df.Date).dt.year #pd.to_datetime takes in the data frames date field and parses it from that dt field we take year in year
#print (year)
''' Creating 3 types sets based on yearly split'''
train_df = raw_df[year < 2015]
val_df = raw_df [ year == 2015]
test_df = raw_df[ year > 2015]
''' defining input/ output columbs'''
input_col = list(train_df.columns)[1:-1] # getting a list of columb names we want to train the model on
target_col ='RainTomorrow' #name of the columb which is the target
'''Creating test train and validation data sets  inputs and targets'''
training_inputs = train_df[input_col].copy()
training_targets = train_df[target_col].copy()
validation_inputs = val_df[input_col].copy()
validation_targets = val_df[target_col].copy()
testing_inputs = test_df [input_col].copy()
testing_targets = test_df[target_col].copy()

''' how to find data types in a data set dataframename.select_dtypes(include='object' or np.numbers).columbs.tolist() o'''
''' devide the data into catagorical and numeric and encode catagorical scale the numeric data'''
numeric_cols = training_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = training_inputs.select_dtypes('object').columns.tolist()
imputer = SimpleImputer(strategy = 'mean').fit(raw_df[numeric_cols])
training_inputs[numeric_cols] = imputer.transform(training_inputs[numeric_cols])
validation_inputs[numeric_cols] = imputer.transform(validation_inputs[numeric_cols])
testing_inputs[numeric_cols] = imputer.transform(testing_inputs[numeric_cols])
'''Bring in a scalar object from preprossesing'''
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler() #creating the scalar object
scalar.fit(raw_df[numeric_cols]) # Fitting for computation
# transforming the data with the scalar (0 to 1)
training_inputs[numeric_cols] = scalar.transform(training_inputs[numeric_cols])
validation_inputs[numeric_cols] = scalar.transform(validation_inputs[numeric_cols])
testing_inputs[numeric_cols]= scalar.transform(testing_inputs[numeric_cols])
'''Encoder for Categorical values'''
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore') # Getting a encoder and making a encoder object
raw2_data_df = raw_df[categorical_cols].fillna('unknown') #creating a dataframe with NaN value = 'unknown'
encoder.fit(raw2_data_df[categorical_cols]) #fitting the encoder to all categorical columbs
#print(encoder.categories_) # checking the encoder categories
'''Generate encoded columb names'''
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#print(encoded_cols) # Testing out new encoded columbs
'''Transforming  and  '''
training_inputs[encoded_cols] = encoder.transform(training_inputs[categorical_cols])
validation_inputs[encoded_cols] = encoder.transform(validation_inputs[categorical_cols])
testing_inputs[encoded_cols] = encoder.transform(testing_inputs[categorical_cols])
'''Removning orignal categorical columbs and fitting in encoded ones '''
X_train = training_inputs[numeric_cols + encoded_cols]
X_val = validation_inputs[numeric_cols + encoded_cols]
X_test = testing_inputs[numeric_cols + encoded_cols]
'''Classifier cause its a YEs or No type problem we could have used Decison treee regressor'''
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3,random_state=43)
model.fit(X_train, training_targets)
'''Evaluation'''
from sklearn.metrics import accuracy_score, confusion_matrix
train_preds = model.predict(X_train)
#print (accuracy_score(training_targets, train_preds))
#checking on validation set too!!
#print(model.score(X_val,validation_targets)) #--> 79% but the training data has 78% no so no learning has takingplace
'''Visilulisation'''
from sklearn.tree import plot_tree, export_text
'''plt.figure(figsize=(80,20))
plot_tree(model, feature_names=X_train.columns, max_depth=2 , filled=True)
plt.show()'''
''' importance'''
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
'''Checking the importantce in bar graph'''
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature')
# plt.show()
''' By reducing the depth of the decision tree we can make good genral '''
# Defining a helper function to dest diffirent depths
def max_depth_error(md):
    model = DecisionTreeClassifier(max_depth=md, random_state=43)
    model.fit(X_train, training_targets)
    train_acc = 1 - model.score(X_train, training_targets)
    val_acc = 1 - model.score(X_val, validation_targets)
    return {'Max Depth': md, 'Training Error': train_acc, 'Validation Error': val_acc}

error_df = pd.DataFrame([max_depth_error(md) for md in range (1,21)])
print(error_df)

plt.figure()
plt.plot(error_df['Max Depth'], error_df['Training Error'])
plt.plot(error_df['Max Depth'], error_df['Validation Error'])
plt.title('Training vs. Validation Error')
plt.xticks(range(0,21, 2))
plt.xlabel('Max. Depth')
plt.ylabel('Prediction Error (1 - Accuracy)')
plt.legend(['Training', 'Validation'])
#plt.show()










