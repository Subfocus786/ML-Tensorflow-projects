import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns





"Bringing in the Data Frame for analysis"
mat_raw_df = pd.read_csv('mat2.csv')
#mat_raw_df.describe().to_excel('summary.xlsx') #made a summary file
''' Checking the DS for Male/Female ratio'''
#sns.barplot(mat_raw_df,x='sex',y='Id',hue='school')
#plt.show()
'''age distribution'''
#sns.barplot(mat_raw_df,x='sex',y='age',hue='school')
#plt.show()
'''living place distribution with sex and urban'''
#sns.barplot(mat_raw_df,x='school',y='Id',hue='address')
#plt.show()

'Male female distro with respect to urban and rulral '
#sns.barplot(mat_raw_df,y='Id',x='address',hue='sex')
#plt.show()
'''Famlisize and students '''
#sns.barplot(mat_raw_df,y='Id',x='famsize',hue='address')
#plt.show()

'''relation between G1 & G2 & G3'''
#sns.lineplot(mat_raw_df,x='G1',y='G3')
#plt.show()
#mat_raw_df['AVG_G1/G2'] = mat_raw_df['G1']/2 + mat_raw_df['G2']/2
#sns.lineplot(mat_raw_df,x="AVG_G1/G2",y='G3')
#plt.show()

'''study time and grades?'''
#mat_raw_df["AVGgrade"] = (mat_raw_df['G1'] + mat_raw_df['G2'] +mat_raw_df['G3'])/3
#sns.barplot(mat_raw_df,x='studytime',y='AVGgrade',hue='sex')
#plt.show()

'''Relationship between grade and the number of absences '''
#mat_raw_df["AVGgrade"] = (mat_raw_df['G1'] + mat_raw_df['G2'] +mat_raw_df['G3'])/3
#mat_raw_df['Percent_absent'] = (mat_raw_df['absences']/185)*100
#sns.lineplot(mat_raw_df,x="Percent_absent",y='AVGgrade')
#plt.show()
'''I have two approaches in mind , one is to use G1,G2 and other factors for predicting G3 results since the data contains many catagoriacal features it would be interesting 
to see the relationship between the desisions, the other approach is to do simple linear regression task on the G1,G2,G3 results'''

"To make the G3 final result easer for a random forest model to predict we need to make it into Binary catagory i.e Pass/Fail ref https://ibs.iscte-iul.pt/contents/the-experience/international-experience/incoming-mobility-exchange-students-faculty/1588/grading-system"
# "Pass/fail model"
X = mat_raw_df # created a new data frame to modify
# converting all G1 G2 G3 data to pass/ fail based on G>10= pass else fail
def pass_fail(score): #defining a pass fail function
    threshold=10
    return 'pass' if score > threshold else 'fail'  # 1= pass , 0=fail

grade_cols=['G1','G2','G3']
for col in grade_cols:
    X[col] = X[col].apply(pass_fail) # using apply function to  rows G1 ,G2 , G3
X=X.drop('Id',axis=1)
fail_col=['failures']
def failure(n): # converting past faliurs to a catogorical Dtype
    return 'acc' if n<3 else 'nacc'
for col in fail_col:
    X[col] = X[col].apply(failure)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(X, test_size=0.50, random_state=42) # intitialising validation/testing data frames, specific size and always a stable seed (random number genrator)
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

#print(X_test)

#print (X_test)
''' getting a random forest class'''
from sklearn.ensemble import RandomForestClassifier
base_model = RandomForestClassifier(n_estimators=100,n_jobs=-1, random_state=42) #making a random forest in object
base_model.fit(X_train, train_targets) # Fitting
base_model.score(X_train, train_targets) #scoring training data
base_model.score(X_test, test_targets) # scoring validations
train_probs = base_model.predict_proba(X_train) # .predict_proba methods()

''' acsessing individual desicion tress '''
#base_model.estimators_[0] # .estimator function
'''Just like decision tree, random forests also assign an "importance" to each feature, by combining the importance values from individual trees'''
#importance_df = pd.DataFrame({
  #  'feature': X_train.columns,
   # 'importance': base_model.feature_importances_
#}).sort_values('importance', ascending=False)
#plt.title('Feature Importance')
#sns.barplot(data=importance_df.head(10), x='importance', y='feature') # plotting this
#plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
def plot_error_vs_estimators(X_train, y_train, X_test, y_test, max_estimators=100, step=10):
    """

    Parameters:
    - X_train: Training feature matrix
    - y_train: Training target variable
    - X_test: Test feature matrix
    - y_test: Test target variable
    - max_estimators: Maximum number of estimators to consider
    - step: Step size for the number of estimators
    """

    # Initialize arrays to store errors
    train_errors = []
    test_errors = []
    estimator_values = list(range(1, max_estimators + 1, step))

    for n_estimators in estimator_values:
        # Create and train the RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators,random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate accuracy and store errors
        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)

        train_errors.append(train_error)
        test_errors.append(test_error)

    # Plot the errors
    plt.figure(figsize=(10, 6))
    plt.plot(estimator_values, train_errors, label='Training Error', marker='o')
    plt.plot(estimator_values, test_errors, label='Test Error', marker='o')
    plt.title('Training and Test Errors vs. Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

'''For cheking the multple parameters'''
def HP_tuning(param1_values,param2_values):
    #param1_values = [20]
    #param2_values = [None, 2, 4, 6, 8, 9, 10]

    # Initialize arrays to store results
    accuracy_matrix = np.zeros((len(param1_values), len(param2_values)))

    # Iterate over hyperparameter values
    for i, param1 in enumerate(param1_values):
        for j, param2 in enumerate(param2_values):
            # Create and train the RandomForestClassifier with current hyperparameters
            model = RandomForestClassifier(n_estimators=param1, max_depth=param2, random_state=42)
            model.fit(X_train, train_targets)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Calculate accuracy and store in the matrix
            accuracy_matrix[i, j] = accuracy_score(test_targets, y_pred)

    # Plot the results using a heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(accuracy_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Accuracy')
    plt.title('Accuracy for Different Hyperparameter Values')
    plt.xlabel('Depth of tree')
    plt.ylabel('Number of estimators')
    plt.grid(True)
    plt.xticks(np.arange(len(param2_values)), param2_values)
    plt.yticks(np.arange(len(param1_values)), param1_values)
    plt.show()
#plot_error_vs_estimators(X_train,train_targets, X_test, test_targets, max_estimators=1000, step=25)

np.random.seed(42)
X = np.random.rand(100, 5)
y = np.random.randint(2, size=100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions with the RF model
y_pred_rf = rf_model.predict(X_test)

# Create a baseline model that always predicts "fail"
y_pred_baseline = np.ones_like(y_test)  # Assuming 1 represents "fail"

# Evaluate the models
print("Random Forest Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(test_targets, y_pred_rf))
print("Classification Report:")
print(classification_report(test_targets, y_pred_rf))

print("\nBaseline Model (Always predicts 'fail'):")
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_baseline))
print("Classification Report:")
print(classification_report(y_test, y_pred_baseline))