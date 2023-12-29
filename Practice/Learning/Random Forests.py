import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
''' getting a random forest class'''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1, random_state=42) #making a random forest in object
from Gradient_boosting import X_train,training_targets,X_val,validation_targets
model.fit(X_train, training_targets) # Fitting
model.score(X_train, training_targets) #scoring training data
model.score(X_val, validation_targets) # scoring validations
train_probs = model.predict_proba(X_train) # .predict_proba methods()

''' acsessing individual desicion tress '''
model.estimators_[0] # .estimator function
'''Just like decision tree, random forests also assign an "importance" to each feature, by combining the importance values from individual trees'''
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature') # can plot this also

''' Hyper parameter tuning'''
''' base model to compare'''
base_model = RandomForestClassifier(random_state=42, n_jobs=-1).fit(X_train, training_targets)
base_train_acc = base_model.score(X_train, training_targets)
base_val_acc = base_model.score(X_val, validation_targets)
base_accs = base_train_acc, base_val_acc
base_accs

'''now n-estimators i.e number of desion trees must be reduced to prevent overfitting'''
model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=10)
model.fit(X_train, training_targets)

def plot_error_vs_estimators(X_train, y_train, X_test, y_test, max_estimators=100, step=10):
    """
    Plot training and test errors as a function of the number of estimators (trees) in a RandomForestClassifier.

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
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
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






























