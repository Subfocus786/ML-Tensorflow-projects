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

''''''






























