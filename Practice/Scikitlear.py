import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
medical_df = pd.read_csv("medical-charges.csv")
#medical_df.info()
#print(medical_df.age.describe())
#fig = px.histogram(medical_df,x="age",y=None,marginal="box",nbins=47,title='Distribution of Age')
#fig.update_layout(bargap=0.1)
#fig.show()
#fig = px.histogram(medical_df,
 #                  x='charges',
  #                 marginal='box',
   #                color='region'
    #               color_discrete_sequence=['blue', 'pink'],
              #     title='Annual Medical Charges')
#fig.update_layout(bargap=0.2)
#fig.show()
#print(medical_df.charges.corr(medical_df.age))
#smoker_values = {'no': 0, 'yes': 1}
#smoker_numeric = medical_df.smoker.map(smoker_values)
#print(medical_df.charges.corr(smoker_numeric))
#print(smoker_numeric)
non_smoker_df = medical_df[medical_df.smoker =='no'] #creating a non smoker data frame
smoker_df = medical_df[medical_df.smoker == 'yes'] #creating a smoker data
#ig=px.scatter(non_smoker_df,x='age',y='charges')
#ig.show()
#plt.title('Age vs Charges')
#sns.scatterplot(non_smoker_df,x='age',y='charges')
#plt.show()
def estimate_charges(age,w=50,b=100):
    return w*age+b
##ages= non_smoker_df.age
#estimated_charges = estimate_charges(ages,w=50, b=100)
#plt.plot(ages, estimated_charges, 'r-o');
#plt.xlabel('Age');
#plt.ylabel('Estimated Charges');
#target =  non_smoker_df.charges
#plt.scatter(ages,target,s=8)
#plt.xlabel('Age')
#plt.ylabel('Charges')
#plt.legend(['Estimate' , 'Actual'])
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    predictions = estimate_charges(ages, w, b)

    plt.plot(ages, predictions, 'r', alpha=0.9);
    plt.scatter(ages, target, s=8, alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual']);
    plt.show()
    loss = rmse(target,predictions)
    print('RMSE Loss:' , loss)
#try_parameters(100,-5000)
#smoker_code={'no':0,'yes':1}
sex_code = {'female':0,'male':1}
#medical_df['smoker_code'] =medical_df.smoker.map(smoker_code)
non_smoker_df['sex_code'] = non_smoker_df.sex.map(sex_code)
#print (medical_df)
enc = preprocessing.OneHotEncoder() #create a encoding class
enc.fit(non_smoker_df[['region']]) #use the encoder's fits method to medical data frames region columb
enc.categories
  # create an array with the encodings
one_hot = enc.transform(non_smoker_df[['region']]).toarray()
non_smoker_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot
print (non_smoker_df)
model = LinearRegression() #create a new model class
inputs = non_smoker_df[['age','bmi','children','sex_code']] # prepare inputs
targets = non_smoker_df.charges #declare the chareges is the function we want to get from linear model
model.fit(inputs, targets) #use Linear Regressin classes .fit() for fitting
predictions = model.predict(inputs)
print(predictions)
print(targets)
loss = rmse(targets,predictions)
print('RMSE Loss:' , loss)


'''plot_1=sns.barplot(data=medical_df,x='smoker',y="charges")
plt.show()'''
'''smoker_code={'no':0,'yes':1}
medical_df['smoker_code'] =medical_df.smoker.map(smoker_code)
coor2 = medical_df.charges.corr(medical_df.smoker_code)


#Create inputs and targets
inputs, targets = medical_df[['age', 'bmi', 'children', 'smoker_code']], medical_df['charges']

# Create and train the model
model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)'''

















