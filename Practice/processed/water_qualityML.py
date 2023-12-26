import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

'''pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)'''
''' Bring in the file'''
#rawdata_df = pd.read_csv('water-qual-processed dataNS01EJ0157-2016-present.csv')
''' making Datetime object'''
'''rawdata_df ['DATE'] = pd.to_datetime(rawdata_df['DATE'],format='mixed')
rawdata_df['YEAR'] = rawdata_df['DATE'].dt.year
rawdata_df['MONTH'] = rawdata_df['DATE'].dt.month
rawdata_df['DAY'] = rawdata_df['DATE'].dt.day
selected_variable = 'TURBIDITY'
variable_df = rawdata_df[rawdata_df['VARIABLE'] == selected_variable]

variablecleaned_df =variable_df.dropna()
excel_file_path = 'turbidty_2016-2017output.xlsx'
variablecleaned_df.to_excel(excel_file_path,index=False)'''

'''DATA ANALYSIS'''
'''
stn1_df = pd.read_excel('NS01EJ0157-2016-present_final.xlsx')
stn1_df = stn1_df.reset_index()
plt.figure(figsize=(10,6))
stn1_df['DATE'] = pd.to_datetime(stn1_df['DATE'])
stn1_df.set_index('DATE',inplace=True)
sns.lineplot(x=stn1_df.index,y='OXYGEN DISSOLVED (MG/L)',data=stn1_df)
y_value = "PH"
#sns.lineplot(x=stn1_df.index,y=y_value,data=stn1_df)
plt.title("Parameter vs time")
plt.xlabel('Date')
plt.ylabel('Parameter')
plt.grid(True)
plt.show()
stn1_df = stn1_df.reset_index()
'''
stn1_df = pd.read_excel('NS01EJ0157-2016-present_final.xlsx')
#describe_df = stn1_df.describe()
#describe_df.to_excel('describe_df.xlsx')
stn1_df['DATE'] = pd.to_datetime(stn1_df['DATE'])
#stn1_reset = stn1_df.reset_index(drop=False)
#print(stn1_df.isna().any().any())
stn1_df = stn1_df.dropna()
#print(stn1_df.isna().any().any())

#stn1_df.set_index('DATE',inplace=True)
#sns.lineplot(stn1_df,x=stn1_df.index,y='')
sns.barplot(stn1_df,x="YEAR",y='PH',hue='MONTH',gap=0.2)
plt.show()




'''seeing the trend in o2'''
'''o2_data = pd.read_excel('o2_2016-2017output.xlsx')
print (o2_data)
o2_data['DATE'] = pd.to_datetime(o2_data['DATE'])

o2_data.set_index('DATE',inplace=True)
plt.figure(figsize=(10, 10))
sns.lineplot(x=o2_data.index, y='OXYGEN DISSOLVED (MG/L)', data=o2_data,)
plt.title('Time Series Plot using Seaborn')
plt.xlabel('Date')
plt.ylabel('Oxygen Content')
plt.grid(True)
plt.show()'''

















