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

'''Data Analysis'''

stn1_df = pd.read_excel('NS01EJ0157-2016-present_final')
print(stn1_df)



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

















