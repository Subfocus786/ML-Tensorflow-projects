import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import seaborn as sns
import numpy as np




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
mat_raw_df["AVGgrade"] = (mat_raw_df['G1'] + mat_raw_df['G2'] +mat_raw_df['G3'])/3
mat_raw_df['Percent_absent'] = (mat_raw_df['absences']/185)*100
sns.lineplot(mat_raw_df,x="Percent_absent",y='AVGgrade')
plt.show()

































