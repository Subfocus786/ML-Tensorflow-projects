import pandas
import pandas as pd
'''Creating merged Data frame from Excel data'''
'''
o2_df=pd.read_excel('o2_2016-2017output.xlsx')
ph_df = pd.read_excel('PH_2016-2017output.xlsx')
sc_df=pd.read_excel('Specific_conductance_2016-2017output.xlsx')
temp_df = pd.read_excel('temp_water_2016-2017output.xlsx')
tur_df=pd.read_excel('turbidty_2016-2017output.xlsx')
merge_col = ['DAY','MONTH','YEAR','Time']
merged_df = pd.merge(o2_df, ph_df, how='left', on=merge_col, suffixes=('_o2', '_ph'))
merged_df = pd.merge(merged_df, sc_df, how='left', on=merge_col, suffixes=('_merged', '_sc'))
merged_df = pd.merge(merged_df, temp_df, how='left', on=merge_col, suffixes=('_merged', '_temp'))
merged_df = pd.merge(merged_df, tur_df, how='left', on=merge_col, suffixes=('_merged', '_tur'))


print (merged_df)
merged_df.to_excel('merged_dfforchecking.xlsx')
'''






