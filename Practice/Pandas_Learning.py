import pandas as pd
new_df = pd.DataFrame(columns=['#','Name','Type 1','Type 2','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Legendary'])
for df in  pd.read_csv("pokemon_data.csv", chunksize=100):
    results = df.groupby(['Type 1']).count()
    new_df = pd.concat([new_df,results])

print (new_df )


#df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk']+ df['Sp. Def']+ df['Speed']
#print(df)
#print(df.loc[(df['Type 1'] == "Grass") & (df['Type 2'] == "Poison")])
#df.to_csv('modified_pokemon_data.csv', index=False, sep="\t")
#df.to_excel("mod_inexcel",index=False)
#df.loc[df['Type 1'].str.contains('Fire|Grass', regex = True)]