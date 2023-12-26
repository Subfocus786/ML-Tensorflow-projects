import pandas as pd

import numpy as np
import cufflinks as cf
import json
from urllib.request import urlopen
import seaborn as sns
import plotly.express as px
#flights = sns.load_dataset("flights")
#df = px.data.gapminder().query('year == 2007')
#print (df)
#fig = px.scatter_geo(df,locations='iso_alpha',color='continent',hover_name="country",size='pop',projection='orthographic')
#fig = px.scatter_3d(flights,x = 'year',y='month',z='passengers',color_continuous_scale = 'Viridis',opacity=0.7 )
#fig.show()
with urlopen ('https://raw.githubusercontent.com/plotly/datasets/masters/geojson-countries-fips.json') as response :
    countries=json.load(response)
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv',dtype={'flips':str})
fig = px.choropleth(df, geojson=countries,locations='fips',color='unemp',range_color=(0,15),scope='usa',labels={'unemp':'unemployment rate'})
fig.show()