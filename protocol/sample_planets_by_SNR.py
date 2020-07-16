import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 


df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/transiting_planet_database.csv')

 

# define condition to sample planets
df['SNR_sort'] =  df['pl_trandep'] * df['pl_trandur'] / (df['pl_orbper'] * df['TESS_mag_uncert'])


df = df.sort_values(by=['SNR_sort'], ascending = False)
print('head ', df['SNR_sort'].head())
#sample_df = df[df['st_optmag'] < 13]
#sample_df = sample_df.sort_values(by=['sampling_condition'], ascending = False)
#sample_df = sample_df[condition > 10e-9 ]

#print('head ', sample_df['pl_trandur'].head())
 
#sample_df = sample_df.drop(['m_M', 'a_R'], axis=1)
df.to_csv(os.path.dirname(os.getcwd()) + '/data/sampled_planets.csv', index = False)

 
# plotting transit depth (S_pl/S_star) vs Period
# original df
#df = df[ ~df['pl_trandep'].isnull() & ~df['pl_orbper'].isnull()]


 # plotting transit depth (S_pl/S_star) vs Period
sample_df = df[ ~df['pl_trandep'].isnull() & ~df['pl_orbper'].isnull()]
 
 


 

fig = go.Figure(data=go.Scatter(x=sample_df['pl_orbper'],
                                y=sample_df['pl_trandep'],
                                mode='markers',
                                marker=dict(
        						size=16,
        						color=sample_df['st_optmag'], #set color equal to a variable
        						colorscale='Viridis', # one of plotly colorscales
        						showscale=True))) # hover text goes here

fig.update_layout(annotations=[
        dict(
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Period [days]",
            xref="paper",
            yref="paper"),
        dict(
            x=-0.12,
            y=0.5,
            showarrow=False,
            text="Depth (%)",
            textangle=-90,
            xref="paper",
            yref="paper")], title='Depth vs Period')
fig.show()






