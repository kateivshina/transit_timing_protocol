import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('planets.csv')

df['pl_bmassj_sol'] = 0.0009543 * df['pl_bmassj'] #planet mass in solar masses
df['pl_orbsmax_sol'] = 215.032 * df['pl_orbsmax'] #semi-major axis in solar radii 

# divide m/M
df[['m_M']] = df[['pl_bmassj_sol']].div(df.st_mass, axis=0)
# divide a/R
df[['a_R']] = df[['pl_orbsmax_sol']].div(df.st_rad, axis=0)
# select rows where the two divisions are not NaN
df = df[ ~df['m_M'].isnull() & ~df['a_R'].isnull()]
# raise a/R to (a/R)^-5
df['a_R'] = np.power((df['pl_orbsmax_sol']), -5)
# define condition to sample planets
df['sampling_condition'] = df['a_R'] * df['m_M']


sample_df = df[df['st_optmag'] < 13]
sample_df = sample_df.sort_values(by=['sampling_condition'], ascending = False)
#sample_df = sample_df[condition > 10e-9 ]

print('head ', sample_df['pl_trandur'].head())
 
sample_df = sample_df.drop(['m_M', 'a_R'], axis=1)
sample_df.to_csv('sampled_planets.csv')

 
# plotting transit depth (S_pl/S_star) vs Period
# original df
df = df[ ~df['pl_trandep'].isnull() & ~df['pl_orbper'].isnull()]


 # plotting transit depth (S_pl/S_star) vs Period
sample_df = sample_df[ ~sample_df['pl_trandep'].isnull() & ~sample_df['pl_orbper'].isnull()]
 
 


 

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
            text="Depth",
            textangle=-90,
            xref="paper",
            yref="paper")], title='Depth vs Period')
fig.show()






