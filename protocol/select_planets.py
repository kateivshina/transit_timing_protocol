import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('planets.csv')

df['pl_bmassj'] = 0.0009543 * df['pl_bmassj'] #planet mass in solar masses
df['pl_orbsmax'] = 215.032 * df['pl_orbsmax'] #semi-major axis in solar radii 

# divide m/M
df[['m_M']] = df[['pl_bmassj']].div(df.st_mass, axis=0)
# divide a/R
df[['a_R']] = df[['pl_orbsmax']].div(df.st_rad, axis=0)
# select rows where the two divisions are not NaN
df = df[ ~df['m_M'].isnull() & ~df['a_R'].isnull()]
# raise a/R to (a/R)^-5
df['a_R'] = np.power((df['pl_orbsmax']), -5)
# define condition to sample planets
condition = df['a_R'] * df['m_M']


print('condition ', condition)
sample_df = df[df['st_optmag'] < 13]
sample_df = sample_df[condition > 10e-9 ]

print('head ', sample_df['a_R'].head())
 
sample_df = sample_df.drop(['m_M', 'a_R'], axis=1)
sample_df.to_csv('sampled_planets.csv')

 
# plotting transit depth (S_pl/S_star) vs Period
# original df
df = df[ ~df['pl_radj'].isnull() & ~df['pl_orbper'].isnull()]
df['pl_radj'] = 0.10049 * df['pl_radj'] #planet radius in solar radii 

df[['depth']] = df[['pl_radj']].div(df.st_rad, axis=0)
df['depth'] = np.power((df['depth']), 2)

# sampled df
 # plotting transit depth (S_pl/S_star) vs Period
sample_df = sample_df[ ~sample_df['pl_radj'].isnull() & ~sample_df['pl_orbper'].isnull()]
sample_df['pl_radj'] = 0.10049 * sample_df['pl_radj'] #planet radius in solar radii 

sample_df[['depth']] = sample_df[['pl_radj']].div(sample_df.st_rad, axis=0)
sample_df['depth'] = np.power((sample_df['depth']), 2)

ax = plt.gca()


#df['st_optmag'] = df['st_optmag']/df['st_optmag'].max()
''' 
fig = go.Figure(data=go.Scatter(x=sample_df['pl_orbper'],
                                y=sample_df['depth'],
                                mode='markers',
                                marker_color=sample_df['st_optmag'],
                                text=sample_df['st_optmag'])) # hover text goes here

fig.update_layout(title='PDepth vs Period')
fig.show()
'''


fig = go.Figure(data=go.Scatter(x=sample_df['pl_orbper'],
                                y=sample_df['depth'],
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
            x=-0.1,
            y=0.5,
            showarrow=False,
            text="Depth",
            textangle=-90,
            xref="paper",
            yref="paper")], title='Depth vs Period')
#fig.write_image("fig1.png")
fig.show()






