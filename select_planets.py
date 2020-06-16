import numpy as np
import pandas as pd


df = pd.read_csv('/Users/ivshina/Desktop/usrp/orbital_decay/exoplanet.eu_catalog.csv')

df['mass'] = 0.0009543 * df['mass'] #planet mass in solar masses
df['semi_major_axis'] = 215.032 * df['semi_major_axis'] #semi-major axis in solar radii 

# divide m/M
df[['m_M']] = df[['mass']].div(df.star_mass, axis=0)
# divide a/R
df[['a_R']] = df[['semi_major_axis']].div(df.star_radius, axis=0)
# select rows where the two divisions are not NaN
df = df[ ~df['m_M'].isnull() & ~df['a_R'].isnull()]
# raise a/R to (a/R)^-5
df['a_R'] = np.power((df['semi_major_axis']), -5)
# define condition to sample planets
condition = df['a_R'] * df['m_M']


print('condition ', condition)
df = df[ condition > 10e-9]

print(df['a_R'].head())
 
df = df.drop(['m_M', 'a_R'], axis=1)
df.to_csv('/Users/ivshina/Desktop/usrp/orbital_decay/sampled_planets.csv')

 