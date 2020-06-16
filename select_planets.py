import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
sample_df = df[ condition > 10e-9]

print('head ', df['a_R'].head())
 
df = df.drop(['m_M', 'a_R'], axis=1)
df.to_csv('/Users/ivshina/Desktop/usrp/orbital_decay/sampled_planets.csv')

 
# plotting transit depth (S_pl/S_star) vs Period
# original df
df = df[ ~df['radius'].isnull() & ~df['orbital_period'].isnull()]
df['radius'] = 0.10049 * df['radius'] #planet radius in solar radii 

df[['depth']] = df[['radius']].div(df.star_radius, axis=0)
df['depth'] = np.power((df['depth']), 2)

# sampled df
 # plotting transit depth (S_pl/S_star) vs Period
sample_df = sample_df[ ~sample_df['radius'].isnull() & ~sample_df['orbital_period'].isnull()]
sample_df['radius'] = 0.10049 * sample_df['radius'] #planet radius in solar radii 

sample_df[['depth']] = sample_df[['radius']].div(sample_df.star_radius, axis=0)
sample_df['depth'] = np.power((sample_df['depth']), 2)

ax = plt.gca()
# a scatter plot comparing num_children and num_pets
df.plot(kind='scatter',x='orbital_period',y='depth',color='blue', ax=ax)
sample_df.plot(kind='scatter',x='orbital_period',y='depth',color='red', ax=ax)
#plt.xlim(time.min(), time.max());
#plt.ylim(0, 0.04);
plt.xlabel("Period [days]")
plt.ylabel("Depth")
plt.show()