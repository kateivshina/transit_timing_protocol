import numpy as np
import pandas as pd
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
sample_df = df[ condition > 10e-9]

print('head ', df['a_R'].head())
 
df = df.drop(['m_M', 'a_R'], axis=1)
df.to_csv('sampled_planets.csv')

 
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


df['st_optmag'] = df['st_optmag']/df['st_optmag'].max()


df.plot(kind='scatter',x='pl_orbper',y='depth',c='st_optmag', ax=ax)
sample_df.plot(kind='scatter',x='pl_orbper',y='depth', ax=ax, c='st_optmag')
plt.xlim(0, 500);
#plt.ylim(0, 0.04);
plt.xlabel("Period [days]")
plt.ylabel("Depth")
plt.show()