import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import batman
import emcee
import os, sys, time
import pandas as pd
import corner
import time as timing
from argparse import ArgumentParser
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import sys

# parse data about the planet
parser = ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--mission')
parser.add_argument('--pl_hostname')
parser.add_argument('--pl_letter') 
parser.add_argument('--cadence')
parser.add_argument('--N')
parser.add_argument('--parent_dir')
parser.add_argument('--path_to_data_file')
parser.add_argument('--refolded')


args = parser.parse_args()

action = args.refolded



 
# path info
MISSION = args.mission
cadence = args.cadence
planet_name = args.pl_hostname + args.pl_letter
path_to_data_file =args.path_to_data_file
# Path 
parent_dir = args.parent_dir
directory = planet_name.replace(" ", "_") 
path = f'{parent_dir}' + f'/{directory}'  
path2table = parent_dir + '/LD.csv'

# load CSV file with the exoplanet data
df = pd.read_csv('sampled_planets.csv')
df = df.loc[df['pl_hostname'] == f'{args.pl_hostname.replace(" ", "-")}']
df = df.loc[df['pl_letter'] == f'{args.pl_letter}']
pl_trandur = df['st_teff'].iloc[0]


logg = df['st_logg'].iloc[0]
Teff = df['st_teff'].iloc[0]
Z = df['st_metfe'].iloc[0]
per = df['pl_orbper'].iloc[0]



if np.isnan(logg) or np.isnan(Teff) or np.isnan(Z) or np.isnan(per) :
  print('Could not find logg or Teff or Z or period')
  sys.exit(0)

 
def match(g, t, z, path2table):
  table = pd.read_csv(path2table, delimiter=',')
  arr = np.array([g, t, z]).reshape([1,3])
  x = table['logg']
  y = table['Teff']
  z = table['Z']
  data = np.stack((x,y,z), axis = 1)
  data = np.vstack((data, arr))
  dist = euclidean_distances(data, data)
  euclid_dist = np.delete(dist[-1], 0)
  min_idx = np.argmin(euclid_dist)
  closest_u1 = table['aLSM'].iloc[min_idx]
  closest_u2 = table['bLSM'].iloc[min_idx]
  return closest_u1, closest_u2



if action == 'True':
  flux = np.load(path + '/data/transit/corrected_flux_refolded.npy', allow_pickle=True)
  time = np.load(path + '/data/transit/individual_time_folded_array_clean_refolded.npy', allow_pickle=True) 
  stds = np.load(path + '/data/transit/stds_refolded.npy', allow_pickle = True)
  theta = np.loadtxt(path + '/data/transit/theta_max.txt')
  rp_i, a_i, b_i, u1_i, u2_i  = theta[0], theta[1], theta[2], theta[3], theta[4]
else:
  flux = np.load(path + '/data/transit/corrected_flux_clean.npy', allow_pickle=True)
  time = np.load(path + '/data/transit/individual_time_folded_array_clean.npy', allow_pickle=True) 
  stds = np.load(path + '/data/transit/stds_clean.npy', allow_pickle = True)
  u1_i, u2_i  = match(logg, Teff, Z, path2table) 

  rp_i = df['pl_radj'].iloc[0] * 0.10049 # planet's radius in solar radii
  R_star = df['st_rad'].iloc[0] # star's radius in solar radii
  rp_i = rp_i/R_star # Rp/R* (planet's radius in terms of stellar radii)
  a_i = df['pl_orbsmax'].iloc[0] * 215.032 / R_star #(semi-major axis in terms of stellar radii)
  inclination =  df['pl_orbincl'].iloc[0] 
  b_i = a_i * np.cos(np.radians(inclination))

  if np.isnan(rp_i) or np.isnan(R_star) or np.isnan(a_i) or np.isnan(inclination) :
    print('Could not find the radius of the planet or stellar radius or semi-major axis or inclination')
    sys.exit(0)


def lnlike(theta, x, y, sigma, per=per):
  r, a, b, u1, u2 = theta
  # From Claret et al. 2012/13
  u1 = u1 # Limb Darkening coefficient 1
  u2 = u2 # Limb Darkening coefficient 2
  # Set up transit parameters.
  params = batman.TransitParams()
  params.t0 = 0
  params.per = per
  params.rp = r
  params.a = a
  params.inc = np.arccos(b/a)*(180./np.pi)
  params.ecc = 0
  params.w = 96
  params.u = [u1, u2]
  params.limb_dark = 'quadratic'
  # Initialize the transit model.
  m_init = batman.TransitModel(params, x)
  model = m_init.light_curve(params)  
  inv_sigma2 = 1.0 / (sigma**2)
  return -0.5*(np.sum((y-model)**2*inv_sigma2))
  
 


sigma = np.mean(stds, axis=0)


flux =  np.concatenate(flux, axis=0)
time =  np.concatenate(time, axis=0)
print('u1 i ', u1_i)
print('u2 i ', u2_i)
# MCMC parameters
nsteps = 5000 
burn_in = 2000
ndim = 5
nwalkers = 100

nll = lambda *args: -lnlike(*args)
initial = np.array([rp_i, a_i, b_i, u1_i, u2_i]) #+ 1e-5*np.random.randn(ndim)
soln = minimize(nll, initial, args=(time, flux, sigma))
rp_ml, a_ml, b_ml, u1_ml, u2_ml = soln.x

print("Maximum likelihood estimates:")
print("rp = {0:.3f}".format(rp_ml))
print("a = {0:.3f}".format(a_ml))
print("b = {0:.3f}".format(b_ml))
print("u1 = {0:.3f}".format(u1_ml))
print("u2 = {0:.3f}".format(u2_ml))
yerr = np.full((time.shape[0]), sigma) 


# Overplot the phase binned light curve
bins = np.linspace(np.min(time), np.max(time), 200)

arr1inds = time.argsort()
sorted_arr1 = time[arr1inds[::-1]]
sorted_arr2 = flux[arr1inds[::-1]]
#bins = np.linspace(-0.4, 0.4, 10)
#print('bins ', bins)
denom, _ = np.histogram(sorted_arr1, bins)
num, _ = np.histogram(sorted_arr1, bins, weights=sorted_arr2)
denom[num == 0] = 1.0
#plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, '.k', color="C1", lw=2)
#plt.xlabel("Time since transit")
#plt.ylabel("Flux");
#plt.show()
 
yerr = np.full((num.shape[0]), sigma)  
#fig, ax = plt.subplots()
#ax.plot(0.5 * (bins[1:] + bins[:-1]), num , color="C1")

 
#ax.set_ylabel("de-trended flux")
#ax.set_xlabel("time");
#plt.show()

#time = 0.5 * (bins[1:] + bins[:-1])
#flux =  num / denom

# Plot optimized transit model.
params_final = batman.TransitParams()
params_final.t0 = 0
params_final.per = per
params_final.rp = rp_ml
params_final.a = a_ml
params_final.inc =  np.arccos(b_ml/a_ml)*(180./np.pi)#np.arccos(theta_max[3] / theta_max[2]) * (180. / np.pi)
params_final.ecc = 0
params_final.w = 96
params_final.u = [u1_ml, u2_ml]
params_final.limb_dark = "quadratic"
tl = np.linspace(min(time),max(time),5000)
m = batman.TransitModel(params_final, tl)
f_final = m.light_curve(params_final)
final_fig, ax = plt.subplots(figsize=(10,8))
ax.set_title(planet_name)
#ax.errorbar(time,flux,yerr=yerr,fmt='.k',capsize=0,alpha=0.4,zorder=1)
ax.errorbar(0.5 * (bins[1:] + bins[:-1]), num / denom, yerr = yerr, fmt = '.k', color="C1", lw=1)
#ax.plot(time, flux, 'k.',alpha=0.8,lw=3,zorder=2)
ax.plot(tl, f_final, 'r-',alpha=0.8,lw=3,zorder=2)
ax.set_xlabel("Time")
ax.set_ylabel("Relative Flux")
ax.legend(('BATMAN','TESS'), loc=2)
plt.show()
     

save_to = path + '/figures'
final_fig.savefig(save_to + f'/MCMCfit_{action}.png', bbox_inches='tight')

m = batman.TransitModel(params_final, time)
f_final = m.light_curve(params_final)

final_fig, ax = plt.subplots(figsize=(10,8))
ax.set_title(planet_name)
ax.plot(time, flux-f_final, 'k.',alpha=0.8,lw=3,zorder=2)
ax.set_xlabel("Time")
ax.set_ylabel("Residuals")
plt.show()
     
 
save_to = path + '/figures'
final_fig.savefig(save_to + f'/residuals_{action}.png', bbox_inches='tight')

 

def lnprior(theta, u1_0, u2_0):
  rp, a, b, u1, u2 = theta
  if (0. < rp) \
  and (0. <= a) \
  and (0. <= b < a) \
  and (0. < u1) \
  and (0. < u1+2*u2) \
  and (u1+u2 < 1):
    return -100*((u1-u1_0)**2)*(u2-u2_0)**2
  return -np.inf

# Define log of probability function.
def lnprob(theta, x, y, sigma, u1_0, u2_0):
  lp = lnprior(theta, u1_0, u2_0)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(theta, x, y, sigma)


u1_0, u2_0  = match(logg, Teff, Z, path2table) 

initial_params = rp_i, a_i, b_i, u1_i, u2_i 
# Initialize walkers around maximum likelihood.
pos = [initial_params + 1e-1*np.random.randn(ndim) for i in range(nwalkers)]



# Set up sampler
t0 = timing.time()
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, flux, sigma, u1_0, u2_0))
# Run MCMC for n steps and display progress bar.
width = 50
for m, result in enumerate(sampler.sample(pos, iterations=nsteps)):
  n = int((width+1) * float(m) / nsteps)
  sys.stdout.write("\r{}[{}{}]{}".format('sampling... ', '#' * n, ' ' * (width - n), ' (%s%%)' % str(100. * float(m) / nsteps)))
sys.stdout.write("\n")

t1 = timing.time()

print('Execution time (min): {:.2f}'.format((t1-t0)/60))

samples = sampler.chain
 
# Discard burn-in. 
samples = samples[:, burn_in:, :].reshape((-1, ndim))

# Final params and uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
rp_i, a_i, b_i, u1_i, u2_i  = map(
	    lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

theta_percentiles = []
theta_percentiles.append(rp_i)
theta_percentiles.append(a_i)
theta_percentiles.append(b_i)
theta_percentiles.append(u1_i)
theta_percentiles.append(u2_i)   
 
theta_max = []
theta_max.append(rp_ml)
theta_max.append(a_ml)
theta_max.append(b_ml)
theta_max.append(u1_ml)   
theta_max.append(u2_ml) 
# save rp, a, b, u1, u2 
np.savetxt(path + '/data/transit/theta_max.txt', theta_max)
np.savetxt(path + '/data/transit/theta_percentiles.txt', theta_percentiles)

#samples = sampler.flatchain

param_names = ["$rp$", "$a$", "$b$", "u1", "u2"]
corn_fig = corner.corner(samples, labels=param_names)
corn_fig.savefig(save_to + f'/corner_folded_transit_{action}.png', bbox_inches='tight')


fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["$rp$", "$a$", "$b$", "u1", "u2"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
fig.savefig(save_to + f'/random_walkers_{action}.png', bbox_inches='tight') 
