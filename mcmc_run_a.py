import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import batman
import emcee
import os, sys, time
import pandas as pd
import corner
from argparse import ArgumentParser

# parse data about the planet
parser = ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--mission')
parser.add_argument('--planet')
parser.add_argument('--cadence')
parser.add_argument('--radius') #, nargs='*')
parser.add_argument('--semi_major_axis')
parser.add_argument('--inclination')
parser.add_argument('--period')
parser.add_argument('--parent_dir')
parser.add_argument('--path_to_data_file')
parser.add_argument('--refolded')

args = parser.parse_args()

action = args.refolded



 
# path info
MISSION = args.mission
cadence = args.cadence
planet_name = args.planet
path_to_data_file =args.path_to_data_file
# Path 
parent_dir = args.parent_dir
directory = planet_name.replace(" ", "_") 
path = f'{parent_dir}' + f'/{directory}'  


# change this to the derived values
# planet info
per = float(args.period)

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
  u1_i = 0.5  # Limb Darkening coefficient 1
  u2_i = 0.1144 # Limb Darkening coefficient 2
  rp_i = float(args.radius) # Rp/R* (planet's radius in terms of stellar radii)
  a_i = float(args.semi_major_axis) #(semi-major axis in terms of stellar radii)
  b_i = float(args.inclination) # impact parameter




sigma = np.mean(stds, axis=0)


flux =  np.concatenate(flux, axis=0)
time =  np.concatenate(time, axis=0)


# MCMC parameters
nsteps = 2000 
burn_in = 500
ndim = 5
nwalkers = 100

# Plot the final transit model.
params_final = batman.TransitParams()
params_final.t0 = 0
params_final.per = per
params_final.rp = rp_i
params_final.a = a_i
params_final.inc =b_i 
params_final.ecc = 0
params_final.w = 96
params_final.u = [u1_i, u2_i]
params_final.limb_dark = "quadratic"
tl = np.linspace(min(time),max(time),5000)
m = batman.TransitModel(params_final, tl)
f_final = m.light_curve(params_final)
final_fig, ax = plt.subplots(figsize=(10,8))
ax.set_title(planet_name)
#ax.errorbar(times,fluxes,yerr=ferr,fmt='k.',capsize=0,alpha=0.4,zorder=1)
ax.plot(time, flux, 'k.',alpha=0.8,lw=3,zorder=2)
ax.plot(tl, f_final, 'r-',alpha=0.8,lw=3,zorder=2)
ax.set_xlabel("Time")
ax.set_ylabel("Relative Flux")
ax.legend(('BATMAN','TESS'), loc=2)
plt.show()

# Priors
def lnprior(theta):
	rp, a, b, u1, u2 = theta
	if (0. < rp) \
  and (0. <= a) \
	and (0. <= b < 180):
		return 0
	return -np.inf


def lnlike(theta, x, y, sigma, per=per):
  r, a, b, u1, u2 = theta
  # From Claret et al. 2012/13
  u1 = u1	# Limb Darkening coefficient 1
  u2 = u2 # Limb Darkening coefficient 2
  # Set up transit parameters.
  params = batman.TransitParams()
  params.t0 = 0
  params.per = per
  params.rp = r
  params.a = a
  params.inc = b
  params.ecc = 0
  params.w = 96
  params.u = [u1, u2]
  params.limb_dark = 'quadratic'
  # Initialize the transit model.
  m_init = batman.TransitModel(params, x)
  model = m_init.light_curve(params)  
  inv_sigma2 = 1.0 / (sigma**2)
  return -0.5*(np.sum((y-model)**2*inv_sigma2))
	
 

# Define log of probability function.
def lnprob(theta, x, y, sigma):
  lp = lnprior(theta)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(theta, x, y, sigma)



initial_params = rp_i, a_i, b_i, u1_i, u2_i 

# Initialize walkers around maximum likelihood.
pos = [initial_params + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

# Set up sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, flux, sigma))
# Run MCMC for n steps and display progress bar.
width = 50
for m, result in enumerate(sampler.sample(pos, iterations=nsteps)):
  n = int((width+1) * float(m) / nsteps)
  sys.stdout.write("\r{}[{}{}]{}".format('sampling... ', '#' * n, ' ' * (width - n), ' (%s%%)' % str(100. * float(m) / nsteps)))
sys.stdout.write("\n")


samples = sampler.chain
 
# Discard burn-in. 
samples = samples[:, burn_in:, :].reshape((-1, ndim))

# Final params and uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
rp_i, a_i, b_i, u1_i, u2_i  = map(
	    lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
     
#t0s.append([round(t0_mcmc[0],4),round(t0_mcmc[1],4), round(t0_mcmc[2],4)])
#np.savetxt('t0s.txt', np.array(t0s))

samples = sampler.flatchain
theta_max  = samples[np.argmax(sampler.flatlnprobability)]
    
# save rp, a, b, u1, u2 
np.savetxt(path + '/data/transit/theta_max.txt', theta_max)



# Plot the final transit model.
params_final = batman.TransitParams()
params_final.t0 = 0
params_final.per = per
params_final.rp = theta_max[0]
params_final.a = theta_max[1]
params_final.inc = theta_max[2] #np.arccos(theta_max[3] / theta_max[2]) * (180. / np.pi)
params_final.ecc = 0
params_final.w = 96
params_final.u = [theta_max[3], theta_max[4]]
params_final.limb_dark = "quadratic"
tl = np.linspace(min(time),max(time),5000)
m = batman.TransitModel(params_final, tl)
f_final = m.light_curve(params_final)
final_fig, ax = plt.subplots(figsize=(10,8))
ax.set_title(planet_name)
#ax.errorbar(times,fluxes,yerr=ferr,fmt='k.',capsize=0,alpha=0.4,zorder=1)
ax.plot(time, flux, 'k.',alpha=0.8,lw=3,zorder=2)
ax.plot(tl, f_final, 'r-',alpha=0.8,lw=3,zorder=2)
ax.set_xlabel("Time")
ax.set_ylabel("Relative Flux")
ax.legend(('BATMAN','TESS'), loc=2)
plt.show()
     
 
save_to = path + '/data/transit'
final_fig.savefig(save_to + '/MCMCfit.png', bbox_inches='tight')
print('finished')

param_names = ["$rp$", "$a$", "$i$", "u1", "u2"]
corn_fig = corner.corner(samples, labels=param_names)
corn_fig.savefig(save_to + '/corner.png', bbox_inches='tight')
 