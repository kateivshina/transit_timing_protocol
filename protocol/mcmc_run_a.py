import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import batman
import emcee
import os, sys, time
import pandas as pd
import corner
from argparse import ArgumentParser
from scipy.optimize import minimize
# parse data about the planet
parser = ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--mission')
parser.add_argument('--planet')
parser.add_argument('--cadence')
parser.add_argument('--radius') #, nargs='*')
parser.add_argument('--semi_major_axis')
parser.add_argument('--b')
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
  b_i = float(args.b) # impact parameter


def lnlike(theta, x, y, sigma, per=per):
  r, a, b, u1, u2 = theta
  # From Claret et al. 2012/13
 
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


# MCMC parameters
nsteps = 5000 
burn_in = 2000
ndim = 5
nwalkers = 100

nll = lambda *args: -lnlike(*args)
initial = np.array([rp_i, a_i, b_i, u1_i, u2_i]) + 1e-5*np.random.randn(ndim)
soln = minimize(nll, initial, args=(time, flux, sigma))
rp_ml, a_ml, b_ml, u1_ml, u2_ml = soln.x

print("Maximum likelihood estimates:")
print("rp = {0:.3f}".format(rp_ml))
print("a = {0:.3f}".format(a_ml))
print("b = {0:.3f}".format(b_ml))
print("u1 = {0:.3f}".format(u1_ml))

yerr = np.full((time.shape[0]), sigma) 


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
ax.errorbar(time,flux,yerr=yerr,fmt='.k',capsize=0,alpha=0.4,zorder=1)
#ax.plot(time, flux, 'k.',alpha=0.8,lw=3,zorder=2)
ax.plot(tl, f_final, 'r-',alpha=0.8,lw=3,zorder=2)
ax.set_xlabel("Time")
ax.set_ylabel("Relative Flux")
ax.legend(('BATMAN','TESS'), loc=2)
plt.show()
     
 
save_to = path + '/figures'
final_fig.savefig(save_to + '/MCMCfit.png', bbox_inches='tight')

 

# Priors
def lnprior(theta):
	rp, a, b, u1, u2 = theta
	if (0. < rp) \
  and (0. <= a) \
	and (0. <= b < 1+rp) \
  and (0. <= u1 < 1) \
  and (0. <= u2 < 1) \
  and (0. <= u1+u2 < 1):
		return 0
	return -np.inf




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
corn_fig.savefig(save_to + '/corner_folded_transit.png', bbox_inches='tight')


fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["$rp$", "$a$", "$b$", "u1", "u2"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
fig.savefig(save_to + '/random_walkers.png', bbox_inches='tight') 
