import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import batman
import emcee
import corner
import os, sys, time
import pandas as pd
from scipy.optimize import minimize
from argparse import ArgumentParser

# parse data about the planet
parser = ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--mission')
parser.add_argument('--pl_hostname')
parser.add_argument('--pl_letter') 
parser.add_argument('--cadence')
parser.add_argument('--N')
parser.add_argument('--degree')
parser.add_argument('--parent_dir')
parser.add_argument('--path_to_data_file')
parser.add_argument('--refolded')



args = parser.parse_args()

 
# path info
MISSION = args.mission
cadence = args.cadence
planet_name = args.planet
path_to_data_file =args.path_to_data_file
# Path 
parent_dir = args.parent_dir
directory = planet_name.replace(" ", "_") 
path = f'{parent_dir}' + f'/{directory}'  



# MCMC parameters
nsteps = 20000
burn_in = 5000
ndim = 2
nwalkers = 100

t0_w_uncert = np.loadtxt(path + '/data/transit/t0_w_uncert.txt')
err = t0_w_uncert[:,1]
t0_k_b = np.loadtxt(path + '/data/transit/t0_k_b.txt')
t0s = t0_k_b[:, 0]

# epoch number 
#N = np.array(range(0, t0_k_b.shape[0]))
# wasp 12
#N_1 = np.array(range(0, 11))
#N_2 = np.array(range(15, t0_k_b.shape[0]+4))
#wasp 4
N_1 = np.array(range(0, 9))
N_2 = np.array(range(11, t0_k_b.shape[0]+2))
N = np.concatenate((N_1, N_2), axis=0)

# need to input actual stds

sigma = np.mean(err)
per_i = float(args.period)
t0_i = t0s[0]


# Priors.
def lnprior(theta, t0_init): 
	per, t0 = theta
	if (0. < per < 1.1) and \
	(t0_init - 0.25 < t0 < t0_init + 0.25):
		return 0
	return -np.inf


def lnlike(theta, x, y, sigma):
  per, t0  = theta   
  model = t0 + x*per  
  inv_sigma2 = 1.0 / (sigma**2)
  return -0.5*(np.sum((y-model)**2*inv_sigma2))

# Define log of probability function.
def lnprob(theta, x, y, sigma, t0_init):
  lp = lnprior(theta, t0_init)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(theta, x, y, sigma)




initial_params = per_i, t0_i 
 
nll = lambda *args: -lnlike(*args) 
initial = np.array([per_i, t0_i]) + 1e-5*np.random.randn(ndim)
soln = minimize(nll, initial, args=(N, t0s, sigma))  
per_ml, t0_ml  = soln.x 
# Initialize walkers around maximum likelihood.
pos = [initial_params + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

# Set up sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(N, t0s, sigma, t0_i))

# Run MCMC for n steps and display progress bar.
width = 50
for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    n = int((width+1) * float(i) / nsteps)
    sys.stdout.write("\r{}[{}{}]{}".format('sampling... ', '#' * n, ' ' * (width - n), ' (%s%%)' % str(100. * float(i) / nsteps)))
sys.stdout.write("\n")
print ('Sampling complete!')


samples = sampler.flatchain

# Final params and uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
period, t0  = map(
	    lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
     
print('period, t0 ', period, t0)     
theta_max  = samples[np.argmax(sampler.flatlnprobability)]
print('theta max', theta_max)
period, t0 = theta_max[0], theta_max[1]

calculated = N*per_ml + t0_ml
o_c = t0s-calculated
plt.figure()
plt.errorbar(N, o_c*24*60, yerr = err*24*60, fmt='o', mew = 1)   # O-C_1 plot (mcmc fitted params)

#plt.plot(N, o_c*24*60, '.k')
plt.xlabel('Epoch')
plt.ylabel('Time deviation [min]')
plt.title(f'{planet_name} transits (constant period model)')
plt.savefig(path + '/figures/o_c.png')
plt.show()

