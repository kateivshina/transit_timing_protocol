import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from scipy.stats import binned_statistic
import batman
import emcee
import corner
import os, sys, time
import pandas as pd
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
nsteps = 1000
burn_in = 100
ndim = 2
nwalkers = 100


t0_k_b = np.loadtxt(path + '/data/transit/t0_k_b.txt')
t0s = t0_k_b[:, 0]

# epoch number 
N = np.array(range(0, t0_k_b.shape[0]))
# need to input actual stds

sigma = 0.00065836
per_i = float(args.period)
t0_i = t0s[0]


# Priors.
def lnprior(theta, t0_init): 
	per, t0 = theta
	if (0. < per < 2) and \
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

theta_max  = samples[np.argmax(sampler.flatlnprobability)]

period, t0 = theta_max[0], theta_max[1]

calculated = N*period + t0
o_c = t0s-calculated
plt.plot(N, o_c, '.k')
plt.show()

