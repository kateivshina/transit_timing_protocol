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

 

def o_c_combo(planet_name,
               pl_hostname,
               pl_letter,
               parent_dir):



  directory = planet_name.replace(" ", "_") 
  path = f'{parent_dir}' + f'/{directory}'  
  path2data = path + '/data/transit/o.csv'

  # load CSV file with the exoplanet data
  df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/sampled_planets.csv')
  df = df.loc[df['pl_hostname'] == f'{pl_hostname.replace(" ", "-")}']
  df = df.loc[df['pl_letter'] == f'{pl_letter}']
  per_i = df['pl_orbper'].iloc[0]



  # MCMC parameters
  nsteps = 1000 
  burn_in = 500 
  ndim = 2
  nwalkers = 100

  df = pd.read_csv(path2data)
  err = df['err']
  t0s = df['t0']
  epoch = df['Epoch']

  # need to input actual stds
  sigma = np.mean(err)
  t0_i = 2456021.70374

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
   
  nll = lambda *args: -lnlike(*args) 
  initial = np.array([per_i, t0_i]) + 1e-5*np.random.randn(ndim)
  soln = minimize(nll, initial, args=(epoch, t0s, sigma))  
  per_ml, t0_ml  = soln.x 
  # Initialize walkers around maximum likelihood.
  pos = [initial_params + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

  # Set up sampler.
  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(epoch, t0s, sigma, t0_i))

  # Run MCMC for n steps and display progress bar.
  width = 50
  for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
      n = int((width+1) * float(i) / nsteps)
      sys.stdout.write("\r{}[{}{}]{}".format('sampling... ', '#' * n, ' ' * (width - n), ' (%s%%)' % str(100. * float(i) / nsteps)))
  sys.stdout.write("\n")
  print ('Sampling complete!')


  samples = sampler.chain
  # Discard burn-in. 
  samples = samples[:, burn_in:, :].reshape((-1, ndim))
  # Final params and uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
  period, t0  = map(
  	    lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
  
  samples = sampler.flatchain     
  theta_max  = samples[np.argmax(sampler.flatlnprobability)]
  period_mcmc, t0_mcmc = theta_max[0], theta_max[1]

  calculated = epoch*per_ml + t0_ml
  o_c = t0s-calculated
  
  plt.figure()
  plt.errorbar(epoch, o_c*24*60, yerr = err*24*60, fmt='o', mew = 1)  

  #plt.plot(N, o_c*24*60, '.k')
  plt.xlabel('Epoch')
  plt.ylabel('Time deviation [min]')
  plt.title(f'{planet_name} transits (constant period model)')
  plt.legend(f'$t0$: {t0_ml} Period: {per_ml}')
  plt.savefig(path + '/figures/o_c_combined.png')
  plt.show()

