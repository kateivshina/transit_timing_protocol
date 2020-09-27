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

 

# Priors.
def lnprior(theta, t0_init): 
	per, t0 = theta
	if (1.2 < per < 4) and \
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


def o_c(pl_hostname,
        pl_letter,
        parent_dir):



  # Path 
  planet_name = pl_hostname + pl_letter
  directory = planet_name.replace("-", "_") 
  path = f'{parent_dir}' + f'/{directory}'  

  df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/hot_jupyter_sample.csv')
  df = df.loc[df['System'] == pl_hostname]#f'{pl_hostname.replace(" ", "-")}']
  #df = df.loc[df['pl_letter'] == f'{pl_letter}']
  period = df['Period'].iloc[0]
  print('period ', period)

  # MCMC parameters
  nsteps = 5000
  burn_in = 2000
  ndim = 2
  nwalkers = 100

  t0_w_uncert = np.loadtxt(path + '/data/transit/t0_w_uncert.txt')
  err = t0_w_uncert[:,1]
  t0_k_b = np.loadtxt(path + '/data/transit/t0_k_b.txt')
  t0s = t0_k_b[:, 0]
 
  t0_i = t0s[0]
 

  # epoch number 
  #N = np.array(range(0, t0_k_b.shape[0]))
  # wasp 12
  #N_1 = np.array(range(0, 11))
  #N_2 = np.array(range(15, t0_k_b.shape[0]+4))
  #wasp 4
  #N_1 = np.array(range(0, 9))
  #N_2 = np.array(range(11, t0_k_b.shape[0]+2))
  #N = np.concatenate((N_1, N_2), axis=0)
  deltas = []
  for i in range(t0s.shape[0]-1):
    #print('delta ',round((t0s[i+1] - t0s[i])/period))
    deltas.append(round((t0s[i+1] - t0s[i])/period))
  N = np.zeros(t0s.shape[0])
  deltas = np.array(deltas)
  for i in range(1, N.shape[0]):
    N[i] = np.sum(deltas[:i])
  print('N: ', N)


 # N = np.array([0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 13, 14, 17, 18])   #kelt-9
  #N = np.array([0, 1, 2, 3, 6, 7, 8, 9])   
  
      
 # print('N ', N.shape, t0s.shape)

  # need to input actual stds
  sigma = np.mean(err)

  initial_params = period, t0_i 
   
  nll = lambda *args: -lnlike(*args) 
  initial = np.array([period, t0_i]) + 1e-5*np.random.randn(ndim)
  soln = minimize(nll, initial, args=(N, t0s, sigma), method='Nelder-Mead')  
  per_ml, t0_ml  = soln.x 
  print('Optimization success: ', soln.success)
  print(per_ml, t0_ml)
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


  samples = sampler.chain
  # Discard burn-in. 
  samples = samples[:, burn_in:, :].reshape((-1, ndim))

  # Final params and uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
  period, t0  = map(
  	    lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

  samples = sampler.flatchain        
  theta_max  = samples[np.argmax(sampler.flatlnprobability)]
  period, t0 = theta_max[0], theta_max[1]

  calculated = N*per_ml + t0_ml
  o_c = t0s-calculated
  plt.figure()
  plt.errorbar(N, o_c*24*60, yerr = t0_w_uncert[:,1]*24*60, fmt='o', mew = 1)   

  #plt.plot(N, o_c*24*60, '.k')
  plt.xlabel('Epoch')
  plt.ylabel('Time deviation [min]')
  plt.title(f'{planet_name} transits (constant period model)')
  legend = f't0 = %.4f Period = %.4f d' % (t0_ml, per_ml)
  #print('legend ', legend)
  #print('legend ', f't = {t0_ml} Period = {per_ml} d')
  plt.text(0.1,0.3, legend, fontsize=8)

  #plt.legend(legend)
  print('saving o-c to ', path + '/figures/tess_o_c.png')
  plt.savefig(path + '/figures/tess_o_c.png')

