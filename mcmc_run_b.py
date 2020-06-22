import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import batman
import emcee
import os, sys, time
import pandas as pd


theta = np.loadtxt('/Users/ivshina/Desktop/theta_max.txt')

rp, a, b, u1, u2  = theta[0], theta[1], theta[2], theta[3], theta[4]

# Exoplanetary system parameters
targ_name = 'WASP-12'


# Data
t0s = []
 
stds = np.load('/Users/ivshina/Desktop/usrp/orbital_decay/WASP_12b/data/transit/stds_clean.npy', allow_pickle = True)
k = np.load('/Users/ivshina/Desktop/usrp/orbital_decay/WASP_12b/data/transit/ks.npy', allow_pickle = True)
c = np.load('/Users/ivshina/Desktop/usrp/orbital_decay/WASP_12b/data/transit/bs.npy', allow_pickle = True)


flux = np.load('/Users/ivshina/Desktop/usrp/orbital_decay/WASP_12b/data/transit/individual_flux_array_clean.npy', allow_pickle = True)
time = np.load('/Users/ivshina/Desktop/usrp/orbital_decay/WASP_12b/data/transit/individual_time_array_clean.npy', allow_pickle = True)


# wasp-12
per = 1.09142
sigma = 0.00065836

 

# MCMC parameters
nsteps = 20000
burn_in = 1000
ndim = 3
nwalkers = 100


save_results = True	# Save plots?
show_plots = True	# Show plots?
annotate_plot = True



# Priors
def lnprior(theta):
	t0, k, c  = theta
	if True:
		return 0
	return -np.inf


def lnlike(theta, x, y, sigma, rp, a, b, u1, u2, per=per):
  t0, k, c = theta
  # From Claret et al. 2012/13
  u1 = u1	# Limb Darkening coefficient 1
  u2 = u2 # Limb Darkening coefficient 2
  # Set up transit parameters.
  params = batman.TransitParams()
  params.t0 = t0
  params.per = per
  params.rp = rp
  params.a = a
  params.inc = 82
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
def lnprob(theta, x, y, sigma, rp, a, b, u1, u2):
  lp = lnprior(theta)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(theta, x, y, sigma, rp, a, b, u1, u2)




params_final = []


for i in range(flux.shape[0]):
    time_i = time[i]
    flux_i =flux[i]
    
    idx = np.argmin(flux_i)
    t0_i = times[idx][0]
    
    print('initial t0: ', t0_i)
    sigma = stds[i]
    k_i = k[i]
    c_i = c[i]
  

    initial_params = t0_i, k_i, c_i 

    # Initialize walkers around maximum likelihood.
    pos = [initial_params + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

    # Set up sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(np.array(time_i), np.array(flux_i), sigma, rp, a, b, u1, u2))

    # Run MCMC for n steps and display progress bar.
    width = 50
    for m, result in enumerate(sampler.sample(pos, iterations=nsteps)):
      n = int((width+1) * float(m) / nsteps)
      sys.stdout.write("\r{}[{}{}]{}".format('sampling... ', '#' * n, ' ' * (width - n), ' (%s%%)' % str(100. * float(m) / nsteps)))
    sys.stdout.write("\n")
    print ('Sampling complete!')

    samples = sampler.chain
 
    # Discard burn-in. 
    samples = samples[:, burn_in:, :].reshape((-1, ndim))

    # Final params and uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
     t0, k, c  = map(
      lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
     
 
    samples = sampler.flatchain
    theta_max  = samples[np.argmax(sampler.flatlnprobability)]
    params_final.append(theta_max)

np.savetxt('/Users/ivshina/Desktop/t0_k_c.txt', np.array(params_final))

'''
    # create array for flux uncertanties
    ferr = []
    for k in range(times.shape[0]):
      ferr.append(stds[i])
    ferr = np.asarray(ferr)

    params_final = batman.TransitParams()
    params_final.t0 = theta_max[0]
    params_final.per = per_i
    params_final.rp = theta_max[1]
    params_final.a = theta_max[2]
    params_final.inc = np.arccos(theta_max[3] / theta_max[2]) * (180. / np.pi)
    params_final.ecc = 0.0092
    params_final.w = 96
    params_final.u = [u1, u2]
    params_final.limb_dark = "quadratic"
    tl = np.linspace(min(times),max(times),5000)
    m = batman.TransitModel(params_final, tl)
    f_final = m.light_curve(params_final)
    final_fig, ax = plt.subplots(figsize=(10,8))
    ax.set_title(targ_name)
    ax.errorbar(times,fluxes,yerr=ferr,fmt='k.',capsize=0,alpha=0.4,zorder=1)
    ax.plot(tl, f_final, 'r-',alpha=0.8,lw=3,zorder=2)
    if annotate_plot == True:
      ant = AnchoredText('$T_0 = %s^{+%s}_{-%s}$ \n $R_p/R = %s^{+%s}_{-%s}$' % (round(t0_mcmc[0],4),round(t0_mcmc[1],4),
      round(t0_mcmc[2],4),round(rp_mcmc[0],4),round(rp_mcmc[1],4),round(rp_mcmc[2],4)), prop=dict(size=11), frameon=True, loc=3)
      ant.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
      ax.add_artist(ant)
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative Flux")
    ax.legend(('BATMAN','TESS'), loc=2)
     
    if save_results == True:
        save_to = 'kelt16_figures'
        final_fig.savefig('kelt16_figures/MCMCfit%d.png' %i, bbox_inches='tight')



''' 

'''
# Plot the final transit model.
# create array for flux uncertanties
ferr = []
    for k in range(times.shape[0]):
	    ferr.append(stds[i])
    ferr = np.asarray(ferr)

    params_final = batman.TransitParams()
    params_final.t0 = theta_max[0]
    params_final.per = per_i
    params_final.rp = theta_max[1]
    params_final.a = theta_max[2]
    params_final.inc = np.arccos(theta_max[3] / theta_max[2]) * (180. / np.pi)
    params_final.ecc = 0.0092
    params_final.w = 96
    params_final.u = [u1, u2]
    params_final.limb_dark = "quadratic"
    tl = np.linspace(min(times),max(times),5000)
    m = batman.TransitModel(params_final, tl)
    f_final = m.light_curve(params_final)
    final_fig, ax = plt.subplots(figsize=(10,8))
    ax.set_title(targ_name)
    ax.errorbar(times,fluxes,yerr=ferr,fmt='k.',capsize=0,alpha=0.4,zorder=1)
    ax.plot(tl, f_final, 'r-',alpha=0.8,lw=3,zorder=2)
    if annotate_plot == True:
    	ant = AnchoredText('$T_0 = %s^{+%s}_{-%s}$ \n $R_p/R = %s^{+%s}_{-%s}$' % (round(t0_mcmc[0],4),round(t0_mcmc[1],4),
    	round(t0_mcmc[2],4),round(rp_mcmc[0],4),round(rp_mcmc[1],4),round(rp_mcmc[2],4)), prop=dict(size=11), frameon=True, loc=3)
    	ant.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    	ax.add_artist(ant)
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative Flux")
    ax.legend(('BATMAN','TESS'), loc=2)
     
    if save_results == True:
        save_to = 'kelt16_figures'
        final_fig.savefig('kelt16_figures/MCMCfit%d.png' %i, bbox_inches='tight')

'''
