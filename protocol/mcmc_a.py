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
import matplotlib.backends.backend_pdf
import sys
from scipy.optimize import Bounds
from lmfit import Model
from lmfit import Minimizer, Parameters, report_fit

def find_nearest_z(array, value):
  epsilon = 0.05
  indx = []
  for i in range(array.shape[0]):
    if np.abs(array[i] - value) < epsilon:
      indx.append(i)

  return indx

def find_nearest_g(array, value):
  epsilon = 0.35
  indx = []
  for i in range(array.shape[0]):
    if np.abs(array[i] - value) < epsilon:
      indx.append(i)

  return indx

def find_nearest_T(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

 
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 
  

def match(g, t, z_star, path2table):
  print(f'g {g}, t {t}, z_star {z_star}')

  table = pd.read_csv(path2table, delimiter=',')
  arr = np.array([g, t, z_star]).reshape([1,3])
  
  x = table['Z']
  y = table['logg']
  z = table['Teff']

  data = np.stack((x,y,z), axis = 1)
  #data = np.vstack((data, arr))

  indx_z = find_nearest_z(x, z_star)
  indx_g = find_nearest_g(y, g)
  indx = intersection(indx_z, indx_g)
 
  res_list_T = [z[i] for i in indx]
  res_list_z = [x[i] for i in indx]
  res_list_g = [y[i] for i in indx]

  table_a = [table['aLSM'].iloc[i] for i in indx]
  table_b = [table['bLSM'].iloc[i] for i in indx]

  min_idx = find_nearest_T(np.array(res_list_T), t)
 
  #print('idx ', min_idx)
  print('z, g, t ', res_list_z[min_idx], res_list_g[min_idx], res_list_T[min_idx])

  closest_u1 = table_a[min_idx]
  closest_u2 = table_b[min_idx]
  return closest_u1, closest_u2





def lnlike(theta, x, y, sigma, per, u1, u2):
  r, a, b = theta
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
  
 
 


def lnprior(theta):
  rp, a, b = theta
  #if (0. < rp < 0.2) \
  #and (0 <= a < 10) \
  #and (0. <= b < 0.5):
  if (0. < rp) \
  and (0 <= a) \
  and (0. <= b < 1):
    return 0
  return -np.inf



# Define log of probability function.
def lnprob(theta, x, y, sigma, u1_0, u2_0, per):
  lp = lnprior(theta)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(theta, x, y, sigma, per, u1_0, u1_0)





def run_mcmc_a(pl_hostname,
               pl_letter,
               parent_dir,
               action):

 
  # path info
  planet_name = pl_hostname + pl_letter
  directory = planet_name.replace("-", "_") 
  path = f'{parent_dir}' + f'/{directory}'  
  path2table = parent_dir + '/data/LD.csv'

  # load CSV file with the exoplanet data
  df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/hot_jupyter_sample.csv')
 
  df = df.loc[df['System'] == pl_hostname]#f'{pl_hostname.replace(" ", "-")}']
  #df = df.loc[df['pl_letter'] == f'{pl_letter}']


  logg = df['st_logg'].iloc[0]
  Teff = df['Teff'].iloc[0]
  print('Effective T: ', Teff)
  Z = df['Z'].iloc[0]
  per = df['Period'].iloc[0]
 
  flux = np.load(path + '/data/transit/corrected_flux.npy', allow_pickle=True)
 

  
  
  

  

  if action == True:
    out_pdf = path + '/figures/mcmc_a_refolded.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
    flux = np.load(path + '/data/transit/corrected_flux_refolded.npy', allow_pickle=True)
    time = np.load(path + '/data/transit/individual_time_folded_array_refolded.npy', allow_pickle=True) 
    stds = np.load(path + '/data/transit/stds_refolded.npy', allow_pickle = True)
    theta = np.loadtxt(path + '/data/transit/theta_max.txt')
    rp_i, a_i, b_i, u1_st, u2_st = theta[0], theta[1], theta[2], theta[3], theta[4]
   
    

  else:
    
    out_pdf =  path + '/figures/mcmc_a.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
    flux = np.load(path + '/data/transit/corrected_flux.npy', allow_pickle=True)
    time = np.load(path + '/data/transit/individual_time_folded_array.npy', allow_pickle=True) 
    stds = np.load(path + '/data/transit/stds.npy', allow_pickle = True)
 
    if np.isnan(logg) or np.isnan(Teff) or np.isnan(Z):
      print('Could not find logg or Teff or Z')
      u1_st, u2_st  = 0.65, 0.1 
    else:
      u1_st, u2_st = match(logg, Teff, Z, path2table)  
     

    print("Estimates from stellar models:") 
    print('u1 ', u1_st)
    print('u2 ', u2_st)

    #rp_i = df['pl_radj'].iloc[0] * 0.10049 # planet's radius in solar radii
    R_star = df['st_rad'].iloc[0] # star's radius in solar radii
    rp_i = df['depth'].iloc[0]**0.5 #rp_i/R_star # Rp/R* (planet's radius in terms of stellar radii)
    a_i = df['pl_orbsmax'].iloc[0] * 215.032 / R_star #(semi-major axis in terms of stellar radii)
    inclination =  df['pl_orbincl'].iloc[0] 
    b_i = a_i * np.cos(np.radians(inclination))
 

    if np.isnan(rp_i) or np.isnan(R_star) or np.isnan(a_i) or np.isnan(inclination) :
      raise Exception('Could not find the radius of the planet or stellar radius or semi-major axis or inclination')
      sys.exit(0)
    
  
 
  # MCMC parameters
  nsteps = 4000 
  burn_in = 3000
  ndim = 3
  nwalkers = 100

  initial_params = rp_i, a_i, b_i 
  # Initialize walkers around maximum likelihood.
  pos = [initial_params + 1e-10*np.random.randn(ndim) for i in range(nwalkers)]
  sigma = np.mean(stds, axis=0)
  flux =  np.concatenate(flux, axis=0)
  
  time =  np.concatenate(time, axis=0)
  plt.plot(time, flux, 'b.')

  def transit_model(x, r, a, b, u1, u2):
     
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
    return model
    

  # define objective function: returns the array to be minimized
  def fcn2min(params, x, data, u1, u2):
    r = params['r']
    a = params['a']
    b = params['b']
   
    model = transit_model(x, r, a, b, u1, u2)
    return model - data
   

  #nll = lambda *args: -lnlike_fit(*args)
  #initial = np.array([rp_i, a_i, b_i])  
  #soln = minimize(nll, initial, args=(time, flux, sigma, per, u1_i, u2_i), method='Nelder-Mead')
  #print('Optimization success: ', soln.success)
  #rp_ml, a_ml, b_ml = soln.x

#pars = Parameters()
#pars.add('x', value=5, vary=True)
#pars.add('delta', value=5, max=10, vary=True)
#pars.add('y', expr='delta-x')

  #gmodel = Model(transit_model)
  pars = Parameters()

  pars.add('r', value = rp_i, min = 0)
  pars.add('a', value = a_i, min = 0)
  pars.add('delta_ab', value = b_i/a_i, min = 0, max = 1)
  pars.add('b', expr='delta_ab * a')
  pars.add('u1', value = u1_st, vary = False)
  pars.add('u2', value = u2_st, vary = False)


  # do fit, here with the default leastsq algorithm
  minner = Minimizer(fcn2min, pars, fcn_args=(time, flux, u1_st, u2_st))
  result = minner.minimize()

  # calculate final result
  final = flux + result.residual

  plt.plot(time, flux, 'b.')
  plt.plot(time, final, 'r.', label='best fit')
   

  print(report_fit(result))
 
  

  #plt.plot(time, flux, '.b')
  #plt.plot(time, result.init_fit, 'k--', label='initial fit')
  #plt.plot(time, result.best_fit, '.k')
  #plt.show()
  rp_ml = result.params['r'].value
  a_ml = result.params['a'].value
  b_ml = result.params['b'].value

 
  theta_max = []
  theta_max.append(rp_ml)
  theta_max.append(a_ml)
  theta_max.append(b_ml)
  theta_max.append(u1_st)
  theta_max.append(u2_st)
 
  
  print("Maximum likelihood estimates:")
  print(f"rp_ml = {rp_ml} vs rp_i = {rp_i} ")
  print(f"a_ml = {a_ml} vs {a_i}")
  print(f"b_ml = {b_ml} vs {b_i}")
 
  yerr = np.full((time.shape[0]), sigma) 
  np.savetxt(path + '/data/transit/theta_max.txt', theta_max)
 
  # choose k - the number of bins - such that the bin's width is about 1 minute
  k = 0
  for i in range(100, 2000, 20):
    current_bin_size = (np.max(time)-np.min(time))/i 
    if 1.7/1440 > current_bin_size > 1.2/1440:
      k = i
      break

  if 1.7/1440 < (np.max(time)-np.min(time))/k or (np.max(time)-np.min(time))/k < 1.2/1440:
    raise Exception('Could not select the bin width to be ~1 minute; increase the number of steps')
    sys.exit(0)

  print(f'# of bins: {k}')
  # Overplot the phase binned light curve
  bins = np.linspace(np.min(time), np.max(time), 200)

  arr1inds = time.argsort()
  sorted_time = time[arr1inds[::-1]] 
  sorted_flux = flux[arr1inds[::-1]] 
 
  #denom, _ = np.histogram(sorted_time, bins)
  #num, _ = np.histogram(sorted_time, bins, weights=sorted_flux)
  #denom[num == 0] = 1.0
  binned_flux = [] #np.zeros(sorted_time.shape[0])
  #binned_time = #np.zeros(sorted_time.shape[0])
  for i in range(bins.shape[0]-1):
    mask = (sorted_time < bins[i+1]) & (sorted_time > bins[i])
    #mask = np.where(condition, True, False)
    #condition = sorted_time > bins[i]
    #mask = np.where(condition, True, False)
    current_flux = np.mean(sorted_flux[mask])
    binned_flux.append(current_flux)
 


  # Plot optimized transit model.
  fig, ax = plt.subplots(2, 1, figsize = (12, 10)) 
  params_final = batman.TransitParams()
  params_final.t0 = 0
  params_final.per = per
  params_final.rp = rp_ml
  params_final.a = a_ml
  params_final.inc =  np.arccos(b_ml/a_ml)*(180./np.pi)
  params_final.ecc = 0
  params_final.w = 96
  params_final.u = [u1_st, u2_st]
  params_final.limb_dark = "quadratic"
  tl = np.linspace(min(time),max(time),5000)
  m = batman.TransitModel(params_final, tl)
  f_final = m.light_curve(params_final)
  ax[0].set_title(planet_name)
  #ax.errorbar(time,flux,yerr=yerr,fmt='.k',capsize=0,alpha=0.4,zorder=1)
  ax[0].plot(0.5 * (bins[1:] + bins[:-1]), binned_flux, '.k')
  ax[0].plot(tl, f_final, 'r-',alpha=0.8,lw=3,zorder=2)
  ax[0].set_xlabel("Time")
  ax[0].set_ylabel("Relative Flux")
  ax[0].legend(('TESS','BATMAN'), loc=0)


   

  m = batman.TransitModel(params_final, 0.5*(bins[1:] + bins[:-1]))
  f_final = m.light_curve(params_final)

  ax[1].set_title(planet_name)
  ax[1].plot(0.5*(bins[1:] + bins[:-1]),  binned_flux - f_final, 'k.', alpha=0.8, lw=3, zorder=2)
  ax[1].set_xlabel("Time")
  ax[1].set_ylabel("Residuals")
  pdf.savefig(fig)
 



  
  # Set up sampler
  t0 = timing.time()
  # fit all data
  #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, flux, sigma, u1_i, u2_i, per))
  # fit binned lc
  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(0.5*(bins[1:] + bins[:-1]), binned_flux, sigma, u1_st, u2_st, per))

  
  # Run MCMC for n steps and display progress bar.
  width = 50
  for m, result in enumerate(sampler.sample(pos, iterations=nsteps)):
    n = int((width+1) * float(m) / nsteps)
    sys.stdout.write("\r{}[{}{}]{}".format('sampling... ', '#' * n, ' ' * (width - n), ' (%s%%)' % str(100. * float(m) / nsteps)))
  sys.stdout.write("\n")

  
  
  samples = sampler.chain
  #print('!!!!!!!! Autocorrelation time: ', sampler.get_autocorr_time())
  samples = samples[:, burn_in:, :].reshape((-1, ndim))

   
  # Discard burn-in. 
 # samples = samples[:, burn_in:, :].reshape((-1, ndim))

  # Final params and uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
  rp_i, a_i, b_i  = map(
        lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))

  theta_percentiles = []
  theta_percentiles.append(rp_i)
  theta_percentiles.append(a_i)
  theta_percentiles.append(b_i)
 
  np.savetxt(path + '/data/transit/theta_percentiles.txt', theta_percentiles) 

  #samples = sampler.flatchain
  
  param_names = ["$rp$", "$a$", "$b$"]
  corn_fig = corner.corner(samples, labels=param_names)
  pdf.savefig(corn_fig)


  #fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
  #samples = sampler.get_chain()
  #labels = ["$rp$", "$a$", "$b$", "u1", "u2"]
  #for i in range(ndim):
  #    ax = axes[i]
  #    ax.plot(samples[:, :, i], "k", alpha=0.3)
  #    ax.set_xlim(0, len(samples))
  #    ax.set_ylabel(labels[i])
  #    ax.yaxis.set_label_coords(-0.1, 0.5)

  #axes[-1].set_xlabel("step number");
  #pdf.savefig(fig)
  np.set_printoptions(precision=6)

  ml_values = np.around(np.array([rp_ml, a_ml, b_ml, u1_st, u2_st]), decimals=6)
  #mcmc_values = np.around(np.array([rp_mcmc, a_mcmc, b_mcmc, u1_mcmc, u2_mcmc]), decimals=6)
  uncrt = np.around(np.array([rp_i[1], a_i[1], b_i[1]]), decimals=6)


  data = np.array([['Value (max likelihood)', ml_values[0], ml_values[1], ml_values[2], ml_values[3], ml_values[4]], 
    ['Uncertainty', uncrt[0], uncrt[1], uncrt[2], 0, 0]])

  #data = np.array([['Value (max likelihood)', ml_values[0], ml_values[1], ml_values[2], ml_values[3], ml_values[4]], 
  #  ['Value (MCMC)', mcmc_values[0], mcmc_values[1], mcmc_values[2], mcmc_values[3], mcmc_values[4]], 
  #  ['Uncertainty', uncrt[0], uncrt[1], uncrt[2], uncrt[3], uncrt[4]]])
  df = pd.DataFrame(data, columns = (" ", "r", "a", "b", 'u1', 'u2'))


  fig, ax =plt.subplots(figsize=(12,4))
  ax.axis('tight')
  ax.axis('off')
  the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')


  pdf.savefig(fig, bbox_inches='tight')

  firstPage = plt.figure(figsize=(11.69,8.27))
  firstPage.clf()

  txt = f'g: {logg}, T: {Teff}, z: {Z}'

  firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=16, ha="center")
  pdf.savefig()

  pdf.close()

  t1 = timing.time()
  print('Execution time (mcmc_a) {:.2f} min'.format((t1-t0)/60))
 
