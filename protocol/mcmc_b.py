import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import batman
import emcee
import os, sys, time
import pandas as pd
from argparse import ArgumentParser
import scipy.optimize  
import corner
import time as timing
 
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages
 
 


# Priors
def lnprior(theta, t0_init):
  t0, a, b, c = theta
  if np.isfinite(t0) and np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and (t0_init - 0.25 < t0 < t0_init + 0.25):
    return 0
  return -np.inf


 
def lnlike(theta, x, y, sigma, r, a, b, u1, u2, per, t0_init):
  t0, a1, b1, c1 = theta

    # Limb Darkening coefficient 1
  # Limb Darkening coefficient 2
  # Set up transit parameters.
  params = batman.TransitParams()
  params.t0 = t0
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
  if np.isfinite(t0) and np.isfinite(a1) and np.isfinite(b1) and np.isfinite(c1) and (t0_init - 0.25 < t0 < t0_init + 0.25):
    return -0.5*(np.sum((y/(a1*x*x+b1*x+c1)-model)**2*inv_sigma2))
  else:
    return -np.inf
  
 
 

# Define log 

# Define log of probability function.
def lnprob(theta, x, y, sigma, r, a, b, u1, u2, per, t0_init):
  lp = lnprior(theta, t0_init)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(theta, x, y, sigma, r, a, b, u1, u2, per, t0_init)





def run_mcmc_b(pl_hostname, 
               pl_letter, 
               parent_dir, 
               action):

  # path info
  planet_name = pl_hostname + pl_letter
  directory = planet_name.replace("-", "_") 
  path = f'{parent_dir}' + f'/{directory}'  
  # Planet info
  # load CSV file with the exoplanet data
  df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/hot_jupyter_sample.csv')
  df = df.loc[df['System'] == pl_hostname]#f'{pl_hostname.replace(" ", "-")}']
  #df = df.loc[df['pl_letter'] == f'{pl_letter}']
  per = df['Period'].iloc[0]


  theta = np.loadtxt(path + '/data/transit/theta_max.txt')
  r, a, b, u1, u2  = theta[0], theta[1], theta[2], theta[3], theta[4]

  if action == True:
    out_pdf = path + '/figures/mcmc_b_refolded.pdf'
    out_corner_pdf =  path + '/figures/mcmc_b_corner_refolded.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
    corner_pdf = matplotlib.backends.backend_pdf.PdfPages(out_corner_pdf)
    flux = np.load(path + '/data/transit/individual_flux_array_refolded.npy', allow_pickle=True)
    time = np.load(path + '/data/transit/individual_time_array_refolded.npy', allow_pickle=True) 
    stds = np.load(path + '/data/transit/stds_refolded.npy', allow_pickle = True)
    
  else:
    out_pdf = path + '/figures/mcmc_b.pdf'
    out_corner_pdf =  path + '/figures/mcmc_b_corner.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
    corner_pdf = matplotlib.backends.backend_pdf.PdfPages(out_corner_pdf)
    flux = np.load(path + '/data/transit/individual_flux_array.npy', allow_pickle = True)
    time = np.load(path + '/data/transit/individual_time_array.npy', allow_pickle = True)
    stds = np.load(path + '/data/transit/stds.npy', allow_pickle = True)
 

  coeffs = np.loadtxt(path + '/data/transit/coeffs.txt')
   

  # MCMC parameters
  nsteps = 5000
  burn_in = 2000
  ndim = 4
  nwalkers = 100


  t0 = timing.time()


  params_final = []
  t0_w_uncert = []
  t0s_ml = []
  t0s_mcmc = []
  t0s_unc = []

  fig, ax = plt.subplots(4, 3)

  #cols = ['Original light curve', 'De-trended light curve','Residuals of transit model']
  #for axi, col in zip(ax[0], cols):
  #  axi.set_title(col, fontsize=6)

  for i in range(flux.shape[0]):

      if i % 6 == 0 and i != 0:
        pdf.savefig(fig)
        fig, ax = plt.subplots(4, 3)
        #cols = ['Original light curve', 'De-trended light curve','Residuals of transit model']
        #for axi, col in zip(ax[0], cols):
        #  axi.set_title(col, fontsize=6)

      time_i = time[i]
      flux_i = flux[i]

      idx = np.argmin(flux_i)
      t0_i = time_i[idx]

      
      sigma = stds[i]
      a_i = coeffs[i, 0]
      b_i = coeffs[i, 1]
      c_i = coeffs[i, 2]
   
      initial_params = t0_i, a_i, b_i, c_i
      nll = lambda *args: -lnlike(*args) 
      initial = np.array([t0_i, a_i, b_i, c_i]) 
      soln = scipy.optimize.minimize(nll, initial, args=(time_i, flux_i, sigma, r, a, b, u1, u2, per, t0_i), method='Nelder-Mead')  
      t0_ml, a_ml, b_ml, c_ml = soln.x 
      print(f'Event: {i} Initial t0: {t0_i} vs max likelihood: {t0_ml}')


      # Initialize walkers around maximum likelihood.
      pos = [initial_params + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

      # Set up sampler.
      sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(np.array(time_i), np.array(flux_i), sigma, r, a, b, u1, u2, per, t0_i))

      # Run MCMC for n steps and display progress bar.
      width = 50
      for m, result in enumerate(sampler.sample(pos, iterations=nsteps)):
        n = int((width+1) * float(m) / nsteps)
        sys.stdout.write("\r{}[{}{}]{}".format('sampling... ', '#' * n, ' ' * (width - n), ' (%s%%)' % str(100. * float(m) / nsteps)))
      sys.stdout.write("\n")

      #samples = sampler.chain
     
   
      
      samples = sampler.flatchain
      #mcmc_max  = samples[np.argmax(sampler.flatlnprobability)]
      #t0_mcmc, k_mcmc, b_mcmc = mcmc_max[0], mcmc_max[1], mcmc_max[2]

      samples = sampler.chain
      # Discard burn-in. 
      samples = samples[:, burn_in:, :].reshape((-1, ndim))
      # Final params and uncertainties based on the 16th, 50th, and 84th percentiles of the samples in the marginalized distributions.
      t01, a1, b1, c1  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
      t0_w_uncert.append(t01)  
      t0s_ml.append(t0_ml)
      t0s_unc.append(t01[1])
      params_final.append([t0_ml, a_ml, b_ml, c_ml])


      fit = a_ml * time_i * time_i + b_ml * time_i + c_ml
      corrected_flux = np.divide(flux_i, fit)
      #sigma = np.std(corrected_flux)
      yerr = np.full(corrected_flux.shape[0], sigma)



      # Plot optimized transit model.
      params_f = batman.TransitParams()
      params_f.t0 = t0_ml
      params_f.per = per
      params_f.rp = r
      params_f.a = a
      params_f.inc =  np.arccos(b/a)*(180./np.pi)
      params_f.ecc = 0
      params_f.w = 96
      params_f.u = [u1, u2]
      params_f.limb_dark = "quadratic"
      tl = np.linspace(min(time_i),max(time_i),5000)
      m = batman.TransitModel(params_f, tl)
      f_final = m.light_curve(params_f)
  
   
      #ax[i % 6, 1].plot(time_i, corrected_flux, '.b', markersize = 0.8) #yerr=yerr, fmt='.b', capsize=0, alpha=0.5, zorder=1, markersize = 0.6)
      #ax[i % 6, 1].plot(tl, f_final, 'r-', linewidth=0.8)
      #ax[i % 6, 1].set_xlabel("Time [days]",  fontsize=3)
      #ax[i % 6, 1].set_ylabel("Relative Flux",  fontsize=3)
      #ax[i % 6, 1].legend(('BATMAN','TESS'), loc=2, prop={'size': 2})
      #ax[i % 6, 1].xaxis.set_tick_params(labelsize=3)
      #ax[i % 6, 1].yaxis.set_tick_params(labelsize=3)
      #ax[i % 6, 1].ticklabel_format(useOffset=False)

      # plot residuals 
      m = batman.TransitModel(params_f, time_i)
      f_ = m.light_curve(params_f)
      residuals = flux_i/fit - f_

      if 0 <= i % 6 <3:

        ax[0, i % 6].plot(time_i, flux_i, '.b', markersize = 0.8)
        ax[0, i % 6].plot(time_i, fit, 'r', linewidth=0.8)      
        ax[0, i % 6].set_xlabel("Time [days]",  fontsize=6)
        ax[0, i % 6].set_ylabel("Flux",  fontsize=6)
        ax[0, i % 6].xaxis.set_tick_params(labelsize=4)
        ax[0, i % 6].yaxis.set_tick_params(labelsize=4)
        ax[0, i % 6].ticklabel_format(useOffset=False)
       # ax[0, i % 6].set_xticklabels(time_i.round(decimals=3), rotation=40)

        ax[1, i % 6].plot(time_i, residuals, '.b', markersize = 0.8)
        ax[1, i % 6].set_xlabel("Time [days]",  fontsize=6)
        ax[1, i % 6].set_ylabel("Residuals",  fontsize=6)
        ax[1, i % 6].xaxis.set_tick_params(labelsize=4)
        ax[1, i % 6].yaxis.set_tick_params(labelsize=4)
        ax[1, i % 6].ticklabel_format(useOffset=False)
      #  ax[1, i % 6].set_xticklabels(time_i.round(decimals=3), rotation=40)
 
      else:

        ax[2, (i % 6) % 3].plot(time_i, flux_i, '.b', markersize = 0.8)
        ax[2, (i % 6) % 3].plot(time_i, fit, 'r', linewidth=0.8)      
        ax[2, (i % 6) % 3].set_xlabel("Time [days]",  fontsize=6)
        ax[2, (i % 6) % 3].set_ylabel("Flux",  fontsize=6)
        ax[2, (i % 6) % 3].xaxis.set_tick_params(labelsize=4)
        ax[2, (i % 6) % 3].yaxis.set_tick_params(labelsize=4)
        ax[2, (i % 6) % 3].ticklabel_format(useOffset=False)
       # ax[2, (i % 6) % 3].set_xticklabels(time_i.round(decimals=3), rotation=40)
        

        ax[3, (i % 6) % 3].plot(time_i, residuals, '.b', markersize = 0.8)
        ax[3, (i % 6) % 3].set_xlabel("Time [days]",  fontsize=6)
        ax[3, (i % 6) % 3].set_ylabel("Residuals",  fontsize=6)
        ax[3, (i % 6) % 3].xaxis.set_tick_params(labelsize=4)
        ax[3, (i % 6) % 3].yaxis.set_tick_params(labelsize=4)
        ax[3, (i % 6) % 3].ticklabel_format(useOffset=False)
      #  ax[3, (i % 6) % 3].set_xticklabels(time_i.round(decimals=3), rotation=40)
 

      fig.tight_layout()    


      samples = sampler.flatchain
      #theta_max  = samples[np.argmax(sampler.flatlnprobability)]
      
      param_names = ["$t_0$", "$a$", "$b$", "$c$"]
      corn_fig = corner.corner(samples, labels=param_names)
      corner_pdf.savefig(corn_fig)
      

      #corner_fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
      #samples = sampler.get_chain()
      #labels = ["$t_0$", "$k$", "$b$"]
      #for k in range(ndim):
      #  axi = axes[k]
      #  axi.plot(samples[:, :, k], "k", alpha=0.3)
      #  axi.set_xlim(0, len(samples))
      #  axi.set_ylabel(labels[k])
      #  axi.yaxis.set_label_coords(-0.1, 0.5)

      #axes[-1].set_xlabel("step number")
      #corner_pdf.savefig(corner_fig)

          
  pdf.savefig(fig)



  t1 = timing.time()
  print('Execution time (mcmc b): {:.2f} min'.format((t1-t0)/60))


  np.savetxt(path + '/data/transit/t0_k_b.txt', np.array(params_final))
  np.savetxt(path + '/data/transit/t0_w_uncert.txt', np.array(t0_w_uncert))
  
   
  df = pd.DataFrame(columns = ("$t_0$ (max likelihood)", "Uncertainty"))
  df["$t_0$ (max likelihood)"] = np.around(np.array(t0s_ml), decimals = 6) 
  #df["$t_0$ (mcmc)"] =  np.around(np.array(t0s_mcmc), decimals = 6) 
  df[ "Uncertainty"] =  np.around(np.array(t0s_unc), decimals = 6) 

  fig, ax =plt.subplots(figsize=(12,4))
  ax.axis('tight')
  ax.axis('off')
  the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
  pdf.savefig(fig, bbox_inches='tight')
  pdf.close()
  corner_pdf.close()



