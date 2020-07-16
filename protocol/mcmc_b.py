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
from ctypes import RTLD_GLOBAL
import matplotlib.backends.backend_pdf
from matplotlib.backends.backend_pdf import PdfPages
 


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

planet_name = args.pl_hostname + args.pl_letter
path_to_data_file =args.path_to_data_file
# Path 
parent_dir = args.parent_dir
directory = planet_name.replace(" ", "_") 
path = f'{parent_dir}' + f'/{directory}'  



action = args.refolded

# Planet info
# load CSV file with the exoplanet data
df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/sampled_planets.csv')
df = df.loc[df['pl_hostname'] == f'{args.pl_hostname.replace(" ", "-")}']
df = df.loc[df['pl_letter'] == f'{args.pl_letter}']
per = df['pl_orbper'].iloc[0]


theta = np.loadtxt(path + '/data/transit/theta_max.txt')

if action == 'True':
  out_pdf = '/Users/kate/Desktop/3_refolded.pdf'
  pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
  flux = np.load(path + '/data/transit/individual_flux_array_clean_refolded.npy', allow_pickle=True)
  time = np.load(path + '/data/transit/individual_time_array_clean_refolded.npy', allow_pickle=True) 
  stds = np.load(path + '/data/transit/stds_refolded.npy', allow_pickle = True)
  
else:
  out_pdf = '/Users/kate/Desktop/3.pdf'
  pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)
  flux = np.load(path + '/data/transit/individual_flux_array_clean.npy', allow_pickle = True)
  time = np.load(path + '/data/transit/individual_time_array_clean.npy', allow_pickle = True)

  stds = np.load(path + '/data/transit/stds_clean.npy', allow_pickle = True)


r, a, b, u1, u2  = theta[0], theta[1], theta[2], theta[3], theta[4]


df = pd.DataFrame(np.array([r, a, b, u1, u2]).reshape(1,-1), columns = ("r", "a", "b", "u1", "u2"))


fig, ax =plt.subplots(figsize=(12,4))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')

pp = PdfPages(path + '/figures/model_min_chisquare.pdf')
pp.savefig(fig, bbox_inches='tight')
pp.close()




coeffs = np.loadtxt(path + '/data/transit/coeffs.txt')

 

# MCMC parameters
nsteps = 5000
burn_in = 3000
ndim = 3
nwalkers = 100

 


# Priors
def lnprior(theta):
  t0, k, c = theta
  if np.isfinite(t0) and np.isfinite(k) and np.isfinite(c):
    return 0
  return -np.inf


#def lnlike(theta, x, y, sigma, r, a, b, u1, per=per):
def lnlike(theta, x, y, sigma, r, a, b, u1, u2):
  t0, k, c = theta

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
  return -0.5*(np.sum((y/(k*x+c)-model)**2*inv_sigma2))
  
 
 

# Define log 

# Define log of probability function.
def lnprob(theta, x, y, sigma, r, a, b, u1, u2):
  lp = lnprior(theta)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(theta, x, y, sigma, r, a, b, u1, u2)


t0 = timing.time()


params_final = []
t0_w_uncert = []

for i in range(flux.shape[0]):
    time_i = time[i]
    flux_i =flux[i]

    idx = np.argmin(flux_i)
    t0_i = time_i[idx]

    print('initial t0: ', t0_i)
    sigma = stds[i]
    k_i = coeffs[i, 0]
    c_i = coeffs[i, 1]

 
    initial_params = t0_i, k_i, c_i
    nll = lambda *args: -lnlike(*args) 
    initial = np.array([t0_i, k_i, c_i]) + 1e-5*np.random.randn(ndim)
    soln = scipy.optimize.minimize(nll, initial, args=(time_i, flux_i, sigma, r, a, b, u1, u2))  
    t0_ml, k_ml, c_ml = soln.x 
     

    # Initialize walkers around maximum likelihood.
    pos = [initial_params + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

    # Set up sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(np.array(time_i), np.array(flux_i), sigma, r, a, b, u1, u2))

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
    t01, k1, b1  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    t0_w_uncert.append(t01)  
 
 
    samples = sampler.flatchain
    #theta_max  = samples[np.argmax(sampler.flatlnprobability)]
    params_final.append([t0_ml, k_ml, c_ml])
  

    param_names = ["$t_0$", "$k$", "$b$"]


    corn_fig = corner.corner(samples, labels=param_names)
    pdf.savefig(corn_fig)
    

    fit = k_ml * time_i + c_ml
    flux = flux_i/fit
    

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
    final_fig, ax = plt.subplots(figsize=(10,8))
    ax.set_title(planet_name)
    #ax.errorbar(time,flux,yerr=yerr,fmt='.k',capsize=0,alpha=0.4,zorder=1)
    ax.plot(time_i, flux, '.k')
    ax.plot(tl, f_final, 'r-',alpha=0.8,lw=3,zorder=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative Flux")
    ax.legend(('BATMAN','TESS'), loc=2)
    
    pdf.savefig(final_fig)

    # plot fit and the data itself 
    fig = plt.figure()

    plt.plot(time_i, fit, 'r')
    plt.plot(time_i.reshape(1, -1), flux_i.reshape(1, -1), '.b')
    pdf.savefig(fig)

    plt.close(fig)

    # plot residuals 
    fig2 = plt.figure()
    residuals = flux_i - fit
    plt.plot(time_i, residuals, '.k')
    plt.xlabel("Time [days]")
    plt.ylabel("Residuals")
    pdf.savefig(fig2)
    plt.close(fig2)
        

df = pd.DataFrame(np.array(params_final), columns = ("$t_0$", "$k$", "$b$"))
fig, ax =plt.subplots(figsize=(12,4))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
pp = PdfPages(path + '/figures/t0_k_b.pdf')
pp.savefig(fig, bbox_inches='tight')
pp.close()

df = pd.DataFrame(np.array(t0_w_uncert), columns = ("$t_0$", "$err1$", "$err2$"))
fig, ax =plt.subplots(figsize=(12,4))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')
pp = PdfPages(path + '/figures/t0_w_uncert.pdf')
pp.savefig(fig, bbox_inches='tight')
pp.close()


t1 = timing.time()

print('Execution time (min): {:.2f}'.format((t1-t0)/60))


np.savetxt(path + '/data/transit/t0_k_b.txt', np.array(params_final))
np.savetxt(path + '/data/transit/t0_w_uncert.txt', np.array(t0_w_uncert))



fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["$t_0$", "$k$", "$b$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

pdf.savefig(fig)
pdf.close()
