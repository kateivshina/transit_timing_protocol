import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import batman
import emcee
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
parser.add_argument('--impact_parameter')
parser.add_argument('--period')
parser.add_argument('--parent_dir')
parser.add_argument('--path_to_data_file')

args = parser.parse_args()
#print(args)
 
# path info
MISSION = args.mission
cadence = args.cadence
planet_name = args.planet
path_to_data_file =args.path_to_data_file
# Path 
parent_dir = args.parent_dir
directory = planet_name.replace(" ", "_") 
path = f'{parent_dir}' + f'/{directory}'  


# planet info
per = float(args.period)
rp_i = float(args.radius) # Rp/R* (planet's radius in terms of stellar radii)
a_i = float(args.semi_major_axis) #(semi-major axis in terms of stellar radii)
b_i = float(args.impact_parameter) # impact parameter

flux = np.load(path + '/data/transit/corrected_flux_clean.npy', allow_pickle=True)
time = np.load(path + '/data/transit/individual_time_folded_array_clean.npy', allow_pickle=True) 

stds = np.load(path + '/data/transit/stds_clean.npy', allow_pickle = True)
sigma = np.mean(stds, axis=0)

u1_i = 0.5	# Limb Darkening coefficient 1
u2_i = 0.1144 # Limb Darkening coefficient 2



flux =  np.concatenate(flux, axis=0)
time =  np.concatenate(time, axis=0)

print('radius ', rp_i)



# Plot the final transit model.
params_final = batman.TransitParams()
params_final.t0 = 0
params_final.per = 1
params_final.rp = 0.1
params_final.a = 4
params_final.inc = 87 #np.arccos(theta_max[3] / theta_max[2]) * (180. / np.pi)
params_final.ecc = 0
params_final.w = 90
params_final.u = [u1_i, u2_i]
params_final.limb_dark = "quadratic"
tl = np.linspace(min(time),max(time),5000)
m = batman.TransitModel(params_final, tl)
f_final = m.light_curve(params_final)
final_fig, ax = plt.subplots(figsize=(10,8))
ax.set_title(planet_name)
#ax.errorbar(times,fluxes,yerr=ferr,fmt='k.',capsize=0,alpha=0.4,zorder=1)
#ax.plot(time, flux, 'k.',alpha=0.8,lw=3,zorder=2)
ax.plot(tl, f_final, 'r-',alpha=0.8,lw=3,zorder=2)
ax.set_xlabel("Time")
ax.set_ylabel("Relative Flux")
ax.legend(('BATMAN','TESS'), loc=2)
plt.show()
     
 

