import numpy as np
from astropy.io import fits
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter
import exoplanet as xo
import os
from argparse import ArgumentParser



flux = np.load('/Users/ivshina/Desktop/usrp/orbital_decay/WASP_12b/data/transit/individual_flux_array_clean.npy', allow_pickle=True) 
time = np.load('/Users/ivshina/Desktop/usrp/orbital_decay/WASP_12b/data/transit/individual_time_folded_array_clean.npy', allow_pickle=True) 
 

flux =  np.concatenate(flux, axis=0)
time =  np.concatenate(time, axis=0)


fig = plt.figure()
plt.plot(time, flux, '.k')
plt.xlabel("Time [days]")
plt.ylabel("Relative flux [ppt]")
plt.show()

np.savetxt('/Users/ivshina/Desktop/usrp/orbital_decay/WASP_12b/data/transit/individual_flux_array_clean_flatten.txt', flux)
np.savetxt('/Users/ivshina/Desktop/usrp/orbital_decay/WASP_12b/data/transit/individual_time_folded_array_clean_flatten.txt', time)


'''
print('flux  ', flux.ravel())

print('flux shape ', flux.ravel().shape)
print('time shape ', time.ravel().shape)
#flux = np.loadtxt('/Users/ivshina/Desktop/flux.txt')
#time = np.loadtxt('/Users/ivshina/Desktop/times.txt')
 
fluxes = []
times = []

fig = plt.figure()
for i in range(20):
	flux_i = flux[i]
	time_i = time[i]
	print('flux i ', flux_i)
	fluxes.append(flux_i)
	times.append(time_i)
	#plt.plot(time_i, flux_i, '.k')
	#plt.xlabel("Time [days]")
	#plt.ylabel("Relative flux [ppt]")
	#plt.show()

fluxes = np.array(fluxes).ravel()
print('fluxes' , np.concatenate( fluxes, axis=0 ))
print('shape ' , np.concatenate( fluxes, axis=0 ).shape)
'''