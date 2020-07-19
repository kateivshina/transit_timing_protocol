import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter
import exoplanet as xo
import os
import pandas as pd
from argparse import ArgumentParser


def refold(planet_name, 
           pl_hostname, 
           pl_letter, 
           parent_dir,
           N):

    # path info
    directory = planet_name.replace(" ", "_") 
    path = f'{parent_dir}' + f'/{directory}'  
    path = path + '/data/transit'
    # load CSV file with the exoplanet data
    df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/sampled_planets.csv')
    df = df.loc[df['pl_hostname'] == f'{pl_hostname.replace(" ", "-")}']
    df = df.loc[df['pl_letter'] == f'{pl_letter}']
    pl_trandur = df['pl_trandur'].iloc[0]
    bls_period = df['pl_orbper'].iloc[0]

    G = 6.67e-10 # gravitational constant
    if np.isnan(pl_trandur):
        # estimate transit duration
        M = df['st_mass'].iloc[0] * 1.989e+30 # in kg
        a = df['pl_orbsmax'].iloc[0] * 1.496e+11 # in meters
        R = df['st_rad'].iloc[0] * 696.34 * 10e6 # in meters
        if np.isnan(M) or np.isnan(a) or np.isnan(R):
        	print('Could not estimate transit duration')
        	sys.exit(0)
        v = (G*M/a)**0.5
        pl_trandur = (2*R/v)/86400 # transit duration in days

    t0_k_b = np.loadtxt(path + '/t0_k_b.txt')
    t0s = t0_k_b[:,0] # mid-transit times
    ks = t0_k_b[:,1] # k coefficients of the fit
    bs = t0_k_b[:,2] # b coefficients of the fit

    #load flux and time data
    flux = np.load(path + '/individual_flux_array.npy', allow_pickle=True)
    time = np.load(path + '/individual_time_array.npy', allow_pickle=True) 

    flux_array = []
    detrended_flux_array = []
    time_array = []
    folded_time_array = []
    stds = []

    for i in range(flux.shape[0]):
    	time_i = time[i]
    	flux_i = flux[i]
    	bls = BoxLeastSquares(time_i, flux_i)
    	bls_t0 = t0s[i]
    	k = ks[i]
    	b = bs[i]

    	# create transit mask
    	x_fold = (time_i - bls_t0 + 0.5 * bls_period) % bls_period - 0.5 * bls_period
    	m = np.abs(x_fold) < N*0.5*pl_trandur
    	transit_mask =  np.abs(x_fold) < 0.1*pl_trandur
    	not_transit = ~transit_mask
    	# folded data with transit masked:
    	total_mask = m & not_transit

    	time_folded_ = x_fold[m] # folded transit masked time
    	time_ = time_i[m] # transit masked time 
    	flux_ = flux_i[m]

    	fit = k*time_+b
    	detrended_flux_ = flux_/fit
    	# calculate std for out of transit data
    	flux_out_ = flux_i[total_mask]
    	time_out = time_i[total_mask]
    	std = np.std(flux_out_/(k*time_out+b))

    	# append the data
    	flux_array.append(flux_)
    	time_array.append(time_)
    	folded_time_array.append(time_folded_)
    	detrended_flux_array.append(detrended_flux_)
    	stds.append(std)

    detrended_flux_array = np.array(detrended_flux_array, dtype=object, copy=False)
    flux_array = np.array(flux_array, dtype=object, copy=False)
    time_array = np.array(time_array, dtype=object, copy=False)
    folded_time_array = np.array(folded_time_array, dtype=object, copy=False)

    np.save(path + '/corrected_flux_refolded.npy', detrended_flux_array)
    np.save(path + '/stds_refolded.npy', stds)
    np.save(path + '/individual_flux_array_refolded.npy', flux_array)
    np.save(path + '/individual_time_array_refolded.npy', time_array)
    np.save(path + '/individual_time_folded_array_refolded.npy', folded_time_array)