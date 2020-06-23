import numpy as np
from astropy.io import fits
import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter
import exoplanet as xo
import os
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


MISSION = args.mission
planet_name = args.planet
cadence = args.cadence
path_to_data_file =args.path_to_data_file
# Path 
parent_dir = args.parent_dir
directory = planet_name.replace(" ", "_") 
path = f'{parent_dir}' + f'/{directory}'  
path = path + '/data/transit/'

bls_period = float(args.period)

t0_k_b = np.loadtxt(path + '/t0_k_b.txt')
t0s = t0_k_b[:,0] # mid-transit times
ks = t0_k_b[:,1] # k coefficients of the fit
bs = t0_k_b[:,2] # b coefficients of the fit

#load flux and time data
flux = np.load(path + '/individual_flux_array_clean.npy', allow_pickle=True)
time = np.load(path + '/individual_time_array_clean.npy', allow_pickle=True) 

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
	m = np.abs(x_fold) <= 0.4
	transit_mask = bls.transit_mask(time_i, bls_period, 0.2, bls_t0)
	not_transit = ~transit_mask
	# folded data with transit masked:
	total_mask = m & not_transit

	
	time_folded_ = x_fold[m] # folded transit masked time
	time_ = time_i[m] # transit masked time 
	flux_ = flux_i[m]

	fit = k*time_+b
	detrended_flux_ = flux_/fit
	#plt.plot(time_, detrended_flux_, '.k')
	#plt.show()

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

np.save(path + '/corrected_flux_refolded.npy', detrended_flux_array)
np.save(path + '/stds_refolded.npy', stds)
np.save(path + '/individual_flux_array_clean_refolded.npy', flux_array)
np.save(path + '/individual_time_array_clean_refolded.npy', time_array)
np.save(path + '/individual_time_folded_array_clean_refolded.npy', folded_time_array)