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
parser.add_argument('--parent_dir')
parser.add_argument('--delete_arrs', nargs='*')

args = parser.parse_args()
 
# Path 
planet_name = args.planet
parent_dir = args.parent_dir
directory = planet_name.replace(" ", "_") 
path = f'{parent_dir}' + f'/{directory}'  
path = path + '/data'
path = path + '/transit_masked'

# extract the indices of the rows to remove
arrays2remove = [int(x) for x in args.delete_arrs[0].split(' ')]



flux_array = np.load(path +'/individual_flux_array.npy', allow_pickle = True)
time_array = np.load(path + '/individual_time_array.npy', allow_pickle = True)
time_folded_array = np.load(path + '/individual_time_folded_array.npy', allow_pickle = True)

num_of_rows_before = time_folded_array.shape[0]

for indx in reversed(arrays2remove):
	print(indx)
	flux_array = np.delete(flux_array, indx, axis=0)
	time_array = np.delete(time_array, indx, axis=0)
	time_folded_array = np.delete(time_folded_array, indx, axis=0)

print('new shape', flux_array.shape, ' vs expected ', num_of_rows_before - len(arrays2remove))
print('new shape', time_array.shape, ' vs expected ', num_of_rows_before - len(arrays2remove))
print('new shape', time_folded_array.shape, ' vs expected ', num_of_rows_before - len(arrays2remove))

# save the data by overwriting the original files
np.save(path +'/individual_flux_array_clean.npy', flux_array)
np.save(path + '/individual_time_array_clean.npy', time_array)
np.save(path + '/individual_time_folded_array_clean.npy',time_folded_array)


