import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter
import exoplanet as xo
import pandas as pd
import os
import sys
from argparse import ArgumentParser

# parse input data
parser = ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--mission')
parser.add_argument('--pl_hostname')
parser.add_argument('--pl_letter') 
parser.add_argument('--cadence')
parser.add_argument('--N')
parser.add_argument('--parent_dir')
parser.add_argument('--path_to_data_file')
parser.add_argument('--refolded')


args = parser.parse_args()


 
MISSION = args.mission
planet_name = args.pl_hostname + args.pl_letter
cadence = args.cadence
N = args.N
path_to_data_file =args.path_to_data_file
# Path 
parent_dir = args.parent_dir
directory = planet_name.replace(" ", "-") 
path = f'{parent_dir}' + f'/{directory}'  

# load CSV file with the exoplanet data
df = pd.read_csv('sampled_planets.csv')
df = df.loc[df['pl_hostname'] == f'{args.pl_hostname.replace(" ", "-")}']
#print('df ', df)
df = df.loc[df['pl_letter'] == f'{args.pl_letter}']
pl_trandur = df['pl_trandur'].iloc[0]

G = 6.67e-10 # gravitational constant
if np.isnan(pl_trandur):
    # estimate transit duration
    M = df['st_mass'].iloc[0] * 1.989e+30 # in kg
    a = df['pl_orbsmax'].iloc[0] * 1.496e+11 # in meters
    R = df['st_rad'].iloc[0] * 696.34 * 10e6 # in meters
    if np.isnan(M) or np.isnan(a) or np.isnan(R):
        print('Could not estimate transit duration')
        sys.exit(0)
    print(f'Mass {M}',  df['st_mass'].iloc[0])
    print(f'a {a}', df['pl_orbsmax'].iloc[0] )
    print(f'R {R}',  df['st_rad'].iloc[0])


    v = (G*M/a)**0.5
    pl_trandur = (2*R/v)/86400 # transit duration in days
    print(f'v {v}')
    
    print('duration ', pl_trandur)

if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(path +'/figures'):
    os.mkdir(path +'/figures')

path_to_fig = path +'/figures'
path_to_data = path + '/data'

if not os.path.isdir(path_to_data):
    os.mkdir(path_to_data)
    os.mkdir(path_to_data + '/transit_masked')
    os.mkdir(path_to_data + '/transit')
