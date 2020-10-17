from astroquery.mast import Observations
from astropy.io import fits
import matplotlib.pyplot as plt
from lightkurve import search_lightcurvefile  
import pandas as pd
import os 
import numpy as np

df = pd.read_csv('hot_jypiter_sample.csv')
pl_names = df['System']

not_found = []
 
for i in range(df.shape[0]):
	print(i)
	
	host_name = pl_names.iloc[i]
	 
 
	search_result = search_lightcurvefile(host_name, mission = 'TESS').table   #.download()   
	try:
		url = search_result['dataURI'].pformat()[2]
		path_to_data_file = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=' + url

		with fits.open(path_to_data_file, mode="readonly") as hdulist:
			hdulist.writeto(f'{host_name}.fits')


	except:
		not_found.append(host_name)
np.savetxt('no_light_curve_found.csv', not_found, fmt='%s')