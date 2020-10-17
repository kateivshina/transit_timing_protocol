import os
#from argparse import ArgumentParser
from process_tess_data import process_tess
from mcmc_a import run_mcmc_a
from mcmc_b import run_mcmc_b
from refolding import refold
from o_c import o_c
from o_c_combo import o_c_combo
import sys
#from lightkurve import search_lightcurvefile  
import pandas as pd
import numpy as np
import faulthandler; faulthandler.enable()

 
def main():
	cadence = 2
	N = 2
	degree = 2

	df = pd.read_csv(os.path.dirname(os.getcwd()) + '/data/hot_jupyter_sample.csv')
	idx = int(sys.argv[-1])
 
	pl_hostname =  df['System'].iloc[idx]
	pl_letter = 'b'
	planet_name = pl_hostname + pl_letter  

	parent_dir = os.path.dirname(os.getcwd())
	directory = planet_name.replace("-", "_") 
	path = f'{parent_dir}' + f'/{directory}'  

	if not os.path.isdir(path): 
		os.mkdir(path)
		os.mkdir(path + '/data')
		os.mkdir(path + '/figures')
		os.mkdir(path + '/data/transit')
		os.mkdir(path + '/data/transit_masked')

		
	#process_tess(pl_hostname, pl_letter, cadence, N, degree, os.path.dirname(os.getcwd()) + '/light_curves' + f'/{pl_hostname}.fits', parent_dir)
	run_mcmc_a(pl_hostname, pl_letter, parent_dir, False)
	#run_mcmc_b(pl_hostname, pl_letter, parent_dir, False)

	#refold(pl_hostname, pl_letter, parent_dir, N)

	run_mcmc_a(pl_hostname, pl_letter, parent_dir, True)
	#run_mcmc_b( pl_hostname, pl_letter, parent_dir, True)

	#o_c(pl_hostname, pl_letter, parent_dir)
		#o_c_combo(pl_hostname, pl_letter, parent_dir)



if __name__ == "__main__":
	main()
