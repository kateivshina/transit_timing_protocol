from argparse import ArgumentParser
from process_tess_data import process_tess
from mcmc_a import run_mcmc_a
from mcmc_b import run_mcmc_b
from refolding import refold
from o_c import o_c
from o_c_combo import o_c_combo

# parse input data
parser = ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--mission')
parser.add_argument('--pl_hostname')
parser.add_argument('--pl_letter') 
parser.add_argument('--cadence')
parser.add_argument('--N')
parser.add_argument('--degree')
parser.add_argument('--parent_dir')
parser.add_argument('--path_to_data_file')
args = parser.parse_args()



def main():
	MISSION = args.mission
	pl_hostname = args.pl_hostname
	pl_letter = args.pl_letter
	planet_name = args.pl_hostname + args.pl_letter
	cadence = args.cadence
	N = float(args.N)
	degree = int(args.degree)
	path_to_data_file =args.path_to_data_file
	# Path 
	parent_dir = args.parent_dir
	directory = planet_name.replace(" ", "_") 
	path = f'{parent_dir}' + f'/{directory}'  
	
	process_tess(planet_name, pl_hostname, pl_letter, cadence, N, degree, path_to_data_file, parent_dir)
	run_mcmc_a(planet_name, pl_hostname, pl_letter, parent_dir, False)
	run_mcmc_b(planet_name, pl_hostname, pl_letter, parent_dir, False)

	refold(planet_name, pl_hostname, pl_letter, parent_dir, N)

	run_mcmc_a(planet_name, pl_hostname, pl_letter, parent_dir, True)
	run_mcmc_b(planet_name, pl_hostname, pl_letter, parent_dir, True)

	o_c(planet_name, pl_hostname, pl_letter, parent_dir)
	o_c_combo(planet_name, pl_hostname, pl_letter, parent_dir)



if __name__ == "__main__":
	main()