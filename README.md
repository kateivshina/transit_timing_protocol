# Transit Timing Protocol

Code for reading, fitting and analyzing TESS light curves + running MCMC to see how orbital period of hot Jupiters changes with epoch number.


## How to run the code:

1. Clone the GitHub repository:

```
git clone https://github.com/kateivshina/transit_timing_protocol.git
```

Go to the directory of the repository:

```
cd transit_timing_protocol/protocol
```

1a. Prepare anaconda environment:
```
module load anaconda3
```
```
conda create -n ttp python=3.7 numpy pandas emcee corner scikit-learn astropy matplotlib batman-package scipy
```

```
conda activate ttp
```
2. To run the protocol, type in the terminal:

```
python3 main.py 1
```
where 1 specifies that the first planet from our exoplanet databse is analyzed.

3. To run the protocol as a job array on multiple planets using a computing cluster, submit the job as a SLURM script:
```
sbatch job.sh
```
You should modify *job.sh* file in the */protocol* folder before executing the script. For example, *--mail-user* should be set to your email address and *--array* should be set to the range of indices of the planets in our database that we would like to analyze.
```
#!/bin/bash
#SBATCH --job-name=array-job     # create a short name for your job
#SBATCH --output=slurm-%N.%j.out # STDOUT file
#SBATCH --error=slurm-%N.%j.err  # STDERR file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-99            # job array with index values 0, 1, 2, 3, 4
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=eivshina@princeton.edu

module load anaconda3
source activate s
python main.py $SLURM_ARRAY_TASK_ID
```

## What the protocol does:
This protocol does the following:
1. selects individual transits from the TESS light curve; 
2. de-trends them; 
3. finds maximum likelihood {a, b, r, u1, u2}; 
4. runs MCMC to estimate the uncertainties of {a, b, r, u1,u2}; 
5. fits each transit event with a transit model with the previously derived {a, b, r, u1,u2} parameters and finds the maximum likelihood {t0, a, b, c} (i.e. the mid-transit time and coefficients of the quadratic de-trending polynomial) as well as their uncertainties; 
6. re-folds the data using the found mid-transit times; 
7. repeats steps  (3)-(6) to re-derive the mid-transit times;
8. plots O-C just for the TESS data;
9. plots O-C including the historical data;

The following figures are stored in the ~/figures folder:
1. *preprocess.pdf* contains TESS light curve of a given planet, the selected aperture, a figure showing the transit and out-of-transit data selected by the algorithm, figures containing individual transit events, their de-trended light curves and the residuals.
2. *mcmc_a.pdf* contains initial transit model fit, corner plots and figures of random walkers of  {a, b, r, u1,u2} as well as a table with the estimated {a, b, r, u1,u2} values and their uncertainties.
3. *mcmc_b.pdf* contains figures of individual transit events, their de-trended light curves fitted with the transit model and the residuals. This file also contains a table of the found mid-transit times and their uncertainties.
4.  *mcmc_b_corner.pdf* contains corner plots and figures of random walkers for {t0, k, b} for each transit event.
5. *mcmc_a_refolded.pdf*, *mcmc_b_refolded.pdf*, *mcmc_b_corner_refolded* contains the same information as files 2-4 but for the re-folded data.
6. *tess_o_c.png* contains O-C just for the TESS data.
7. *o_c_combined.png* contains O-C for TESS as well as historical data.

## TESS Data

1) [TESS cut](https://mast.stsci.edu/tesscut/) 
2) [TICs for each sector](https://tess.mit.edu/observations/sector-17/)
3) [2-minute cadence data](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)

