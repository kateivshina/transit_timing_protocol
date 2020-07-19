# README

Code for reading, fitting and analyzing TESS data + running MCMC to see how orbital period changes with epoch number.

TESS data:
1) [TESS cut](https://mast.stsci.edu/tesscut/) 
2) [TICs for each sector](https://tess.mit.edu/observations/sector-17/)
3) [2-minute cadence data](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)

## Running the code:

0. Clone the GitHub repository:

```
git clone https://github.com/kateivshina/transit_timing_protocol.git
```

Go to the directory of the repository:

```
cd transit_timing_protocol/protocol
```

1. In pl_params.txt file, specify the parameters of the planetary system as well as where to store the output data.

**Example**
- --mission=TESS
- --pl_hostname=WASP 19
- --pl_letter=b
- --cadence=2
- --N=0.5
- --degree=1
- --parent_dir=/Users/kate/Documents/usrp/TTP
- --path_to_data_file=/Users/kate/Documents/usrp/TTP/lc/wasp19_lc.fits

2. To run the protocol, type in the terminal:

```
python3 main.py @pl_params.txt
```

This protocol does the following:
1. selects individual transits from the TESS light curve; 
2. de-trends them; 
3. finds maximum likelihood {a, b, r, u1,u2}; 
4. runs MCMC to estimate the uncertainties of {a, b, r, u1,u2}; 
5. fits each transit event with a transit model with the previously derived {a, b, r, u1,u2} parameters and finds the maximum likelihood {t0, k, b} as well as their uncertainties; 
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
6. *tess_o_c.png* contains O-C just for the TESS data
7. *o_c_combined.png* contains O-C for TESS as well as historical data.
