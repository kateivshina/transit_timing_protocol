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
N = float(args.N)
path_to_data_file =args.path_to_data_file
# Path 
parent_dir = args.parent_dir
directory = planet_name.replace(" ", "_") 
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

    v = (G*M/a)**0.5
    pl_trandur = (2*R/v)/86400 # transit duration in days

pl_trandur = np.float64(pl_trandur)

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


#################################################################

def select_transits(transit, path, path_to_times, path_to_times_folded, path_to_flux, path_to_figures):
    '''
    This function applies transit mask to TESS data to select individual transits 
    and stores fluxes and times of transits in .txt file

    Inputs:
    transit (True/False) = boolean variable to check if we import data inside or outside
    of transits 
    path = folder where to save the output of the function
    path_to_times = path to file storing times
    path_to_times_folded = path to file storing transit mask
    path_to_flux = path to file storing fluxes

    '''

    # load data
    times = np.loadtxt(path_to_times)
    time_folded = np.loadtxt(path_to_times_folded)
    flux = np.loadtxt(path_to_flux)
     

    indx = []
    indx.append(0)
    for i in range(1,time_folded.shape[0]):
        if (time_folded[i] < 0 and time_folded[i-1] > 0):
            indx.append(i)

     
    flux_array = []
    time_array = []
    time_folded_array = []

    PATH_TO_FIGURES = path_to_figures + '/individual_transits_figures'
    if os.path.isdir(PATH_TO_FIGURES) == False and transit == True:
        os.mkdir(PATH_TO_FIGURES)
   

    for i in range(len(indx)):
        if indx[i] != max(indx):
            data_ = flux[indx[i]:indx[i+1]]
            time_ = times[indx[i]:indx[i+1]]
            time_folded_ = time_folded[indx[i]:indx[i+1]]
            flux_array.append(data_)
            time_array.append(time_)
            time_folded_array.append(time_folded_)
            if transit == True:
                fig = plt.figure()
                plt.plot(time_, data_, '.k')
                plt.xlabel("Time [days]")
                plt.ylabel("Relative flux")
                plt.savefig(PATH_TO_FIGURES + f'/transit_{i}')
                plt.close(fig)

            
        else:
            data_ = flux[indx[i]:]
            time_ = times[indx[i]:]
            time_folded_ = time_folded[indx[i]:]
            flux_array.append(data_)
            time_array.append(time_)
            time_folded_array.append(time_folded_)
            if transit == True:
                fig = plt.figure()
                plt.plot(time_, data_, '.k')
                plt.xlabel("Time [days]")
                plt.ylabel("Relative flux")
                plt.savefig(PATH_TO_FIGURES + f'/transit_{i}')
                plt.close(fig)
    
    flux_array = np.array(flux_array, dtype=object, copy=False)
    time_array = np.array(time_array, dtype=object, copy=False)
    time_folded_array = np.array(time_folded_array, dtype=object, copy=False)

    np.save(path +'/individual_flux_array.npy', flux_array)
    np.save(path + '/individual_time_array.npy', time_array)
    np.save(path + '/individual_time_folded_array.npy', time_folded_array)
 
    # to load .npy, use np.load(path + '/transit_flux.npy', allow_pickle=True))



#################################################################

def detrend(path, path_to_times, path_to_flux, path_to_time_masked, path_to_flux_masked, path_to_figures):
    '''
    This function de-trends individual transits by fitting a polynomial of degree 1
    to the data points outside of each transit and dividing each transit's data by 
    the best fit.

    Inputs:
    path = folder where to save the output of the function
    path_to_times = path to file storing times for each transit
    path_to_flux = path to file storing fluxes for each transit
    path_to_flux_masked = path to file storing fluxes outside of transit for each transit
    path_to_time_masked = path to file storing times corresponding to fluxes in path_to_flux_masked
    file.

    Output:
    .npy file storing de-trended indiivdual transit fluxes and .txt file storing stds of fluxes 
    outside of transits for each transit. 
    '''

    flux_masked = np.load(path_to_flux_masked, allow_pickle=True)
    time_masked = np.load(path_to_time_masked, allow_pickle=True)
    flux = np.load(path_to_flux, allow_pickle=True)
    time = np.load(path_to_times, allow_pickle=True)

    #print('flux_masked shape ', flux_masked.shape)
    #print('time_masked shape ', time_masked.shape)
    #print('flux shape ', flux.shape)
    #print('time shape ', time.shape)

    stds = [] #list to store stds of fluxes
    corrected_flux = []

    PATH_TO_FIGURES = path_to_figures + '/transits_after_detrending'
    if os.path.isdir(PATH_TO_FIGURES) == False:
        os.mkdir(PATH_TO_FIGURES)

    PATH_TO_FIT = path_to_figures + '/fits'
    if os.path.isdir(PATH_TO_FIT) == False:
        os.mkdir(PATH_TO_FIT)

    PATH_TO_RESIDUALS = path_to_figures + '/residuals'
    if os.path.isdir(PATH_TO_RESIDUALS) == False:
        os.mkdir(PATH_TO_RESIDUALS)

    coeffs = []
  


    for i in range(flux_masked.shape[0]):
        flux_i_out = flux_masked[i]
        time_i_out = time_masked[i]
        flux_i = flux[i]
        time_i = time[i]  
        # find the best linear fit
        k, b = np.polyfit(time_i_out, flux_i_out, deg=1)
        fit = k*time_i+b
        # divide the data by the best fit
        corrected_flux_i = flux_i/fit
        # fit for the points outside of transit
        fit_out = k*time_i_out+b 
        y = flux_i_out/fit_out
        # append the data to list
        corrected_flux.append(corrected_flux_i)
        stds.append(np.std(y))
        coeffs.append([k, b])
        

        fig = plt.figure()
        plt.plot(time_i, fit, 'r')
        plt.plot(time_i, flux_i, '.b')
        plt.savefig(PATH_TO_FIT + f'/transit_{i}')
        plt.close(fig)

        fig1 = plt.figure()
        plt.plot(time_i, corrected_flux_i, '.k')
        plt.xlabel("Time [days]")
        plt.ylabel("Relative flux")
        plt.savefig(PATH_TO_FIGURES + f'/transit_{i}')
        plt.close(fig1)

        fig2 = plt.figure()
        residuals = flux_i - fit
        plt.plot(time_i, residuals, '.k')
        plt.xlabel("Time [days]")
        plt.ylabel("Residuals")
        plt.savefig(PATH_TO_RESIDUALS + f'/transit_{i}')
        plt.close(fig2)
        
    corrected_flux = np.array(corrected_flux, dtype=object, copy=False)

    np.save(path + '/corrected_flux.npy', corrected_flux)
    np.save(path + '/stds.npy', stds)
    np.savetxt(path + '/coeffs.txt', coeffs)





#################################################################
# Read fits file
#################################################################

if int(cadence) == 2:
    with fits.open(path_to_data_file, mode="readonly") as hdulist:
        tess_bjds = hdulist[1].data['TIME']
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
        #print('Header:', hdulist[0].header)

    # Start figure and axis.
    fig, ax = plt.subplots()

    # Plot the timeseries in black circles.
    ax.plot(tess_bjds, pdcsap_fluxes, '.k')
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.savefig(path_to_fig + '/lc_'+f'{planet_name.replace(" ", "_")}')
    plt.close(fig)

    with fits.open(path_to_data_file, mode="readonly") as hdulist:
        aperture = hdulist[2].data

    # Start figure and axis.
    fig, ax = plt.subplots()
    # Display the pixels as an image.
    cax = ax.imshow(aperture, cmap=plt.cm.YlGnBu_r, origin="lower")
    # Add a color bar.
    cbar = fig.colorbar(cax)
    # Add a title to the plot.
    fig.suptitle("Aperture")
    plt.savefig(path_to_fig + '/selected_aperture'+f'{ planet_name.replace(" ", "_")}')



    time = tess_bjds

    m = np.isfinite(pdcsap_fluxes)
    time = np.ascontiguousarray(time[m])
    pdcsap_fluxes = np.ascontiguousarray(pdcsap_fluxes[m])

    ##############################################
    # Periodogram
    model = BoxLeastSquares(time, pdcsap_fluxes)
    periodogram = model.autopower(0.2)
    plt.plot(periodogram.period, periodogram.power)  
    plt.xlabel("Period [day]")
    plt.ylabel("Power")
    plt.text(10,2117,
        "period = {0:.4f} d".format(periodogram.period[np.argmax(periodogram.power)]))


 
    period_grid = np.exp(np.linspace(np.log(0.05), np.log(15), 50000))

    bls = BoxLeastSquares(time, pdcsap_fluxes)
    bls_power = bls.power(period_grid, 0.02, oversample=20)
 
    # Save the highest peak as the planet candidate
    index = np.argmax(bls_power.power)
    bls_period = bls_power.period[index]
    bls_t0 = bls_power.transit_time[index]
    bls_depth = bls_power.depth[index]
    transit_mask = bls.transit_mask(time, bls_period, 0.6*pl_trandur, bls_t0)


    x = np.ascontiguousarray(time, dtype=np.float64)
    y = np.ascontiguousarray(pdcsap_fluxes, dtype=np.float64) 


     
    # Plot the folded transit
     
    x_fold = (time - bls_t0 + 0.5 * bls_period) % bls_period - 0.5 * bls_period

    m = np.abs(x_fold) < N*pl_trandur
    transit_mask =  np.abs(x_fold) < 0.6*pl_trandur
    not_transit = ~transit_mask

    # folded data with transit masked:
    total_mask = m & not_transit
    flux_masked = pdcsap_fluxes[total_mask]
    time_masked = x_fold[total_mask]
    times_masked = time[total_mask] # times (not relative)
    # folded data with transit included:
    flux_folded = pdcsap_fluxes[m]
    time_folded = x_fold[m]
    times = time[m]
  

 

else:
    with fits.open(path_to_data_file, mode="readonly") as hdu: 
        tpf = hdu[1].data
        tpf_hdr = hdu[1].header

    texp = tpf_hdr["FRAMETIM"] * tpf_hdr["NUM_FRM"]
    texp /= 60.0 * 60.0 * 24.0
    time = tpf["TIME"]
    flux = tpf["FLUX"]
     

    m = np.any(np.isfinite(flux), axis=(1, 2)) & (tpf["QUALITY"] == 0)
    ref_time = 0.5 * (np.min(time[m]) + np.max(time[m]))
    #time = np.ascontiguousarray(time[m] - ref_time, dtype=np.float64) # time w.r.t. reference time
    time = time[m] #apply mask
    flux = np.ascontiguousarray(flux[m], dtype=np.float64) #store flux as contiguous array in memory
    mean_img = np.median(flux, axis=0) # for each pixel, calculate median flux over the period of observations
     
    plt.figure()
    plt.imshow(mean_img.T, cmap="gray_r")
    plt.title(f"{MISSION} image of {planet_name}")
    plt.xticks([])
    plt.yticks([]);
    plt.savefig(path_to_fig + '/image_of_'+f'{ planet_name.replace(" ", "_")}')
    #plt.show()

    #################################################################
    # Aperture selection
    #################################################################

    # Sort pixels  
    order = np.argsort(mean_img.flatten())[::-1]
     
    # Estimate the windowed scatter in a lightcurve
    def estimate_scatter_with_mask(mask):
        f = np.sum(flux[:, mask], axis=-1)
        #smooth data
        smooth = savgol_filter(f, 1001, polyorder=5)
        return 1e6 * np.sqrt(np.median((f / smooth - 1) ** 2))


    # Loop over pixels ordered by brightness and add them one-by-one
    # to the aperture
    masks, scatters = [], []
    for i in range(10, 100):
        msk = np.zeros_like(mean_img, dtype=bool)
        msk[np.unravel_index(order[:i], mean_img.shape)] = True
        scatter = estimate_scatter_with_mask(msk)
        masks.append(msk)
        scatters.append(scatter)

    # Choose the aperture that minimizes the scatter
    pix_mask = masks[np.argmin(scatters)]
     
    # Plot the selected aperture
    plt.imshow(mean_img.T, cmap="gray_r")
    plt.imshow(pix_mask.T, cmap="Reds", alpha=0.3)
    plt.title("selected aperture")
    plt.xticks([])
    plt.yticks([]);
    plt.savefig(path_to_fig + '/selected_aperture'+f'{ planet_name.replace(" ", "_")}')
    #plt.show()


     
    plt.figure(figsize=(10, 5))
    sap_flux = np.sum(flux[:, pix_mask], axis=-1)
    sap_flux = (sap_flux / np.median(sap_flux) - 1) * 1e3
    plt.plot(time, sap_flux, "k")
    plt.xlabel("time [days]")
    plt.ylabel("relative flux [ppt]")
    plt.title("raw light curve")
    plt.xlim(time.min(), time.max());
    plt.savefig(path_to_fig + '/raw_light_curve_'+f'{planet_name.replace(" ", "_")}')
    #plt.show()

    #####################################################################
    # De-trending (systematic and random noise sources)
    #####################################################################

    # Build the first order PLD basis
    X_pld = np.reshape(flux[:, pix_mask], (len(flux), -1))
    X_pld = X_pld / np.sum(flux[:, pix_mask], axis=-1)[:, None]

    # Build the second order PLD basis and run PCA to reduce the number of dimensions
    X2_pld = np.reshape(X_pld[:, None, :] * X_pld[:, :, None], (len(flux), -1))
    U, _, _ = np.linalg.svd(X2_pld, full_matrices=False)
    X2_pld = U[:, : X_pld.shape[1]]

    # Construct the design matrix and fit for the PLD model
    X_pld = np.concatenate((np.ones((len(flux), 1)), X_pld, X2_pld), axis=-1)
    XTX = np.dot(X_pld.T, X_pld)
    w_pld = np.linalg.solve(XTX, np.dot(X_pld.T, sap_flux))
    pld_flux = np.dot(X_pld, w_pld)

    # Plot the de-trended light curve
    plt.figure(figsize=(10, 5))
    plt.plot(time, sap_flux - pld_flux, "k")
    plt.xlabel("time [days]")
    plt.ylabel("de-trended flux [ppt]")
    plt.title("initial de-trended light curve")
    plt.xlim(time.min(), time.max());
    plt.savefig(path_to_fig + '/initial_de-trended_lc_of'+f'{planet_name.replace(" ", "_")}')
    #plt.show()



    #####################################################################
    # Periodogram
    #####################################################################

    model = BoxLeastSquares(time, sap_flux - pld_flux)
    periodogram = model.autopower(0.2) #0.2 - duration
    plt.plot(periodogram.period, periodogram.power)  
    plt.xlabel("Period [day]")
    plt.ylabel("Power")
    plt.text(10,2117,
        "period = {0:.4f} d".format(periodogram.period[np.argmax(periodogram.power)]))
    #print(periodogram.period[np.argmax(periodogram.power)])
     
    period_grid = np.exp(np.linspace(np.log(0.05), np.log(15), 50000))
     
    bls = BoxLeastSquares(time, sap_flux - pld_flux)
    bls_power = bls.power(period_grid, 0.02, oversample=20)
    plt.xlabel("time [days]")
    plt.ylabel("de-trended flux [ppt]")

    # Save the highest peak as the planet's period
    index = np.argmax(bls_power.power)
    bls_period = bls_power.period[index]
    bls_t0 = bls_power.transit_time[index]
     
    bls_depth = bls_power.depth[index]
    transit_mask = bls.transit_mask(time, bls_period, 0.2, bls_t0) #0.2 duration of the transit 

     

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the periodogram
    ax = axes[0]
    ax.axvline(np.log10(bls_period), color="C1", lw=5, alpha=0.8)
    ax.plot(np.log10(bls_power.period), bls_power.power, "k")
    ax.annotate(
        "period = {0:.4f} d".format(bls_period),
        (0, 1),
        xycoords="axes fraction",
        xytext=(5, -5),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=12,
    )
    ax.set_ylabel("bls power")
    ax.set_yticks([])
    ax.set_xlim(np.log10(period_grid.min()), np.log10(period_grid.max()))
    ax.set_xlabel("log10(period)")
     
    # Plot the folded transit
    ax = axes[1]
    x_fold = (time - bls_t0 + 0.5 * bls_period) % bls_period - 0.5 * bls_period

    m = np.abs(x_fold) < 0.4 #transit mask
    no_transit = ~transit_mask #no transit mask

    #####################################################################
    #plot
    #####################################################################

    ax.plot(x_fold[m], sap_flux[m] - pld_flux[m], ".k")
    # Overplot the phase binned light curve
    bins = np.linspace(-0.41, 0.41, 32)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=sap_flux - pld_flux)
    denom[num == 0] = 1.0
    ax.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, color="C1")

    ax.set_xlim(-0.3, 0.3)
    ax.set_ylabel("de-trended flux [ppt]")
    ax.set_xlabel("time since transit");
    plt.savefig(path_to_fig + '/de-trended_lc_of'+f'{planet_name.replace(" ", "_")}')
    #plt.show()


    #####################################################################
    # more de-trending
    #####################################################################

    XTX = np.dot(X_pld[no_transit].T, X_pld[no_transit])
    w_pld = np.linalg.solve(XTX, np.dot(X_pld[no_transit].T, sap_flux[no_transit]))
    pld_flux = np.dot(X_pld, w_pld)



    x = np.ascontiguousarray(time, dtype=np.float64)
    y = np.ascontiguousarray(sap_flux - pld_flux, dtype=np.float64)

    plt.figure(figsize=(10, 5))
    plt.plot(time, y, "k")
    plt.xlabel("time [days]")
    plt.ylabel("de-trended flux [ppt]")
    plt.title("final de-trended light curve")
    plt.xlim(time.min(), time.max());
    plt.savefig(path_to_fig + '/final_de-trended_lc_of'+f'{planet_name.replace(" ", "_")}')
    #plt.show()



     
    #####################################################################
    # fit
    #####################################################################

#    plt.figure(figsize=(10, 5))

 #   x_fold = (x - bls_t0 + 0.5 * bls_period) % bls_period - 0.5 * bls_period
 #   m = np.abs(x_fold) < 0.3
 #   plt.plot(x_fold[m], pld_flux[m], ".k", ms=4)

 #  bins = np.linspace(-0.5, 0.5, 60)
 #   denom, _ = np.histogram(x_fold, bins)
 #   num, _ = np.histogram(x_fold, bins, weights=pld_flux)
 #   denom[num == 0] = 1.0
 #   plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, color="C1", lw=2)
 #   plt.xlim(-0.2, 0.2)
 #   plt.xlabel("time since transit")
 #   plt.ylabel("PLD model flux");
    #plt.show()

    #####################################################################
    # save data of masked transits and folded transits
    #####################################################################


    

    pdcsap_fluxes = sap_flux - pld_flux
    #pdcsap_fluxes = sap_flux # this is data before any de-trending

    # folded data with transit masked:
    total_mask = m & no_transit
    flux_masked = pdcsap_fluxes[total_mask]
    time_masked = x_fold[total_mask]
    times_masked = time[total_mask] # times (not relative)
    # folded data with transit included:
    flux_folded = pdcsap_fluxes[m]
    time_folded = x_fold[m]
    times = time[m]




# save folded transits
np.savetxt(path_to_data + '/transit/times.txt', times)
np.savetxt(path_to_data + '/transit/time_folded.txt', time_folded)
np.savetxt(path_to_data + '/transit/flux.txt', flux_folded)

# save masked transits
np.savetxt(path_to_data + '/transit_masked/folded_time_masked.txt', time_masked)
np.savetxt(path_to_data + '/transit_masked/time_masked.txt', times_masked)
np.savetxt(path_to_data + '/transit_masked/flux_masked.txt', flux_masked)


# transits
select_transits(True,
                path_to_data + '/transit', 
                path_to_data + '/transit/times.txt',
                path_to_data + '/transit/time_folded.txt',
                path_to_data + '/transit/flux.txt', 
                path_to_fig)

# out of transits
select_transits(False,
                path_to_data + '/transit_masked', 
                path_to_data + '/transit_masked/time_masked.txt',
                path_to_data + '/transit_masked/folded_time_masked.txt',
                path_to_data + '/transit_masked/flux_masked.txt',
                path_to_fig)


detrend(path_to_data + '/transit', 
		path_to_data + '/transit/individual_time_array.npy', 
		path_to_data + '/transit/individual_flux_array.npy',
		path_to_data + '/transit_masked/individual_time_array.npy',
		path_to_data + '/transit_masked/individual_flux_array.npy',
        path_to_fig)


 
 
