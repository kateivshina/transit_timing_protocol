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
parser.add_argument('--mass')
parser.add_argument('--parent_dir')
parser.add_argument('--path_to_data_file')

args = parser.parse_args()
print(args)
 
ID = 'TIC146264536'
MISSION = args.mission
planet_name = args.planet
cadence = args.cadence
path_to_data_file =args.path_to_data_file
# Path 
parent_dir = args.parent_dir
directory = planet_name.replace(" ", "_") 
path = f'{parent_dir}' + f'/{directory}' #os.path.join(parent_dir, directory) 

if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(path +'/figures'):
    os.mkdir(path +'/figures')

path_to_fig = path +'/figures'

#################################################################
# Read fits file
#################################################################


 
#wasp-12
tpf_file = path_to_data_file

if cadence == 2:
    pass

else:
    #with tpf_file.hdu as hdu:
    with fits.open(tpf_file, mode="readonly") as hdu: 
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
    #plt.title("{} image of {}".format(MISSION, planet_name))
    plt.title(f"{MISSION} image of {planet_name}")
    plt.xticks([])
    plt.yticks([]);
    plt.savefig(path_to_fig + '/image_of_'+f'{ planet_name.replace(" ", "_")}')
    plt.show()

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
    plt.show()


     
    plt.figure(figsize=(10, 5))
    sap_flux = np.sum(flux[:, pix_mask], axis=-1)
    sap_flux = (sap_flux / np.median(sap_flux) - 1) * 1e3
    plt.plot(time, sap_flux, "k")
    plt.xlabel("time [days]")
    plt.ylabel("relative flux [ppt]")
    plt.title("raw light curve")
    plt.xlim(time.min(), time.max());
    plt.savefig(path_to_fig + '/raw_light_curve_'+f'{planet_name.replace(" ", "_")}')
    plt.show()

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
    plt.show()



    #####################################################################
    # Periodogram
    #####################################################################

    model = BoxLeastSquares(time, sap_flux - pld_flux)
    periodogram = model.autopower(0.2) #0.2 - duration?
    plt.plot(periodogram.period, periodogram.power)  
    plt.xlabel("Period [day]")
    plt.ylabel("Power")
    plt.text(10,2117,
        "period = {0:.4f} d".format(periodogram.period[np.argmax(periodogram.power)]))
    print(periodogram.period[np.argmax(periodogram.power)])
     
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
    plt.show()


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
    plt.show()



     
    #####################################################################
    # fit
    #####################################################################

    plt.figure(figsize=(10, 5))

    x_fold = (x - bls_t0 + 0.5 * bls_period) % bls_period - 0.5 * bls_period
    m = np.abs(x_fold) < 0.3
    plt.plot(x_fold[m], pld_flux[m], ".k", ms=4)

    bins = np.linspace(-0.5, 0.5, 60)
    denom, _ = np.histogram(x_fold, bins)
    num, _ = np.histogram(x_fold, bins, weights=pld_flux)
    denom[num == 0] = 1.0
    plt.plot(0.5 * (bins[1:] + bins[:-1]), num / denom, color="C1", lw=2)
    plt.xlim(-0.2, 0.2)
    plt.xlabel("time since transit")
    plt.ylabel("PLD model flux");
    #plt.show()

    #####################################################################
    # save data of masked transits and folded transits
    #####################################################################


    path = path + '/data'

    if not os.path.isdir(path):
        os.mkdir(path)
        os.mkdir(path + '/transit_masked')
        os.mkdir(path + '/transit')

    pdcsap_fluxes = sap_flux - pld_flux
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
    np.savetxt(path + '/transit/times.txt', times)
    np.savetxt(path + '/transit/flux.txt', flux_folded)
    np.savetxt(path + '/transit/time_folded.txt', time_folded)

    # save masked transits
    np.savetxt(path + '/transit_masked/folded_time_masked.txt', time_masked)
    np.savetxt(path + '/transit_masked/time_masked.txt', times_masked)
    np.savetxt(path + '/transit_masked/flux_masked.txt', flux_masked)



 
