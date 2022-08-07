import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import batman
from algorithms import reject_beginning, bin_data, get_mad, \
    get_data_pickle, print_stats
import emcee
import argparse
from configparser import ConfigParser
from emcee_methods import lnprob_transit_limited as lnprob
from emcee_methods import get_batman_params, run_emcee
from scipy.ndimage import median_filter
import astropy.stats
import time
import corner
import pdb
import copy
import os.path

def reject_outliers(data, errors, times, y, sigma=4):
    detrended = data - median_filter(data, int(len(data) / 100))
    mask = astropy.stats.sigma_clip(detrended, sigma).mask
    #plt.plot(times, detrended, '.')
    #plt.plot(times[~mask], detrended[~mask], '.')
    #plt.show()    

    return data[~mask], errors[~mask], times[~mask], y[~mask]


def estimate_limb_dark(wavelength, filename="limb_dark.txt"):
    all_wave, all_c1, all_c2, all_c3 = np.loadtxt(filename, usecols=(6, 8, 10, 12), skiprows=2, unpack=True)
    coeffs = [0,
              np.interp(wavelength, all_wave, all_c1),
              np.interp(wavelength, all_wave, all_c2),
              np.interp(wavelength, all_wave, all_c3)]
    return coeffs


def correct_lc(wavelengths, fluxes, errors, bjds, y, t0, per, rp, a, inc,
               limb_dark_coeffs, output_file_prefix="chain",
               nwalkers=100, burn_in_runs=100, production_runs=1000, output_txt="result.txt"):
    print("Median is", np.median(fluxes))
    fluxes /= np.median(fluxes)

    #We initialize transit and eclipse models so that lnprob doesn't have to.
    #Speeds up lnprob dramatically.
    initial_batman_params = get_batman_params(t0, per, rp, a, inc, limb_dark_coeffs)
    batman_params = copy.deepcopy(initial_batman_params)
    transit_model = batman.TransitModel(batman_params, bjds)
    error_factor = 2
    print("Error factor", error_factor)
    initial_params = np.array([rp**2, error_factor, 0, 1, 0, 0, 30./1440, 0, 4./1440])

    #All arguments, aside from the parameters, that will be passed to lnprob
    w = 2*np.pi/per
    lnprob_args = (initial_batman_params, transit_model, bjds, fluxes, errors, y, t0)

    #Plot initial
    #residuals = lnprob(initial_params, *lnprob_args, plot_result=True, return_residuals=True)
    #plt.show()
    
    best_step, chain = run_emcee(lnprob, lnprob_args, initial_params, nwalkers, output_file_prefix, burn_in_runs, production_runs)
    residuals = lnprob(best_step, *lnprob_args, plot_result=True, return_residuals=True)
    chain = chain[int(len(chain)/2):]

    print_stats(chain[:,0], "depth")
    print_stats(chain[:,1], "error")
    print_stats(chain[:,2], "slope")
    print_stats(chain[:,3], "Fstar")
    print_stats(chain[:,4], "c_y")
    print_stats(chain[:,5], "Aramp")
    print_stats(chain[:,6], "tau")
    plt.figure()
    
    corner.corner(chain, range=[0.99] * chain.shape[1], labels=["rp", "error", "slope", "Fstar", "c_y", "A1", "tau1", "A2", "tau2"])
    plt.show()
    
    if not os.path.exists(output_txt):
        with open(output_txt, "w") as f:
            f.write("#min_wavelength max_wavelength RpRs_med RpRs_lower_err RpRs_upper_err\n")
    
    with open(output_txt, "a") as f:
        f.write("{} {} ".format(wavelengths[0], wavelengths[-1]))
        for var in chain.T:
            f.write("{} {} {} ".format(np.median(var), np.median(var) - np.percentile(var, 16), np.percentile(var, 84) - np.median(var)))
        f.write("\n")



parser = argparse.ArgumentParser(description="Extracts phase curve and transit information from light curves")
parser.add_argument("config_file", help="Contains transit, eclipse, and phase curve parameters")
parser.add_argument("start_wave", type=float)
parser.add_argument("end_wave", type=float)
parser.add_argument("-b", "--bin-size", type=int, default=16, help="Bin size to use on data")
parser.add_argument("--burn-in-runs", type=int, default=1000, help="Number of burn in runs")
parser.add_argument("--production-runs", type=int, default=1000, help="Number of production runs")
parser.add_argument("--num-walkers", type=int, default=100, help="Number of walkers in the ensemble sampler")
parser.add_argument("-o", "--output", type=str, default="chain", help="Directory to store the chain and lnprob arrays")


args = parser.parse_args()

bjds, fluxes, flux_errors, wavelengths, y = get_data_pickle(args.start_wave, args.end_wave, 0)
fluxes, flux_errors, bjds, y = reject_outliers(fluxes, flux_errors, bjds, y)
print("wavelengths", wavelengths)

bin_size = args.bin_size
binned_fluxes = bin_data(fluxes, bin_size)
factor = np.median(binned_fluxes)
if factor <= 0:
    #This means there's no signal
    factor = 1
binned_fluxes /= factor
binned_errors = np.sqrt(bin_data(flux_errors**2, bin_size) / bin_size) / factor
binned_bjds = bin_data(bjds, bin_size)
binned_y = bin_data(y, bin_size)

#plt.scatter(binned_bjds, binned_fluxes)
binned_fluxes, binned_errors, binned_bjds, binned_y = reject_outliers(binned_fluxes, binned_errors, binned_bjds, binned_y)


#plt.scatter(bjds, fluxes)
#plt.figure()
#plt.scatter(binned_bjds, binned_fluxes)
#plt.figure()
#plt.scatter(binned_bjds, binned_y)
#plt.show()

#get values from configuration file
default_section_name = "DEFAULT"
config = ConfigParser()
config.read(args.config_file)
items = dict(config.items(default_section_name))
limb_dark_coeffs = estimate_limb_dark(np.mean(wavelengths))
print("Found limb dark coeffs", limb_dark_coeffs)

correct_lc(wavelengths, binned_fluxes, binned_errors, binned_bjds, binned_y, float(items["t0"]), float(items["per"]),
           float(items["rp"]), float(items["a"]), float(items["inc"]), limb_dark_coeffs,
           args.output, args.num_walkers, args.burn_in_runs, args.production_runs)
