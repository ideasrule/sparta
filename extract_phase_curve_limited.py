import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import batman
import astropy.stats
from algorithms import reject_beginning, bin_data, get_mad, \
    get_data_pickle, print_stats
import emcee
import argparse
from scipy.ndimage import median_filter
from configparser import ConfigParser
from emcee_methods import lnprob_limited as lnprob
from emcee_methods import get_batman_params, run_emcee
import time
import corner
import copy
import os.path
import pdb
from scipy.ndimage import uniform_filter

def reject_outliers(data, errors, times, x, sigma=4):
    detrended = data - median_filter(data, int(len(data) / 100))
    mask = astropy.stats.sigma_clip(detrended, sigma).mask
    #plt.plot(times, detrended, '.')
    #plt.plot(times[~mask], detrended[~mask], '.')
    #plt.show()    

    return data[~mask], errors[~mask], times[~mask], x[~mask]


def estimate_limb_dark(wavelength, filename="limb_dark.txt"):
    #return [0.00000,     0.847989,     -1.05762,     0.414449]
    
    all_wave, all_c1, all_c2, all_c3 = np.loadtxt(filename, usecols=(0, 2, 3, 4), skiprows=2, unpack=True)
    coeffs = [0,
              np.interp(wavelength, all_wave, all_c1),
              np.interp(wavelength, all_wave, all_c2),
              np.interp(wavelength, all_wave, all_c3)]
    return coeffs


def correct_lc(start_wave, end_wave, fluxes, errors, bjds, x, t0, t_secondary, per, rp, a, inc,
               limb_dark_coeffs, fp, C1, D1, C2, D2, output_file_prefix="chain",
               nwalkers=100, burn_in_runs=100, production_runs=1000, extra_phase_terms=False, output_txt="result.txt"):
    print("Median is", np.median(fluxes))
    fluxes /= np.median(fluxes)

    #We initialize transit and eclipse models so that lnprob doesn't have to.
    #Speeds up lnprob dramatically.
    initial_batman_params = get_batman_params(t0, per, rp, a, inc, limb_dark_coeffs, t_secondary=t_secondary)
    batman_params = copy.deepcopy(initial_batman_params)
    transit_model = batman.TransitModel(batman_params, bjds)
    eclipse_model = batman.TransitModel(batman_params, bjds, transittype='secondary')

    error_factor = 1.3
    print("Error factor", error_factor)
    if extra_phase_terms:
        initial_params = np.array([fp, C1, D1, C2, D2, rp, error_factor, 1, 0, 10, 0, 0])
        labels = ["Fp", "C1", "D1", "C2", "D2", "rp", "error", "Fstar", "A", "1/tau", "cx", "m"]
        rp_index = 5
    else:
        initial_params = np.array([fp, C1, D1, rp, error_factor, 1, 0, 10, 0, 10, 0, 0, 0])
        labels = ["Fp", "C1", "D1", "rp", "error", "Fstar", "Aramp", "1/tau", "cx", "m"]
        rp_index = 3

    #All arguments, aside from the parameters, that will be passed to lnprob
    w = 2*np.pi/per
    lnprob_args = (initial_batman_params, transit_model, eclipse_model, bjds, fluxes, errors, x, t0, None, extra_phase_terms)

    #Plot initial
    #residuals = lnprob(initial_params, *lnprob_args, plot_result=True, return_residuals=True)
    #plt.show()
    
    _, chain, lnprobs = run_emcee(lnprob, lnprob_args, initial_params, nwalkers, output_file_prefix, burn_in_runs, production_runs)
    length = len(chain)
    chain = chain[int(length/2):]
    lnprobs = lnprobs[int(length/2):]
    best_step = chain[np.argmax(lnprobs)]
    best_lnprob = lnprobs[np.argmax(lnprobs)]
    
    _, residuals = lnprob(best_step, *lnprob_args, plot_result=True, return_residuals=True, wavelength=(start_wave + end_wave) / 2)
    print("Best lnprob", best_lnprob)

    A = np.sqrt(chain[:,1]**2 + chain[:,2]**2)
    phi = np.arctan2(chain[:,2], chain[:,1]) * 180 / np.pi

    print_stats(A, "A")
    print_stats(phi, "phi")
    for i in range(chain.shape[1]):
        print_stats(chain[:,i], labels[i])

    plt.figure()
    corner.corner(chain, range=[0.99] * chain.shape[1], labels=labels)
    plt.show()
    
    night_Fp = chain[:,0] - 2*chain[:,1]
    print("Night Fp", np.percentile(night_Fp, 16), np.median(night_Fp), np.percentile(night_Fp, 84))
    if not os.path.exists(output_txt):
        with open(output_txt, "w") as f:
            f.write("#min_wavelength max_wavelength A_med A_lower_err A_upper_err phi_med phi_lower_err phi_upper_err Fp_med Fp_lower_err Fp_upper_err RpRs_med RpRs_lower_err RpRs_upper_err night_Fp_med night_Fp_lower_err night_Fp_upper_err lnprob\n")
    
    with open(output_txt, "a") as f:
        f.write("{} {} ".format(start_wave, end_wave))
        for var in (A, phi, chain[:,0], chain[:,rp_index], night_Fp):
            f.write("{} {} {} ".format(np.median(var), np.median(var) - np.percentile(var, 16), np.percentile(var, 84) - np.median(var)))
        f.write("{}".format(best_lnprob))
        f.write("\n")
    return chain, lnprobs

parser = argparse.ArgumentParser(description="Extracts phase curve and transit information from light curves")
parser.add_argument("config_file", help="Contains transit, eclipse, and phase curve parameters")
parser.add_argument("start_wave", type=float)
parser.add_argument("end_wave", type=float)
parser.add_argument("-b", "--bin-size", type=int, default=1, help="Bin size to use on data")
parser.add_argument("--burn-in-runs", type=int, default=1000, help="Number of burn in runs")
parser.add_argument("--production-runs", type=int, default=1000, help="Number of production runs")
parser.add_argument("--num-walkers", type=int, default=100, help="Number of walkers in the ensemble sampler")
parser.add_argument("-o", "--output", type=str, default="chain", help="Directory to store the chain and lnprob arrays")
parser.add_argument("--extra-phase-terms", action="store_true", help="Include C2 and D2 in phase curve fit")
parser.add_argument("-e", "--exclude", type=int, default=263, help="Number of integrations to trim")

args = parser.parse_args()
bjds, fluxes, flux_errors, wavelengths, _, x = get_data_pickle(args.start_wave, args.end_wave, args.exclude)
fluxes, flux_errors, bjds, x = reject_outliers(fluxes, flux_errors, bjds, x)
print("Wavelengths", wavelengths)


bin_size = args.bin_size
binned_fluxes = bin_data(fluxes, bin_size)
factor = np.median(binned_fluxes)
if factor <= 0:
    #This means there's no signal
    factor = 1
binned_fluxes /= factor
binned_errors = np.sqrt(bin_data(flux_errors**2, bin_size) / bin_size) / factor
binned_bjds = bin_data(bjds, bin_size)
binned_x = bin_data(x, bin_size)

binned_fluxes, binned_errors, binned_bjds, binned_x = reject_outliers(binned_fluxes, binned_errors, binned_bjds, binned_x)
delta_t = binned_bjds - np.median(binned_bjds)
coeffs = np.polyfit(delta_t, binned_x, 1)
smoothed_binned_x = np.polyval(coeffs, delta_t)

#get values from configuration file
default_section_name = "DEFAULT"
config = ConfigParser()
config.read(args.config_file)
items = dict(config.items(default_section_name))
limb_dark_coeffs = estimate_limb_dark(np.mean(wavelengths))
print("Found limb dark coeffs", limb_dark_coeffs)
print("# points", len(binned_fluxes))

chain, lnprobs = correct_lc(args.start_wave/1000, args.end_wave/1000, binned_fluxes, binned_errors, binned_bjds, binned_x - smoothed_binned_x, float(items["t0"]), float(items["t_secondary"]), float(items["per"]),
           float(items["rp"]), float(items["a"]), float(items["inc"]), limb_dark_coeffs,
           float(items["fp"]), float(items["c1"]), float(items["d1"]), float(items["c2"]), float(items["d2"]),
           args.output, args.num_walkers, args.burn_in_runs, args.production_runs, args.extra_phase_terms)

np.save("chain_{}_{}.npy".format(int(args.start_wave), int(args.end_wave)), chain)
np.save("lnprobs_{}_{}.npy".format(int(args.start_wave), int(args.end_wave)), lnprobs)
