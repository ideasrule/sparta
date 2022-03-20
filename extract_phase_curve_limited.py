import numpy as np
import sys
import matplotlib.pyplot as plt
import batman
from algorithms import reject_beginning, bin_data, get_mad, \
    get_data, print_stats
import emcee
import argparse
from configparser import ConfigParser
from emcee_methods import lnprob_limited as lnprob
from emcee_methods import get_batman_params, run_emcee
import time
import corner
import copy
import os.path

def estimate_limb_dark(wavelength, filename="limb_dark.txt"):
    all_wave, all_c1, all_c2, all_c3 = np.loadtxt(filename, usecols=(6, 8, 10, 12), skiprows=2, unpack=True)
    coeffs = [0,
              np.interp(wavelength, all_wave, all_c1),
              np.interp(wavelength, all_wave, all_c2),
              np.interp(wavelength, all_wave, all_c3)]
    return coeffs


def correct_lc(wavelengths, fluxes, errors, bjds, t0, t_secondary, per, rp, a, inc,
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

    error_factor = 0.2 #get_mad(fluxes) / np.median(errors)
    print("Error factor", error_factor)
    if extra_phase_terms:
        initial_params = np.array([fp, C1, D1, C2, D2, rp, error_factor, 1])
    else:
        initial_params = np.array([fp, C1, D1, rp, error_factor, 1])

    #All arguments, aside from the parameters, that will be passed to lnprob
    w = 2*np.pi/per
    lnprob_args = (initial_batman_params, transit_model, eclipse_model, bjds, fluxes, errors, t0, extra_phase_terms)

    #Plot initial
    #residuals = lnprob(initial_params, *lnprob_args, plot_result=True, return_residuals=True)
    #plt.show()
    
    best_step, chain = run_emcee(lnprob, lnprob_args, initial_params, nwalkers, output_file_prefix, burn_in_runs, production_runs)
    residuals = lnprob(best_step, *lnprob_args, plot_result=True, return_residuals=True, wavelength=np.mean(wavelengths))
    chain = chain[int(len(chain)/2):]

    A = np.sqrt(chain[:,1]**2 + chain[:,2]**2)
    phi = np.arctan2(chain[:,2], chain[:,1]) * 180 / np.pi
    
    print_stats(A, "A")
    print_stats(phi, "phi")
    print_stats(chain[:,0], "Fp")
    print_stats(chain[:,1], "C1")
    print_stats(chain[:,2], "D1")
    print_stats(chain[:,3], "rp")
    print_stats(chain[:,4], "error")
    print_stats(chain[:,5], "Fstar")
    plt.figure()
    corner.corner(chain, range=[0.99] * chain.shape[1], labels=["Fp", "C1", "D1", "rp", "error", "Fstar"])
    #plt.show()

    if not os.path.exists(output_txt):
        with open(output_txt, "w") as f:
            f.write("#min_wavelength max_wavelength A_med A_lower_err A_upper_err phi_med phi_lower_err phi_upper_err Fp_med Fp_lower_err Fp_upper_err RpRs_med RpRs_lower_err RpRs_upper_err\n")
    
    with open(output_txt, "a") as f:
        f.write("{} {} ".format(wavelengths[-1], wavelengths[0]))
        for var in (A, phi, chain[:,0], chain[:,3]):
            f.write("{} {} {} ".format(np.median(var), np.median(var) - np.percentile(var, 16), np.percentile(var, 84) - np.median(var)))
        f.write("\n")

parser = argparse.ArgumentParser(description="Extracts phase curve and transit information from light curves")
parser.add_argument("config_file", help="Contains transit, eclipse, and phase curve parameters")
parser.add_argument("start_bin", type=int)
parser.add_argument("end_bin", type=int)
parser.add_argument("-b", "--bin-size", type=int, default=1, help="Bin size to use on data")
parser.add_argument("--burn-in-runs", type=int, default=1000, help="Number of burn in runs")
parser.add_argument("--production-runs", type=int, default=1000, help="Number of production runs")
parser.add_argument("--num-walkers", type=int, default=100, help="Number of walkers in the ensemble sampler")
parser.add_argument("-o", "--output", type=str, default="chain", help="Directory to store the chain and lnprob arrays")
parser.add_argument("--extra-phase-terms", action="store_true", help="Include C2 and D2 in phase curve fit")

args = parser.parse_args()

bjds, fluxes, flux_errors, wavelengths = get_data(args.start_bin, args.end_bin)
#plt.scatter(bjds, fluxes)
bjds, fluxes, flux_errors = reject_beginning(bjds, fluxes, flux_errors)

#plt.show()


bin_size = args.bin_size
binned_fluxes = bin_data(fluxes, bin_size)
factor = np.median(binned_fluxes)
if factor <= 0:
    #This means there's no signal
    factor = 1
binned_fluxes /= factor
binned_errors = np.sqrt(bin_data(flux_errors**2, bin_size) / bin_size) / factor
binned_bjds = bin_data(bjds, bin_size)


#plt.errorbar(binned_bjds, binned_fluxes, yerr=binned_errors, fmt='.')
#plt.show()

#get values from configuration file
default_section_name = "DEFAULT"
config = ConfigParser()
config.read(args.config_file)
items = dict(config.items(default_section_name))
limb_dark_coeffs = estimate_limb_dark(np.mean(wavelengths))
print("Found limb dark coeffs", limb_dark_coeffs)

correct_lc(wavelengths, binned_fluxes, binned_errors, binned_bjds, float(items["t0"]), float(items["t_secondary"]), float(items["per"]),
           float(items["rp"]), float(items["a"]), float(items["inc"]), limb_dark_coeffs,
           float(items["fp"]), float(items["c1"]), float(items["d1"]), float(items["c2"]), float(items["d2"]),
           args.output, args.num_walkers, args.burn_in_runs, args.production_runs, args.extra_phase_terms)
