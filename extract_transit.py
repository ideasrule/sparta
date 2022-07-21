import numpy as np
import sys
import matplotlib.pyplot as plt
import batman
from algorithms import reject_beginning, bin_data, get_mad, \
    get_data_txt, print_stats
import emcee
import argparse
from configparser import ConfigParser
from emcee_methods import lnprob_transit as lnprob
from emcee_methods import get_batman_params, run_emcee
import time
import corner
import pdb

def correct_lc(wavelengths, fluxes, errors, bjds, t0, per, rp, a, inc,
               limb_dark_coeffs, output_file_prefix="chain",
               nwalkers=100, burn_in_runs=100, production_runs=1000):
    print("Median is", np.median(fluxes))
    fluxes /= np.median(fluxes)

    #We initialize transit and eclipse models so that lnprob doesn't have to.
    #Speeds up lnprob dramatically.
    initial_batman_params = get_batman_params(t0, per, rp, a, inc, limb_dark_coeffs)
    transit_model = batman.TransitModel(initial_batman_params, bjds)

    #First guess for PLD corrections based on PCA components
    error_factor = 1.1 #Initial guess; slightly larger than 1 is usually good
    print("Error factor", error_factor)
    b = a*np.cos(inc*np.pi/180)
    initial_params = np.array([0, rp, a, b, error_factor, 0, 1, 0, 1./24, 0, 0.05/24])

    #All arguments, aside from the parameters, that will be passed to lnprob
    w = 2*np.pi/per
    lnprob_args = (initial_batman_params, transit_model, bjds, fluxes, errors, t0)

    #Plot initial
    #residuals = lnprob(initial_params, *lnprob_args, plot_result=True, return_residuals=True)
    #plt.show()
    
    best_step, chain = run_emcee(lnprob, lnprob_args, initial_params, nwalkers, output_file_prefix, burn_in_runs, production_runs)
    print("Best step", best_step)
    residuals = lnprob(best_step, *lnprob_args, plot_result=True, return_residuals=True)
    print("STD of residuals", np.std(residuals))
    chain = chain[int(len(chain)/2):]

    print_stats(t0 + chain[:,0], "transit")
    print_stats(chain[:,1], "rp")
    print_stats(chain[:,2], "a_star")
    print_stats(chain[:,3], "b")
    print_stats(chain[:,4], "error")
    print_stats(chain[:,5], "slope")
    plt.figure()
    corner.corner(chain, range=[0.99] * chain.shape[1], labels=["t0", "rp", "a_star", "b", "error", "slope", "Fstar", "A", "tau", "A2", "tau2"])
    plt.show()

    with open("result.txt", "a") as f:
        f.write("{} {} ".format(wavelengths[-1], wavelengths[0]))
        for var in (chain[:,0], chain[:,1], chain[:,2], chain[:,3]):
            f.write("{} {} {} ".format(np.median(var), np.median(var) - np.percentile(var, 16), np.percentile(var, 84) - np.median(var)))
        f.write("\n")
    pdb.set_trace()

parser = argparse.ArgumentParser(description="Extracts phase curve and transit information from light curves")
parser.add_argument("config_file", help="Contains transit, eclipse, and phase curve parameters")
parser.add_argument("start_bin", type=int)
parser.add_argument("end_bin", type=int)
parser.add_argument("-b", "--bin-size", type=int, default=16, help="Bin size to use on data")
parser.add_argument("--burn-in-runs", type=int, default=1000, help="Number of burn in runs")
parser.add_argument("--production-runs", type=int, default=1000, help="Number of production runs")
parser.add_argument("--num-walkers", type=int, default=100, help="Number of walkers in the ensemble sampler")
parser.add_argument("-o", "--output", type=str, default="chain", help="Directory to store the chain and lnprob arrays")


args = parser.parse_args()

bjds, fluxes, flux_errors, wavelengths = get_data_txt(args.start_bin, args.end_bin)

bin_size = args.bin_size
binned_fluxes = bin_data(fluxes, bin_size)
factor = np.median(binned_fluxes)
binned_fluxes /= factor
binned_errors = np.sqrt(bin_data(flux_errors**2, bin_size) / bin_size) / factor
binned_bjds = bin_data(bjds, bin_size)

#get values from configuration file
default_section_name = "DEFAULT"
config = ConfigParser()
config.read(args.config_file)
items = dict(config.items(default_section_name))

correct_lc(wavelengths, binned_fluxes, binned_errors, binned_bjds, float(items["t0"]), float(items["per"]),
           float(items["rp"]), float(items["a"]), float(items["inc"]), eval(items["limb_dark_coeffs"]),
           args.output, args.num_walkers, args.burn_in_runs, args.production_runs)
