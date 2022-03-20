import numpy as np
import sys
import matplotlib.pyplot as plt
import batman
from algorithms import reject_beginning, bin_data, get_mad, \
    get_data, print_stats
import emcee
import argparse
from configparser import ConfigParser
from emcee_methods import lnprob as lnprob
from emcee_methods import get_batman_params, run_emcee
import time
import corner

def correct_lc(wavelengths, fluxes, errors, bjds, t0, per, rp, a, inc,
               limb_dark_coeffs, fp, C1, D1, C2, D2, output_file_prefix="chain",
               nwalkers=100, burn_in_runs=100, production_runs=1000, extra_phase_terms=False):
    print("Median is", np.median(fluxes))
    fluxes /= np.median(fluxes)

    #We initialize transit and eclipse models so that lnprob doesn't have to.
    #Speeds up lnprob dramatically.
    initial_batman_params = get_batman_params(t0, per, rp, a, inc, limb_dark_coeffs)
    transit_model = batman.TransitModel(initial_batman_params, bjds)
    eclipse_model = batman.TransitModel(initial_batman_params, bjds, transittype='secondary')

    #First guess for PLD corrections based on PCA components
    error_factor = 0.2 #get_mad(fluxes) / np.median(errors)
    print("Error factor", error_factor)
    b = a*np.cos(inc*np.pi/180)
    if extra_phase_terms:
        initial_params = np.array([0, 0, fp, C1, D1, C2, D2, rp, a, b, error_factor, 0, 1])
    else:
        initial_params = np.array([0, 0, fp, C1, D1, rp, a, b, error_factor, 0, 1])

    #All arguments, aside from the parameters, that will be passed to lnprob
    w = 2*np.pi/per
    lnprob_args = (initial_batman_params, transit_model, eclipse_model, bjds, fluxes, errors, t0, extra_phase_terms)

    #Plot initial
    #residuals = lnprob(initial_params, *lnprob_args, plot_result=True, return_residuals=True)
    #plt.show()
    
    best_step, chain = run_emcee(lnprob, lnprob_args, initial_params, nwalkers, output_file_prefix, burn_in_runs, production_runs)
    print("Best step", best_step)
    residuals = lnprob(best_step, *lnprob_args, plot_result=True, return_residuals=True)
    chain = chain[int(len(chain)/2):]

    A = np.sqrt(chain[:,3]**2 + chain[:,4]**2)
    phi = np.arctan2(chain[:,4], chain[:,3]) * 180 / np.pi
    
    print_stats(t0 + chain[:,0], "transit")
    #print_stats(eclipse_phase, "eclipse_phase")
    print_stats(chain[:,1], "eclipse")
    print_stats(A, "A")
    print_stats(phi, "phi")
    print_stats(chain[:,2], "Fp")
    print_stats(chain[:,3], "C1")
    print_stats(chain[:,4], "D1")
    print_stats(chain[:,5], "rp")
    print_stats(chain[:,6], "a_star")
    print_stats(chain[:,7], "b")
    print_stats(np.arccos(chain[:,7]/chain[:,6])*180/np.pi, "inc")
    print_stats(chain[:,8], "error")
    print_stats(chain[:,9], "slope")
    plt.figure()
    corner.corner(chain, range=[0.99] * chain.shape[1], labels=["t0", "eclipse", "Fp", "C1", "D1", "rp", "a_star", "b", "error", "slope", "offset"])
    plt.show()

    with open("result.txt", "a") as f:
        f.write("{} {} ".format(wavelengths[-1], wavelengths[0]))
        for var in (A, phi, chain[:,2], chain[:,5]):
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
#plt.scatter(bjds, fluxes)
#plt.show()


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
           float(items["fp"]), float(items["c1"]), float(items["d1"]), float(items["c2"]), float(items["d2"]),
           args.output, args.num_walkers, args.burn_in_runs, args.production_runs, args.extra_phase_terms)
