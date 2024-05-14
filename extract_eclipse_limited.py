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
from emcee_methods import lnprob_eclipse_limited as lnprob
from emcee_methods import get_batman_params, run_emcee
import time
import corner

def correct_lc(start_wave, end_wave, fluxes, errors, bjds, y, x, t0, t_secondary, per, rp, a, inc,
               limb_dark_coeffs, fp, output_file_prefix="chain",
               nwalkers=100, burn_in_runs=100, production_runs=1000, output_txt="result.txt"):
    print("Median is", np.median(fluxes))
    fluxes /= np.median(fluxes)
    #plt.plot(fluxes)
    #plt.show()
    
    initial_batman_params = get_batman_params(t0, per, rp, a, inc, limb_dark_coeffs, limb_dark_law="quadratic", t_secondary=t_secondary)
    transit_model = batman.TransitModel(initial_batman_params, bjds, transittype='secondary')
    
    #First guess for PLD corrections based on PCA components
    error_factor = 1.1 #Initial guess; slightly larger than 1 is usually good
    print("Error factor", error_factor)
    b = a*np.cos(inc*np.pi/180)
    initial_params = np.array([fp, error_factor, 1, 0, 1./24, 0, 0, 0, 0])
    labels = ["Fp", "error", "Fstar", "Aramp", "tau", "cy", "cx", "m", "eclipse_offset"]

    #All arguments, aside from the parameters, that will be passed to lnprob
    w = 2*np.pi/per
    lnprob_args = (initial_batman_params, transit_model, bjds, fluxes, errors, y, x, t_secondary, (start_wave + end_wave) / 2)
    
    _, chain, lnprobs = run_emcee(lnprob, lnprob_args, initial_params, nwalkers, output_file_prefix, burn_in_runs, production_runs)
    length = len(chain)
    chain = chain[int(length/2):]
    lnprobs = lnprobs[int(length/2):]
    best_step = chain[np.argmax(lnprobs)]
    best_lnprob = lnprobs[np.argmax(lnprobs)]    
    
    print("Best step", best_step)
    _, residuals = lnprob(best_step, *lnprob_args, plot_result=True, return_residuals=True)
    
    for i in range(chain.shape[1]):
        print_stats(chain[:,i], labels[i])

    plt.figure()
    corner.corner(chain, range=[0.99] * chain.shape[1], labels=labels)
    plt.show()

    
    with open(output_txt, "a") as f:
        f.write("{} {} ".format(start_wave, end_wave))
        for var in (chain[:,0],):
            f.write("{} {} {} ".format(np.median(var), np.median(var) - np.percentile(var, 16), np.percentile(var, 84) - np.median(var)))
        f.write("{}".format(best_lnprob))
        f.write("\n")

    return chain, lnprobs
            
parser = argparse.ArgumentParser(description="Extracts phase curve and transit information from light curves")
parser.add_argument("config_file", help="Contains transit parameters")
parser.add_argument("start_wave", type=float)
parser.add_argument("end_wave", type=float)
parser.add_argument("-b", "--bin-size", type=int, default=1, help="Bin size to use on data")
parser.add_argument("--burn-in-runs", type=int, default=2000, help="Number of burn in runs")
parser.add_argument("--production-runs", type=int, default=1000, help="Number of production runs")
parser.add_argument("--num-walkers", type=int, default=100, help="Number of walkers in the ensemble sampler")
parser.add_argument("-o", "--output", type=str, default="chain", help="Directory to store the chain and lnprob arrays")
parser.add_argument("-e", "--exclude-beginning", type=int, default=1122, help="How many integrations to exclude at the beginning")
parser.add_argument("--white-lc", type=str, default="white_lightcurve.txt", help="Location of the white light curve, with astro model")

args = parser.parse_args()

bjds, fluxes, flux_errors, wavelengths, y, x, _ = get_data_pickle(args.start_wave, args.end_wave, args.exclude_beginning)

bin_size = args.bin_size
binned_fluxes = bin_data(fluxes, bin_size)
factor = np.median(binned_fluxes)
binned_fluxes /= factor
binned_errors = np.sqrt(bin_data(flux_errors**2, bin_size) / bin_size) / factor
binned_bjds = bin_data(bjds, bin_size)
binned_y = bin_data(y, bin_size)
binned_x = bin_data(x, bin_size)

#Divide by white systematics
if args.white_lc is not None:
    white_bjds, white_flux, white_err, white_astro = np.loadtxt(args.white_lc, usecols=(0,1,2,4), unpack=True)
    binned_fluxes /= np.interp(binned_bjds,
                               bin_data(white_bjds, bin_size),
                               bin_data(white_flux / white_astro, bin_size))
    print("Dividing by white flux")

delta_t = binned_bjds - np.median(binned_bjds)
coeffs = np.polyfit(delta_t, binned_y, 1)
smoothed_binned_y = np.polyval(coeffs, delta_t)

coeffs = np.polyfit(delta_t, binned_x, 1)
smoothed_binned_x = np.polyval(coeffs, delta_t)

print("Num points", len(binned_fluxes))
#get values from configuration file
default_section_name = "DEFAULT"
config = ConfigParser()
config.read(args.config_file)
items = dict(config.items(default_section_name))

chain, lnprobs = correct_lc(args.start_wave/1000, args.end_wave/1000, binned_fluxes, binned_errors, binned_bjds, binned_y - smoothed_binned_y, binned_x - smoothed_binned_x, float(items["t0"]), float(items["t_secondary"]), float(items["per"]),
           float(items["rp"]), float(items["a"]), float(items["inc"]), eval(items["limb_dark_coeffs"]), float(items["fp"]),
           args.output, args.num_walkers, args.burn_in_runs, args.production_runs)
np.save("chain_{}_{}.npy".format(int(args.start_wave), int(args.end_wave)), chain)
np.save("lnprobs_{}_{}.npy".format(int(args.start_wave), int(args.end_wave)), lnprobs)
