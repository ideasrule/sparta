import sys
import numpy as np
import emcee
import batman
import pickle
from astropy.stats import sigma_clip
from read_config import *
from emcee_methods import *
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import corner
from dynesty import NestedSampler
import ast
import pandas as pd
import os
parser = argparse.ArgumentParser(description="Extracts phase curve and transit information from light curves")
parser.add_argument("config_file", help="Contains transit, eclipse, and phase curve parameters")
args = parser.parse_args()



def get_data_pickle(min_wavelength, max_wavelength, bad_intervals=[[0,300]], filename="data.pkl"):
    result = pickle.load(open(filename, "rb"))
    cond = np.logical_and(result["wavelengths"] >= min_wavelength/1000,
                          result["wavelengths"] < max_wavelength/1000)
    mask = np.ones(result["data"].shape[0], dtype=bool)
    bad_intervals = ast.literal_eval(bad_intervals)
    bad_intervals = [[int(start), int(end)] for start, end in bad_intervals]
    for i in range(len(bad_intervals)):
        interval = bad_intervals[i]
        start = interval[0]
        end = interval[1]
        mask[start:end] = False
    ix = np.ix_(mask, cond)
    data = np.sum(result["data"][ix], axis=1)

    var = result["errors"]**2
    errors = np.sqrt(np.sum(var[ix], axis=1))
    median = np.median(data)
    data /= median
    errors /= median
    y = result["y"][mask]
    x = result["x"][mask]
    time = result["times"][mask]
    wavelength = result["wavelengths"][cond]
    
    #import pdb
    #pdb.set_trace()

    return time, data, errors, wavelength, y, x

def bin_lightcurve(time, flux, bin_size, err=None):
    """
    Bin the light curve data.

    Parameters:
    - time: array of time values
    - flux: array of flux values
    - bin_size: number of points in each bin
    - err: optional, array of errors for the flux values

    Returns:
    - binned_time: binned time values
    - binned_flux: binned flux values
    - binned_err: optional, binned errors for the flux values
    """
    n_bins = len(time) // bin_size
    binned_time = np.zeros(n_bins)
    binned_flux = np.zeros(n_bins)
    
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size
        binned_time[i] = np.mean(time[start:end])
        binned_flux[i] = np.mean(flux[start:end])
    
    if err is not None:
        binned_err = np.sqrt(np.sum(err[start:end]**2)) / bin_size
        return binned_time, binned_flux, binned_err
    
    return binned_time, binned_flux
    
def reject_outliers(time, data, error, sigma=20):
    """Reject outliers using astropy-based method."""

    mask = sigma_clip(data, sigma=sigma, maxiters=5, masked=True).mask
    if error is not None:
        mask |= sigma_clip(error, sigma=sigma, maxiters=5, masked=True).mask

    return mask

def load_data_pkl(data_path, start_wave, end_wave, exclude=[[0,300]]):
    bjds, fluxes, flux_errors, wavelengths, y, x = get_data_pickle(start_wave, end_wave, bad_intervals = exclude, filename=data_path)

    factor = np.median(fluxes)
    fluxes /= factor
    
    delta_t = bjds - np.median(bjds)
    coeffs = np.polyfit(delta_t, x, 1)
    smoothed_x = np.polyval(coeffs, delta_t)
    
    coeffs = np.polyfit(delta_t, y, 1)
    smoothed_y = np.polyval(coeffs, delta_t)

    mask = reject_outliers(bjds, fluxes, flux_errors, sigma=20)
    deltax = x - smoothed_x
    deltay = y - smoothed_y
    return wavelengths, bjds[~mask], fluxes[~mask], flux_errors[~mask], deltax[~mask], deltay[~mask]
    
    
def load_data_csv(data_path, start_wave, end_wave, exclude=[[0,300]]):

    df = pd.read_csv(data_path)
    N = len(df)

    mask = np.ones(N, dtype=bool)

    if isinstance(exclude, str):
        try:
            exclude = ast.literal_eval(exclude)
        except Exception as e:
            raise RuntimeError(f"Failed to parse exclude interval: {e}")
    for start, end in exclude:
        print(start, end)
        start = max(0, int(start))
        end   = min(N, int(end))
        mask[start:end] = False
        
    if "opt" in data_path:
        bjds, fluxes, flux_errors, x, y, xw, yw = (
        df["time"].to_numpy()[mask],
        df["flux_opt"].to_numpy()[mask],
        df["error_opt"].to_numpy()[mask],
        df["xc"].to_numpy()[mask],
        df["yc"].to_numpy()[mask],
        df["xwidth"].to_numpy()[mask],
        df["ywidth"].to_numpy()[mask],
    )
    else:
    	bjds, fluxes, flux_errors, x, y, xw, yw = (
        df["time"].to_numpy()[mask],
        df["flux_subbkg"].to_numpy()[mask],
        df["error"].to_numpy()[mask],
        df["xc"].to_numpy()[mask],
        df["yc"].to_numpy()[mask],
        df["xwidth"].to_numpy()[mask],
        df["ywidth"].to_numpy()[mask],
    )


    clean_flux = sigma_clip(fluxes, sigma=5)
    mask = clean_flux.mask
    fluxes = fluxes[~mask]
    factor = np.median(fluxes)
    fluxes /= factor
    bjds = bjds[~mask]
    flux_errors = flux_errors[~mask]
    flux_errors /= factor
    x = x[~mask]
    y = y[~mask]
    xw = xw[~mask]
    yw = yw[~mask]

    delta_t = bjds - np.median(bjds)
    coeffs = np.polyfit(delta_t, y, 1)
    smoothed_y = np.polyval(coeffs, delta_t)

    coeffs = np.polyfit(delta_t, x, 1)
    smoothed_x = np.polyval(coeffs, delta_t)
    
    deltax = x - smoothed_x
    deltay = y - smoothed_y
    
    deltaxw = xw - np.median(xw)
    deltayw = yw - np.median(yw)
    wavelengths = [start_wave, end_wave]
    return wavelengths, bjds, fluxes, flux_errors, deltax, deltay, deltaxw, deltayw


def build_initial_batman_params(param_info, free_dict, fixed_dict):
    """
    Construct a batman.TransitParams() object using both free and fixed fields from the config.
    You must ensure that any geometry‐related parameters (e.g. per, ecc, w, limb_dark) appear in
    either free_dict or fixed_dict.

    Returns a batman.TransitParams instance with at least:
        per, ecc, w, limb_dark, u (two‐coefficient list if using 'kipping2013'), and any other
        fields you require (e.g. Rs, Ms, etc.).  The fields that are free will be overwritten
        inside lnprob_wrapper during each likelihood evaluation.
    """
    tp = batman.TransitParams()
    if "t0" in free_dict:
        tp.t0 = free_dict["t0"]
    else:
        tp.t0 = fixed_dict["t0"]
    if "t_secondary" in free_dict:
        tp.t_secondary = free_dict["t_secondary"]
    else:
        tp.t_secondary = fixed_dict["t_secondary"] 
    if "rp" in free_dict:
        tp.rp = free_dict["rp"]
    else:
        tp.rp = fixed_dict["rp"]
    if "a_star" in free_dict:
        tp.a = free_dict["a_star"]
    else:
        tp.a = fixed_dict["a_star"]
    if "per" in free_dict:
        tp.per = free_dict["per"]
    else:
        tp.per = fixed_dict["per"]
    if "sqrt_ecosw" in free_dict:
        ecosw = free_dict["sqrt_ecosw"]**2
    else:
        ecosw = fixed_dict["sqrt_ecosw"]**2
    if "sqrt_esinw" in free_dict:
        esinw = free_dict["sqrt_esinw"]**2
    else:
        esinw = fixed_dict["sqrt_esinw"]**2
    if 'inc'in free_dict:
        tp.inc = free_dict["inc"]
    elif 'inc' in fixed_dict:
        tp.inc = fixed_dict["inc"]
    if 'b' in free_dict:
        tp.inc = np.arccos(free_dict["b"]/tp.a) * 180/np.pi
    elif 'b' in fixed_dict:
        tp.inc = np.arccos(fixed_dict["b"]/tp.a) * 180/np.pi
    tp.ecc = ecosw ** 2 + esinw ** 2
    tp.w = np.arctan2(esinw, ecosw) * 180.0 / np.pi  # w in radians

    tp.fp = free_dict["fp"]

    # Limb‐darkening law
    if fixed_dict["limb_dark"] == "kipping2013":
        fixed_dict["limb_dark"] = "quadratic"  # Kipping2013 is a quadratic law in batman
        if "q1" in free_dict:
            q1 = free_dict["q1"]
            q2 = free_dict["q2"]
        else:
            q1 = fixed_dict["q1"]
            q2 = fixed_dict["q2"]
        
        u1 = 2*np.sqrt(q1) * q2
        u2 = np.sqrt(q1) * (1 - 2*q2)
        tp.u = [u1, u2]
    elif fixed_dict["limb_dark"] == "quadratic":
        # Quadratic law: u1, u2 are free or fixed
        if "q1" in free_dict:
            tp.u = [free_dict["q1"], free_dict["q2"]]
        else:
            tp.u = [fixed_dict["q1"], fixed_dict["q2"]]
    elif fixed_dict["limb_dark"] == "uniform":
        tp.u = []
    tp.limb_dark = fixed_dict["limb_dark"]

    
    # If you have any other fixed fields that batman.Params requires (e.g. Rs, Ms), set them here.
    # For example, if “Rs” was fixed in config:
    if "Rs" in fixed_dict:
        tp.Rs = fixed_dict["Rs"]
    if "Ms" in fixed_dict:
        tp.Ms = fixed_dict["Ms"]


    return tp


def main():

    # ------------------------------------------------------------------------
    # 1) Read the configuration file
    # ------------------------------------------------------------------------
    try:
        param_info = read_config(args.config_file)
    except Exception as e:
        print(f"Error reading configuration file '{args.config_file}':\n  {e}")
        sys.exit(1)

    # Split into dictionaries of free vs fixed parameters
    free_dict  = get_free_params(param_info)
    fixed_dict = get_fixed_params(param_info)
    priors     = get_priors(param_info)

    # List of names of free parameters (alphabetical order)
    free_names  = sorted(free_dict.keys())
    fixed_names = sorted(fixed_dict.keys())
    ndim        = len(free_names)

    print("=== Free parameters (initial guess & prior) ===")
    for nm in free_names:
        print(f"  {nm:<12s}  init={free_dict[nm]:<12}   prior={priors[nm]}")
    print("\n=== Fixed/Shared/Independent parameters ===")
    for nm in fixed_names:
        print(f"  {nm:<12s} = {fixed_dict[nm]}")

    initial_theta = np.zeros(ndim, dtype=float)
    for i, nm in enumerate(free_names):
        init_val = free_dict[nm]
        prior_spec = priors[nm]
        if prior_spec is None:
            raise RuntimeError(f"Parameter '{nm}' is free but has no prior!")

        if prior_spec["type"] == "LU":
            # Store the exponent = log10(initial_value)
            initial_theta[i] = init_val
        else:
            # For U or N, store the parameter value directly
            initial_theta[i] = init_val


    try:
        datapath_field = fixed_dict.get("data_path")
        joint_fit = str(fixed_dict.get("joint_fit"))
        if joint_fit == "True":
            joint_fit = True
        elif joint_fit == "False":
            joint_fit = False
        N_visits = int(fixed_dict.get("N_visits", 1))
        start_wave = fixed_dict.get("start_wave")
        end_wave = fixed_dict.get("end_wave")
        exclude = fixed_dict.get("exclude")
        try:
            parsed_exclude = ast.literal_eval(exclude) if isinstance(exclude, str) else exclude
        except Exception as e:
            raise RuntimeError(f"Failed to parse 'exclude' from config: {e}")

        if joint_fit:
            data_paths = [p.strip() for p in datapath_field.split(",") if p.strip()]
            if len(data_paths) != N_visits:
                raise RuntimeError(f"joint_fit=True but number of entries in data_path ({len(data_paths)}) != N_visits ({N_visits})")

            if not isinstance(parsed_exclude, list):
                raise RuntimeError(f"joint_fit=True but 'exclude' is not a list: got {type(parsed_exclude)}")
            if len(parsed_exclude) != N_visits:
                raise RuntimeError(f"joint_fit=True but number of entries in exclude ({len(parsed_exclude)}) != N_visits ({N_visits})")
            exclude_list = parsed_exclude  # e.g., [[0,800], [100,900], ...]
        else:
            # Single visit: allow either [[a,b]] or [a,b] or empty
            data_paths = [datapath_field.strip()]
            if parsed_exclude and isinstance(parsed_exclude[0], list):
                exclude_list = [parsed_exclude[0]]
            else:
                exclude_list = [parsed_exclude] if parsed_exclude else [[]]

        lc_savepath = fixed_dict.get("lc_savepath", "lightcurve.txt")
        params_savepath = fixed_dict.get("best_fit_savepath", "params.txt")

        if os.path.exists(lc_savepath): os.remove(lc_savepath)
        if os.path.exists(params_savepath): os.remove(params_savepath)

        all_bjds = []
        all_fluxes = []
        all_errors = []
        all_delta_x = []
        all_delta_y = []
        all_delta_xw = []
        all_delta_yw = []
        all_wavelengths = []
        for idx, path in enumerate(data_paths, start=1):
            this_exclude = exclude_list[idx - 1]
            print(f"Loading data from {path} for wavelengths {start_wave} to {end_wave} with exclusion {this_exclude} (visit {idx})")
            if '.pkl' in path:
                wavelengths, bjds, fluxes, errors, delta_x, delta_y, delta_xw, delta_yw = load_data_pkl(path, start_wave, end_wave, [this_exclude])
            elif '.csv' in path:
                wavelengths, bjds, fluxes, errors, delta_x, delta_y, delta_xw, delta_yw = load_data_csv(path, start_wave, end_wave, [this_exclude])
            else:
                raise RuntimeError(f"Un-recognized data type for path '{path}'")
        
            all_wavelengths.append(wavelengths)
            all_bjds.append(bjds)
            all_fluxes.append(fluxes)
            all_errors.append(errors)
            all_delta_x.append(delta_x)
            all_delta_y.append(delta_y)
            all_delta_xw.append(delta_xw)
            all_delta_yw.append(delta_yw)
        bjds = np.concatenate(all_bjds)
        fluxes = np.concatenate(all_fluxes)
        errors = np.concatenate(all_errors)
        delta_x = np.concatenate(all_delta_x)
        delta_y = np.concatenate(all_delta_y)
        delta_xw = np.concatenate(all_delta_xw)
        delta_yw = np.concatenate(all_delta_yw)
        
    except Exception as e:
        print(f"Error loading data file '{datapath_field}':\n  {e}")
        sys.exit(1)
    
    visit_indices = []
    for vi, arr in enumerate(all_bjds, start=1):
        visit_indices.extend([vi] * len(arr))
    visit_indices = np.array(visit_indices)

    plt.figure()
    plt.errorbar(bjds - bjds[0], fluxes, yerr = errors, ms = 1, color='gray', alpha=0.2)
    b = len(bjds) // 15
    plt.errorbar(uniform_filter(bjds - bjds[0], b)[::b], uniform_filter(fluxes, b)[::b], yerr = errors[::b] / np.sqrt(b), fmt='.')    
    plt.show()
    
    if "t0" in free_dict:
        initial_t0 = free_dict["t0"]
    else:
        initial_t0 = fixed_dict["t0"]
    if "ln_sigma_gp" in free_dict:
        fit_gp = True
    else:
        fit_gp = False
    if "ln_w0_gp" in free_dict:
        fit_period_gp = True
    else:
        fit_period_gp = False


    initial_batman_params = build_initial_batman_params(param_info, free_dict, fixed_dict)

    eclipse_model = batman.TransitModel(initial_batman_params, bjds, transittype='secondary')


    def lnprob(theta, plot_result=False):

        return lnprob_wrapper_eclipse(
            theta,
            free_names,
            fixed_dict,
            priors,
            initial_batman_params,
            eclipse_model,
            bjds,
            fluxes,
            errors,
            initial_t0=initial_t0,
            x = delta_x,
            y = delta_y,
            xw = delta_xw,
            yw = delta_yw,
            fit_gp=fit_gp,               # set to False to disable GP, True to include GP
            fit_period_gp=fit_period_gp,           
            plot_result=plot_result,
            wavelengths = wavelengths,
            lc_savepath = lc_savepath,
            joint_fit = joint_fit,
            visit_indices = visit_indices,
            N_visits = N_visits
        )
    
    fit_method = fixed_dict.get("fitting_method")
    if fit_method == "emcee":
        print("\nUsing MCMC for fitting...")
        nwalkers = int(fixed_dict.get("mcmc_nwalkers"))  # Default nwalkers is 100 if not specified
        nsteps   = int(fixed_dict.get("mcmc_production"))
        burnin   = int(fixed_dict.get("mcmc_burnin"))  # Default burnin is 0 if not specified
        thin     = 1
        # Initialize the walker positions by adding a small random perturbation
        p0 = np.zeros((nwalkers, ndim))
        for i in range(nwalkers):
            p0[i, :] = initial_theta + 1e-4 * np.random.randn(ndim)
    
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        print("\nRunning MCMC...")
        sampler.run_mcmc(p0, nsteps, progress=True)
        tau = sampler.get_autocorr_time(tol=0)
        print("Autocorrelation times:", tau)
        converged = sampler.iteration > 50 * np.max(tau)
        print("Converged ", converged)
        samples = sampler.get_chain()
        nwalkers, nsteps, ndim = samples.shape
        #fig, axes = plt.subplots(ndim, figsize=(10, 2.5 * ndim), sharex=True)
        #for i in range(ndim):
        #    ax = axes[i]
        #    for j in range(nwalkers):
    	#        ax.plot(samples[j, :, i], alpha=0.3, lw=0.5)
        #    ax.set_ylabel(f"{free_names[i]}")
        #    ax.grid(True)

        #axes[-1].set_xlabel("Step number")
        #plt.tight_layout()
        #plt.savefig("mcmc_trace.png")
    
        flat_chain   = sampler.get_chain(discard=burnin, thin=thin, flat=True)
        flat_lnprob  = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
            # Optionally, save the full chain to disk
        np.save("chain.npy", flat_chain)
        np.save("lnprob.npy", flat_lnprob)
    
    
        best_idx    = np.argmax(flat_lnprob)
        best_theta  = flat_chain[best_idx]

    if fit_method == "dynesty":
        import scipy.stats as stats
        from dynesty import utils as dyfunc
        print("\nUsing Dynesty for fitting...")
        nlive = int(fixed_dict.get("dynesty_nlive"))
        bounds = fixed_dict.get("dynesty_bound")
        dlogz = float(fixed_dict.get("dynesty_dlogz", 0.1))
        sampling = fixed_dict.get("dynesty_sampling")
        def prior_transform(u):
            """
            Map unit-cube u[i] in [0,1] to 
            theta[i] exactly as lnprob_wrapper_pc expects:
              - U:   real in [p1,p2]
              - LU:  log10(real) in [p1,p2]
              - N:   real from N(p1,p2)
            """
            theta = np.zeros(ndim)
            for i, name in enumerate(free_names):
                p = priors[name]
                p1, p2 = p["p1"], p["p2"]
                if p["type"] == "U":
                    theta[i] = p1 + u[i] * (p2 - p1)
                elif p["type"] == "LU":
                    theta[i] = p1 + u[i] * (p2 - p1)
                elif p["type"] == "N":
                    theta[i] = stats.norm(loc=p1, scale=p2).ppf(u[i])
                else:
                    raise ValueError(f"Unknown prior type {p['type']} for {name}")
            return theta
        def loglike(theta):
            all_params = {}
            for i, name in enumerate(free_names):
                p = priors[name]
                if p["type"] == "LU":
                    all_params[name] = 10.0 ** theta[i]
                else:
                    all_params[name] = theta[i]
            all_params.update(fixed_dict)

            return compute_lnprob_eclipse(
                all_params, free_names, initial_batman_params,
                eclipse_model,
                bjds, fluxes, errors, 
                initial_t0,
                x = delta_x,
                y = delta_y,
                xw = delta_xw,
                yw = delta_yw,
                fit_gp=fit_gp,
                fit_period_gp=fit_period_gp,
                plot_result=False,
                visit_indices=visit_indices,
                joint_fit=joint_fit,
                N_visits=N_visits,
            )

        sampler = NestedSampler(
            loglike, prior_transform, ndim,
            nlive=nlive, bound=bounds, sample=sampling
        )
        print("\nRunning dynesty nested sampling...")
        sampler.run_nested(dlogz = dlogz)
        res = sampler.results
        best_idx   = np.argmax(res.logl)
        best_theta = res.samples[best_idx]

        res = sampler.results
        weights = np.exp(res['logwt'] - res['logz'][-1])
        flat_chain = dyfunc.resample_equal(res.samples, weights)
        np.savetxt("dynesty_flat_chain.txt", flat_chain)


    best_params = {}
    for i, nm in enumerate(free_names):
        prior_spec = priors[nm]
        if prior_spec["type"] == "LU":

            best_params[nm] = 10.0 ** best_theta[i]
        else:
            best_params[nm] = best_theta[i]
    for nm in fixed_names:
        best_params[nm] = fixed_dict[nm]

    best_lnprob = lnprob(best_theta, plot_result=True)

    print("\n=== Best‐Fit Parameters ===")
    free_best_params = {nm: best_params[nm] for nm in free_names}
    for nm in sorted(free_best_params.keys()):
        print(f"  {nm:<12s} = {free_best_params[nm]}")
    print("\n=== Median and +-1σ Uncertainties ===")
    for nm in free_names:
        if priors[nm]["type"] == "LU":
            median = 10.0 ** np.median(flat_chain[:, free_names.index(nm)])
            lower = median - 10.0 ** np.percentile(flat_chain[:, free_names.index(nm)], 16)
            upper = 10.0 ** np.percentile(flat_chain[:, free_names.index(nm)], 84) - median
        else:
            median = np.median(flat_chain[:, free_names.index(nm)])
            lower = median - np.percentile(flat_chain[:, free_names.index(nm)], 16)
            upper = np.percentile(flat_chain[:, free_names.index(nm)], 84) - median
        print(f"  {nm:<12s} = {median:.6f} +{upper:.6f} -{lower:.6f}")
    print("Saving parameter results to  ", params_savepath)
    with open(params_savepath, "w") as f:
        f.write("# Free parameters:\n")
        for nm in free_names:
            if priors[nm]["type"] == "LU":
                median = 10.0 ** np.median(flat_chain[:, free_names.index(nm)])
                lower = median - 10.0 ** np.percentile(flat_chain[:, free_names.index(nm)], 16)
                upper = 10.0 ** np.percentile(flat_chain[:, free_names.index(nm)], 84) - median
            else:
                median = np.median(flat_chain[:, free_names.index(nm)])
                lower = median - np.percentile(flat_chain[:, free_names.index(nm)], 16)
                upper = np.percentile(flat_chain[:, free_names.index(nm)], 84) - median
            f.write(f"{nm} {median:.6f} +{upper:.6f} -{lower:.6f}\n")
        f.write("\n# Fixed parameters:\n")
        for nm in fixed_names:
            f.write(f"{nm} {fixed_dict[nm]}\n")





    if "sqrt_ecosw" and "sqrt_esinw" in free_dict:
        index1 = free_names.index("sqrt_ecosw")
        index2 = free_names.index("sqrt_esinw")
        ecc_chain = flat_chain[:, index1]**2 + flat_chain[:, index2]**2
        w_chain = np.arctan2(flat_chain[:, index2], flat_chain[:, index1]) * 180.0 / np.pi
        corner_names = free_names.copy()
        corner_names.append("ecc")
        corner_names.append("w")
        corner_chain = np.column_stack((flat_chain, ecc_chain, w_chain))
    else: 
        corner_chain = flat_chain
        corner_names = free_names
    
    fig = plt.figure(figsize=(30,30), dpi = 200)
    corner.corner(corner_chain, labels=corner_names, range=[0.99] * corner_chain.shape[1], show_titles=True, fig = fig, tight_layout=True, quantiles=[0.16, 0.5, 0.84])
    plt.savefig("corner_plot.png")
    #plt.show()

main()
