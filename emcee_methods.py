import batman
import numpy as np
import emcee
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
import copy
import traceback
import os
import errno
import os.path
import pdb
from scipy.ndimage import uniform_filter
import pywt
from dynesty import NestedSampler
from dynesty import plotting as dyplot
import dynesty.utils
import celerite2
from celerite2 import terms

def reduced_chisq(obs, model, err=None, n_free_params=None) :
    """
    Compute reduced chi-squared:
        χ²_r = χ² / dof

    """
    obs = np.asarray(obs)
    model = np.asarray(model)
    N = obs.size

    p = int(n_free_params)

    dof = N - p
    if dof <= 0:
        raise ValueError(f"Degrees of freedom non-positive: N={N}, p={p} gives dof={dof}")

    delta = obs - model

    if err is None:
        raise ValueError("Either err or cov must be provided.")
    err = np.asarray(err)
    if np.any(err <= 0):
        raise ValueError("All uncertainties must be positive.")
    chi2 = np.sum((delta / err) ** 2)

    chi2_r = chi2 / dof
    return chi2_r, chi2, dof


def bin_lightcurve(time, flux, bin_size, err=None):

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

def get_batman_params(t0, per, rp, a, inc, limb_dark_coeffs, \
                      t_secondary=None, w=0, ecc=0, limb_dark_law="nonlinear"):
    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = t0  # time of inferior conjunction
    params.per = per  # orbital period
    params.rp = rp  # planet radius (in units of stellar radii)
    params.a = a  # semi-major axis (in units of stellar radii)
    params.inc = inc  # orbital inclination (in degrees)
    params.ecc = ecc  # eccentricity
    params.w = w  # longitude of periastron (in degrees)
    params.limb_dark = limb_dark_law  # limb darkening model
    params.u = limb_dark_coeffs
    params.fp = 1 #we want eclipse to have no planet light

    if t_secondary is None:
        params.t_secondary = params.t0 + params.per / 2
    else:
        params.t_secondary = t_secondary

    return params

def get_planet_flux(eclipse_model, batman_params, t0, period, bjds, Fp, C1, D1,
                    C2=None, D2=None, t_secondary=None, pdb=False):
    batman_params = copy.deepcopy(batman_params)
    if t_secondary is None:
        t_secondary = t0 + period/2.0
        
    w = 2*np.pi/period
    #if planet_sine is None:
    #    planet_sine = np.sin(w*bjds)
    #if planet_cosine is None:
    #    planet_cosine = np.cos(w*bjds)

    fine_args = w*(bjds - t_secondary)
    coarse_args = np.linspace(fine_args[0], fine_args[-1], 1000)

    assert(np.max(np.diff(fine_args)) < 2*(coarse_args[1] - coarse_args[0]))
    
    planet_sine = np.sin(coarse_args)
    planet_cosine = np.cos(coarse_args)

    
    Lplanet_coarse = Fp + C1*planet_cosine + D1*planet_sine - C1
    
    if C2 is not None:        
        planet_cosine2 = np.cos(2*coarse_args)
        Lplanet_coarse += C2*planet_cosine2 - C2

    if D2 is not None:
        planet_sine2 = np.sin(2*coarse_args)
        Lplanet_coarse += D2*planet_sine2


    Lplanet = np.interp(fine_args, coarse_args, Lplanet_coarse)
    Lplanet *= (eclipse_model.light_curve(batman_params) - 1)
    
    if pdb:
        pdb.set_trace()
    return Lplanet


def plot_fit_and_residuals(times, data, err, model, binsize, plot_phase = False, period = None, t0 = None, astro_model=None, systematics=None):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    res = data - model
    if plot_phase and period is not None and t0 is not None:
        phase = ((times - t0) % period) / period
        ax[0].scatter(phase, data, s=1, label='Data', size=1, color='black', zorder = 0, alpha = 0.3)
        binned_times, binned_data, binned_err = bin_lightcurve(phase, data, binsize, err)
        ax[0].errorbar(binned_times, binned_data, yerr=binned_err, fmt='o', color='red', label='Binned Data', zorder = 1, ls='')
        ax[0].plot(phase, model, color='blue', label='Model', zorder = 2)
        
        ax[1].scatter(phase, res, s=1, color='black', zorder = 0, alpha = 0.3)
        binned_times, binned_res = bin_lightcurve(phase, res, binsize)
        ax[1].scatter(binned_times, binned_res, size=5, color='red', label='Binned Residuals', zorder = 1)
        ax[1].axhline(0, color='blue', ls='--', zorder = 2)
        ax[1].set_xlabel('Phase')


    else:
        ax[0].scatter(times, data, s=1, label='Data', color='black', zorder = 0, alpha = 0.3)
        binned_times, binned_data, binned_err = bin_lightcurve(times, data, binsize, err)
        ax[0].errorbar(binned_times, binned_data, yerr=binned_err, fmt='o', color='red', label='Binned Data', zorder = 1, ls='')
        ax[0].plot(times, model, color='blue', label='Model', zorder = 2)

        ax[1].scatter(times, res, s=1, color='black', zorder = 0, alpha = 0.3)
        binned_times, binned_res = bin_lightcurve(times, res, binsize)
        ax[1].scatter(binned_times, binned_res, s=5, color='red', label='Binned Residuals', zorder = 1)
        ax[1].axhline(0, color='blue', ls='--', zorder = 2)
        ax[1].set_xlabel('Phase')
    plt.savefig('fit_and_residuals.png', dpi=300)
    plt.show()
    if astro_model is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(times, astro_model, color='green', label='Astro Model')
        plt.xlabel('Time')
        plt.savefig('astro_model.png', dpi=300)
        #plt.show()
    if systematics is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(times, systematics, color='orange', label='Systematics')
        plt.xlabel('Time')
        plt.savefig('systematics.png', dpi=300)
        #plt.show()

def lnprob_transit(params, initial_batman_params, transit_model, bjds,
                   fluxes, errors, y, x, initial_t0, plot_result=False,
                   return_residuals=False,
                   output_filename="white_lightcurve.txt"):
    transit_offset, rp, a_star, b, error_factor, Fstar, A, tau, y_coeff, x_coeff, m, q1, q2 = params
    inc = np.arccos(b/a_star) * 180/np.pi

    u1 = 2*np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2*q2)
    if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1: return -np.inf
    batman_params = initial_batman_params
    batman_params.u = [u1, u2]
    batman_params.t0 = initial_t0 + transit_offset
    batman_params.rp = rp
    batman_params.a = a_star
    batman_params.inc = inc
    
    #now account for prior
    if (np.abs(params[0])) >= batman_params.per/4.0:
        return -np.inf
    if tau <= 1e-3 or tau > 0.1: return -np.inf
        
    if error_factor <= 0 or error_factor >= 5: return -np.inf
    if rp <= 0 or rp >= 1 or a_star <= 0 or b <= 0 or b >= 1: return -np.inf
    scaled_errors = error_factor * errors

    delta_t = bjds - bjds[0]
    systematics = Fstar * (1 + A*np.exp(-delta_t/tau) + y_coeff * y + x_coeff * x + m * (bjds - np.mean(bjds)))
    astro = transit_model.light_curve(batman_params)
    model = systematics * astro
    
    residuals = fluxes - model

    
    result = -0.5*(np.sum(residuals**2/scaled_errors**2 - np.log(1.0/scaled_errors**2)))

    if plot_result:
        with open(output_filename, "w") as f:
            f.write("#time flux uncertainty systematics_model astro_model total_model residuals\n")        
            for i in range(len(residuals)):
                f.write("{} {} {} {} {} {} {}\n".format(
                    bjds[i], fluxes[i] / Fstar, scaled_errors[i] / Fstar,
                    systematics[i] / Fstar, astro[i], model[i] / Fstar,
                    residuals[i] / Fstar))
            f.write("\n")
        plot_fit_and_residuals(bjds, fluxes, scaled_errors, model, binsize=len(fluxes)//100, 
                               systematics=systematics)
        
        print("STD", np.std(residuals))
        
    if np.random.randint(0, 1000) == 0:
        print(result/len(residuals), error_factor, np.median(np.abs(residuals)), rp, a_star, inc, b)

    if return_residuals:
        return result, residuals

    if np.isnan(result):
        pdb.set_trace()
    return result


def lnprob_transit_limited(params, batman_params, transit_model, bjds,
                           fluxes, errors, y, x, fix_tau=None, plot_result=False,
                           return_residuals=False, wavelength=None, output_filename="lightcurves.txt"):
    batman_params = copy.deepcopy(batman_params)
    depth, error_factor, Fstar, A, tau, y_coeff, x_coeff, m, q1, q2 = params
    
    if Fstar <= 0:
        return -np.inf

    if error_factor <= 0: return -np.inf
    if depth <= 0 or depth >= 1: return -np.inf
    if tau < 0 or tau > 0.1: return -np.inf
    if fix_tau is not None:
        tau = fix_tau
    if q1 <= 0 or q1 >= 1 or q2 <= 0 or q2 >= 1: return -np.inf


    
    u1 = 2*np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2*q2)

    rp = np.sqrt(depth)


    lnprior = -0.5 * ((u1 - batman_params.u[0])**2 + (u2 - batman_params.u[1])**2) / 0.1**2

    

    batman_params.u = [u1, u2]
    batman_params.rp = rp
    delta_t = bjds - bjds[0]
    systematics = Fstar * (1 + A*np.exp(-delta_t/tau) + y_coeff * y + x_coeff * x + m * (bjds - np.mean(bjds)))
    
    astro = transit_model.light_curve(batman_params)
    model = systematics * astro
    residuals = fluxes - model
    scaled_errors = errors * error_factor

    phases = (bjds-batman_params.t0)/batman_params.per
    phases -= np.round(np.median(phases))

        
    result = lnprior -0.5*(np.sum(residuals**2/scaled_errors**2 - np.log(1.0/scaled_errors**2)))

    if plot_result:
        print("lnprob", result)
        if not os.path.exists(output_filename):
             with open(output_filename, "w") as f:
                 f.write("#wavelength time flux uncertainty systematics_model astro_model total_model residuals\n")
        
        with open(output_filename, "a") as f:
            for i in range(len(residuals)):
                f.write("{} {} {} {} {} {} {} {}\n".format(wavelength, bjds[i], fluxes[i] / Fstar, scaled_errors[i] / Fstar,
                                                     systematics[i] / Fstar, astro[i], model[i] / Fstar,
                                                     residuals[i] / Fstar))

                


        plot_fit_and_residuals(bjds, fluxes, scaled_errors, model, binsize=len(fluxes)//100, 
                               systematics=systematics)
    
        plt.figure()
        plt.scatter(bjds, systematics)
        plt.title("systematics")


        print("STD of residuals", np.std(residuals))
        
    if np.random.randint(0, 1000) == 0:
        print(result/len(residuals), rp**2, np.median(scaled_errors), error_factor, Fstar, np.median(np.abs(residuals)), rp)

    if return_residuals:
        return result, residuals    

    if np.isnan(result):
        pdb.set_trace()
        print("result")
    return result


def compute_lnprob_pc(
    all_params,
    free_names,
    initial_batman_params,
    transit_model,
    eclipse_model,
    bjds,
    fluxes,
    errors,
    y,
    x,
    initial_t0,
    extra_phase_terms=False,
    fit_gp=False, 
    fit_period_gp=True,
    plot_result=False, 
    wavelengths = None,
    lc_savepath = None,
    residuals_blue = None,
):



    C1 = all_params["C1"]
    D1 = all_params["D1"]

    if extra_phase_terms:
        C2 = all_params["C2"]
        D2 = all_params["D2"]
    else:
        C2 = None
        D2 = None
    t0 = all_params["t0"]
    t_secondary = all_params["t_secondary"]
    rp           = all_params["rp"]
    fp           = all_params["fp"]
    a_star       = all_params["a_star"]
    error_factor = all_params["error_factor"]
    q1      = all_params["q1"]
    q2      = all_params["q2"]
    per     = all_params["per"]

    if fit_gp and fit_period_gp:
        ln_sigma_gp = all_params["ln_sigma_gp"]
        ln_Q_gp     = all_params["ln_Q_gp"]
        ln_w0_gp    = all_params["ln_w0_gp"]
        #ln_jit_gp   = all_params["ln_jit_gp"]

        sigma_gp = np.exp(ln_sigma_gp)
        Q_gp     = np.exp(ln_Q_gp)
        w0_gp    = np.exp(ln_w0_gp)
        #jitter_gp= np.exp(ln_jit_gp)
        jitter_gp = 0
    elif fit_gp and not fit_period_gp:
        ln_sigma_gp = all_params["ln_sigma_gp"]
        ln_Q_gp     = all_params["ln_Q_gp"]
        ln_w0_gp    = all_params["ln_w0_gp"]
        #ln_jit_gp   = all_params["ln_jit_gp"]

        sigma_gp = np.exp(ln_sigma_gp)
        Q_gp     = np.exp(ln_Q_gp)
        fixed_w0_gp    = np.exp(ln_w0_gp)
        #jitter_gp= np.exp(ln_jit_gp)
        jitter_gp = 0
    else:
        sigma_gp = Q_gp = w0_gp = jitter_gp = None


    batman_params = copy.deepcopy(initial_batman_params)
    if all_params['limb_dark'] == 'kipping2013':
        batman_params.limb_dark = 'quadratic'
        batman_params.u   = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1 - 2*q2)]
    elif all_params['limb_dark'] == 'quadratic':
        batman_params.limb_dark = 'quadratic'
        batman_params.u = [q1, q2]
    
    if "t0" in free_names:
        batman_params.t0 = t0
    if "rp" in free_names:
        batman_params.rp = rp
    if "per" in free_names:
        batman_params.per = per
    if "a_star" in free_names:
        batman_params.a = a_star
    if "b" in free_names:
        batman_params.inc = np.arccos(all_params["b"]/a_star) * 180/np.pi
    elif "inc" in free_names:
        batman_params.inc = all_params["inc"]
    if "t_secondary" in free_names:
        batman_params.t_secondary = t_secondary
    else: 
        batman_params.t_secondary = t0 + batman_params.per / 2.0

    delta_t = bjds - bjds[0]
    systematics = 1.0
    if "m" in all_params:
        systematics += all_params["m"] * (bjds - np.mean(bjds))
    if "A" in all_params and "tau" in all_params:
        systematics += all_params["A"] * np.exp(-delta_t / all_params["tau"])
    if "y_coeff" in all_params:
        systematics += all_params["y_coeff"] * y
    if "x_coeff" in all_params:
        systematics += all_params["x_coeff"] * x
    if residuals_blue is not None and 'res_coeff' in all_params:
        systematics += residuals_blue * all_params['res_coeff']
    if "amp1" in all_params and "amp2" in all_params and "log_om" in all_params:
        log_om = all_params["log_om"]
        om = np.exp(log_om)
        systematics += all_params["amp1"] * np.cos(om * delta_t) + all_params["amp2"] * np.sin(om * delta_t)
    if "Fstar" in all_params:
        Fstar = all_params["Fstar"]
        systematics *= Fstar
    else: 
        Fstar = 1.0  ### Needed for plotting later

    # Phase‐curve (planet) flux:
    if extra_phase_terms:
        Lplanet = get_planet_flux(
            eclipse_model,
            batman_params,
            batman_params.t0,
            batman_params.per,
            bjds,
            fp,
            C1, D1,
            C2, D2,
            t_secondary=batman_params.t_secondary,
        )
    else:
        Lplanet = get_planet_flux(
            eclipse_model,
            batman_params,
            batman_params.t0,
            batman_params.per,
            bjds,
            fp,
            C1, D1,
            t_secondary=batman_params.t_secondary,
        )

    astro = transit_model.light_curve(batman_params) + Lplanet

    model = systematics * astro

    residuals = fluxes - model
    scaled_errors = error_factor * errors
    # 2d) Compute log‐likelihood
    if fit_gp:
        if fit_period_gp:
            kernel_gp = terms.SHOTerm(sigma=sigma_gp, Q=Q_gp, w0=w0_gp)
        if not fit_period_gp:
            kernel_gp = terms.SHOTerm(sigma=sigma_gp, Q=Q_gp, w0=fixed_w0_gp)
        gp = celerite2.GaussianProcess(kernel_gp, mean=0.0)
        #diag = scaled_errors**2 + jitter_gp**2
        diag = scaled_errors**2
        gp.compute(bjds, diag=diag)
        bf_model = model + gp.predict(residuals, return_cov=False)
        if plot_result:
            print("-----------Plotting...----------")
            plot_fit_and_residuals(bjds, fluxes, scaled_errors, bf_model, binsize=100, plot_phase=False, astro_model=astro, systematics=systematics)
            if not os.path.exists(lc_savepath):
                with open(lc_savepath, "w") as f:
                    f.write("startwave endwave time flux uncertainty systematics_model astro_model total_model residuals\n")
        
                with open(lc_savepath, "a") as f:
                    for i in range(len(residuals)):
                        f.write("{} {} {} {} {} {} {} {} {}\n".format(
                            wavelengths[0], wavelengths[-1], bjds[i], fluxes[i] / Fstar, scaled_errors[i] / Fstar,
                            systematics[i] / Fstar, astro[i], model[i] / Fstar,
                            residuals[i] / Fstar))
        
        return gp.log_likelihood(residuals)

    else:
        var = scaled_errors**2
        bf_model = model
        if plot_result:
            print("-----------Plotting...----------")
            plot_fit_and_residuals(bjds, fluxes, scaled_errors, bf_model, binsize=100, plot_phase=False, astro_model=astro, systematics=systematics)
            if not os.path.exists(lc_savepath):
                with open(lc_savepath, "w") as f:
                    f.write("startwave endwave time flux uncertainty systematics_model astro_model total_model residuals\n")
        
                with open(lc_savepath, "a") as f:
                    for i in range(len(residuals)):
                        f.write("{} {} {} {} {} {} {} {} {}\n".format(
                            wavelengths[0], wavelengths[-1], bjds[i], fluxes[i] / Fstar, scaled_errors[i] / Fstar,
                            systematics[i] / Fstar, astro[i], model[i] / Fstar,
                            residuals[i] / Fstar))
        return -0.5 * np.sum((residuals**2) / var + np.log(2 * np.pi * var))


def lnprob_wrapper_pc(
    theta,
    free_names,
    fixed_dict,
    priors,
    initial_batman_params,
    transit_model,
    eclipse_model,
    bjds,
    fluxes,
    errors,
    y,
    x,
    initial_t0,
    extra_phase_terms=False,
    fit_gp=False,
    fit_period_gp=True,
    plot_result=False,
    wavelengths = None,
    lc_savepath = None,
    residuals_blue = None,
):


    all_params = {}

    lp = 0.0
    for i, name in enumerate(free_names):
        prior_spec = priors[name]
        val_theta  = theta[i]

        if prior_spec is None:
            # This shouldn’t happen: free parameters always have a prior.
            raise ValueError(f"Parameter '{name}' is free but has no prior in config.")

        ptype = prior_spec["type"]
        p1    = prior_spec["p1"]
        p2    = prior_spec["p2"]

        if ptype == "U":
            if not (p1 <= val_theta <= p2):
                return -np.inf
            real_val = val_theta
        elif ptype == "LU":
            if not (p1 <= val_theta <= p2):
                return -np.inf
            real_val = 10.0 ** val_theta
        elif ptype == "N":
            real_val = val_theta
            lp += -0.5 * ((val_theta - p1) / p2) ** 2
        else:
            raise ValueError(f"Unknown prior type '{ptype}' for parameter '{name}'")

        all_params[name] = real_val

    for name, val in fixed_dict.items():
        all_params[name] = val


    lnlike = compute_lnprob_pc(
        all_params,
        free_names,
        initial_batman_params,
        transit_model,
        eclipse_model,
        bjds,
        fluxes,
        errors,
        y,
        x,
        initial_t0,
        extra_phase_terms=extra_phase_terms,
        fit_gp=fit_gp,
        fit_period_gp=fit_period_gp,
        plot_result=plot_result,
        wavelengths=wavelengths,
        lc_savepath=lc_savepath,
        residuals_blue=residuals_blue,
    )
    if not np.isfinite(lnlike):
        return -np.inf

    return lp + lnlike
    
def compute_lnprob_eclipse(
    all_params,
    free_names,
    initial_batman_params,
    eclipse_model,
    bjds,
    fluxes,
    errors,
    initial_t0,
    x,
    y,
    xw,
    yw,
    fit_gp=False, 
    fit_period_gp=True,
    plot_result=False, 
    wavelengths = None,
    lc_savepath = None,
    visit_indices=None,
    joint_fit=False,
    N_visits=1,

):



    batman_params = copy.deepcopy(initial_batman_params)
    

    if fit_gp and fit_period_gp:
        ln_sigma_gp = all_params["ln_sigma_gp"]
        ln_Q_gp     = all_params["ln_Q_gp"]
        ln_w0_gp    = all_params["ln_w0_gp"]
        #ln_jit_gp   = all_params["ln_jit_gp"]

        sigma_gp = np.exp(ln_sigma_gp)
        Q_gp     = np.exp(ln_Q_gp)
        w0_gp    = np.exp(ln_w0_gp)
        #jitter_gp= np.exp(ln_jit_gp)
        jitter_gp = 0
    elif fit_gp and not fit_period_gp:
        ln_sigma_gp = all_params["ln_sigma_gp"]
        ln_Q_gp     = all_params["ln_Q_gp"]
        ln_w0_gp    = all_params["ln_w0_gp"]
        #ln_jit_gp   = all_params["ln_jit_gp"]

        sigma_gp = np.exp(ln_sigma_gp)
        Q_gp     = np.exp(ln_Q_gp)
        fixed_w0_gp    = np.exp(ln_w0_gp)
        #jitter_gp= np.exp(ln_jit_gp)
        jitter_gp = 0
    else:
        sigma_gp = Q_gp = w0_gp = jitter_gp = None

    # Shared parameters
    
    t0 = all_params["t0"]
    t_secondary = all_params["t_secondary"]
    rp = all_params["rp"]
    fp = all_params["fp"]
    a_star = all_params["a_star"]
    error_factor = all_params["error_factor"]
    if "Fstar" in all_params:
        Fstar = all_params["Fstar"]
    per = all_params["per"]
    
    # Initialize per-visit systematics containers
    # Default to single-visit behavior:
    if joint_fit:
        # Expect visit-specific names, e.g., A1...AN, tau1...tauN, m1...mN, x_coeff1...x_coeffN, y_coeff1...y_coeffN, Fstar1... etc.
        systematics = np.zeros_like(bjds, dtype=float)
        Fstar_per_point = np.zeros_like(bjds, dtype=float) 
        for v in range(1, N_visits + 1):
            # Mask of points belonging to visit v
            mask_v = (visit_indices == v)
            delta_t_v = bjds[mask_v] - bjds[mask_v][0]
            # Pull visit-specific parameters, fall back or error if missing
            try:
                A_v = all_params[f"A{v}"]
                tau_v = all_params[f"tau{v}"]
                m_v = all_params[f"m{v}"]
                x_coeff_v = all_params.get(f"x_coeff{v}", 0.0)
                y_coeff_v = all_params.get(f"y_coeff{v}", 0.0)
                xw_coeff_v = all_params.get(f"xw_coeff{v}", 0.0)
                yw_coeff_v = all_params.get(f"yw_coeff{v}", 0.0)
                Fstar_v = all_params.get(f"Fstar{v}", 1.0)
                #error_factor_v = all_params.get(f"error_factor{v}", 1.0)
            except KeyError as e:
                raise KeyError(f"Missing expected visit-specific parameter for joint fit: {e}")
            sys_v = 1.0 + m_v * (bjds[mask_v] - np.mean(bjds[mask_v]))
            sys_v += A_v * np.exp(-delta_t_v / tau_v)
            sys_v += y_coeff_v * y[mask_v]
            sys_v += x_coeff_v * x[mask_v]
            sys_v += yw_coeff_v * yw[mask_v] 
            sys_v += xw_coeff_v * xw[mask_v]

            # Apply Fstar_v scaling per visit
            sys_v *= Fstar_v

            systematics[mask_v] = sys_v
            Fstar_per_point[mask_v] = Fstar_v

    
    else:
        if "A" in all_params:
            A = all_params["A"]
        if "log_tau" in all_params:
            tau = np.exp(all_params["log_tau"])
        if "y_coeff" in all_params:
            y_coeff = all_params["y_coeff"]
        if "x_coeff" in all_params:
            x_coeff = all_params["x_coeff"]
        if "xw_coeff" in all_params:
            xw_coeff = all_params["xw_coeff"]
        if "yw_coeff" in all_params:
            yw_coeff = all_params["yw_coeff"]
        if "m" in all_params:
            m = all_params["m"]
        systematics = 1.0 + m * (bjds - np.mean(bjds)) + A * np.exp(-(bjds-bjds[0])/tau)
        if 'y_coeff' in free_names:
            systematics += y_coeff * y
        if 'x_coeff' in free_names:
            systematics += x_coeff * x
        if 'xw_coeff' in free_names:
            systematics += xw_coeff * xw
        if 'yw_coeff' in free_names:
            systematics += yw_coeff * yw
        systematics *= Fstar

    if all_params['limb_dark'] == 'kipping2013':
        q1 = all_params["q1"]
        q2 = all_params["q2"]
        batman_params.limb_dark = 'quadratic'
        batman_params.u   = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1 - 2*q2)]
    elif all_params['limb_dark'] == 'quadratic':
        q1 = all_params["q1"]
        q2 = all_params["q2"]
        batman_params.limb_dark = 'quadratic'
        batman_params.u = [q1, q2]
    elif all_params['limb_dark'] == 'linear':
        q1 = all_params["q1"]
        batman_params.limb_dark = 'linear'
        batman_params.u = [q1]
    elif all_params['limb_dark'] == 'uniform':
        batman_params.limb_dark = 'uniform'
        batman_params.u = []
    

    batman_params.t0 = t0
    if "rp" in free_names:
        batman_params.rp = rp
    if "per" in free_names:
        batman_params.per = per
    if "a_star" in free_names:
        batman_params.a = a_star
    if "b" in free_names and "inc" in free_names:
        raise ValueError("Cannot have both 'b' and 'inc' as free parameters. Please fix one.")
    if "b" in free_names:
        batman_params.inc = np.arccos(all_params["b"]/a_star) * 180/np.pi
    elif "inc" in free_names:
        batman_params.inc = all_params["inc"]
    if "t_secondary" in free_names:
        batman_params.t_secondary = t_secondary
    batman_params.fp = fp
    
    if "sqrt_ecosw" in free_names and "sqrt_esinw" in free_names:
        ecosw = all_params["sqrt_ecosw"]
        esinw = all_params["sqrt_esinw"]
        #print(ecosw, esinw, ecosw**2 + esinw**2, np.arctan2(esinw, ecosw) * 180.0/np.pi)
        batman_params.ecc = ecosw**2 + esinw**2
        batman_params.w = np.arctan2(esinw, ecosw) * 180.0/np.pi
    
    

    astro = eclipse_model.light_curve(batman_params)
    model = systematics * astro

 
    residuals = fluxes - model
    scaled_errors = error_factor * errors

    # 2d) Compute log‐likelihood
    if fit_gp:
        raise NotImplementedError("GP fitting not yet implemented for fitting eclipses")
    
        if fit_period_gp:
            kernel_gp = terms.SHOTerm(sigma=sigma_gp, Q=Q_gp, w0=w0_gp)
        if not fit_period_gp:
            kernel_gp = terms.SHOTerm(sigma=sigma_gp, Q=Q_gp, w0=fixed_w0_gp)
        gp = celerite2.GaussianProcess(kernel_gp, mean=0.0)
        #diag = scaled_errors**2 + jitter_gp**2
        diag = scaled_errors**2
        gp.compute(bjds, diag=diag)
        bf_model = model + gp.predict(residuals, return_cov=False)
        if plot_result:
            print("-----------Plotting...----------")
            binsize = len(residuals) // 3
            plot_fit_and_residuals(bjds, fluxes, scaled_errors, bf_model, binsize=binsize, plot_phase=False, astro_model=astro, systematics=systematics)
            chi2_r, chi2, dof = reduced_chisq(fluxes, bf_model, scaled_errors, n_free_params=len(free_names))
            print(f"Reduced chi-squared: {chi2_r:.3f} (chi2={chi2:.3f}, dof={dof})")
            if not os.path.exists(lc_savepath):
                with open(lc_savepath, "w") as f:
                    f.write("startwave endwave time flux uncertainty systematics_model astro_model total_model residuals visit\n")
        
                with open(lc_savepath, "a") as f:
                    for i in range(len(residuals)):
                        visit_str = f"{visit_indices[i]}" if (joint_fit and visit_indices is not None) else "1"
                        this_Fstar = Fstar_per_point[i] if (joint_fit and visit_indices is not None) else all_params.get("Fstar", 1.0)
                        scaled_err = scaled_errors[i]  # assume this already includes the shared error_factor
                        f.write(
                            "{} {} {} {} {} {} {} {} {} {}\n".format(
                                wavelengths[0],
                                wavelengths[-1],
                                bjds[i],
                                fluxes[i] / this_Fstar,
                                scaled_err / this_Fstar,
                                systematics[i] / this_Fstar,
                                astro[i],
                                model[i] / this_Fstar,
                                residuals[i] / this_Fstar,
                                visit_str,
                            )
                        )
        
        return gp.log_likelihood(residuals)

    else:
        var = scaled_errors**2
        bf_model = model
        if plot_result:
            print("-----------Plotting...----------")
            plot_fit_and_residuals(bjds, fluxes, scaled_errors, bf_model, binsize=100, plot_phase=False, astro_model=astro, systematics=systematics)
            chi2_r, chi2, dof = reduced_chisq(fluxes, bf_model, scaled_errors, n_free_params=len(free_names))
            print(f"Reduced chi-squared: {chi2_r:.3f} (chi2={chi2:.3f}, dof={dof})")
            if not os.path.exists(lc_savepath):
                with open(lc_savepath, "w") as f:
                    f.write("startwave endwave time flux uncertainty systematics_model astro_model total_model residuals visit\n")
                with open(lc_savepath, "a") as f:
                    for i in range(len(residuals)):
                        visit_str = f"{visit_indices[i]}" if (joint_fit and visit_indices is not None) else "1"
                        this_Fstar = Fstar_per_point[i] if (joint_fit and visit_indices is not None) else all_params.get("Fstar", 1.0)
                        scaled_err = scaled_errors[i]  # assume this already includes the shared error_factor
                        f.write(
                            "{} {} {} {} {} {} {} {} {} {}\n".format(
                                wavelengths[0],
                                wavelengths[-1],
                                bjds[i],
                                fluxes[i] / this_Fstar,
                                scaled_err / this_Fstar,
                                systematics[i] / this_Fstar,
                                astro[i],
                                model[i] / this_Fstar,
                                residuals[i] / this_Fstar,
                                visit_str,
            
            # optionally log the shared error_factor for diagnostics
                            )
                        )
        return -0.5 * np.sum((residuals**2) / var + np.log(2 * np.pi * var))


def lnprob_wrapper_eclipse(
    theta,
    free_names,
    fixed_dict,
    priors,
    initial_batman_params,
    eclipse_model,
    bjds,
    fluxes,
    errors,
    initial_t0,
    x=None,
    y=None,
    xw=None,
    yw=None,
    fit_gp=False,
    fit_period_gp=True,
    plot_result=False,
    wavelengths = None,
    lc_savepath = None,
    visit_indices=None,
    joint_fit=False,
    N_visits=1,
):


    all_params = {}

    lp = 0.0
    for i, name in enumerate(free_names):
        prior_spec = priors[name]
        val_theta  = theta[i]

        if prior_spec is None:
            # This shouldn’t happen: free parameters always have a prior.
            raise ValueError(f"Parameter '{name}' is free but has no prior in config.")

        ptype = prior_spec["type"]
        p1    = prior_spec["p1"]
        p2    = prior_spec["p2"]

        if ptype == "U":
            if not (p1 <= val_theta <= p2):
                return -np.inf
            real_val = val_theta
        elif ptype == "LU":
            if not (p1 <= val_theta <= p2):
                return -np.inf
            real_val = 10.0 ** val_theta
        elif ptype == "N":
            real_val = val_theta
            lp += -0.5 * ((val_theta - p1) / p2) ** 2
        else:
            raise ValueError(f"Unknown prior type '{ptype}' for parameter '{name}'")

        all_params[name] = real_val

    for name, val in fixed_dict.items():
        all_params[name] = val
        if joint_fit:
            shared_keys = ["rp", "t0", "per", "ecc", "w", "inc", "a_star", "t_secondary", "fp", "limb_dark"]



    lnlike = compute_lnprob_eclipse(
        all_params,
        free_names,
        initial_batman_params,
        eclipse_model,
        bjds,
        fluxes,
        errors,
        initial_t0,
        y=y,
        x=x,
        yw=yw,
        xw=xw,
        fit_gp=fit_gp,
        fit_period_gp=fit_period_gp,
        plot_result=plot_result,
        wavelengths=wavelengths,
        lc_savepath=lc_savepath,
        visit_indices=visit_indices,
        joint_fit=joint_fit,
        N_visits=N_visits,
    )
    if not np.isfinite(lnlike):
        return -np.inf

    return lp + lnlike





def get_initial_positions(initial_params, lnprob, lnprob_args, nwalkers):
    initial_params = np.array(initial_params)
    ndim = len(initial_params)

    positions = []
    for i in range(nwalkers):
        curr_lnprob = -np.inf
        curr_pos = None
        while np.isinf(curr_lnprob):
            curr_pos = initial_params + 1e-2*np.random.randn(ndim) + 1e-2*np.random.randn(ndim)*initial_params
            #curr_pos[7] = initial_params[7] + 0.5*np.random.rand()*initial_params[7]
            #curr_pos[5] = initial_params[5] + 0.5*np.random.rand()*initial_params[5]
            curr_lnprob = lnprob(curr_pos, *lnprob_args)
        positions.append(curr_pos)

    lnprobs = np.array([lnprob(p, *lnprob_args) for p in positions])
    assert(np.sum(np.isinf(lnprobs)) == 0)
    return positions

def run_sampler_return_best(sampler, total_runs, output_dir, output_prefix, init_positions, chunk_size=5000):
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        
    run_counter = 0
    chunk_counter = 0
    positions = init_positions
    best_lnprob = -np.inf
    best_step = None
    
    while run_counter < total_runs:
        sampler.reset()
        runs_in_chunk = min(total_runs - run_counter, chunk_size)        
        for i, (positions, lnp, _) in enumerate(sampler.sample(positions, iterations=runs_in_chunk)):
            run_counter += 1
            if (run_counter + 1) % 10 == 0:
                print("Progress: {0}/{1}".format(run_counter + 1, total_runs))
        
        if np.max(sampler.flatlnprobability) > best_lnprob:
            index = np.argmax(sampler.flatlnprobability)
            best_lnprob = sampler.flatlnprobability[index]
            best_step = sampler.flatchain[index]

        chain_name = os.path.join(output_dir, output_prefix + str(chunk_counter) + "_chain.npy")
        lnprob_name = os.path.join(output_dir, output_prefix + str(chunk_counter) + "_lnprob.npy")
        np.save(chain_name, sampler.flatchain)
        np.save(lnprob_name, sampler.flatlnprobability)
        print("Burn in acceptance fraction for chunk {0}: {1}".format(chunk_counter, np.median(sampler.acceptance_fraction)))
        chunk_counter += 1

    return best_step


def run_dynesty(lnlike, lnlike_args, prior_transform, num_dim, output_dir, nlive=200):
    def dynesty_ln_like(params):
        return lnlike(params, *lnlike_args)
    
    sampler = NestedSampler(dynesty_ln_like, prior_transform, num_dim, bound='multi', nlive=nlive)
    sampler.run_nested()
    result = sampler.results
    best_step = result.samples[np.argmax(result.logl)]
    normalized_weights = np.exp(result.logwt - np.max(result.logwt))
    normalized_weights /= np.sum(normalized_weights)
    
    equal_samples = dynesty.utils.resample_equal(result.samples, normalized_weights)
    np.random.shuffle(equal_samples)
    equal_logl = np.zeros(len(equal_samples))
    for i in range(len(equal_samples)):
        index = np.argwhere((result.samples == equal_samples[i]).all(-1))[0,0]
        equal_logl[i] = result.logl[index]
    return best_step, equal_samples, equal_logl
    
    

def run_emcee(lnprob, lnprob_args, initial_params, nwalkers, output_dir, burn_in_runs, production_runs):
    ndim = len(initial_params)
    positions = get_initial_positions(initial_params, lnprob, lnprob_args, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=lnprob_args)
    print("Starting burn-in run")
    best_step = run_sampler_return_best(sampler, burn_in_runs, output_dir, "burnin", positions)
    
    latest_positions = [best_step + 1e-4*np.random.randn(ndim)*best_step
                        for i in range(nwalkers)]    
    sampler.reset() #Should already be reset, but be safe
    print("Resetting...")
    print("Starting production run")

    best_step = run_sampler_return_best(sampler, production_runs, output_dir, "production", latest_positions)
    return best_step, sampler.flatchain, sampler.flatlnprobability
    
