import copy
import numpy as np
import celerite2
from celerite2 import terms
import batman
import matplotlib.pyplot as plt
import os
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
    coarse_args = np.linspace(fine_args[0], fine_args[-1], 500)
    #import pdb
    #pdb.set_trace()
    diff = np.diff(fine_args).tolist()
    diff.sort(reverse = True)
    #print(diff[1], diff[0])

    assert(diff[1] < 2*(coarse_args[1] - coarse_args[0]))
    
    planet_sine = np.sin(coarse_args)
    planet_cosine = np.cos(coarse_args)

    #planet_sine = np.sin(w*bjds)
    #planet_cosine = np.cos(w*bjds)
    
    Lplanet_coarse = Fp + C1*planet_cosine + D1*planet_sine - C1
    
    if C2 is not None:        
        planet_cosine2 = np.cos(2*coarse_args)
        Lplanet_coarse += C2*planet_cosine2 - C2
        #assert(False)
        #print("C2")
    if D2 is not None:
        planet_sine2 = np.sin(2*coarse_args)
        Lplanet_coarse += D2*planet_sine2
        #assert(False)
        #print("D2")

    Lplanet = np.interp(fine_args, coarse_args, Lplanet_coarse)
    Lplanet *= (eclipse_model.light_curve(batman_params) - 1)
    #plt.plot(eclipse_model.light_curve(batman_params) - 1)
    #plt.show()
    
    if pdb:
        pdb.set_trace()
    return Lplanet

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
    #Fstar   = all_params["Fstar"]
    #A       = all_params["A"]
    #tau     = all_params["tau"]
    #y_coeff = all_params["y_coeff"]
    #x_coeff = all_params["x_coeff"]
    #m       = all_params["m"]
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

    if "Fstar" in free_names:    
        Fstar = all_params["Fstar"]
    if "A" in free_names:
        A = all_params["A"]
    if "tau" in free_names:
        tau = all_params["tau"]
    if "m" in free_names:
        m = all_params["m"]
    if "y_coeff" in free_names:
        y_coeff = all_params["y_coeff"]
    if "x_coeff" in free_names:
        x_coeff = all_params["x_coeff"]
    if "res_coeff" in free_names:
        #res_coeff = all_params["res_coeff"]
        res_coeff = all_params.get("res_coeff", 0.0)

    if "amp1" in free_names and "amp2" in free_names and "log_om" in free_names:
        amp1 = all_params["amp1"]
        amp2 = all_params["amp2"]
        log_om = all_params["log_om"]
        om = np.exp(log_om)

    delta_t = bjds - bjds[0]
    
    
    systematics = 1.0 + m * (bjds - np.mean(bjds))
    if 'A' and 'tau' in free_names:
        systematics += A * np.exp(-delta_t / tau)
    if 'y_coeff' in free_names:
        systematics += y_coeff * y
    if 'x_coeff' in free_names:
        systematics += x_coeff * x
    if residuals_blue is not None and 'res_coeff' in free_names:
        systematics += residuals_blue * res_coeff
    if "amp1" in free_names and "amp2" in free_names and "om" in free_names:
        systematics += amp1 * np.cos(om * delta_t) + amp2 * np.sin(om * delta_t)
        
    systematics *= Fstar
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
    #import pdb
    #pdb.set_trace()
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
            import pdb
            pdb.set_trace()
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
            # Uniform in [p1, p2] for the parameter itself:
            if not (p1 <= val_theta <= p2):
                return -np.inf
            real_val = val_theta
            # log‐prior is constant within bounds → no change to lp
        elif ptype == "LU":
            # val_theta is log10(real_val).  Must lie in [p1, p2].
            if not (p1 <= val_theta <= p2):
                return -np.inf
            real_val = 10.0 ** val_theta
            # log‐prior is constant within bounds → no change to lp
        elif ptype == "N":
            # val_theta is the real parameter, with Gaussian prior mean=p1, sigma=p2
            real_val = val_theta
            lp += -0.5 * ((val_theta - p1) / p2) ** 2
        else:
            raise ValueError(f"Unknown prior type '{ptype}' for parameter '{name}'")

        all_params[name] = real_val

    # 2b) Next, add all fixed parameters directly:
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

    
        # Use a representative error_factor/Fstar for modeling scaling if needed globally, but per-visit was applied above.
    else:
        # Single visit (original behavior)
        if "A" in all_params:
            A = all_params["A"]
        if "tau" in all_params:
            tau = all_params["tau"]
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
        systematics = 1.0 + m * (bjds - np.mean(bjds))
        if 'A' in free_names and 'tau' in free_names:
            systematics += A * np.exp(- (bjds - bjds[0]) / tau)
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
    if "b" in free_names:
        batman_params.inc = np.arccos(all_params["b"]/a_star) * 180/np.pi
    elif "inc" in free_names:
        batman_params.inc = all_params["inc"]
    if "t_secondary" in free_names:
        batman_params.t_secondary = t_secondary
    #else:
    #    batman_params.t_secondary = 0
    batman_params.fp = fp
    
    if "sqrt_ecosw" in free_names and "sqrt_esinw" in free_names:
        ecosw = all_params["sqrt_ecosw"]
        esinw = all_params["sqrt_esinw"]
        #print(ecosw, esinw, ecosw**2 + esinw**2, np.arctan2(esinw, ecosw) * 180.0/np.pi)
        batman_params.ecc = ecosw**2 + esinw**2
        batman_params.w = np.arctan2(esinw, ecosw) * 180.0/np.pi
    
    

    astro = eclipse_model.light_curve(batman_params)
    model = systematics * astro
    #plt.figure()
    #plt.plot(bjds, astro)
    #plt.show()
    #print(eclipse_model.get_t_secondary(batman_params))
 
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
            print(eclipse_model.get_t_secondary(batman_params))
            
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
    #print(free_names)
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
            # Uniform in [p1, p2] for the parameter itself:
            if not (p1 <= val_theta <= p2):
                return -np.inf
            real_val = val_theta
            # log‐prior is constant within bounds → no change to lp
        elif ptype == "LU":
            # val_theta is log10(real_val).  Must lie in [p1, p2].
            if not (p1 <= val_theta <= p2):
                return -np.inf
            real_val = 10.0 ** val_theta
            # log‐prior is constant within bounds → no change to lp
        elif ptype == "N":
            # val_theta is the real parameter, with Gaussian prior mean=p1, sigma=p2
            real_val = val_theta
            lp += -0.5 * ((val_theta - p1) / p2) ** 2
        else:
            raise ValueError(f"Unknown prior type '{ptype}' for parameter '{name}'")

        all_params[name] = real_val

    # 2b) Next, add all fixed parameters directly:
    for name, val in fixed_dict.items():
        all_params[name] = val
        if joint_fit:
        # Shared astrophysical parameters: keep as-is (e.g., rp, t0, per, ecc, w, inc, a_star, t_secondary, fp, limb_dark, etc.)
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
