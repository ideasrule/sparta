import batman
import numpy as np
import emcee
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from algorithms import bin_data
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
    #import pdb
    #pdb.set_trace()
    assert(np.max(np.diff(fine_args)) < 2*(coarse_args[1] - coarse_args[0]))
    
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


def plot_fit_and_residuals(phases, binsize, Lobserved, Lexpected, gp=None, bjds=None):
    if gp is not None:
        Lexpected += gp.predict(Lobserved-Lexpected, bjds, return_cov=False)
    residuals = Lobserved - Lexpected
    
    f, axarr = plt.subplots(2, sharex=True)
    #f.tight_layout()
    fontsize = 10
    plt.xlabel("Orbital phase", fontsize=fontsize)
    #plt.xlim(-0.6, 0.6)
    
    '''#axarr[0].set_ylim([0.997, 1.001])
    #axarr[0].set_ylim([0.985, 1.006])
    axarr[0].scatter(bin_data(phases,binsize), bin_data(Lobserved - star_variation, binsize), s=10, c='blue', edgecolors='black')
    axarr[0].plot(bin_data(phases,binsize), bin_data(Lexpected - star_variation,binsize), color="r")
    axarr[0].ticklabel_format(useOffset=False)
    axarr[0].set_ylabel("Relative flux", fontsize=fontsize)'''

    #axarr[1].set_ylim([0.9996, 1.0008])
    #axarr[1].set_ylim([0.998, 1.006])
    axarr[0].scatter(bin_data(phases,binsize), bin_data(Lobserved, binsize), s=10, c='blue', edgecolors='black')
    axarr[0].plot(bin_data(phases,binsize), bin_data(Lexpected, binsize), color="r")
    axarr[0].set_ylabel("Relative flux", fontsize=fontsize)
    axarr[0].ticklabel_format(useOffset=False)
    axarr[0].set_ylim([0.9985, max(Lobserved)])
    #axarr[0].set_ylim([0.9985, 1.0015])

    #axarr[2].set_ylim([-0.0005, 0.0005])
    #axarr[2].set_ylim([-0.002, 0.002])
    axarr[1].scatter(bin_data(phases,binsize), bin_data(residuals,binsize), s=10, c='blue', edgecolors='black')
    axarr[1].set_ylabel("Relative flux", fontsize=fontsize)
    
def lnprob_transit(params, initial_batman_params, transit_model, bjds,
                   fluxes, errors, y, x, initial_t0, plot_result=False,
                   return_residuals=False,
                   output_filename="white_lightcurve.txt"):
    transit_offset, rp, a_star, b, error_factor, Fstar, A, tau, y_coeff, x_coeff, m, q1, q2 = params
    #a_star = 7.61
    inc = np.arccos(b/a_star) * 180/np.pi

    u1 = 2*np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2*q2)
    if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1: return -np.inf
    batman_params = initial_batman_params
    batman_params.u = [u1, u2]
    #batman_params.u[1] = u2 # = [initial_batman_params.u[0], u2]
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

    phases = (bjds-batman_params.t0)/batman_params.per
    phases -= np.round(np.median(phases))
    
    residuals = fluxes - model
    if abs(batman_params.t0 - 59820.937747) < 1e-3:
        #assert(False)
        residuals[np.logical_and(phases > -0.0024, phases < 0.007)] = 0
    
    result = -0.5*(np.sum(residuals**2/scaled_errors**2 - np.log(1.0/scaled_errors**2)))

    if plot_result:
        with open(output_filename, "w") as f:
            f.write("#time flux uncertainty systematics_model astro_model total_model residuals\n")        
            for i in range(len(residuals)):
                f.write("{} {} {} {} {} {} {}\n".format(
                    bjds[i], fluxes[i] / Fstar, scaled_errors[i] / Fstar,
                    systematics[i] / Fstar, astro[i], model[i] / Fstar,
                    residuals[i] / Fstar))
                
        binsize = len(fluxes) // 100

        plot_fit_and_residuals(phases, binsize, fluxes / systematics, astro)
        plt.figure()
        plt.scatter(bjds, systematics)
        
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
    depth, error_factor, Fstar, A, tau, y_coeff, x_coeff, m, u1, u2 = params
    
    if Fstar <= 0:
        return -np.inf

    if error_factor <= 0: return -np.inf
    if depth <= 0 or depth >= 1: return -np.inf
    if tau < 0 or tau > 0.1: return -np.inf
    if fix_tau is not None:
        tau = fix_tau
    #if q1 <= 0 or q1 >= 1 or q2 <= 0 or q2 >= 1: return -np.inf

    #q1 = 0.0334
    #q2 = 0.445

    #q1 = 0.050
    #q2 = 0.08

    #q1 = 0.04

    #q1_init = (batman_params.u[0] + batman_params.u[1])**2
    #q2_init = 0.5 * batman_params.u[0] / (batman_params.u[0] + batman_params.u[1])
    
    #u1 = 2*np.sqrt(q1) * q2
    #u2 = np.sqrt(q1) * (1 - 2*q2)

    rp = np.sqrt(depth)

    #lnprior = -0.5 * ((q1 - q1_init)**2 + (q2 - q2_init)**2) / 0.1**2
    lnprior = -0.5 * ((u1 - batman_params.u[0])**2 + (u2 - batman_params.u[1])**2) / 0.1**2
    #lnprior = 0
    
    #batman_params.u[1] = u2# = [u1, u2]
    #batman_params.u = [u1, u2]
    batman_params.rp = rp
    delta_t = bjds - bjds[0]
    systematics = Fstar * (1 + A*np.exp(-delta_t/tau) + y_coeff * y + x_coeff * x + m * (bjds - np.mean(bjds)))
    
    astro = transit_model.light_curve(batman_params)
    model = systematics * astro
    residuals = fluxes - model
    scaled_errors = errors * error_factor

    phases = (bjds-batman_params.t0)/batman_params.per
    phases -= np.round(np.median(phases))

    if abs(batman_params.t0 - 59820.937747) < 1e-3:
        #assert(False)
        #residuals[np.logical_and(phases > -0.0024, phases < 0.007)] = 0
        residuals[np.logical_and(phases > -5.19e-4, phases < 6.95e-3)] = 0
        
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

                
        binsize = len(fluxes) // 100
        #residuals -= np.mean(residuals)
        phases = (bjds-batman_params.t0)/batman_params.per
        phases -= np.round(np.median(phases))

        plot_fit_and_residuals(phases, binsize, fluxes / systematics, astro)
    
        plt.figure()
        plt.scatter(bjds, systematics)
        plt.title("systematics")


        print("STD of residuals", np.std(residuals))
        
    if np.random.randint(0, 1000) == 0:
        print(result/len(residuals), rp**2, np.median(scaled_errors), error_factor, Fstar, np.median(np.abs(residuals)), rp)
    #if result/len(residuals) > 4.7:
    #    Lplanet = get_planet_flux(eclipse_model, batman_params, batman_params.t0, batman_params.per, bjds, Fp, C1, D1, t_secondary=batman_params.t_secondary, pdb=True)
        
    #print(result/len(residuals), Fp, np.median(np.abs(residuals)), rp, a_star, inc, b)
    if return_residuals:
        return result, residuals    

    if np.isnan(result):
        pdb.set_trace()
        print("result")
    return result

def lnprob_eclipse(params, initial_batman_params, eclipse_model, bjds,
                   fluxes, errors, y, x, initial_t_secondary, plot_result=False,
                   return_residuals=False, max_Fp=0.02,
                   output_filename="white_lightcurve.txt"):
    eclipse_offset, Fp, error_factor, Fstar, A, tau, y_coeff, x_coeff, m = params
    #x_coeff = -0.0152
    batman_params = initial_batman_params
    batman_params.t_secondary = initial_t_secondary + eclipse_offset
    
    #now account for prior
    if (np.abs(params[0])) >= batman_params.per/4.0:
        return -np.inf
    if tau <= 0 or tau > 0.9: return -np.inf
    if Fp < 0 or Fp > max_Fp: return -np.inf
        
    if error_factor <= 0 or error_factor >= 5: return -np.inf
    scaled_errors = error_factor * errors

    delta_t = bjds - bjds[0]
    systematics = Fstar * (1 + A*np.exp(-delta_t/tau) + 0*y_coeff * y + x_coeff * x + m * (bjds - np.mean(bjds)))
    astro = 1 + get_planet_flux(eclipse_model, batman_params, batman_params.t0, batman_params.per, bjds, Fp, 0, 0, 0, 0, t_secondary=batman_params.t_secondary)
    model = systematics * astro
    
    residuals = fluxes - model
    result = -0.5*(np.sum(residuals**2/scaled_errors**2 - np.log(1.0/scaled_errors**2)))
    
    if plot_result:
        print("lnprob of plotted", result)
        with open(output_filename, "w") as f:
            f.write("#time flux uncertainty systematics_model astro_model total_model residuals\n")        
            for i in range(len(residuals)):
                f.write("{} {} {} {} {} {} {}\n".format(
                    bjds[i], fluxes[i] / Fstar, scaled_errors[i] / Fstar,
                    systematics[i] / Fstar, astro[i], model[i] / Fstar,
                    residuals[i] / Fstar))
        
        binsize = len(fluxes) // 100
        phases = (bjds-batman_params.t0)/batman_params.per
        phases -= np.round(np.median(phases))

        plot_fit_and_residuals(phases, binsize, fluxes / systematics, astro)
        plt.figure()
        plt.scatter(bjds, systematics)
        print("Residuals STD", np.std(residuals))
        
    if np.random.randint(0, 1000) == 0:
        print(result/len(residuals), Fp, error_factor, np.median(np.abs(residuals)))

    if return_residuals:
        return result, residuals

    if np.isnan(result):
        pdb.set_trace()
    return result

def lnprob_eclipse_limited(params, initial_batman_params, eclipse_model, bjds,
                           fluxes, errors, y, x, wavelength=None,
                           plot_result=False,
                           return_residuals=False, max_Fp=0.02,
                           output_filename="lightcurves.txt"):
    
    Fp, error_factor, Fstar, A, tau, y_coeff, x_coeff, m = params
    batman_params = initial_batman_params
    
    #now account for prior
    if (np.abs(params[0])) >= batman_params.per/4.0:
        return -np.inf
    if tau <= 0.01 or tau > 0.9: return -np.inf
    if Fp < -max_Fp or Fp > max_Fp: return -np.inf
        
    if error_factor <= 0 or error_factor >= 5: return -np.inf
    scaled_errors = error_factor * errors

    delta_t = bjds - bjds[0]
    
    #Hack in x_coeff
    center_waves = np.array([5.3645, 5.9735, 6.5825, 7.1915, 7.8005, 8.4095, 9.0185, 9.6275, 10.2365, 10.8455, 11.4545, 12.0635])

    #Slopes
    #No correction for saturated pixels
    #theoretical_x_coeffs = np.array([-0.0062, -0.018, -0.019, -0.020, -0.018, -0.011, -0.015, -0.012, -0.020, -0.021, -0.017, -0.021])

    #Opt
    theoretical_x_coeffs = np.array([-0.00142, -0.00328, -0.00831, -0.0151, -0.0186, -0.0109, -0.0153, -0.0121, -0.0197, -0.0207, -0.0162, -0.0209])

    #GJ 367b
    #theoretical_x_coeffs = np.array([-0.033, -0.023, -0.024, -0.017, -0.021, -0.011, -0.013, -0.012, -0.020, -0.020, -0.017, -0.020])

    #Simple, window 3
    #theoretical_x_coeffs = np.array([-0.0014, -0.0033, -0.0083, -0.015, -0.019, -0.011, -0.015, -0.012, -0.020, -0.021, -0.016, -0.021])

    #0th group
    #theoretical_x_coeffs = np.array([-0.031, -0.024, -0.016, -0.017, -0.019, -0.010, -0.015, -0.012, -0.021, -0.022, -0.014, -0.027])
    
    curr_theoretical_x_coeff = theoretical_x_coeffs[np.argmin(np.abs(wavelength - center_waves))]
    #x_coeff = curr_theoretical_x_coeff

    
    systematics = Fstar * (1 + A*np.exp(-delta_t/tau) + 0*y_coeff * y + x_coeff * x + m * (bjds - np.mean(bjds)))
    astro = 1 + get_planet_flux(eclipse_model, batman_params, batman_params.t0, batman_params.per, bjds, Fp, 0, 0, 0, 0, t_secondary=batman_params.t_secondary)
    model = systematics * astro
    
    residuals = fluxes - model
    phases = (bjds - batman_params.t0) / batman_params.per
    phases -= np.round(np.median(phases))
    #scaled_errors[phases > 0.544] = 10 #GJ 486b visit 1 only
    
    result = -0.5*(np.sum(residuals**2/scaled_errors**2 - np.log(1.0/scaled_errors**2)))

    if plot_result:
        print("lnprob of plotted", result)
        if not os.path.exists(output_filename):
             with open(output_filename, "w") as f:
                 f.write("#wavelength time flux uncertainty systematics_model astro_model total_model residuals\n")
        
        with open(output_filename, "a") as f:
            for i in range(len(residuals)):
                f.write("{} {} {} {} {} {} {} {}\n".format(
                    wavelength, bjds[i], fluxes[i] / Fstar, scaled_errors[i] / Fstar,
                    systematics[i] / Fstar, astro[i], model[i] / Fstar,
                    residuals[i] / Fstar))
        
        binsize = len(fluxes) // 100

        plot_fit_and_residuals(phases, binsize, fluxes / systematics, astro)
        plt.figure()
        plt.scatter(bjds, systematics)
        print("Residuals STD", np.std(residuals))
        
    if np.random.randint(0, 1000) == 0:
        print(result/len(residuals), Fp, error_factor, np.median(np.abs(residuals)))

    if return_residuals:
        return result, residuals

    if np.isnan(result):
        pdb.set_trace()
    return result

def lnprob(params, initial_batman_params, transit_model, eclipse_model, bjds,
           fluxes, errors, y, x, initial_t0, 
           extra_phase_terms=False, plot_result=False, max_Fp=1,
           return_residuals=False, output_filename="white_lightcurve.txt"):
    transit_offset = params[0]
    eclipse_offset = params[1]
    Fp = params[2]
    C1 = params[3]
    D1 = params[4]
    if extra_phase_terms:
        C2 = params[5]
        D2 = params[6]
        end_phase_terms = 7
    else:
        end_phase_terms = 5

    rp, a_star, b, error_factor, Fstar, A, tau, y_coeff, x_coeff, m = params[end_phase_terms:]    
    inc = np.arccos(b/a_star) * 180/np.pi
    
    batman_params = initial_batman_params
    batman_params.t0 = initial_t0 + transit_offset
    batman_params.rp = rp
    batman_params.a = a_star
    batman_params.inc = inc
    #batman_params.ecc = np.sqrt(ecosw**2 + esinw**2)    
    #batman_params.w = 180/np.pi * np.arctan2(esinw, ecosw)
    batman_params.t_secondary = initial_t0 + batman_params.per/2 + eclipse_offset
    if b <= 0 or b >= 1: return -np.inf

    #now account for prior
    if (np.abs(params[0])) >= batman_params.per/20.0:
        return -np.inf
        
    if (np.abs(params[1])) >= batman_params.per/20.0:
        return -np.inf

    if Fstar <= 0:
        return -np.inf
    
    if tau < 1e-2 or tau > 0.3: return -np.inf
    #if one_over_tau < 5 or one_over_tau > 100: return -np.inf
    #tau = 1./one_over_tau

    if Fp <= 0 or Fp >= max_Fp: return -np.inf
    if error_factor <= 0 or error_factor >= 5: return -np.inf
    #if rp <= 0 or rp >= 1 or a_star <= 0 or b <= 0 or b >= 1: return -np.inf
    
    lnprior = -0.5 * A**2 / 0.1**2
    scaled_errors = error_factor * errors

    delta_t = bjds - bjds[0]
    systematics = Fstar * (1 + A*np.exp(-delta_t/tau) + y_coeff * y + x_coeff * x + m * (bjds - np.mean(bjds)))
    
    if extra_phase_terms:
        Lplanet = get_planet_flux(eclipse_model, batman_params, batman_params.t0, batman_params.per, bjds, Fp, C1, D1, C2, D2, t_secondary=batman_params.t_secondary)
    else:
        Lplanet = get_planet_flux(eclipse_model, batman_params, batman_params.t0, batman_params.per, bjds, Fp, C1, D1, t_secondary=batman_params.t_secondary)

    #if np.min(Lplanet) < 0: return -np.inf
    astro = transit_model.light_curve(batman_params) + Lplanet    
    model = systematics * astro

    phases = (bjds - initial_t0) / batman_params.per
    phases -= np.round(np.median(phases))
    #scaled_errors[np.logical_and(phases > -0.06, phases < -0.011)] = 10
    
    residuals = fluxes - model
    #residuals[np.logical_and(phases > -0.3506, phases < -0.3451)] = 0
    #residuals[np.logical_and(phases > 0.19879, phases < 0.206)] = 0
    result = lnprior -0.5*(np.sum(residuals**2/scaled_errors**2 - np.log(1.0/scaled_errors**2)))

    if plot_result:
        print("lnprob of plotted", result)
        with open(output_filename, "w") as f:
            f.write("#time flux uncertainty systematics_model astro_model total_model residuals\n")        
            for i in range(len(residuals)):
                f.write("{} {} {} {} {} {} {}\n".format(bjds[i], fluxes[i] / Fstar, scaled_errors[i] / Fstar,
                                                        systematics[i] / Fstar, astro[i], model[i] / Fstar,
                                                        residuals[i] / Fstar))
        
        binsize = len(bjds) // 200
        phases = (bjds-batman_params.t0)/batman_params.per
        phases -= np.round(np.median(phases))

        plot_fit_and_residuals(phases, binsize, fluxes / systematics, astro)
        plt.figure()
        plt.scatter(bjds, Lplanet)

        plt.figure()
        plt.scatter(bjds[::binsize], uniform_filter(systematics, binsize)[::binsize])


        print("STD including & excluding near-transit:", np.std(residuals), np.std(residuals[np.abs(phases) > 0.1]))
              
    if np.random.randint(0, 1000) == 0:
        print(result/len(residuals), error_factor, Fp, np.median(np.abs(residuals)), rp)
   
    if return_residuals:
        return result, residuals    

    if np.isnan(result):
        print("result")
    return result

def norm_lnlike(residuals, sigma_sqr):
    return -0.5 * np.sum(residuals**2 / sigma_sqr + np.log(2 * np.pi * sigma_sqr))

def wavelet_lnlike(residuals, sigma_w, sigma_r, gamma=1):
    if np.log2(len(residuals)).is_integer():                                                                                                                                                            
        N = len(residuals)                                                                                                                                                                                 
        padded_residuals = np.copy(residuals)                                                                                                                                                              
    else:                                                                                                                                                                                                  
        power = np.ceil(np.log2(len(residuals)))                                                                                                                                                           
        N = int(2**power)                                                                                                                                                                                  
        padded_residuals = np.zeros(N)                                                                                                                                                                     
        padded_residuals[0:len(residuals)] = residuals                                                                                                                                                     

    assert(np.log2(N).is_integer())
    level = int(np.log2(N / 2))
    result = pywt.wavedec(padded_residuals, 'db2', mode='periodization', level=level)
    cA = result[0]
    cDs = result[1:]

    if gamma == 1:
        g_gamma = 1.0 / (2 * np.log(2))
    else:
        g_gamma = 1.0 / (2**(1-gamma) - 1)

    sigma_cA_sqr = sigma_r**2 * 2**(-gamma) * g_gamma + sigma_w**2
    lnlike = norm_lnlike(cA, sigma_cA_sqr)
    for m in range(1, level + 1):
        sigma_cD_sqr = sigma_r**2 * 2**(-gamma*m) + sigma_w**2
        lnlike += norm_lnlike(cDs[m-1], sigma_cD_sqr)
        
    return lnlike



def lnprob_limited(params, batman_params, transit_model, eclipse_model, bjds,
                           fluxes, errors, y, x, initial_t0, fix_tau, extra_phase_terms=False, wavelength=None, plot_result=False, max_Fp=1,
                           return_residuals=False, output_filename="lightcurves.txt"):
    batman_params = copy.deepcopy(batman_params)
    Fp = params[0]
    C1 = params[1]
    D1 = params[2]
    if extra_phase_terms:
        C2 = params[3]
        D2 = params[4]
        end_phase_terms = 5
    else:
        end_phase_terms = 3
    rp, error_factor, Fstar, A, tau, y_coeff, x_coeff, m = params[end_phase_terms:]

    '''#Hack in x_coeff
    center_waves = np.array([5.3645, 5.9735, 6.5825, 7.1915, 7.8005, 8.4095, 9.0185, 9.6275, 10.2365, 10.8455, 11.4545, 12.0635])
    theoretical_x_coeffs = np.array([-0.033, -0.023, -0.024, -0.017, -0.021, -0.011, -0.013, -0.013, -0.012, -0.019, -0.020, -0.017, -0.020])
    curr_theoretical_x_coeff = theoretical_x_coeffs[np.argmin(np.abs(wavelength - center_waves))]
    x_coeff = curr_theoretical_x_coeff'''
    
    if Fstar <= 0:
        return -np.inf

    if Fp >= max_Fp: return -np.inf
    if error_factor <= 0: return -np.inf
    if rp <= 0 or rp >= 1: return -np.inf
    #if A <= 0: return -np.inf

    #if one_over_tau < 5 or one_over_tau > 100: return -np.inf
    batman_params.rp = rp
    
    #if tau < 0.01 or tau > 1: return -np.inf    
    if tau < 0.01 or tau > 0.2: return -np.inf
    #if tau_power <= 0 or tau_power > 3: return -np.inf
    #tau = 1./one_over_tau
    #tau2 = 1./one_over_tau2
    #if tau2 > tau/2: return -np.inf

    lnprior = 0
    #lnprior = -0.5 * A**2 / 0.1**2 #-0.5 * A2**2 / 0.1**2
    
    delta_t = bjds - bjds[0]

    #Fix ramp values
    #A = 3e-4
    #tau = 0.12
    
    systematics = Fstar * (1 + A*np.exp(-delta_t/tau) + y_coeff * y + x_coeff * x + m * (bjds - np.mean(bjds)))

    if extra_phase_terms:
        Lplanet = get_planet_flux(eclipse_model, batman_params, batman_params.t0, batman_params.per, bjds, Fp, C1, D1, C2, D2, t_secondary=batman_params.t_secondary)
    else:
        Lplanet = get_planet_flux(eclipse_model, batman_params, batman_params.t0, batman_params.per, bjds, Fp, C1, D1, t_secondary=batman_params.t_secondary)

    #if np.min(Lplanet) < 0: return -np.inf
    
    astro = transit_model.light_curve(batman_params) + Lplanet
    model = systematics * astro
    residuals = fluxes - model
    scaled_errors = errors * error_factor

    phases = (bjds-batman_params.t0)/batman_params.per
    phases -= np.round(np.median(phases))

    residuals[np.logical_and(phases > -0.3506, phases < -0.3451)] = 0
    #residuals[np.logical_and(phases > -0.28154, phases < -0.246)] = 0
    residuals[np.logical_and(phases > 0.19879, phases < 0.206)] = 0
    
    result = lnprior - 0.5*(np.sum(residuals**2/scaled_errors**2 - np.log(1.0/scaled_errors**2)))

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
                
        binsize = len(bjds) // 200
        #residuals -= np.mean(residuals)
        phases = (bjds-batman_params.t0)/batman_params.per
        phases -= np.round(np.median(phases))

        plot_fit_and_residuals(phases, binsize, fluxes / systematics, astro)
    
        plt.figure()
        plt.scatter(bjds, Lplanet)
        plt.title("Lplanet")

        plt.figure()
        plt.scatter(bjds, systematics)
        plt.title("systematics")


        print(np.std(residuals[phases < -0.09]), np.std(residuals[phases > -0.09]))
    if np.random.randint(0, 1000) == 0:
        print(result/len(residuals), np.median(scaled_errors), error_factor, Fstar, Fp, np.median(np.abs(residuals)), rp)
    #if result/len(residuals) > 4.7:
    #    Lplanet = get_planet_flux(eclipse_model, batman_params, batman_params.t0, batman_params.per, bjds, Fp, C1, D1, t_secondary=batman_params.t_secondary, pdb=True)
        
    #print(result/len(residuals), Fp, np.median(np.abs(residuals)), rp, a_star, inc, b)
    if return_residuals:
        return result, residuals    

    if np.isnan(result):
        pdb.set_trace()
        print("result")

    return result


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
    
