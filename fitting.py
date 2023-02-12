import numpy as np
import scipy.interpolate
import scipy.signal
from scipy.optimize import curve_fit
import astropy.stats
import matplotlib.pyplot as plt
import time


def fit_gaussian(xs, ys, errors=None, std_guess=1):
    if errors is None:
        errors = np.ones(len(xs))
        
    #If xs or ys are masked arrays, get only valid entries
    mask = np.zeros(len(xs), dtype=bool)
    if xs is np.ma.MaskedArray:
        mask = xs.mask
    if ys is np.ma.MaskedArray:
        mask = np.logical_or(mask, ys.mask)
    xs = xs[~mask]
    ys = ys[~mask]
    errors = errors[~mask]

    #print(errors)
    def gauss(xs, *p):
        A, mu, sigma, pedestal = p
        return A*np.exp(-(xs-mu)**2/(2.*sigma**2)) + pedestal

    amplitude_guess = np.max(ys)
    mean_guess = xs[np.argmax(ys)]
    p0 = [amplitude_guess, mean_guess, std_guess, 0]
    coeff, var_matrix = curve_fit(gauss, xs, ys, p0=p0, sigma=errors)
    if np.any(var_matrix < 0):
        #ill conditioned fit, set to infinite variance
        fit_errors = np.ones(len(p0)) * np.inf
    else:
        assert(not np.any(np.diag(var_matrix) < 0))
        fit_errors = np.sqrt(np.diag(var_matrix))
    return coeff, fit_errors, gauss(xs, *coeff)


def robust_polyfit(xs, ys, deg, target_xs=None, include_residuals=False, inverse_sigma=None):
    if target_xs is None: target_xs = xs
    ys = astropy.stats.sigma_clip(ys)
    residuals = ys - np.polyval(np.ma.polyfit(xs, ys, deg), xs)
    ys.mask = astropy.stats.sigma_clip(residuals).mask
    last_mask = np.copy(ys.mask)
    while True:
        coeffs = np.ma.polyfit(xs, ys, deg, w=inverse_sigma)
        predicted_ys = np.polyval(coeffs, xs)
        residuals = ys - predicted_ys
        ys.mask = astropy.stats.sigma_clip(residuals).mask
        if np.all(ys.mask == last_mask):
            break
        else:
            last_mask = np.copy(ys.mask)
        
    result = np.polyval(coeffs, target_xs)
    if include_residuals:
        return result, residuals
    return result

