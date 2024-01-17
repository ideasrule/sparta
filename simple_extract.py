from astropy.io import fits
import astropy.stats
import sys
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebval
import scipy.linalg
import os.path
import pdb
from multiprocessing import Pool
from constants import HIGH_ERROR, TOP_MARGIN, X_MIN, X_MAX, SUM_EXTRACT_WINDOW, BAD_GRPS, BKD_REG_TOP, BKD_REG_BOT, INSTRUMENT, FILTER, SUBARRAY, Y_CENTER
from scipy.stats import median_abs_deviation
from wave_sol import get_wavelengths
from fitting import robust_polyfit, fit_gaussian

def get_trace(image):
    col_nums = []
    trace_ys = []

    for x in range(image.shape[1]):
        y_indices = np.arange(image.shape[0])
        extracted_data = image[y_indices, x]
        try:
            coeffs, _, _ = fit_gaussian(y_indices, extracted_data)
            trace_y = coeffs[1]
        except RuntimeError:
            continue
            
        trace_ys.append(trace_y)
        col_nums.append(x)

    col_nums = np.array(col_nums)
    trace_ys = np.array(trace_ys)
    if len(col_nums) == 0: return None
    
    '''plt.figure(0, figsize=(20,4))
    plt.clf()
    plt.plot(col_nums, trace_ys)
    plt.savefig("trace.png")'''

    trace, residuals = robust_polyfit(col_nums, trace_ys, 2, include_residuals=True, target_xs=np.arange(image.shape[1]))
    '''plt.figure(figsize=(20,4))
    plt.clf()
    plt.plot(col_nums, residuals)
    plt.axhline(0, color='r')
    plt.savefig("trace_residuals.png")
    np.save("trace.npy", trace)
    #plt.show()'''
    return trace

def get_pixel_sum(image, min_y, max_y, x):
    result = np.sum(image[int(min_y) : int(max_y), x])
    result -= (min_y - int(min_y)) * image[int(min_y), x]
    result += (max_y - int(max_y)) * image[int(max_y), x]
    return result

def simple_extract(image, err, window=2):
    if INSTRUMENT=="MIRI" or INSTRUMENT=="NIRSPEC":
        #Do the simple thing of ignoring trace curvature
        spectrum = image.sum(axis=0)
        variance = (err**2).sum(axis=0)
        return spectrum, variance
    
    trace = get_trace(image)
    spectrum = np.zeros(image.shape[1])
    variance = np.zeros(image.shape[1])
    for c in range(image.shape[1]):
        min_y = trace[c] - window
        max_y = trace[c] + window + 1
        spectrum[c] = get_pixel_sum(image, min_y, max_y, c)
        variance[c] = get_pixel_sum(err**2, min_y, max_y, c)

    return spectrum, variance
        
def process_one(filename):
    print("Processing", filename)
    with fits.open(filename) as hdul:
        assert(hdul[0].header["INSTRUME"] == INSTRUMENT and hdul[0].header["FILTER"] == FILTER and hdul[0].header["SUBARRAY"] == SUBARRAY)
        wavelengths = get_wavelengths(hdul[0].header["INSTRUME"], hdul[0].header["FILTER"])
        hdulist = [hdul[0], hdul["INT_TIMES"]]
    
        for i in range(len(hdul["SCI"].data)):
            print("Processing integration", i)
            
            data = hdul["SCI"].data[i,:,X_MIN:X_MAX]
            err = hdul["ERR"].data[i,:,X_MIN:X_MAX]
            data[:TOP_MARGIN] = 0
            
            s = np.s_[Y_CENTER - SUM_EXTRACT_WINDOW : Y_CENTER + SUM_EXTRACT_WINDOW + 1, X_MIN:X_MAX]

            spectrum, variance = simple_extract(
                hdul["SCI"].data[i][s],
                hdul["ERR"].data[i][s]            
            )

            bkd = np.mean(hdul["BKD"].data[i][s], axis=0)
            bkd_var = np.mean(hdul["BKD_ERR"].data[i][s]**2, axis=0)
            variance += bkd_var * (2*SUM_EXTRACT_WINDOW + 1)**2
        
            hdulist.append(fits.BinTableHDU.from_columns([
                fits.Column(name="WAVELENGTH", format="D", unit="um", array=wavelengths[X_MIN:X_MAX]),
                fits.Column(name="FLUX", format="D", unit="Electrons/group", array=spectrum),
                fits.Column(name="ERROR", format="D", unit="Electrons/group", array=np.sqrt(variance)),
                fits.Column(name="BKD", format="D", unit="Electrons/group", array=bkd)
            ]))

            if i == 20:            
                spectra_filename = "spectra_{}_" + filename[:-4] + "png"
                N = hdul[0].header["NGROUPS"] - 1 - BAD_GRPS
                plt.clf()
                plt.plot(spectrum * N, label="Spectra")
                plt.plot(variance * N**2, label="Variance")
                plt.savefig(spectra_filename.format(i))
    
        output_hdul = fits.HDUList(hdulist)
        output_hdul.writeto("x1d_" + os.path.basename(filename), overwrite=True)
    
filenames = sys.argv[1:]
with Pool() as pool:
    pool.map(process_one, filenames)
    
