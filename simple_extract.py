from astropy.io import fits
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebval
import scipy.linalg
import os.path
import pdb
from constants import HIGH_ERROR, LEFT_MARGIN, EXTRACT_Y_MIN, EXTRACT_Y_MAX, SUM_EXTRACT_WINDOW, SLITLESS_TOP, SLITLESS_BOT, BAD_GRPS, WCS_FILE, BKD_WIDTH
from scipy.stats import median_abs_deviation


def fix_outliers(data, badpix):
    final = np.copy(data)
    for c in range(LEFT_MARGIN, data.shape[1]):
        rows = np.arange(data.shape[0])
        good = ~badpix[:,c]
        repaired = np.interp(rows, rows[good], data[:,c][good])
        final[:,c] = repaired
    return final

def simple_extract(image, err):
    spectrum = np.sum(image, axis=1)
    variance = np.sum(err**2, axis=1)
    #pdb.set_trace()
    fractions = np.sum(image, axis=0) / np.sum(spectrum)
    return spectrum, variance, fractions[2], fractions[3], fractions[4]
    
def get_wavelengths():
    with fits.open(WCS_FILE) as hdul:
        all_ys = np.arange(SLITLESS_TOP, SLITLESS_BOT)
        wavelengths = np.interp(all_ys,
                                (hdul[0].header["IMYSLTL"] + hdul[1].data["Y_CENTER"] - 1)[::-1],
                                hdul[1].data["WAVELENGTH"][::-1])
        return wavelengths
                                


    
print("Applying simple extraction")
filename = sys.argv[1]
with fits.open(filename) as hdul:
    wavelengths = get_wavelengths()
    #pdb.set_trace()
    second_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="integration_number", format="i4"),
        fits.Column(name="int_start_MJD_UTC", format="f8"),
        fits.Column(name="int_mid_MJD_UTC", format="f8"),
        fits.Column(name="int_end_MJD_UTC", format="f8"),
        fits.Column(name="int_start_BJD_TDB", format="f8"),
        fits.Column(name="int_mid_BJD_TDB", format="f8"),
        fits.Column(name="int_end_BJD_TDB", format="f8")])
    
    hdulist = [hdul[0], second_hdu]
    
    for i in range(len(hdul["SCI"].data)):
        print("Processing integration", i)

        #Manually input bad pixels
        hdul["DQ"].data[i,382,39] = 1
        hdul["DQ"].data[i,382,58] = 1
        
        hdul["SCI"].data[i] = fix_outliers(hdul["SCI"].data[i], hdul["DQ"].data[i] != 0) 
        
        data = hdul["SCI"].data[i][EXTRACT_Y_MIN:EXTRACT_Y_MAX]
        err = hdul["ERR"].data[i][EXTRACT_Y_MIN:EXTRACT_Y_MAX]
        data[:, 0:LEFT_MARGIN] = 0
        bkd = np.mean(data[:, -BKD_WIDTH:], axis=1)
        bkd_var = np.sum(err[:, -BKD_WIDTH:]**2, axis=1) / BKD_WIDTH**2
        
        profile = np.sum(data, axis=0)
        trace_loc = np.argmax(profile)
        s = np.s_[EXTRACT_Y_MIN:EXTRACT_Y_MAX, trace_loc - SUM_EXTRACT_WINDOW : trace_loc + SUM_EXTRACT_WINDOW + 1]
        
        spectrum, variance, fl, fc, fr = simple_extract(
            hdul["SCI"].data[i][s] - bkd[:, np.newaxis],
            hdul["ERR"].data[i][s]            
        )
        variance += bkd_var * (2*SUM_EXTRACT_WINDOW + 1)**2
        
        hdulist.append(fits.BinTableHDU.from_columns([
            fits.Column(name="WAVELENGTH", format="D", unit="um", array=wavelengths[EXTRACT_Y_MIN:EXTRACT_Y_MAX]),
            fits.Column(name="FLUX", format="D", unit="Electrons/group", array=spectrum),
            fits.Column(name="ERROR", format="D", unit="Electrons/group", array=np.sqrt(variance)),
            fits.Column(name="BKD", format="D", unit="Electrons/group", array=bkd)
        ]))
        hdulist[-1].header["fl"] = fl
        hdulist[-1].header["fc"] = fc
        hdulist[-1].header["fr"] = fr

        if i == 20:            
            spectra_filename = "spectra_{}_" + filename[:-4] + "png"
            N = hdul[0].header["NGROUPS"] - 1 - BAD_GRPS
            plt.clf()
            plt.plot(spectrum * N, label="Spectra")
            plt.plot(variance * N**2, label="Variance")
            plt.savefig(spectra_filename.format(i))

    
    output_hdul = fits.HDUList(hdulist)
    
    output_hdul.writeto("x1d_" + os.path.basename(filename), overwrite=True)
