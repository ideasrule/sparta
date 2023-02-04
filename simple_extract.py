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

def simple_extract(image, err):
    spectrum = np.sum(image, axis=0)
    variance = np.sum(err**2, axis=0)
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
    
