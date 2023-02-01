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
from constants import HIGH_ERROR, TOP_MARGIN, X_MIN, X_MAX, SUM_EXTRACT_WINDOW, LEFT, RIGHT, BAD_GRPS, WCS_FILE, BKD_REG_TOP, BKD_REG_BOT
from scipy.stats import median_abs_deviation

def simple_extract(image, err):
    spectrum = np.sum(image, axis=0)
    variance = np.sum(err**2, axis=0)
    return spectrum, variance
    
def get_wavelengths(instrument, instrument_filter):
    #Work in progress
    if instrument == "MIRI":
        with fits.open(WCS_FILE) as hdul:
            #print(hdul[1].data["WAVELENGTH"].shape)
            #Subtraction is necessary because image is rotated
            all_ys = np.arange(1024 - LEFT - 1, 1024 - RIGHT - 1, -1)
            wavelengths = np.interp(all_ys,
                                    (hdul[0].header["IMYSLTL"] + hdul[1].data["Y_CENTER"] - 1)[::-1],
                                    hdul[1].data["WAVELENGTH"][::-1])
            return wavelengths
    if instrument == "NIRCAM":
        print("WARNING: wavelength solution to be implemented.  Using hard-coded wavelengths.")
        wavelengths = np.load("{}_wavelength_solution.npy".format(instrument_filter))
        return wavelengths
        
                                
def process_one(filename):
    print("Processing", filename)
    with fits.open(filename) as hdul:
        wavelengths = get_wavelengths(hdul[0].header["INSTRUME"], hdul[0].header["FILTER"])
        hdulist = [hdul[0], hdul["INT_TIMES"]]
    
        for i in range(len(hdul["SCI"].data)):
            print("Processing integration", i)
            
            data = hdul["SCI"].data[i,:,X_MIN:X_MAX]
            err = hdul["ERR"].data[i,:,X_MIN:X_MAX]
            data[:TOP_MARGIN] = 0

            bkd_rows = np.vstack([
                data[BKD_REG_TOP[0]:BKD_REG_TOP[1]],
                data[BKD_REG_BOT[0]:BKD_REG_BOT[1]]])

            bkd_rows = astropy.stats.sigma_clip(bkd_rows, axis=1)
            bkd_err_rows = np.vstack([
                err[BKD_REG_TOP[0]:BKD_REG_TOP[1]],
                err[BKD_REG_BOT[0]:BKD_REG_BOT[1]]])

            bkd = np.ma.mean(bkd_rows, axis=0)
            bkd_var = np.sum(bkd_err_rows**2, axis=0) / bkd_err_rows.shape[1]**2
        
            profile = np.sum(data, axis=1)
            trace_loc = np.argmax(profile)
            s = np.s_[trace_loc - SUM_EXTRACT_WINDOW : trace_loc + SUM_EXTRACT_WINDOW + 1, X_MIN:X_MAX]

            spectrum, variance = simple_extract(
                hdul["SCI"].data[i][s] - bkd,
                hdul["ERR"].data[i][s]            
            )
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
process_one(filenames[0])

with Pool() as pool:
    pool.map(process_one, filenames)
    
