from astropy.io import fits
import os.path
import numpy as np
from constants import WCS_FILE, LEFT, RIGHT

def get_wavelengths(instrument, instrument_filter):
    #Work in progress
    if instrument == "MIRI":
        with fits.open(WCS_FILE) as hdul:
            #Subtraction is necessary because image is rotated
            all_ys = np.arange(1024 - LEFT - 1, 1024 - RIGHT - 1, -1)
            wavelengths = np.interp(all_ys,
                                    (hdul[0].header["IMYSLTL"] + hdul[1].data["Y_CENTER"] - 1)[::-1],
                                    hdul[1].data["WAVELENGTH"][::-1])
            return wavelengths
    if instrument == "NIRCAM":
        print("WARNING: wavelength solution to be implemented.  Using hard-coded wavelengths.")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        wavelengths = np.load("{}/{}_wavelength_solution.npy".format(script_dir, instrument_filter))
        return wavelengths
        
