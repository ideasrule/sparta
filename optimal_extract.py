from astropy.io import fits
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebval
import scipy.linalg
import os.path
import astropy.stats
import pdb
from constants import HIGH_ERROR, TOP_MARGIN, X_MIN, X_MAX, OPT_EXTRACT_WINDOW, BAD_GRPS, BKD_REG_TOP, BKD_REG_BOT, Y_CENTER, INSTRUMENT, FILTER, SUBARRAY
from scipy.stats import median_abs_deviation
from wave_sol import get_wavelengths

def horne_iteration(image, bkd, spectrum, M, V, badpix, read_noise, n_groups_used, smoothed_profile, sigma=5):
    #N is the number of groups used, minus one
    V[image == 0] = HIGH_ERROR**2 
    cols = np.arange(image.shape[1])
    
    model_image = smoothed_profile * spectrum
    l = np.arccosh(1 + np.abs(model_image + bkd) / read_noise**2 / 2)
    N = n_groups_used - 1
    V = 1 / (read_noise**-2 * np.exp(l) * (-N*np.exp(-l*N) + np.exp(2*l)*N + np.exp(l-l*N)*(2+N) - np.exp(l)*(2+N)) / (np.exp(l) - 1)**3 / (np.exp(-l*N) + np.exp(l)))
    V[badpix] = HIGH_ERROR**2

    #plt.figure(0, figsize=(18,3))
    #plt.clf() 
    #plt.imshow(badpix, aspect='auto')
    #plt.show()

    z_scores = (image - model_image)/np.sqrt(V)
    M = np.array(z_scores**2 < sigma**2, dtype=bool)
    V[~M] = HIGH_ERROR**2
    original_spectrum = np.copy(spectrum)
    spectrum = np.sum(smoothed_profile * image / V, axis=0) / np.sum(smoothed_profile**2 / V, axis=0)
    spectrum_variance = np.sum(smoothed_profile, axis=0) / np.sum(smoothed_profile**2 / V, axis=0)

    #import pdb
    #pdb.set_trace()
    '''plt.imshow(z_scores, vmin=-5, vmax=5, aspect='auto')
    plt.figure()
    plt.imshow(M)
    plt.show()'''
    
    return spectrum, spectrum_variance, V, M, z_scores

def optimal_extract(image, bkd, badpix, read_noise, n_groups_used, P, max_iter=10):
    #plt.imshow(image, aspect='auto', vmin=0, vmax=20)
    #plt.show()
    
    if badpix is None:
        badpix = np.zeros(image.shape, dtype=bool)
    print("Num badpix", np.sum(badpix))
        
    spectrum = np.sum(image, axis=0)
    simple_spectrum = np.copy(spectrum)
    
    V = np.ones(image.shape)
    M = np.ones(image.shape, dtype=bool)
    counter = 0
    
    while True:        
        spectrum, spectrum_variance, V, new_M, z_scores = horne_iteration(image, bkd, spectrum, M, V, badpix, read_noise, n_groups_used, P)
        #plt.figure(figsize=(16,16))
        #plt.imshow(z_scores, vmin=-10, vmax=10)
        #plt.figure()
        
        print("Iter, num bad:", counter, np.sum(~new_M))
        if np.all(M == new_M) or counter > max_iter: break
        M = new_M
        counter += 1
    print("Final std of z_scores (should be around 1)", np.std(z_scores[M]))
    return spectrum, spectrum_variance, z_scores, simple_spectrum

def get_profile(filename="median_image.npy"):
    median_image = np.load(filename)[Y_CENTER - OPT_EXTRACT_WINDOW : Y_CENTER + OPT_EXTRACT_WINDOW + 1, X_MIN : X_MAX]
    median_spectrum = np.sum(median_image, axis=0)
    P = median_image / median_spectrum       
    P[P < 0] = 0
    P /= np.sum(P, axis=0)
    return P


print("Applying optimal extraction")
P = get_profile()    

for filename in sys.argv[1:]:
    with fits.open(filename) as hdul:
        assert(hdul[0].header["INSTRUME"] == INSTRUMENT and hdul[0].header["FILTER"] == FILTER and hdul[0].header["SUBARRAY"] == SUBARRAY)
        wavelengths = get_wavelengths(hdul[0].header["INSTRUME"], hdul[0].header["FILTER"])
        hdulist = [hdul[0], hdul["INT_TIMES"]]

        for i in range(len(hdul["SCI"].data)):
            print("Processing integration", i)

            data = hdul["SCI"].data[i,:,X_MIN:X_MAX]
            err = hdul["ERR"].data[i,:,X_MIN:X_MAX]
            data[:TOP_MARGIN] = 0

            s = np.s_[Y_CENTER - OPT_EXTRACT_WINDOW : Y_CENTER + OPT_EXTRACT_WINDOW + 1, X_MIN : X_MAX]
          
            spectrum, variance, z_scores, simple_spectrum = optimal_extract(
                hdul["SCI"].data[i][s],
                hdul["BKD"].data[i][s],
                hdul["DQ"].data[i][s] != 0,
                hdul["RNOISE"].data[s],
                hdul[0].header["NGROUPS"] - BAD_GRPS,
                P)
            bkd = hdul["BKD"].data[i][s].mean(axis=0)
            hdulist.append(fits.BinTableHDU.from_columns([
                fits.Column(name="WAVELENGTH", format="D", unit="um", array=wavelengths[X_MIN:X_MAX]),
                fits.Column(name="FLUX", format="D", unit="Electrons/group", array=spectrum),
                fits.Column(name="ERROR", format="D", unit="Electrons/group", array=np.sqrt(variance)),
                fits.Column(name="SIMPLE FLUX", format="D", unit="Electrons/group", array=simple_spectrum),
                fits.Column(name="BKD", format="D", unit="Electrons/group", array=bkd)
            ]))

            if i == 20: 
                z_scores_filename = "zscores_{}_" + filename[:-4] + "png"
                plt.clf()
                plt.figure(0, figsize=(18,3))
                plt.imshow(z_scores, vmin=-5, vmax=5, aspect='auto')
                plt.savefig(z_scores_filename.format(i))
                #plt.show()

                spectra_filename = "optspectra_{}_" + filename[:-4] + "png"
                N = hdul[0].header["NGROUPS"] - 1 - BAD_GRPS
                plt.clf()
                plt.plot(spectrum * N, label="Spectra")
                plt.plot(variance * N**2, label="Variance")
                plt.savefig(spectra_filename.format(i))


        output_hdul = fits.HDUList(hdulist)    
        output_hdul.writeto("optx1d_" + os.path.basename(filename), overwrite=True)
