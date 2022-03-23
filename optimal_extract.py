from astropy.io import fits
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebval
import scipy.linalg
import os.path
from constants import HIGH_ERROR, LEFT_MARGIN, EXTRACT_WINDOW, SLITLESS_TOP, SLITLESS_BOT, BAD_GRPS
from scipy.stats import median_abs_deviation

def fit_spectrum(image_row, spectrum, weights, num_ord=5):
    cols = np.arange(len(spectrum))
    xs = (cols - np.mean(cols))/len(cols) * 2
    A = []
    for o in range(num_ord):
        cheby_coeffs = np.zeros(o + 1)
        cheby_coeffs[o] = 1
        cheby = chebval(xs, cheby_coeffs)
        A.append(spectrum * cheby)

    A = np.array(A).T
    Aw = A * np.sqrt(weights[:, np.newaxis])
    Bw = image_row * np.sqrt(weights)
    coeffs, residuals, rank, s = scipy.linalg.lstsq(Aw, Bw)    
    predicted = Aw.dot(coeffs)
    smoothed_profile = chebval(xs, coeffs)    
    return smoothed_profile

def horne_iteration(image, bkd, spectrum, M, V, badpix, flat_err, read_noise, n_groups_used, sigma=5):
    #N is the number of groups used, minus one
    V[image == 0] = HIGH_ERROR**2 
    smoothed_profile = np.zeros(image.shape)
    cols = np.arange(image.shape[1])
    
    for r in range(image.shape[0]):
        weights = 1.0/V[r] 
        weights[~M[r]] = 0
        smoothed = fit_spectrum(image[r], spectrum, weights)
        smoothed_profile[r] = smoothed
                            
    smoothed_profile[smoothed_profile < 0] = 0
    smoothed_profile[image == 0] = 0
    smoothed_profile /= np.sum(smoothed_profile, axis=0)[np.newaxis, :]
    
    model_image = smoothed_profile * spectrum
    l = np.arccosh(1 + np.abs(model_image + bkd) / read_noise**2 / 2)
    N = n_groups_used - 1
    V = 1 / (read_noise**-2 * np.exp(l) * (-N*np.exp(-l*N) + np.exp(2*l)*N + np.exp(l-l*N)*(2+N) - np.exp(l)*(2+N)) / (np.exp(l) - 1)**3 / (np.exp(-l*N) + np.exp(l)))
    V += ((model_image + bkd) * flat_err)**2
    V[badpix] = HIGH_ERROR**2

    #plt.figure(0, figsize=(18,3))
    #plt.clf() 
    #plt.imshow(badpix, aspect='auto')
    #plt.show()

    z_scores = (image - model_image)/np.sqrt(V)
    M = np.array(z_scores**2 < sigma**2, dtype=bool)
    V[~M] = HIGH_ERROR**2
    spectrum = np.sum(smoothed_profile * image / V, axis=0) / np.sum(smoothed_profile**2 / V, axis=0)
    spectrum_variance = np.sum(smoothed_profile, axis=0) / np.sum(smoothed_profile**2 / V, axis=0)

    return spectrum, spectrum_variance, V, smoothed_profile, M, z_scores

def optimal_extract(image, bkd, badpix, flat_err, read_noise, n_groups_used, max_iter=10):
    #plt.imshow(image, aspect='auto', vmin=0, vmax=20)
    #plt.show()
    
    if badpix is None:
        badpix = np.zeros(image.shape, dtype=bool)
        
    spectrum = np.sum(image, axis=0)
    simple_spectrum = np.copy(spectrum)
    
    V = np.ones(image.shape)
    M = np.ones(image.shape, dtype=bool)
    counter = 0
    
    while True:        
        spectrum, spectrum_variance, V, P, new_M, z_scores = horne_iteration(image, bkd, spectrum, M, V, badpix, flat_err, read_noise, n_groups_used)
        #plt.figure(figsize=(16,16))
        #plt.imshow(z_scores, vmin=-10, vmax=10)
        #plt.figure()
        
        print("Iter, num bad:", counter, np.sum(~new_M))
        if np.all(M == new_M) or counter > max_iter: break
        M = new_M
        counter += 1
    print("Final std of z_scores (should be around 1)", np.std(z_scores[M]))
    return spectrum, spectrum_variance, z_scores, simple_spectrum

def get_wavelengths(filename="jwst_miri_specwcs_0003.fits"):
    with fits.open(filename) as hdul:
        all_ys = np.arange(SLITLESS_TOP, SLITLESS_BOT)
        wavelengths = np.interp(all_ys,
                                (hdul[0].header["IMYSLTL"] + hdul[1].data["Y_CENTER"] - 1)[::-1],
                                hdul[1].data["WAVELENGTH"][::-1])
        return wavelengths
                                

print("Applying optimal extraction")
filename = sys.argv[1]
with fits.open(filename) as hdul:
    wavelengths = get_wavelengths()    
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
        data = hdul["SCI"].data[i]
        data[:, 0:LEFT_MARGIN] = 0
        profile = np.sum(data, axis=0)
        trace_loc = np.argmax(profile)
        s = np.s_[:, trace_loc - EXTRACT_WINDOW : trace_loc + EXTRACT_WINDOW]
        spectrum, variance, z_scores, simple_spectrum = optimal_extract(
            hdul["SCI"].data[i][s].T,
            hdul["BKD"].data[i][s].T,
            None,
            hdul["FLATERR"].data[s].T,
            hdul["RNOISE"].data[s].T,
            hdul[0].header["NGROUPS"] - BAD_GRPS
        )
        
        hdulist.append(fits.BinTableHDU.from_columns([
            fits.Column(name="WAVELENGTH", format="D", unit="um", array=wavelengths),
            fits.Column(name="FLUX", format="D", unit="Electrons/group", array=spectrum),
            fits.Column(name="ERROR", format="D", unit="Electrons/group", array=np.sqrt(variance)),
            fits.Column(name="SIMPLE FLUX", format="D", unit="Electrons/group", array=simple_spectrum)
        ]))

        if i == int(len(hdul["SCI"].data)/2):
            z_scores_filename = "zscores_{}_" + filename[:-4] + "png"
            plt.clf()
            plt.figure(0, figsize=(18,3))
            plt.imshow(z_scores, vmin=-5, vmax=5, aspect='auto')
            plt.savefig(z_scores_filename.format(i))
            #plt.show()

            spectra_filename = "spectra_{}_" + filename[:-4] + "png"
            N = hdul[0].header["NGROUPS"] - 1 - BAD_GRPS
            plt.clf()
            plt.plot(spectrum * N, label="Spectra")
            plt.plot(variance * N**2, label="Variance")
            plt.savefig(spectra_filename.format(i))

    
    output_hdul = fits.HDUList(hdulist)    
    output_hdul.writeto("x1d_" + os.path.basename(filename), overwrite=True)
