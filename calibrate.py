import numexpr as ne
import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import time
import sys
import os.path
from constants import LEFT_MARGIN, BAD_GRPS, SLITLESS_LEFT, SLITLESS_RIGHT, SLITLESS_TOP, SLITLESS_BOT, NONLINEAR_FILE, DARK_FILE, FLAT_FILE, RNOISE_FILE, RESET_FILE, GAIN

def apply_reset(data):
    after_reset = np.copy(data)
    with astropy.io.fits.open(RESET_FILE) as hdul:
        reset = hdul[1].data
        print("Note: throw away first {} integrations".format(reset.shape[0] - 1))
        N_grp = data.shape[1]
        N_grp_reset = reset.shape[1]
        if N_grp <= N_grp_reset:
            after_reset -= reset[-1, :N_grp]
        else:
            after_reset[:, :N_grp_reset] -= reset[-1]

    return after_reset

#@profile
def apply_nonlinearity(data):
    start = time.time()
    data_float = np.array(data, dtype=float)
    with astropy.io.fits.open(NONLINEAR_FILE) as hdul:
        coeffs = np.copy(hdul[1].data[:, SLITLESS_TOP:SLITLESS_BOT, SLITLESS_LEFT:SLITLESS_RIGHT])
        result = np.zeros(data.shape)
        exp_data = np.ones(data.shape)
        
        for i in range(len(coeffs)):
            result += coeffs[i] * exp_data
            exp_data *= data_float
    end = time.time()
    print("Non linearity took", end - start)
    return result



#Dark current subtraction
def subtract_dark(data):
    with astropy.io.fits.open(DARK_FILE) as dark_hdul:
        dark = dark_hdul[1].data

    N_int_dark = dark.shape[0]
    N_grp_dark = dark.shape[1]
    
    assert(N_grp <= N_grp_dark)
    
    result = np.copy(data)

    if N_int > N_int_dark:
        result[:N_int_dark] -= dark[:, :N_grp]
        result[N_int_dark:] -= dark[-1, :N_grp]
    else:
        result -= dark[:N_int, :N_grp]
    return result

def get_slopes(after_gain, read_noise, initial_run=True, max_iter=50):
    if initial_run:
        max_iter = 1
        
    N = N_grp - 1 - BAD_GRPS
    j = np.array(np.arange(1, N + 1), dtype=float)

    R = read_noise[:, LEFT_MARGIN:]
    cutout = after_gain[:,BAD_GRPS:,:,LEFT_MARGIN:]
    signal_estimate = (cutout[:,-1] - cutout[:, 0]) / N
    error = np.zeros(signal_estimate.shape)
    diff_array = np.diff(cutout, axis=1)
    noise = np.sqrt(2*R[np.newaxis,]**2 + signal_estimate)
    bad_mask = np.zeros(diff_array.shape, dtype=bool)

    for iteration in range(max_iter):
        old_bad_mask = np.copy(bad_mask)
        for i in range(len(cutout)):
            ratio_estimate = signal_estimate[i]  / R**2
            ratio_estimate[ratio_estimate < 1e-6] = 1e-6
            l = np.arccosh(1 + ratio_estimate/2)[:, :, np.newaxis]

            weights = -R[:,:,np.newaxis]**-2 * ne.evaluate("exp(l) * (1 - exp(-j*l)) * (exp(j*l - l*N) - exp(l)) / (exp(l) - 1)**2 / (exp(l) + exp(-l*N))")
            #plt.imshow(R)
            #plt.show()

            #Find cosmic rays and other anomalies
            z_scores = (diff_array[i] - signal_estimate[i, np.newaxis]) / noise[i, np.newaxis]
            if initial_run:
                bad_mask[i] = np.zeros(z_scores.shape, dtype=bool)
            else:
                bad_mask[i] = np.abs(z_scores) > 5
            weights[bad_mask[i].transpose(1,2,0)] = 0

            signal_estimate[i] = np.sum(diff_array[i].transpose(1,2,0) * weights, axis=2) / np.sum(weights, axis=2)
            error[i] = 1. / np.sqrt(np.sum(weights, axis=2))

        num_changed = np.sum(old_bad_mask != bad_mask)
        if num_changed == 0:
            break
        print("Num changed", iteration, num_changed)

    #Fill in borders in order to maintain size
    full_signal_estimate = (after_gain[:, -1] - after_gain[:, 0]) / N
    full_signal_estimate[:,:,LEFT_MARGIN:] = signal_estimate
        
    full_error = np.ones(full_signal_estimate.shape) * np.inf
    full_error[:,:,LEFT_MARGIN:] = error
    return full_signal_estimate, full_error


def apply_flat(signal, error):
    with astropy.io.fits.open(FLAT_FILE) as hdul:
        flat = hdul["SCI"].data
        flat_err = hdul["ERR"].data

    invalid = np.isnan(flat)
    flat[invalid] = 1
    final_signal = signal / flat
    final_error = np.sqrt((error / flat)**2 + final_signal**2 * flat_err**2)
    final_error[:,invalid] = np.inf
    return final_signal, final_error, flat_err

def get_read_noise():
    with astropy.io.fits.open(RNOISE_FILE) as hdul:
        return GAIN * hdul[1].data[SLITLESS_TOP:SLITLESS_BOT, SLITLESS_LEFT:SLITLESS_RIGHT]

filename = sys.argv[1]
hdul = astropy.io.fits.open(filename)
if len(sys.argv) == 2:
    initial_run = True
elif len(sys.argv) == 3:
    initial_run = False
    residual_correction = np.load(sys.argv[2])
else:
    assert(False)

#Assumptions for dark current subtraction
assert(hdul[0].header["NFRAMES"] == 1)
assert(hdul[0].header["GROUPGAP"] == 0)

data = hdul[1].data
N_int, N_grp, N_row, N_col = data.shape

print("Applying reset correction")
after_reset = apply_reset(data)

print("Applying non-linearity correction")
after_nonlinear = apply_nonlinearity(after_reset)

print("Applying dark correction")
after_dark = subtract_dark(after_nonlinear)

print("Applying gain correction")
after_gain = after_dark * GAIN

read_noise = get_read_noise()
print("Getting slopes")
#import pdb
#pdb.set_trace()

signal, error = get_slopes(after_gain, read_noise, False)
print("Applying flat")
final_signal, final_error, flat_err = apply_flat(signal, error)

sci_hdu = astropy.io.fits.ImageHDU(final_signal, name="SCI")
err_hdu = astropy.io.fits.ImageHDU(final_error, name="ERR")
flat_err_hdu = astropy.io.fits.ImageHDU(flat_err, name="FLATERR")
read_noise_hdu = astropy.io.fits.ImageHDU(read_noise, name="RNOISE")
output_hdul = astropy.io.fits.HDUList([hdul[0], sci_hdu, err_hdu, flat_err_hdu, read_noise_hdu])
output_hdul.writeto("rateints_" + os.path.basename(filename), overwrite=True)
