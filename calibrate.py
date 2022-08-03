import numexpr as ne
import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import time
import sys
import os.path
import pdb
import gc
from constants import LEFT_MARGIN, BAD_GRPS, SLITLESS_LEFT, SLITLESS_RIGHT, SLITLESS_TOP, SLITLESS_BOT, NONLINEAR_FILE, DARK_FILE, FLAT_FILE, RNOISE_FILE, RESET_FILE, MASK_FILE, GAIN, NONLINEAR_COEFFS

def get_mask():
    with astropy.io.fits.open(MASK_FILE) as hdul:
        mask = hdul["DQ"].data[SLITLESS_TOP:SLITLESS_BOT, SLITLESS_LEFT:SLITLESS_RIGHT]
    return mask

def apply_reset(data):
    after_reset = np.copy(data)
    with astropy.io.fits.open(RESET_FILE) as hdul:
        reset = hdul[1].data
        dq = hdul["DQ"].data
        print("Note: throw away first {} integrations".format(reset.shape[0] - 1))
        N_grp = data.shape[1]
        N_grp_reset = reset.shape[1]
        if N_grp <= N_grp_reset:
            after_reset -= reset[-1, :N_grp]
        else:
            after_reset[:, :N_grp_reset] -= reset[-1]

    return after_reset, dq

#@profile
def apply_nonlinearity(data):   
    start = time.time()
    c2, c3, c4 = NONLINEAR_COEFFS

    data2 = data * data
    result = data + c2 * data2 + c3 * data2*data + c4 * data2 * data2
    end = time.time()
    print("Non linearity took", end - start)
    mask = np.zeros(data[0,0].shape, dtype=bool)
    return result, mask



#Dark current subtraction
def subtract_dark(data):
    with astropy.io.fits.open(DARK_FILE) as dark_hdul:
        dark = dark_hdul[1].data
        mask = dark[-1,0] == 0

    N_int_dark = dark.shape[0]
    N_grp_dark = dark.shape[1]
    
    assert(N_grp <= N_grp_dark)
    
    result = np.copy(data)

    print("Note: throw away first {} integrations".format(dark.shape[0] - 1))
    result -= dark[-1, :N_grp]
    return result, mask

def get_slopes_initial(after_gain, read_noise):
    N_grp = after_gain.shape[1]
    N = N_grp - 1 - 1 #reject last group
    j = np.array(np.arange(1, N + 1), dtype=float)

    R = read_noise[:, LEFT_MARGIN:]
    cutout = after_gain[:,:-1,:,LEFT_MARGIN:] #reject last group
    signal_estimate = (cutout[:,-1] - cutout[:, 0]) / N
    error = np.zeros(signal_estimate.shape)
    diff_array = np.diff(cutout, axis=1)
    noise = np.sqrt(2*R[np.newaxis,]**2 + np.abs(signal_estimate))

    for i in range(len(cutout)):
        ratio_estimate = signal_estimate[i]  / R**2
        ratio_estimate[ratio_estimate < 1e-6] = 1e-6
        l = np.arccosh(1 + ratio_estimate/2)[:, :, np.newaxis]

        weights = -R[:,:,np.newaxis]**-2 * ne.evaluate("exp(l) * (1 - exp(-j*l)) * (exp(j*l - l*N) - exp(l)) / (exp(l) - 1)**2 / (exp(l) + exp(-l*N))")
        signal_estimate[i] = np.sum(diff_array[i].transpose(1,2,0) * weights, axis=2) / np.sum(weights, axis=2)                
        error[i] = 1. / np.sqrt(np.sum(weights, axis=2))

    #Fill in borders in order to maintain size
    full_signal_estimate = (after_gain[:, -1] - after_gain[:, 0]) / N
    full_signal_estimate[:,:,LEFT_MARGIN:] = signal_estimate
        
    full_error = np.ones(full_signal_estimate.shape) * np.inf
    full_error[:,:,LEFT_MARGIN:] = error

    residuals = after_gain - (full_signal_estimate*np.arange(N_grp)[:,np.newaxis,np.newaxis,np.newaxis]).transpose((1,0,2,3))
    median_residuals = np.median(residuals, axis=0)
    median_residuals -= np.median(median_residuals, axis=0)

    return full_signal_estimate, full_error, median_residuals


def get_slopes(after_gain, read_noise, max_iter=50):
    N = N_grp - 1 - BAD_GRPS
    j = np.array(np.arange(1, N + 1), dtype=float)

    R = read_noise[:, LEFT_MARGIN:]
    cutout = after_gain[:,BAD_GRPS:,:,LEFT_MARGIN:]
    #signal_estimate = (cutout[:,-1] - cutout[:, 0]) / N

    diff_array = np.diff(cutout, axis=1)
    signal_estimate = np.clip(np.median(diff_array, axis=1), 0, None)
    error = np.zeros(signal_estimate.shape)
    noise = np.sqrt(2*R[np.newaxis,]**2 + signal_estimate)
    bad_mask = np.zeros(diff_array.shape, dtype=bool)
    pixel_bad_mask = np.zeros(signal_estimate.shape, dtype=bool)

    #pdb.set_trace()
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
            bad_mask[i] = np.abs(z_scores) > 5
            weights[bad_mask[i].transpose(1,2,0)] = 0
            weights_sum = np.sum(weights, axis=2)
            if np.sum(weights_sum == 0) > 0:
                bad_pos = np.nonzero(weights_sum == 0)
                print("These pixels are bad in integration {}: y,x={}".format(i, bad_pos))
                pixel_bad_mask[i][bad_pos] = True
                weights[bad_pos] += 1

            slightly_bad = np.sum(bad_mask[i], axis=0) > 5
            pixel_bad_mask[i][slightly_bad] = True
                
            signal_estimate[i] = np.sum(diff_array[i].transpose(1,2,0) * weights, axis=2) / np.sum(weights, axis=2)
            error[i] = 1. / np.sqrt(np.sum(weights, axis=2))

        num_changed = np.sum(old_bad_mask != bad_mask)
        if num_changed == 0:
            break
        print("Num changed", iteration, num_changed)

    #gc.collect()
    #Fill in borders in order to maintain size
    full_signal_estimate = (after_gain[:, -1] - after_gain[:, 0]) / N
    full_signal_estimate[:,:,LEFT_MARGIN:] = signal_estimate
        
    full_error = np.ones(full_signal_estimate.shape) * np.inf
    full_error[:,:,LEFT_MARGIN:] = error

    full_pixel_mask = np.zeros(full_signal_estimate.shape, dtype=bool)
    full_pixel_mask[:,:,LEFT_MARGIN:] = pixel_bad_mask

    #Free some memory
    cutout = None
    diff_array = None
    bad_mask = None
    gc.collect()
    
    residuals = after_gain - (full_signal_estimate*np.arange(N_grp)[:,np.newaxis,np.newaxis,np.newaxis]).transpose((1,0,2,3))
    median_residuals = np.median(residuals, axis=0)
    median_residuals -= np.median(median_residuals, axis=0)
    return full_signal_estimate, full_error, full_pixel_mask, median_residuals


def apply_flat(signal, error, include_flat_error=False):
    with astropy.io.fits.open(FLAT_FILE) as hdul:
        flat = hdul["SCI"].data
        flat_err = hdul["ERR"].data

    invalid = np.isnan(flat)
    flat[invalid] = 1
    final_signal = signal / flat
    if include_flat_error:
        final_error = np.sqrt((error / flat)**2 + final_signal**2 * flat_err**2)
    else:
        final_error = error / flat
    final_error[:,invalid] = np.inf
    return final_signal, final_error, flat_err

def get_read_noise():
    with astropy.io.fits.open(RNOISE_FILE) as hdul:
        return GAIN * hdul[1].data[SLITLESS_TOP:SLITLESS_BOT, SLITLESS_LEFT:SLITLESS_RIGHT]

filename = sys.argv[1]
hdul = astropy.io.fits.open(filename)

#Assumptions for dark current subtraction
assert(hdul[0].header["NFRAMES"] == 1)
assert(hdul[0].header["GROUPGAP"] == 0)

data = np.array(hdul[1].data, dtype=float)
N_int, N_grp, N_row, N_col = data.shape

mask = get_mask()

print("Applying reset correction")
data, dq = apply_reset(data)
mask |= dq

print("Applying non-linearity correction")
data, dq = apply_nonlinearity(data)
mask |= dq
gc.collect()

print("Applying dark correction")
data, dark_mask = subtract_dark(data)
mask |= dark_mask

print("Applying gain correction")
data = data * GAIN

read_noise = get_read_noise()
print("Getting slopes 1")

signal, error, residuals1 = get_slopes_initial(data, read_noise)
data -= residuals1

print("Getting slopes 2")
signal, error, per_int_mask, residuals2 = get_slopes(data, read_noise)
print("Applying flat")

final_signal, final_error, flat_err = apply_flat(signal, error)
per_int_mask = per_int_mask | mask

sci_hdu = astropy.io.fits.ImageHDU(final_signal, name="SCI")
err_hdu = astropy.io.fits.ImageHDU(final_error, name="ERR")
dq_hdu = astropy.io.fits.ImageHDU(per_int_mask, name="DQ")
flat_err_hdu = astropy.io.fits.ImageHDU(flat_err, name="FLATERR")
res1_hdu = astropy.io.fits.ImageHDU(residuals1, name="RESIDUALS1")
res2_hdu = astropy.io.fits.ImageHDU(residuals2, name="RESIDUALS2")
read_noise_hdu = astropy.io.fits.ImageHDU(read_noise, name="RNOISE")
output_hdul = astropy.io.fits.HDUList([hdul[0], sci_hdu, err_hdu, dq_hdu, flat_err_hdu, read_noise_hdu, res1_hdu, res2_hdu])
output_hdul.writeto("rateints_" + os.path.basename(filename), overwrite=True)
