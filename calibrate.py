import astropy.io.fits
import matplotlib.pyplot as plt
import _cupy_numpy as np
import scipy.interpolate
import numexpr as ne
import time
import sys
import os.path
import pdb
import gc
import pdb
import argparse
import numpy
import asdf
from scipy.ndimage import uniform_filter
from scipy.signal import correlate
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from constants import INSTRUMENT, EMICORR_FILE, FILTER, SUBARRAY, TOP_MARGIN, BAD_GRPS, \
LEFT, RIGHT, TOP, BOT, NONLINEAR_FILE, DARK_FILE, FLAT_FILE, RNOISE_FILE, MASK_FILE, GAIN_FILE, \
ROTATE, SKIP_SUPERBIAS, SUPERBIAS_FILE, SKIP_FLAT, SKIP_REF, N_REF, ROW_CLOCK, FRAME_CLOCK, XSTART, XSIZE

def destripe(data):
    #data: N_int x N_grp x N_row x N_col
    bkd_pixels = np.concatenate((data[:,:,BKD_REG_TOP[0] : BKD_REG_TOP[1]],
                          data[:,:,BKD_REG_BOT[0] : BKD_REG_BOT[1]]),
                         axis=2)
    bkd = np.median(bkd_pixels, axis=2)
    return data - bkd[:,:,np.newaxis,:]


def linear_fit(data):

    n_groups, ny, nx = data.shape
    t = np.arange(n_groups)  # Time index array

    # Initialize arrays to store results
    slopes = np.zeros((ny, nx))
    intercepts = np.zeros((ny, nx))

    # Compute sum values for linear regression
    sx = np.sum(t)  # Sum of time indices
    sxx = np.sum(t**2)  # Sum of squared time indices

    for j in range(ny):  # Loop over rows
        for i in range(nx):  # Loop over columns
            y = data[:, j, i]  # Extract pixel time series

            sy = np.sum(y)  # Sum of pixel values
            sxy = np.sum(t * y)  # Sum of (time * pixel values)

            # Compute slope (m) and intercept (b) using least-squares formula
            m = (n_groups * sxy - sx * sy) / (n_groups * sxx - sx * sx)
            b = (sxx * sy - sx * sxy) / (n_groups * sxx - sx * sx)

            slopes[j, i] = m
            intercepts[j, i] = b

    return slopes, intercepts

def get_emicorr_ref():
    af = asdf.open(EMICORR_FILE)
    tree = af.tree 
    frequency = tree["frequencies"]['Hz10']['frequency']
    phase_amplitudes = tree["frequencies"]['Hz10']['phase_amplitudes']

    return frequency, phase_amplitudes

def emicorr(data, frequency, ref_pa, bin_numbers = 500, dataname = None):
    print("EmiCorr going on......")
    print("Correct for {} Hz".format(frequency))
    data = np.array(hdul[1].data, dtype=float)
    print(data.shape)
    stacked = 0
    for i in range(4):
        stacked += 0.25 * data[:,:,:,i::4]


    nints, ngroups, ny, nx = data.shape
    nx4 = int(nx/4)
    nsamples = 1
    xsize = XSIZE
    xstart = XSTART
    colstop = int(xsize / 4 + xstart - 1)
    time_read_each_row = ROW_CLOCK
    time_read_frame = FRAME_CLOCK
    phaseall = np.zeros((nints, ngroups, ny, nx4))

    residualsall = np.zeros((nints, ngroups, ny, nx4))
    start_time = 0
    for n in range(nints):
        slopes, intercepts = linear_fit(stacked[n])
    
        times_this_int = np.zeros((ngroups, ny, nx4), dtype = "ulonglong")
        for k in range(ngroups):   # frames
            residualsall[n, k]  = stacked[n, k] - (slopes * k + intercepts)
            for j in range(ny):    # rows
                ### nsamples= 1 for fast, 9 for slow (from metadata)
                times_this_int[k, j, :] = np.arange(nx4, dtype='ulonglong') * nsamples + start_time
    
                if colstop == 258:
                    times_this_int[k,j,nx4-1] = times_this_int[k,j,nx4-1] + 2**32
                    ### Do not include the last row
                start_time += time_read_each_row
        
        if n == 10:
            plt.figure()
            plt.plot(np.arange(ngroups), np.arange(ngroups) * slopes[25, 25] + intercepts[25, 25])
            plt.plot(stacked[n, :,25,25])
            plt.savefig("emicorr_linear_int10.png")

    
    
        period_in_pixels = (1./frequency) / 10.0e-6
        #print("Period in pixels is {}".format(period_in_pixels))
    
        phase_this_int = times_this_int / period_in_pixels
        phaseall[n,:,:,:] = phase_this_int - phase_this_int.astype('ulonglong')
        start_time += time_read_frame


    binned_phases = np.linspace(0, 1, bin_numbers)
    binned_vals = np.zeros(len(binned_phases))
    dp = np.median(np.diff(binned_phases))

    for i in range(len(binned_phases)):
        cond = (phaseall > binned_phases[i] - dp/2) & (phaseall < binned_phases[i] + dp/2)
        mean, median, std = sigma_clipped_stats(residualsall[cond])
        binned_vals[i] = median

    binned_vals -= np.mean(binned_vals)

    #print(ref_pa)
    correlation = correlate(binned_vals - np.mean(binned_vals), ref_pa - np.mean(ref_pa), mode='full')
    lags = np.arange(-len(ref_pa) + 1, len(ref_pa))  # Lag values
    best_lag = lags[np.argmax(correlation)]
    phase_shift = best_lag

    shifted_pa = np.roll(ref_pa, phase_shift)

    phase_amplitudes_period = np.interp(np.linspace(0,1,int(period_in_pixels)+1), np.linspace(0,1,500), shifted_pa)
    m, b = np.polyfit(shifted_pa , binned_vals, 1)
    vals_period = np.interp(np.linspace(0,1,int(period_in_pixels)+1), np.linspace(0,1,500), binned_vals)
    #import pdb
    #pdb.set_trace()
    #pa_final = phase_amplitudes_period * m
    pa_final = vals_period
    #import pdb
    #pdb.set_trace()

    noise4 = pa_final[(phaseall * period_in_pixels).astype(int)]

    noise = np.zeros((nints, ngroups, ny, nx))   # same size as input data
    noise_x = np.arange(nx4) * 4
    for k in range(4):
        noise[:, :, :, noise_x + k] = noise4

    plt.figure()
    plt.plot(np.arange(len(pa_final)), pa_final)
    #plt.plot(np.arange(len(binned_vals)),binned_vals)
    plt.savefig("emicorr_profile_"+dataname+".png")

    return data - noise


def sigma_clip_images(data, sigma=3.0, maxiters=5):
    """
    Perform sigma clipping on the last two dimensions (images) of a 4D array.
    
    Parameters:
    - data: np.ndarray, shape (T1, T2, H, W) where (H, W) are image dimensions
    - sigma: float, the number of standard deviations to use for clipping
    - maxiters: int, maximum number of clipping iterations
    
    Returns:
    - clipped_data: np.ndarray, same shape as data but with outliers replaced with NaN
    - mask: np.ndarray, same shape as data, where True indicates clipped values
    """
    # Apply sigma clipping along the last two dimensions (spatial axes)
    clipped_result = sigma_clip(data, sigma=sigma, maxiters=maxiters, axis=(2,3), masked=True)

    # Extract the clipped data and mask
    clipped_data = clipped_result.filled(0)  # Replace masked values with NaN
    mask = clipped_result.mask  # Boolean mask, True for clipped values
    
    return clipped_data, mask

def row_by_row_background_subtraction(data):
    print("doing row by row subtraction")
    nints, ngroups, nx, ny = data.shape
    bkg_total = numpy.zeros((nints, ngroups, nx, ny))

    center = (69, 61)  # (row, col)
    width = 80  # Square width

    x_start = center[0] - width // 2
    x_end = center[0] + width // 2
    y_start = center[1] - width // 2
    y_end = center[1] + width // 2

    mask_new = np.zeros(data.shape, dtype = bool)
    mask_new[:,:, y_start:y_end, x_start:x_end] = True
    masked_data = np.where(~mask_new, data, 0)

    clipped_data, _ = sigma_clip_images(masked_data)

    bkg = np.median(clipped_data, axis = 2)
    print(bkg.shape)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(masked_data[0,-1], vmax = np.percentile(masked_data[0,-1], 0.95), vmin = np.percentile(masked_data[0,-1], 0.05))
    ax[0].imshow(mask_new[0,-1])
    ax[0].scatter(center[0], center[1])
    ax[1].imshow(np.zeros((data.shape[2], data.shape[3])) + bkg[0,-1,numpy.newaxis,:], vmax = np.percentile(bkg[0,-1], 0.95), vmin = np.percentile(bkg[0,-1], 0.05))
    #plt.colorbar()
    plt.show()

    return data - bkg[:,:,numpy.newaxis,:]

def get_mask():
    with astropy.io.fits.open(MASK_FILE) as hdul:
        mask = np.asarray(np.rot90(hdul["DQ"].data, ROTATE)[TOP:BOT, LEFT:RIGHT])
    return mask


def get_gain():
    with astropy.io.fits.open(GAIN_FILE) as hdul:
        substrt_x = int(hdul[0].header["SUBSTRT1"]) - 1
        substrt_y = int(hdul[0].header["SUBSTRT2"]) - 1
        gain = np.asarray(np.rot90(hdul[1].data, ROTATE)[
            TOP-substrt_y : BOT-substrt_y,
            LEFT-substrt_x : RIGHT-substrt_x])
    return gain


def subtract_superbias(data):
    #There are a LOT of pixels with UNRELIABLE_BIAS flag that seem perfectly
    #fine otherwise, so we ignore the superbias DQ
    with astropy.io.fits.open(SUPERBIAS_FILE) as hdul:
        superbias = np.rot90(hdul[1].data, ROTATE)
        if superbias.shape != data.shape[2:]:
            superbias = superbias[TOP:BOT, LEFT:RIGHT]
        
    superbias[np.isnan(superbias)] = 0
    return data - superbias


def subtract_ref(data, noutputs):
    result = np.copy(data)
    chunk_size = int(data.shape[-1] / hdul[0].header["NOUTPUTS"])

    #Subtract ref along top
    for c in range(int(data.shape[-1]/chunk_size)):
        c_min = c * chunk_size
        c_max = (c + 1) * chunk_size
        mean = np.mean(data[:,:,:N_REF,c_min:c_max], axis=(2,3))
        result[:,:,:,c_min:c_max] -= mean[:,:,np.newaxis,np.newaxis]

    #Subtract ref along sides
    mean = np.mean(result[:,:,:,:N_REF], axis=3) / 2 + np.mean(result[:,:,:,-N_REF:], axis=3) / 2
    result -= mean[:,:,:,np.newaxis]
    return result


def apply_nonlinearity(data):   
    start = time.time()

    with astropy.io.fits.open(NONLINEAR_FILE) as hdul:
        substrt_x = int(hdul[0].header["SUBSTRT1"]) - 1
        substrt_y = int(hdul[0].header["SUBSTRT2"]) - 1
        coeffs = np.array(np.rot90(hdul[1].data, ROTATE, (-2,-1))[:,
            TOP-substrt_y : BOT-substrt_y,
            LEFT-substrt_x : RIGHT-substrt_x], dtype=np.float64)
        dq = np.array(np.rot90(hdul["DQ"].data, ROTATE)[
            TOP-substrt_y : BOT-substrt_y,
            LEFT-substrt_x : RIGHT-substrt_x])
                      
        result = np.zeros(data.shape, dtype=np.float64)
        exp_data = np.ones(data.shape, dtype=np.float64)
        
        for i in range(len(coeffs)):
            print(i)
            result += coeffs[i] * exp_data
            exp_data *= data
    end = time.time()
    print("Non linearity took", end - start)
    return result, dq > 0


def subtract_dark(data, nframes, groupgap):
    with astropy.io.fits.open(DARK_FILE) as hdul:
        #substrt_x = int(hdul[0].header["SUBSTRT1"]) - 1
        #substrt_y = int(hdul[0].header["SUBSTRT2"]) - 1
        dark = np.array(hdul[1].data, dtype=np.float64)
        dq = np.array(hdul["DQ"].data)
        if dark.ndim == 4:
            print("Warning: skipping first {} integrations of dark".format(dark.shape[0] - 1))
            dark = dark[-1]
            dq = dq[0,0]
    
    #Make dark frame the right size
    if not (nframes == 1 and groupgap == 0):
        assert(nframes == 1 or (nframes/2).is_integer())
        total_frames = nframes + groupgap
        indices = np.arange(dark.shape[0]) % total_frames
        include = indices < nframes
        trunc_dark = dark[include]
        final_dark = uniform_filter(trunc_dark, [nframes,1,1])[int(nframes/2)::nframes]
    else:
        final_dark = dark
        
    assert(data.shape[1] <= final_dark.shape[0])
    
    result = data - final_dark[:data.shape[1]]
    if INSTRUMENT == "NIRSPEC":
        #NIRSPEC dark mask has a lot of DQ flags, most of which don't seem to be reflected in actual data anomalies
        mask = np.zeros(dq.shape, dtype=bool)
    else:
        mask = dq > 0
    #import pdb
    #pdb.set_trace()
    return result, mask


def get_slopes_initial(after_gain, read_noise, bad_grps=0):
    N_grp = after_gain.shape[1]
    if N_grp > 3 and "MIRI" in INSTRUMENT:
        ignore_last = 1
        print("Ignore the last group in ramp fitting")
    else:
        ignore_last = 0

    N = N_grp - 1 - ignore_last - bad_grps
    j = np.array(np.arange(1, N + 1), dtype=np.float64)

    R = read_noise[TOP_MARGIN:]
    cutout = after_gain[:,bad_grps:N_grp-ignore_last,TOP_MARGIN:] #reject last group
    
    signal_estimate = (cutout[:,-1] - cutout[:, 0]) / N
    error = np.zeros(signal_estimate.shape)
    diff_array = np.diff(cutout, axis=1)
    noise = np.sqrt(2*R[np.newaxis,]**2 + np.absolute(signal_estimate))

    for i in range(len(cutout)):
        ratio_estimate = signal_estimate[i]  / R**2
        ratio_estimate[ratio_estimate < 1e-6] = 1e-6
        l = np.arccosh(1 + ratio_estimate/2)[:, :, np.newaxis]

        #weights = -R[:,:,np.newaxis]**-2 * -np.ones(len(j))
        #weights = -R[:,:,np.newaxis]**-2 * j*(j - N - 1)/2
        weights = -R[:,:,np.newaxis]**-2 * ne.evaluate("exp(l) * (1 - exp(-j*l)) * (exp(j*l - l*N) - exp(l)) / (exp(l) - 1)**2 / (exp(l) + exp(-l*N))")
        #weights = -R[:,:,np.newaxis]**-2 * np.exp(l) * (1 - np.exp(-j*l)) * (np.exp(j*l - l*N) - np.exp(l)) / (np.exp(l) - 1)**2 / (np.exp(l) + np.exp(-l*N))
        #weights[:,:,0:5] = 0
        #weights[:,:,-1] = 0
        signal_estimate[i] = np.sum(diff_array[i].transpose(1,2,0) * weights, axis=2) / np.sum(weights, axis=2)                
        error[i] = 1. / np.sqrt(np.sum(weights, axis=2))

    #Fill in borders in order to maintain size    
    full_signal_estimate = (after_gain[:, -1] - after_gain[:, 0]) / N
    full_signal_estimate[:,TOP_MARGIN:] = signal_estimate
        
    full_error = np.ones(full_signal_estimate.shape) * np.inf
    full_error[:,TOP_MARGIN:] = error

    if N_grp == 2:
        median_residuals = np.zeros(after_gain.shape[1:])
    else:
        residuals = after_gain - (full_signal_estimate*np.arange(N_grp)[:,np.newaxis,np.newaxis,np.newaxis]).transpose((1,0,2,3))
        median_residuals = np.median(residuals, axis=0)
        median_residuals -= np.median(median_residuals, axis=0)

    return full_signal_estimate, full_error, median_residuals


def get_slopes(after_gain, read_noise, max_iter=100, sigma=5, bad_grps=0):
    N_grp = after_gain.shape[1]
    if N_grp > 3 and "MIRI" in INSTRUMENT:
        ignore_last = 1
        print("Ignore the last group in ramp fitting")
    else:
        ignore_last = 0


    N = N_grp - 1 - ignore_last - bad_grps
    j = np.array(np.arange(1, N + 1), dtype=np.float64)
    R = read_noise[TOP_MARGIN:]
    cutout = after_gain[:,bad_grps:N_grp-ignore_last,TOP_MARGIN:]

    diff_array = np.diff(cutout, axis=1)

    signal_estimate = np.clip(np.median(diff_array, axis=1), 0, None)
    error = np.zeros(signal_estimate.shape)
    noise = np.sqrt(2*R[np.newaxis,]**2 + signal_estimate)
    bad_mask = np.zeros(diff_array.shape, dtype=bool)
    pixel_bad_mask = np.zeros(signal_estimate.shape, dtype=bool)

    for iteration in range(max_iter):
        old_bad_mask = np.copy(bad_mask)
        for i in range(len(cutout)):
            ratio_estimate = signal_estimate[i]  / R**2
            ratio_estimate[ratio_estimate < 1e-6] = 1e-6
            l = np.arccosh(1 + ratio_estimate/2)[:, :, np.newaxis]
            #weights = -R[:,:,np.newaxis]**-2 * j*(j - N - 1)/2
            weights = -R[:,:,np.newaxis]**-2 * ne.evaluate("exp(l) * (1 - exp(-j*l)) * (exp(j*l - l*N) - exp(l)) / (exp(l) - 1)**2 / (exp(l) + exp(-l*N))")
            #weights = -R[:,:,np.newaxis]**-2 * np.exp(l) * (1 - np.exp(-j*l)) * (np.exp(j*l - l*N) - np.exp(l)) / (np.exp(l) - 1)**2 / (np.exp(l) + np.exp(-l*N))
            #weights[:,:,:BAD_GRPS] = 0
            #weights[:,:,-1] = 0
            
            #Find cosmic rays and other anomalies
            z_scores = (diff_array[i] - signal_estimate[i, np.newaxis]) / noise[i, np.newaxis]
            bad_mask[i] = np.absolute(z_scores) > sigma
            weights[bad_mask[i].transpose(1,2,0)] = 0
            weights_sum = np.sum(weights, axis=2)
            if np.sum(weights_sum == 0) > 0:
                bad_pos = np.nonzero(weights_sum == 0)
                print("These pixels are bad in integration {}: y,x={}".format(i, bad_pos))
                pixel_bad_mask[i][bad_pos] = True
                weights[bad_pos] += 1

            slightly_bad = np.sum(bad_mask[i], axis=0) > sigma
            pixel_bad_mask[i][slightly_bad] = True
                
            signal_estimate[i] = np.sum(diff_array[i].transpose(1,2,0) * weights, axis=2) / np.sum(weights, axis=2)
   
            error[i] = 1. / np.sqrt(np.sum(weights, axis=2))

        num_changed = np.sum(old_bad_mask != bad_mask)
        if num_changed == 0:
            break
        print("Num changed", iteration, num_changed)
        gc.collect()

    #Fill in borders in order to maintain size
    full_signal_estimate = (after_gain[:, -1] - after_gain[:, 0]) / N
    full_signal_estimate[:,TOP_MARGIN:] = signal_estimate

    full_error = np.ones(full_signal_estimate.shape) * np.inf
    full_error[:,TOP_MARGIN:] = error

    full_pixel_mask = np.zeros(full_signal_estimate.shape, dtype=bool)
    full_pixel_mask[:,TOP_MARGIN:] = pixel_bad_mask

    #Free some memory
    cutout = None
    diff_array = None
    bad_mask = None
    gc.collect()

    if N_grp == 2:
        median_residuals = np.zeros(after_gain.shape)
    else:
        residuals = after_gain - (full_signal_estimate*np.arange(N_grp)[:,np.newaxis,np.newaxis,np.newaxis]).transpose((1,0,2,3))
        median_residuals = np.median(residuals, axis=0)
        median_residuals -= np.median(median_residuals, axis=0)
    return full_signal_estimate, full_error, full_pixel_mask, median_residuals


def set_slopes_saturated(after_gain, signal, grps_to_sat):
    grps_to_sat = np.load(grps_to_sat)
        
    for i in range(after_gain.shape[0]):
        for g in range(after_gain.shape[1] - 1, 1, -1):
            saturated = grps_to_sat <= g
            original = np.copy(signal[i])
            signal[i, saturated] = (after_gain[i, g-1, saturated] - after_gain[i, 0, saturated]) / (g - 1)

            
def apply_flat(signal, error, include_flat_error=False):
    with astropy.io.fits.open(FLAT_FILE) as hdul:
        substrt_x = int(hdul[0].header["SUBSTRT1"]) - 1
        substrt_y = int(hdul[0].header["SUBSTRT2"]) - 1
        flat = np.array(np.rot90(hdul['SCI'].data, ROTATE)[
            TOP-substrt_y : BOT-substrt_y,
            LEFT-substrt_x : RIGHT-substrt_x], dtype=np.float64)
        flat_err = np.array(np.rot90(hdul['ERR'].data, ROTATE)[
            TOP-substrt_y : BOT-substrt_y,
            LEFT-substrt_x : RIGHT-substrt_x], dtype=np.float64)

    invalid = np.isnan(flat)
    flat[invalid] = 1
    final_signal = signal / flat
    if include_flat_error:
        final_error = np.sqrt((error / flat)**2 + final_signal**2 * flat_err**2)
    else:
        final_error = error / flat
    final_error[:,invalid] = np.inf
    return final_signal, final_error, flat_err


def get_read_noise(gain):    
    with astropy.io.fits.open(RNOISE_FILE) as hdul:
        substrt_x = int(hdul[0].header["SUBSTRT1"]) - 1
        substrt_y = int(hdul[0].header["SUBSTRT2"]) - 1
        return gain * np.array(np.rot90(hdul[1].data, ROTATE)[
            TOP-substrt_y : BOT-substrt_y,
            LEFT-substrt_x : RIGHT-substrt_x], dtype=np.float64) / np.sqrt(2)

    
def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)

parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs="+")
parser.add_argument("--median-residuals")
parser.add_argument("--grps-to-sat")
args = parser.parse_args()

for filename in args.filenames:
    print("Processing", filename)
    hdul = astropy.io.fits.open(filename)

    #assert(hdul[0].header["INSTRUME"] == INSTRUMENT and hdul[0].header["FILTER"] == FILTER and hdul[0].header["SUBARRAY"] == SUBARRAY)
    
    #Assumptions for dark current subtraction
    nframes = hdul[0].header["NFRAMES"]
    groupgap = hdul[0].header["GROUPGAP"]
    assert(is_power_of_two(nframes))

    data = np.array(
        np.rot90(hdul[1].data, ROTATE, axes=(2,3)),
        dtype=np.float64)

        

    start = time.time()
    frequency, ref_pa = get_emicorr_ref()

    data = emicorr(data, frequency, ref_pa, dataname=filename.split("/")[-1].split(".")[0])
    end = time.time()

    print("EmiCorr takes", end - start)
    if SUBARRAY == 'FULL': 
        data = data[:,:,TOP:BOT,LEFT:RIGHT]
        print("Data cropped to", TOP, BOT, LEFT, RIGHT)

    
    #data = data[:,5:]
    print("data shape is", data.shape)
    N_int, N_grp, N_row, N_col = data.shape
    mask = get_mask()
    #import pdb
    #pdb.set_trace()
    #data = row_by_row_background_subtraction(data)

    if not SKIP_SUPERBIAS:
        print("Subtracting superbias")
        data = subtract_superbias(data)

    if not SKIP_REF:
        print("Subtracting ref pixels")
        data = subtract_ref(data, hdul[0].header["NOUTPUTS"])

    print("Applying non-linearity correction")

    data, dq = apply_nonlinearity(data)
    np.save(filename.split(".")[-2].split("_")[-2]+"_linearity", data)
    mask |= dq
    gc.collect()

    if INSTRUMENT=="NIRSPEC":
        #Do other instruments benefit from this? Haven't checked
        print("Subtracting 1/f noise group by group")
        data = destripe(data)

    print("Applying dark correction")
    data, dark_mask = subtract_dark(data, nframes, groupgap)
    mask |= dark_mask

    print("Applying gain correction")
    gain = get_gain()

    data = data * gain
    

    read_noise = get_read_noise(gain)
    print("Getting slopes 1")

    #original_data = np.copy(data)

    signal, error, residuals1 = get_slopes_initial(data, read_noise)
    if args.median_residuals is not None:
        data -= np.load(args.median_residuals)
        print("Subtracting median residuals")
    else:
        print("Not subtracting median residuals")
        data -= residuals1

    print("Getting slopes 2")
    data[:,:,:,-1] = 0 #sometimes anomalous
    signal, error, per_int_mask, residuals2 = get_slopes(data, read_noise)


    if args.grps_to_sat is not None:
        set_slopes_saturated(data, signal, args.grps_to_sat)
    
    if not SKIP_FLAT:
        print("Applying flat")
        signal, error, flat_err = apply_flat(signal, error)
    signal *= (data.shape[1] - 1)
    error *= (data.shape[1] - 1)
    #signal *= 46
    per_int_mask = per_int_mask | mask
    per_int_mask = per_int_mask | np.isnan(signal)
    sci_hdu = astropy.io.fits.ImageHDU(np.cpu(signal), name="SCI")
    err_hdu = astropy.io.fits.ImageHDU(np.cpu(error), name="ERR")
    dq_hdu = astropy.io.fits.ImageHDU(np.cpu(per_int_mask), name="DQ")
    res1_hdu = astropy.io.fits.ImageHDU(np.cpu(residuals1), name="RESIDUALS1")
    res2_hdu = astropy.io.fits.ImageHDU(np.cpu(residuals2), name="RESIDUALS2")
    read_noise_hdu = astropy.io.fits.ImageHDU(np.cpu(read_noise), name="RNOISE")
    output_hdul = astropy.io.fits.HDUList([hdul[0], sci_hdu, err_hdu, dq_hdu, read_noise_hdu, res1_hdu, res2_hdu, hdul["INT_TIMES"]])
    output_hdul.writeto("rateints_" + os.path.basename(filename).replace("_uncal", ""), overwrite=True)
    output_hdul.close()
    hdul.close()
