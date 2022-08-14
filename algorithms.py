import pdb
import numpy as np
import sys
import scipy.signal
import matplotlib.pyplot as plt
import astropy.stats
from sklearn.neighbors import NearestNeighbors
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from scipy.stats import binned_statistic
from scipy.ndimage import uniform_filter, median_filter
from sklearn.decomposition import PCA
import scipy.stats
import glob
import pickle
from astropy.io import fits

def get_mad(data):
    return np.median(np.abs(data - np.median(data)))

def bin_data(data, bin_width=128, axis=None):
    num_bins = int(len(data) / bin_width)
    if len(data) % num_bins != 0: num_bins += 1
    means = [np.mean(data[int(i * bin_width): int((i + 1) * bin_width)], axis=axis) for i in range(num_bins)]
    return np.array(means)
    
def smooth(data, window_len=101):
    return scipy.signal.medfilt(data, window_len)
    
    
def reject_beginning(bjds, fluxes, errors, cutoff_in_days=0.0, max_separation_in_days=0.01):
    #now make a beginning mask
    start_points = []
    beginning_mask = []
    for i in range(len(bjds)):
        if i==0 or bjds[i]-bjds[i-1] > max_separation_in_days:
            start_points.append(bjds[i])
        if len(start_points)==0 or (bjds[i] > start_points[-1]+cutoff_in_days):
            beginning_mask.append(False)
        else:
            beginning_mask.append(True)
    beginning_mask = np.array(beginning_mask)
    print("Num of starting points: ", len(start_points))
    print("Number of points rejected at beginning: " + str(np.sum(beginning_mask)))
    valid_data = ~beginning_mask    
    return bjds[valid_data], fluxes[valid_data], errors[valid_data]
    

def calc_binned_rms(residuals, photon_noise, min_datapoints = 16):
    bin_sizes = []
    all_rms = []
    photon_noises = []
    log_bin_size = 0

    while len(residuals)/2**log_bin_size > min_datapoints:
        bin_size = 2**log_bin_size
        #binned_residuals = bin_data(residuals, bin_size)
        
        binned_residuals, _, _ = binned_statistic(range(len(residuals)), residuals, bins=int(len(residuals)/2**log_bin_size))
        #rms = astropy.stats.sigma_clipped_stats(binned_residuals)[2]
        rms = np.std(binned_residuals)
        bin_sizes.append(bin_size)
        all_rms.append(rms)
        photon_noises.append(photon_noise/np.sqrt(bin_size))
        log_bin_size += 1
    return bin_sizes, all_rms, photon_noises

def robust_polyfit(xs, ys, deg, target_xs=None, include_residuals=False, inverse_sigma=None):
    if target_xs is None: target_xs = xs
    ys = astropy.stats.sigma_clip(ys)
    residuals = ys - np.polyval(np.ma.polyfit(xs, ys, deg), xs)
    ys.mask = astropy.stats.sigma_clip(residuals).mask
    last_mask = np.copy(ys.mask)
    while True:
        coeffs = np.ma.polyfit(xs, ys, deg, w=inverse_sigma)
        predicted_ys = np.polyval(coeffs, xs)
        residuals = ys - predicted_ys
        ys.mask = astropy.stats.sigma_clip(residuals).mask
        if np.all(ys.mask == last_mask):
            break
        else:
            last_mask = np.copy(ys.mask)
        
    result = np.polyval(coeffs, target_xs)
    if include_residuals:
        return result, residuals
    return result

def get_data_pickle(min_wavelength, max_wavelength, trim_start=525, filename="data.pkl"):
    result = pickle.load(open(filename, "rb"))
    cond = np.logical_and(result["wavelengths"] >= min_wavelength/1000,
                          result["wavelengths"] < max_wavelength/1000)
      
    data = np.sum(result["data"][trim_start:, cond], axis=1)
    var = result["errors"]**2
    errors = np.sqrt(np.sum(var[trim_start:, cond], axis=1))
    median = np.median(data)
    data /= median
    errors /= median
    y = result["y"][trim_start:]

    return result["times"][trim_start:], data, errors, result["wavelengths"][cond], y


def get_data_txt(start_bin, end_bin, trim_start=2000, filename="lightcurve.txt"):
    wavelength, time, flux, error = np.loadtxt(filename, unpack=True)
    unique_wavelengths = np.sort(np.unique(wavelength))
    print(unique_wavelengths[48:54])
    #print("Ind", np.argwhere(unique_wavelengths == 6.6584))
    if end_bin == -1:
        end_bin = len(unique_wavelengths)
    #pdb.set_trace()

    binned_fluxes = 0.
    binned_errors = 0.
    for b in range(start_bin, end_bin):
        cond = wavelength == unique_wavelengths[b]
        binned_fluxes += flux[cond]
        binned_errors += error[cond]**2

    binned_errors = np.sqrt(binned_errors)
    median = np.median(binned_fluxes)

    '''#pdb.set_trace()
    test_fluxes = []
    test_var = []
    test_waves = []
    for i in range(int(len(unique_wavelengths)/6)):
        start = 6*i
        end = start + 6
        lc = 0.
        var = 0.
        test_waves.append(np.mean(unique_wavelengths[start:end]))
        for b in range(start, end):
            lc += flux[wavelength == unique_wavelengths[b]][trim_start:]
            var += error[wavelength == unique_wavelengths[b]][trim_start:]**2
        
        test_fluxes.append(lc)
        test_var.append(var)

    test_fluxes = np.array(test_fluxes)
    test_var = np.array(test_var)
    t1 = 3000
    t2 = 3300
    t3 = 5600
    t4 = 6000
    
    for i in range(test_fluxes.shape[0]):
        out_flux = test_fluxes[i, 0:t1].sum() + test_fluxes[i, t4:].sum()
        out_var = test_var[i, 0:t1].sum() + test_var[i, t4:].sum()
        in_flux = test_fluxes[i, t2:t3].sum()
        in_var = test_var[i, t2:t3].sum()
        in_flux_mean = in_flux / (t3-t2)
        out_flux_mean = out_flux / (test_fluxes.shape[1] - t4 + t1)
        depth = 1 - in_flux_mean / out_flux_mean
        print(test_waves[i], 1e6 * depth, 1e6 * np.sqrt(out_var/out_flux**2 + in_var/in_flux**2))
    
    pdb.set_trace()
    plt.imshow(test_fluxes, aspect='auto', vmin=0.995, vmax=1.005)
    plt.figure()
    plt.plot(test_fluxes[28])
    plt.plot(test_fluxes[29])
    plt.plot(test_fluxes[30])
    plt.show()'''

    
    return np.unique(time)[trim_start:], binned_fluxes[trim_start:] / median, binned_errors[trim_start:] / median, unique_wavelengths[start_bin : end_bin]
    

def get_data(start_bin, end_bin, file_pattern="x1d_bkdsub_rateints_ERS_NGTS10_2022_new_nodrift_seg_???.fits"):
    DAY_TO_SEC = 86400
    filenames = glob.glob(file_pattern)
    mjds = []
    data = []
    errors = []

    for filename in filenames:
        with fits.open(filename) as hdul:        
            header = hdul[0].header
            times = header["EXSEGNUM"] * header["EFFINTTM"] * header["NINTS"] + np.linspace(0, header["EFFINTTM"] * header["NINTS"], header["NINTS"])
            mjds += list(times / DAY_TO_SEC)
            for i in range(2, len(hdul)):
                wavelengths = hdul[i].data["WAVELENGTH"]
                data.append(hdul[i].data["FLUX"])
                errors.append(hdul[i].data["ERROR"])

    argsort = np.argsort(mjds)
    data = np.array(data)[argsort]
    errors = np.array(errors)[argsort]
    mjds = np.array(mjds)[argsort]

    wavelengths = wavelengths[start_bin:end_bin]
    fluxes = np.sum(data[:,start_bin:end_bin], axis=1)
    flux_errors = np.sqrt(np.sum(errors[:, start_bin:end_bin]**2, axis=1))
    
    return mjds, fluxes, flux_errors, wavelengths
    

def print_percentiles(chain):
    percentiles = [5, 16, 50, 84, 95]
    for i in range(chain.shape[1]):
        results = np.array([np.percentile(chain[:,i], p) for p in percentiles])
        print(results)

def print_stats(data, label="Untitled"):
    lower = np.percentile(data, 16)
    median = np.median(data)
    upper = np.percentile(data, 84)
    print(label, median-lower, median, upper-median)

