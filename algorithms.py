import numpy as np
import sys
import scipy.signal
import matplotlib.pyplot as plt
import astropy.stats
from sklearn.neighbors import NearestNeighbors
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from scipy.stats import binned_statistic
from sklearn.decomposition import PCA
import scipy.stats
import glob
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
    
    
def reject_beginning(bjds, fluxes, errors, cutoff_in_days=0.1, max_separation_in_days=0.01):
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


def get_data(start_bin, end_bin, file_pattern="x1d_bkdsub_rateints_ERS_NGTS10_2022_seg_???.fits"):
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

