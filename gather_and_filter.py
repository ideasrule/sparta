import astropy.io.fits
import numpy as np
import matplotlib.pyplot as plt
import glob
import astropy.stats
import pdb
from scipy.stats import median_abs_deviation
from scipy.ndimage import uniform_filter, median_filter
import sys
import pickle

def read_data(filenames):
    data = None
    errors = None
    bkd = None
    times = []
    section_edges = []
    
    for filename in filenames:
        with astropy.io.fits.open(filename) as hdul:
            wavelengths = hdul[2].data["WAVELENGTH"]
            lcs = np.array([hdul[i].data["FLUX"] for i in range(2, len(hdul))])
            lc_errors = np.array([hdul[i].data["ERROR"] for i in range(2, len(hdul))])
            curr_bkd = np.array([hdul[i].data["BKD"] for i in range(2, len(hdul))])
            if data is None:
                data = lcs
                errors = lc_errors
                bkd = curr_bkd
            else:
                data = np.append(data, lcs, axis=0)
                errors = np.append(errors, lc_errors, axis=0)
                bkd = np.append(bkd, curr_bkd, axis=0)
            print(len(times))            
            times += list(hdul["INT_TIMES"].data["int_mid_BJD_TDB"])
            section_edges.append(len(times))
            assert(hdul[0].header["INTEND"] - hdul[0].header["INTSTART"] + 1 == len(lcs))
            
    times = np.array(times)        
    return data, errors, bkd, times, wavelengths, np.array(section_edges)

def repair_rows(data, sigma=4):
    repaired = np.copy(data)
    for c in range(data.shape[1]):
        col = data[:,c] / np.median(data[:,c])
        detrended = col - median_filter(col, int(data.shape[0] / 100))
        mask = astropy.stats.sigma_clip(detrended, sigma).mask
        rows = np.arange(data.shape[0])
        repaired[:,c] = np.interp(rows, rows[~mask], data[~mask,c])
    return repaired
    

def reject_rows(data, errors, bkd, times, x, y, sigma=4, num_discard_beginning=10):    
    summed = np.sum(data, axis=1)
    plt.figure()
    plt.plot(summed)
    plt.xlabel("Integration")
    plt.ylabel("Flux")

    filter_size = int(len(data) / 100)
    detrended = summed - median_filter(summed, filter_size)
    bad_rows = astropy.stats.sigma_clip(detrended, sigma).mask
    bad_rows[0:num_discard_beginning] = True

    #Reject based on position   
    bad_rows[np.isnan(y) | np.isnan(x)] = True
    bad_rows[astropy.stats.sigma_clip(y, sigma).mask] = True
    bad_rows[astropy.stats.sigma_clip(x, sigma).mask] = True

    #For TOI 134, by manual inspection
    #is_bad[2763:2765] = True #manual inspection

    #For GJ 1214b, transit should be good
    #is_bad[10770:11236] = False

    print("Bad rows", np.argwhere(bad_rows))
    plt.figure()
    plt.plot(uniform_filter(np.sum(data[~bad_rows], axis=1),100)[50::100])
    plt.xlabel("Integration")
    plt.ylabel("Flux (bad rows excluded)")
    plt.figure()
    plt.plot(np.sum(data[~bad_rows][:,33:55], axis=1))

    return data[~bad_rows], errors[~bad_rows], bkd[~bad_rows], times[~bad_rows], x[~bad_rows], y[~bad_rows]

def reject_cols(data, errors, wavelengths, threshold=1.2):
    is_bad = np.zeros(data.shape[1], dtype=bool)
    normalized = data / np.mean(data, axis=0)
    stds = np.std(normalized, axis=0)
    plt.figure()
    plt.plot(stds)
    plt.xlabel("Wavelength")
    plt.ylabel("STD")
    
    for i in range(1, len(stds)-1):
        ratio = stds[i] / (stds[i-1] + stds[i+1]) * 2
        if ratio > threshold:
            is_bad[i] = True
            print("Bad col", i, ratio)
    return data[:,~is_bad], errors[:,~is_bad], wavelengths[~is_bad]


    
data, errors, bkd, times, wavelengths, section_edges = read_data(sys.argv[1:])

output = {"uncut_wavelengths": wavelengths,
          "uncut_times": times,
          "uncut_data": data,
          "uncut_errors": errors,
          "uncut_bkd": bkd,
          "uncut_section_edges": section_edges
}

print(wavelengths[244:250])
print(wavelengths[250:])
plt.figure()
plt.imshow(data / np.mean(data, axis=0), aspect='auto', vmin=0.995, vmax=1.005)
for e in section_edges:
    plt.axhline(e, color='k')
 
plt.xlabel("Wavelength")
plt.ylabel("Time")

y, x, A = np.loadtxt("positions.txt", usecols=(2,3,4), unpack=True)
output.update({"uncut_y": y,
               "uncut_x": x})


data, errors, bkd, times, x, y = reject_rows(data, errors, bkd, times, x, y)
#data, errors, wavelengths = reject_cols(data, errors, wavelengths)
#data = repair_rows(data)

output.update({"wavelengths": wavelengths,
          "times": times,
          "data": data,
          "errors": errors,
          "bkd": bkd,
          "x": x,
          "y": y})

with open("data.pkl", "wb") as f:
    pickle.dump(output, f)


plt.figure()
plt.imshow(data / np.mean(data, axis=0), aspect='auto', vmin=0.995, vmax=1.005)
plt.xlabel("Wavelength")
plt.ylabel("Time")
plt.show()
