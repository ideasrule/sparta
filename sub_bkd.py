import numpy as np
import astropy.stats
import matplotlib.pyplot as plt
import sys
import astropy.io.fits
import os.path
from constants import LEFT_MARGIN, RIGHT_MARGIN_BKD

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

def get_bkd(data, trace_width=16):    
    xs = np.arange(data.shape[1])
    profile = np.sum(data, axis=0)
    trace_loc = np.argmax(profile)

    left_xmin = LEFT_MARGIN
    left_xmax = int(trace_loc - trace_width/2)
    right_xmin = int(trace_loc + trace_width/2)
    right_xmax = data.shape[1] - RIGHT_MARGIN_BKD

    background_pixels = np.hstack([data[:, left_xmin:left_xmax], data[:, right_xmin:]])
    bkd_xs = np.append(xs[left_xmin:left_xmax], xs[right_xmin:])
    background = np.zeros(data.shape)
    for i in range(len(background_pixels)):
        background[i] = robust_polyfit(bkd_xs, background_pixels[i], 1, target_xs=xs)
    return background


filename = sys.argv[1]
hdul = astropy.io.fits.open(filename)
data = hdul[1].data
background = np.zeros(data.shape)
for i in range(len(data)):
    print("Integration", i)
    background[i] = get_bkd(data[i])
    
image = hdul[1].data - background
sci_hdu = astropy.io.fits.ImageHDU(image, name="SCI")
bkd_hdu = astropy.io.fits.ImageHDU(background, name="BKD")

output_hdul = astropy.io.fits.HDUList([hdul[0], sci_hdu, hdul["ERR"], hdul["FLATERR"], bkd_hdu, hdul["RNOISE"]])
output_hdul.writeto("bkdsub_" + os.path.basename(filename), overwrite=True)

plt.imshow(background[0])
plt.title("Background (integration 0)")
plt.savefig("background_{}.png".format(filename[:-5]))

plt.imshow(image[0], vmin=0, vmax=10)
plt.title("Background subtracted (integration 0)")
plt.savefig("background_sub_{}.png".format(filename[:-5]))

