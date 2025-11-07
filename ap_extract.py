from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import glob
from scipy.ndimage import median_filter
import numpy.ma as ma
import pandas as pd
from photutils.aperture import ApertureStats as ApertureStats
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAnnulus, CircularAperture
from scipy import optimize
from astropy.stats import SigmaClip
from constants import GAIN_FILE, RNOISE_FILE, TOP, BOT, LEFT, RIGHT, ROTATE
from scipy.stats import median_abs_deviation as mad
import os
import argparse

def plot_image(img, aperture, annulus_aperture):
    plt.imshow(img, interpolation='nearest', vmax = 30000, vmin = 20000, cmap='gray')
    
    ap_patches = aperture.plot(color='white', lw=2,
                               label='Photometry aperture')
    ann_patches = annulus_aperture.plot(color='red', lw=2,
                                        label='Background annulus')
    handles = (ap_patches[0], ann_patches[0])
    plt.legend(loc=(0.17, 0.05), facecolor='#458989', labelcolor='white',
               handles=handles, prop={'weight': 'bold', 'size': 11})
    
def cal_MAD(data):
    return np.median(np.abs(data - np.median(data)))
    

def source_by_peak(sources, i, file_name, manual_centroid = None):
    real_source = sources[sources["peak"] == sources["peak"].max()]
       
    if len(real_source) == 0: 
        print("no source found for int {} in ".format(i)+file_name+", manually defined as {}".format(manual_centroid))
        xcentroid = manual_centroid[0]
        ycentroid = manual_centroid[1]
    if len(real_source) > 1: 
        print("more than one source found for int {}".format(i))
    if len(real_source) == 1:
        
        xcentroid = real_source['xcentroid'].value[0]
        ycentroid = real_source['ycentroid'].value[0]
    return xcentroid, ycentroid
        

def source_by_accurate_moments(data, xc, yc, radius=5.0, max_iter=100, max_pos_error=1e-4):
    """Obtains the centroid in an iterative fashion, given the image data and a first guess.
    Then uses the centroid to calculate second moments.
    Algorithm: calculate sum of x*data, y*data, and data within specified radius of xc,yc.  Update
    xc,yc by dividing the former two numbers by the third.  Repeat for max_iter."""
    data = np.array(data)
    xs = np.array([np.arange(data.shape[1])] * data.shape[0])
    ys = xs.transpose()

    for i in range(max_iter):
        aperture = CircularAperture((xc, yc), radius)
        flux = (aperture_photometry(data, aperture)['aperture_sum'][0])

        old_xc = xc
        old_yc = yc
        xc = (aperture_photometry(data * xs, aperture)['aperture_sum'][0] / flux)
        yc = (aperture_photometry(data * ys, aperture)['aperture_sum'][0] / flux)
        last_pos_error = np.sqrt((xc - old_xc) ** 2 + (yc - old_yc) ** 2)
        if last_pos_error < max_pos_error:
            break

    if last_pos_error < max_pos_error:
        second_x = aperture_photometry(data*(xs-xc)**2, aperture)['aperture_sum'][0]/flux
        second_y = aperture_photometry(data*(ys-yc)**2, aperture)['aperture_sum'][0]/flux
        cross_xy = aperture_photometry(data*(xs*ys - xc*yc), aperture)['aperture_sum'][0]/flux
        return xc, yc, second_x, second_y, cross_xy
    
    raise ValueError("Failed to find centroid for star")

def gaussian2d(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = np.nansum(data)
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fit_gaussian(data, xc, yc, radius=4.0):
    """Fit a 2D Gaussian to the data within 5 pixels of xc,yc"""
    mask = np.ones(data.shape, dtype=bool) * False
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (i - xc) ** 2 + (j - yc) ** 2 > radius ** 2: mask[i][j] = True
    masked_img = ma.array(data / np.abs(np.median(data)), mask=mask)

    params = moments(masked_img)
    errorfunction = lambda p: np.ravel(gaussian2d(*p)(*np.indices(masked_img.shape)) -
                                 masked_img)
    p, success = optimize.leastsq(errorfunction, params)

    return p

def get_gain():
    with fits.open(GAIN_FILE) as hdul:
        substrt_x = int(hdul[0].header["SUBSTRT1"]) - 1
        substrt_y = int(hdul[0].header["SUBSTRT2"]) - 1
        gain = np.asarray(np.rot90(hdul[1].data, ROTATE)[
            TOP-substrt_y : BOT-substrt_y,
            LEFT-substrt_x : RIGHT-substrt_x])
    return gain

def get_read_noise(gain):    
    with fits.open(RNOISE_FILE) as hdul:
        substrt_x = int(hdul[0].header["SUBSTRT1"]) - 1
        substrt_y = int(hdul[0].header["SUBSTRT2"]) - 1
        return gain * np.array(np.rot90(hdul[1].data, ROTATE)[
            TOP-substrt_y : BOT-substrt_y,
            LEFT-substrt_x : RIGHT-substrt_x], dtype=np.float64) / np.sqrt(2)

def ap_extract(filelist, X_WINDOW, Y_WINDOW, ap_size =4, annulus_r_in = 12, annulus_r_out = 26, initial_guess = [128,128]):
    k = 0
    xc_list = []
    yc_list = []
    xwidth_list = []
    ywidth_list = []
    flux_subbkg_list = []
    flux_list = []
    bkg_list = []
    error_list = []
    time_array = []
    var_ap_list = []
    var_bgest_list = []
    
    nint = 0
    for filename in filelist:
        print('Processing ', filename)
        with fits.open(filename) as hdul:
            data = hdul[1].data
            nint += data.shape[0]

    bkg_sub_imgarray = np.zeros((nint, int(X_WINDOW[1] - X_WINDOW[0]), int(Y_WINDOW[1] - Y_WINDOW[0])))
    for file_name in filelist:
        with fits.open(file_name) as hdul:
            time_array.extend(hdul['INT_TIMES'].data['int_mid_BJD_TDB'])


            data = hdul["SCI"].data
            ERROR = hdul["ERR"].data

            cropped_data = data[:,Y_WINDOW[0]:Y_WINDOW[1], X_WINDOW[0]:X_WINDOW[1]]

            for i in range(cropped_data.shape[0]):
                img = cropped_data[i]
                err_img = ERROR[i,Y_WINDOW[0]:Y_WINDOW[1], X_WINDOW[0]:X_WINDOW[1]]
                mean, median, std = sigma_clipped_stats(img, sigma=3.0)  
                daofind = DAOStarFinder(fwhm=3.0, threshold=20.*std)  

                sources = daofind(img - median)  
                if i == 0:
                    xc, yc = initial_guess[0], initial_guess[1]
                xcentroid, ycentroid = source_by_peak(sources, i, file_name, manual_centroid = [xc, yc])
                
                xc, yc, second_x, second_y, cross_xy = source_by_accurate_moments(img, xcentroid, ycentroid)
    
                positions = [(xc, yc)]
                xc_list.append(xc)
                yc_list.append(yc)
                p = fit_gaussian(img.T, xc, yc)
    
                (height, x, y, width_x, width_y) = p
    
                xwidth_list.append(width_x)
                ywidth_list.append(width_y)
                
                aperture = CircularAperture(positions, r=ap_size)
                annulus_aperture = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)

                if i % 200 == 0:
                    plot_image(img, aperture, annulus_aperture)
                    savefilename = file_name.split('/')[-1].split('.')[0]
                    plt.savefig(f'./img/{savefilename}_apertures_{i}.png')
            
                aperstats = ApertureStats(img, annulus_aperture, sigma_clip = SigmaClip(sigma=5.0))

                
                bkg_mean = aperstats.mean
                bkg_sub_imgarray[k] = img - np.ones(bkg_sub_imgarray.shape[1:]) * bkg_mean[0]
                total_bkd = (bkg_mean[0]) * aperture.area
                bkg_list.append(bkg_mean[0])
                phot_table = aperture_photometry(img, aperture)
                phot_bkgsub = phot_table['aperture_sum'] - total_bkd
                print("flux in the aperture is", phot_table['aperture_sum'][0])


                mask_ann     = annulus_aperture.to_mask(method='center')[0]
                ann_data     = mask_ann.multiply(img)
                ann_vals     = ann_data[mask_ann.data.astype(bool)]
                sigma_ann    = np.std(ann_vals, ddof=1)
                var_B        = sigma_ann**2 / len(ann_vals)


                N_ap         = aperture.area
                var_bgest    = (N_ap**2) * var_B

                var_ap = ApertureStats(err_img**2, aperture).sum[0]  # sum of var in ap

                var_total    = var_ap + var_bgest #+ var_rnoise_ap + var_rnoise_ann
                error  = np.sqrt(var_total)
                flux_subbkg_list.append(phot_bkgsub[0])
                flux_list.append(phot_table['aperture_sum'][0])
                error_list.append(error)
                var_ap_list.append(var_ap)
                var_bgest_list.append(var_bgest)
                                      

                k += 1
                

    return flux_list, flux_subbkg_list, bkg_list, error_list, xc_list, yc_list, xwidth_list, ywidth_list, time_array, var_ap_list, var_bgest_list



parser = argparse.ArgumentParser(description="Extract aperture photometry.")
parser.add_argument("-f", "--filenames", nargs="+", required=True,
                        help="One or more input rateints .fits files.")
args = parser.parse_args()
stage1 = args.filenames
stage1.sort()


X_WINDOW = [0,256]
Y_WINDOW = [0,256]

initial_guess = [128,128]
apsize_list = [4]

annulus_r_in_list = [26]
annulus_r_out_list = [30]

# Give a range of annulus sizes to test
# annulus_r_in_list = [10,12,14,16,18,20]
# annulus_r_out_list = [14,16,18,20,22,24,26,28,30,32]

if os.path.exists('mad_dict.txt'):
    os.remove('mad_dict.txt')



with open('mad_dict_new.txt', 'a+') as f:
    f.write('ap_size annulus_r_in annulus_r_out mad\n')
for ap_size in apsize_list:

    for annulus_r_in in annulus_r_in_list:
        for annulus_r_out in annulus_r_out_list:
            if annulus_r_in < annulus_r_out and annulus_r_out - annulus_r_in >= 1: 
                print(f"ap_size: {ap_size}, annulus_r_in: {annulus_r_in}, annulus_r_out: {annulus_r_out}")
                flux_list, flux_subbkg_list, bkg_list, error_list, xc_list, yc_list, xwidth_list, ywidth_list, time_array, \
                var_ap_list, var_bgest_list \
                     = ap_extract(stage1, X_WINDOW, Y_WINDOW, ap_size = ap_size,
                                  annulus_r_in = annulus_r_in, annulus_r_out = annulus_r_out, initial_guess=initial_guess)
                
                normalized_flux_list = flux_subbkg_list / np.median(flux_subbkg_list)
                trend = median_filter(normalized_flux_list, size=30)
                detrended_flux = normalized_flux_list - trend
                mad = cal_MAD(detrended_flux)
                with open('mad_dict.txt', 'a+') as f:
                    f.write(f"{ap_size} {annulus_r_in} {annulus_r_out} {mad}\n")

                plt.figure(figsize=(10, 5))
                plt.plot(time_array, normalized_flux_list, 'o', markersize=1)
                plt.plot(time_array, trend, 'r-', label='median filter')
                plt.title(f'MAD= {int(mad*1e6)} ppm, ap_size={ap_size}, annulus_r_in={annulus_r_in}, annulus_r_out={annulus_r_out}')
                plt.savefig(f'./img/ap{ap_size}_in{annulus_r_in}_out{annulus_r_out}_lightcurve.png')

                df = pd.DataFrame({
                    'flux': flux_list,
                    'flux_subbkg': flux_subbkg_list,
                    'bkg': bkg_list,
                    'error': error_list,
                    'xc': xc_list,
                    'yc': yc_list,
                    'xwidth': xwidth_list,
                    'ywidth': ywidth_list,
                    'time': time_array,
                    'var_ap': var_ap_list,
                    'var_bgest': var_bgest_list,
                })
                df.to_csv(f'./ap_extract_ap{ap_size}_in{annulus_r_in}_out{annulus_r_out}.csv', index=False)
                plt.close()


