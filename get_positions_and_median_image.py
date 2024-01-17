import astropy.io.fits
import astropy.stats
import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
import scipy.optimize
from multiprocessing import Pool
from scipy.interpolate import RectBivariateSpline
from constants import TOP_MARGIN, Y_CENTER, INSTRUMENT, FILTER, SUBARRAY

def fix_outliers(data, badpix, sigma=5):
    for r in range(TOP_MARGIN, data.shape[0]):
        cols = np.arange(data.shape[1])
        good = ~badpix[r]
        data[r] = np.interp(cols, cols[good], data[r][good])

        good = ~astropy.stats.sigma_clip(data[r], sigma).mask
        data[r] = np.interp(cols, cols[good], data[r][good])

def chi_sqr(params, image, error, template, left=31, right=463, top=Y_CENTER-10, bottom=Y_CENTER+10, plot=False):
    delta_y, delta_x, A = params
    ys = np.arange(image.shape[0])
    xs = np.arange(image.shape[1])
    interpolator = RectBivariateSpline(ys, xs, template)
    shifted_template = A * interpolator(ys + delta_y, xs + delta_x)
    residuals = image - shifted_template
    zs = residuals / error
    '''zs[:,64:137] = 0
    zs[7,187] = 0
    zs[0:5,192:199] = 0
    zs[9,257] = 0
    zs[30,269] = 0
    zs[12,344] = 0
    zs[17,397] = 0
    zs[13,478] = 0'''
        
    if plot:
        plt.imshow(zs)
        plt.show()
    return np.sum(zs[top:bottom, left:right]**2)

    
#chi_sqr([0.1, 20], data[0], template)

all_data = []
all_error = []
all_filenames = []
all_int_nums = []
all_y = []
all_x = []
all_A = []
all_badpix = None

for filename in sys.argv[1:]:
    with astropy.io.fits.open(filename) as hdul:
        print(filename)
        assert(hdul[0].header["INSTRUME"] == INSTRUMENT and hdul[0].header["FILTER"] == FILTER and hdul[0].header["SUBARRAY"] == SUBARRAY)
        data = hdul["SCI"].data
        error = hdul["ERR"].data
        dq = hdul["DQ"].data

        for i in range(len(data)):
            fix_outliers(data[i], (dq[i] > 0) | np.isnan(data[i]))
            
        if all_data is None:
            all_data = data
            all_error = error
        else:
            all_data += list(data)
            all_error += list(error)
            
        all_filenames += len(data) * [filename]
        all_int_nums += list(np.arange(data.shape[0]))

all_data = np.array(all_data)        
all_error = np.array(all_error)
all_error[np.isnan(all_error)] = np.inf
print(len(all_data), len(all_filenames), len(all_int_nums))
template = np.median(all_data, axis=0)
fix_outliers(template, np.isnan(template))
np.save("median_image.npy", template)
#pdb.set_trace()

def do_one(i):
    bounds = ((-0.4,0.4), (-0.2,0.2), (0.9, 1.1))
    result = scipy.optimize.minimize(chi_sqr, [0,0,1], args=(all_data[i], all_error[i], template), bounds=bounds, method="Nelder-Mead")

    hit_bounds = False
    for b_ind, b in enumerate(bounds):
        if result.x[b_ind] <= b[0] or result.x[b_ind] >= b[1]:
            hit_bounds = True
            print("WARNING: hitting bounds for filename {}, integration {}, result {}, bound {}".format(all_filenames[i], i, result.x[b_ind], b))
    
    if i % 100 == 0:
        print("Filename {}, integration {}".format(all_filenames[i], all_int_nums[i]))
        
    if not result.success:
        print("Filename {}, integration {} failed".format(all_filenames[i], all_int_nums[i]))

    if not result.success or hit_bounds:
        result.x *= np.nan

    #chi_sqr(result.x, all_data[i], all_error[i], template, plot=True)
    return result.x

#do_one(1000)

with Pool() as p:
    results = p.map(do_one, range(len(all_data)))    

f = open("positions.txt", "w")
f.write("#Filename Integration y x A\n")
for i in range(len(all_data)):
    f.write("{} {} {} {} {}\n".format(all_filenames[i], all_int_nums[i], results[i][0], results[i][1], results[i][2]))
f.close()

#plt.imshow(template)
#plt.show()
