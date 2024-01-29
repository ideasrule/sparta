import sys
import numpy as np
import astropy.io.fits

all_residuals = []

for filename in sys.argv[1:]:
    with astropy.io.fits.open(filename) as hdul:
        print(filename, hdul["RESIDUALS1"].data.shape)
        all_residuals.append(hdul["RESIDUALS1"].data)
    #print(filename)
    
all_residuals = np.array(all_residuals)    
mean = np.mean(all_residuals, axis=0)
np.save("median_residuals.npy", mean)
