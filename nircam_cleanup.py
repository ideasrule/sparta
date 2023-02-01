import sys
import matplotlib.pyplot as plt
import numpy as np
import pdb
import astropy.io.fits
from constants import N_REF, ONE_OVER_F_WINDOW_LEFT, ONE_OVER_F_WINDOW_RIGHT, BKD_REG_TOP, BKD_REG_BOT

for filename in sys.argv[1:]:
    print(filename)
    hdul = astropy.io.fits.open(filename)
    data = hdul[1].data
    for i in range(len(data)):
        #Fix outliers
        mask = np.logical_or(np.isnan(data[i]), hdul["DQ"].data[i] > 0)

        xs = np.arange(data[i].shape[1])
        for y in range(N_REF, data[i].shape[0]):
            if np.sum(~mask[y]) == 0:
                pdb.set_trace()
            data[i,y] = np.interp(xs, xs[~mask[y]], data[i,y][~mask[y]])
   
    data[:,:N_REF] = 0
    subtracted = np.zeros(data.shape)
    subtracted[:,:,:] = np.nanmedian(data[:,:,ONE_OVER_F_WINDOW_LEFT:ONE_OVER_F_WINDOW_RIGHT], axis=2)[:,:,np.newaxis]
    data -= subtracted

    bkd_rows = np.concatenate((data[:,BKD_REG_TOP[0]:BKD_REG_TOP[1]],
                               data[:,BKD_REG_BOT[0]:BKD_REG_BOT[1]]),
                              axis=1)
    bkd = np.nanmedian(bkd_rows, axis=1)
    subtracted += bkd[:,np.newaxis,:]
    data -= bkd[:,np.newaxis,:]
    
    bkd_hdu = astropy.io.fits.ImageHDU(subtracted, name="BKD")    
    hdul.append(bkd_hdu)
    hdul.writeto("cleaned_{}".format(filename), overwrite=True)
    hdul.close()
