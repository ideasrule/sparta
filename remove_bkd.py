import numpy as np
import astropy.stats
import sys
import pdb
from astropy.io.fits import ImageHDU
from constants import *

def remove_bkd_nircam(data, err, dq):
    data = np.copy(data)
    err = np.copy(err)
    
    #Modifies data and err (outlier rejection) before estimating bkd
    for i in range(len(data)):
        #Fix outliers
        mask = np.logical_or(np.isnan(data[i]), dq[i] > 0)

        xs = np.arange(data[i].shape[1])
        for y in range(N_REF, data[i].shape[0]):
            if np.sum(~mask[y]) == 0:
                pdb.set_trace()
            data[i,y] = np.interp(xs, xs[~mask[y]], data[i,y][~mask[y]])
            err[i,y] = np.interp(xs, xs[~mask[y]], err[i,y][~mask[y]])
   
    data[:,:N_REF] = 0
    subtracted = np.zeros(data.shape)
    var_subtracted = np.zeros(data.shape)
    subtracted[:,:,:] = np.nanmedian(data[:,:,ONE_OVER_F_WINDOW_LEFT:ONE_OVER_F_WINDOW_RIGHT], axis=2)[:,:,np.newaxis]
    var_subtracted[:,:,:] = np.sum(err[:,:,ONE_OVER_F_WINDOW_LEFT:ONE_OVER_F_WINDOW_RIGHT]**2, axis=2)[:,:,np.newaxis] / (ONE_OVER_F_WINDOW_RIGHT - ONE_OVER_F_WINDOW_LEFT)**2 * np.pi / 2
    
    data_no_bkd = data - subtracted

    bkd_rows = np.concatenate((data[:,BKD_REG_TOP[0]:BKD_REG_TOP[1]],
                               data[:,BKD_REG_BOT[0]:BKD_REG_BOT[1]]),
                              axis=1)
    num_bkd_cols = np.diff(BKD_REG_TOP) + np.diff(BKD_REG_BOT)
    bkd = np.nanmedian(bkd_rows, axis=1)
    bkd_var = np.sum(np.concatenate((err[:,BKD_REG_TOP[0]:BKD_REG_TOP[1]],
                                     err[:,BKD_REG_BOT[0]:BKD_REG_BOT[1]]),
                                    axis=1)**2,
                     axis=1) / num_bkd_cols**2 * np.pi / 2
    
    subtracted += bkd[:,np.newaxis,:]
    var_subtracted += bkd_var[:,np.newaxis,:]
    data_no_bkd -= bkd[:,np.newaxis,:]
    return data_no_bkd, err, subtracted, np.sqrt(var_subtracted)


def remove_bkd_miri(data, err):
    bkd_im = np.zeros(data.shape)
    bkd_var_im = np.zeros(err.shape)
        
    for i in range(data.shape[0]):
        bkd_rows = np.vstack([
            data[i,BKD_REG_TOP[0]:BKD_REG_TOP[1]],
            data[i,BKD_REG_BOT[0]:BKD_REG_BOT[1]]])

        bkd_rows = astropy.stats.sigma_clip(bkd_rows, axis=0)
        bkd_err_rows = np.vstack([
            err[i,BKD_REG_TOP[0]:BKD_REG_TOP[1]],
            err[i,BKD_REG_BOT[0]:BKD_REG_BOT[1]]])

        bkd = np.ma.mean(bkd_rows, axis=0)
        bkd_var = np.sum(bkd_err_rows**2, axis=0) / bkd_err_rows.shape[1]**2

        bkd_im[i] = bkd
        bkd_var_im[i] = bkd_var
        
    return data - bkd_im, err, bkd_im, np.sqrt(bkd_var_im)

for filename in sys.argv[1:]:
    print(filename)
    hdul = astropy.io.fits.open(filename)
    assert(hdul[0].header["INSTRUME"] == INSTRUMENT and hdul[0].header["FILTER"] == FILTER and hdul[0].header["SUBARRAY"] == SUBARRAY)
    if hdul[0].header["INSTRUME"] == "MIRI":
        data_no_bkd, err, bkd, err_bkd = remove_bkd_miri(
            hdul["SCI"].data, hdul["ERR"].data)
    elif hdul[0].header["INSTRUME"] == "NIRCAM":
        data_no_bkd, err, bkd, err_bkd = remove_bkd_nircam(
            hdul["SCI"].data, hdul["ERR"].data, hdul["DQ"].data)
        
    hdul["SCI"].data = data_no_bkd
    hdul["ERR"].data = err
    hdul.append(ImageHDU(bkd, name="BKD"))
    hdul.append(ImageHDU(err_bkd, name="BKD_ERR"))
    hdul.writeto("cleaned_{}".format(filename), overwrite=True)
    hdul.close()
    
