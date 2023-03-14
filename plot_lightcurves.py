import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import uniform_filter

wavelength, time, flux, unc, sys_model, astro_model, total_model, residuals = np.loadtxt(sys.argv[1], unpack=True)
unique_wavelengths = np.unique(wavelength)

for i,w in enumerate(unique_wavelengths):
    cond = wavelength == w
    plt.clf()
    binsize = len(flux[cond]) // 200

    plt.subplot(211)
    plt.plot(time[cond][::binsize], uniform_filter(flux[cond] / sys_model[cond], binsize)[::binsize])
    plt.plot(time[cond], astro_model[cond])
    plt.title(str(w) + " um")
    #plt.ylim(0.975, 0.976)
    #plt.xlim(5.9816e4 + 0.48, 5.9816e4 + 0.52)
    #plt.savefig("spec_lc_{}_{}.png".format(i,w))    

    #plt.clf()
    plt.subplot(212)
    plt.errorbar(time[cond][::binsize], uniform_filter(residuals[cond], binsize)[::binsize], yerr=unc[cond][::binsize]/np.sqrt(binsize), fmt='.')
    plt.title(str(w) + " um")
    T0 = 59816.500518
    #T0 = 59820.937747
    #plt.axvline(T0 - 0.01275*2.21857519, color='k')
    #plt.axvline(T0 + 0.01275*2.21857519, color='k')
    plt.savefig("spec_lc_{}_{}.png".format(i,w))

    print("Saved spec_lc and residuals pngs for {} um".format(w))
    #print("Expected STD, calculated STD", 1e6*np.median(unc[cond][::binsize]) / np.sqrt(binsize), 1e6*np.std(uniform_filter(residuals[cond], binsize)[::binsize]))
    #print("Expected STD, calculated STD", 1e6*np.mean(unc[cond]), 1e6*np.std(residuals[cond]))
plt.show()    
