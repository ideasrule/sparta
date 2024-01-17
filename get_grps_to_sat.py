import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
from constants import SAT_LEVEL, ROTATE

medians = []
for filename in sys.argv[1:]:
    with astropy.io.fits.open(filename) as hdul:
        median = np.median(np.rot90(hdul[1].data, ROTATE, (-2,-1)), axis=0)
        medians.append(median)

image = np.mean(medians, axis=0)
grps_to_sat = np.ones((image.shape[1], image.shape[2]), dtype=int) * image.shape[0]
for g in range(image.shape[0] - 1, -1, -1):
    grps_to_sat[image[g] > SAT_LEVEL] = g

np.save("grps_to_sat.npy", grps_to_sat)
plt.imshow(grps_to_sat)
plt.show()
