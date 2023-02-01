import astropy.io.fits
import sys
import numpy as np
import matplotlib.pyplot as plt

images = []

for filename in sys.argv[1:]:
    with astropy.io.fits.open(filename) as hdul:
        images += list(hdul[1].data)

images = np.array(images)
median_image = np.median(images, axis=0)
num_nan = np.sum(np.isnan(median_image))
if num_nan > 0:
    print("WARNING: {} pixels are nan!".format(num_nan))

np.save("median_image.npy", median_image)
print("Saved median image to median_image.npy")

plt.imshow(median_image)
plt.show()
