import astropy.io.fits
import sys
import numpy as np
import matplotlib.pyplot as plt

images = []

for filename in sys.argv[1:]:
    with astropy.io.fits.open(filename) as hdul:
        print(hdul[1].data.shape)
        images += list(hdul[1].data)

images = np.array(images)
print(images.shape)
median_image = np.median(images, axis=0)
bkd_cols = np.hstack([median_image[:,10:25], median_image[:,47:62]])
#bkd_cols = median_image[:,-15:]
median_image -= np.mean(bkd_cols, axis=1)[:,np.newaxis]
np.save("median_image.npy", median_image)
plt.imshow(median_image)
plt.show()
