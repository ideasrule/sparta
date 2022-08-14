import os

LEFT_MARGIN = 10
RIGHT_MARGIN_BKD = 5
BAD_GRPS = 0
HIGH_ERROR = 1e10
EXTRACT_Y_MIN = 141
EXTRACT_Y_MAX = 386
SUM_EXTRACT_WINDOW = 3
OPT_EXTRACT_WINDOW = 5
SLITLESS_LEFT = 0
SLITLESS_RIGHT = 72
SLITLESS_TOP = 528
SLITLESS_BOT = 944
GAIN = 5.5
BKD_WIDTH = 15
#NONLINEAR_COEFFS = [-4.99787404e-07, 1.45159521e-10, -8.58359417e-16]
NONLINEAR_COEFFS = [1.8901927e-6, 6.1071504e-12, 3.082711e-16]

REF_DIR = os.path.expanduser("~/jwst_refs/")
NONLINEAR_FILE = REF_DIR + "jwst_miri_linearity_0032.fits"
DARK_FILE = REF_DIR + "jwst_miri_dark_0084.fits"
FLAT_FILE = REF_DIR + "jwst_miri_flat_0789.fits"
RNOISE_FILE = REF_DIR + "jwst_miri_readnoise_0057.fits"
RESET_FILE = REF_DIR + "jwst_miri_reset_0073.fits"
MASK_FILE = REF_DIR + "jwst_miri_mask_0030.fits"
WCS_FILE = REF_DIR + "jwst_miri_specwcs_0004.fits"
