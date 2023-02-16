import os

USE_GPU = False
REF_DIR = os.path.expanduser("~/jwst_refs/")
RIGHT_MARGIN_BKD = 5
HIGH_ERROR = 1e10
OPT_EXTRACT_WINDOW = 5
INSTRUMENT = "NIRCAM" #Change this

if INSTRUMENT == "MIRI":
    SUBARRAY = "SLITLESSPRISM"
    FILTER = "P750L"
    BAD_GRPS = 5
    ROTATE = -1
    SKIP_SUPERBIAS = True
    SKIP_FLAT = False
    LEFT = 80
    RIGHT = 496
    TOP = 0
    BOT = 72
    TOP_MARGIN = 10
    N_REF = 4
    Y_CENTER = 36
    
    GAIN = 3.1
    NONLINEAR_FILE = REF_DIR + "jwst_miri_linearity_0032.fits"
    DARK_FILE = REF_DIR + "jwst_miri_dark_0084.fits"
    FLAT_FILE = REF_DIR + "jwst_miri_flat_0789.fits"
    RNOISE_FILE = REF_DIR + "jwst_miri_readnoise_0085.fits"
    RESET_FILE = REF_DIR + "jwst_miri_reset_0073.fits"
    MASK_FILE = REF_DIR + "jwst_miri_mask_0033.fits"
    WCS_FILE = REF_DIR + "jwst_miri_specwcs_0006.fits"
    SUPERBIAS_FILE = None
    SKIP_REF = True
    SKIP_RESET = False

    X_MIN = 30
    X_MAX = 275
    BKD_REG_TOP = [10, 25]
    BKD_REG_BOT = [-25, -10]
    SUM_EXTRACT_WINDOW = 3    

elif INSTRUMENT == "NIRCAM":
    BAD_GRPS = 0
    ROTATE = 0
    SUBARRAY = "SUBGRISM64" #Change this
    FILTER = "F444W" #Change this"
    
    SKIP_SUPERBIAS = False
    SKIP_FLAT = True
    SKIP_REF = False
    SKIP_RESET = True

    N_REF = 4
    TOP_MARGIN = N_REF
    
    GAIN = 1.82
    Y_CENTER = 33    
    BKD_REG_TOP = [N_REF, N_REF + 7]
    SUM_EXTRACT_WINDOW = 6

    RESET_FILE = None
    FLAT_FILE = None    
    MASK_FILE = REF_DIR + "jwst_nircam_mask_0063.fits"
    NONLINEAR_FILE = REF_DIR + "jwst_nircam_linearity_0052.fits"
    WCS_FILE = None

    if FILTER == "F444W":
        ONE_OVER_F_WINDOW_LEFT = 4
        ONE_OVER_F_WINDOW_RIGHT = 600
        X_MIN = 821
        X_MAX = 1900
    if FILTER == "F322W2":
        ONE_OVER_F_WINDOW_LEFT = 1894
        ONE_OVER_F_WINDOW_RIGHT = 2044
        X_MIN = 50
        X_MAX = 1645

    if SUBARRAY == "SUBGRISM256":
        SUBARR_SIZE = 256
        RNOISE_FILE = REF_DIR + "jwst_nircam_readnoise_0157.fits"
        DARK_FILE = REF_DIR + "jwst_nircam_dark_0355.fits"
        SUPERBIAS_FILE = REF_DIR + "jwst_nircam_superbias_0141.fits"
        BKD_REG_BOT = [57, 253]        
    if SUBARRAY == "SUBGRISM64":
        SUBARR_SIZE = 64
        RNOISE_FILE = REF_DIR + "jwst_nircam_readnoise_0180.fits"
        DARK_FILE = REF_DIR + "jwst_nircam_dark_0364.fits"
        SUPERBIAS_FILE = REF_DIR + "jwst_nircam_superbias_0145.fits"
        BKD_REG_BOT = [57, 64]
        
    LEFT = 0
    RIGHT = 2048 
    TOP = 0
    BOT = SUBARR_SIZE
