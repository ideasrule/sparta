#!/usr/bin/env python
# optimal_extract.py
#
# Full‑frame optimal photometry extractor for JWST/MIRI (or similar) data.
# Now supports TWO weighting profiles:
#   1. 2‑D Gaussian PSF fitted to each integration (default)
#   2. Fixed "median‑signal" profile derived from the cube median (as in
#      the optphot routine) and shifted to the current centroid.
#
# Users choose the profile with the *profile_type* keyword.
#   profile_type = 'gaussian'  -> per‑integration Gaussian fit (Horne‑style)
#   profile_type = 'median'    -> median‑derived empirical profile
#
# If profile_type == 'median', you may supply *saved_profile* (2‑D numpy
# array) to reuse a pre‑computed profile; otherwise the routine will build
# one automatically from the *first* FITS cube.
#
# Author: ChatGPT‑o3, 2025‑07‑22
# ---------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from functools import partial
import glob

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip, sigma_clip
from photutils.detection import DAOStarFinder
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                aperture_photometry, ApertureStats)
from scipy.ndimage import median_filter, shift as ndi_shift
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import pandas as pd
from constants import GAIN_FILE, RNOISE_FILE, TOP, BOT, LEFT, RIGHT, ROTATE
__all__ = [
    "optimal_extract",
]

# ---------------------------------------------------------------
# ---------------------- helper utilities -----------------------
# ---------------------------------------------------------------

def _plot_image(img, aperture, annulus_aperture, title=None, savefile=None):
    """Debug plot of an image with source+sky apertures."""
    plt.imshow(img, interpolation="nearest", cmap="gray", origin="lower")
    ap_patch = aperture.plot(color="white", lw=2)[0]
    ann_patch = annulus_aperture.plot(color="red", lw=2)[0]
    plt.legend([ap_patch, ann_patch], ["source", "sky"], loc="lower left")
    if title:
        plt.title(title)
    if savefile:
        Path(savefile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savefile, dpi=200)
    plt.close()

def cal_MAD(data):
    return np.median(np.abs(data - np.median(data)))
    
def _moments(data):
    """Rough PSF estimates (height, cx, cy, sigx, sigy)."""
    total = np.nansum(data)
    if total == 0:
        return 0, data.shape[1] / 2, data.shape[0] / 2, 2.0, 2.0
    Y, X = np.indices(data.shape)
    cx = (X * data).sum() / total
    cy = (Y * data).sum() / total
    col = data[:, int(cx)]
    wx = np.sqrt(np.abs(((np.arange(col.size) - cx) ** 2 * col).sum() / col.sum()))
    row = data[int(cy), :]
    wy = np.sqrt(np.abs(((np.arange(row.size) - cy) ** 2 * row).sum() / row.sum()))
    height = data.max()
    return height, cx, cy, wx, wy


def _gaussian2d(height, cx, cy, wx, wy):
    wx, wy = float(wx), float(wy)
    return lambda x, y: height * np.exp(-(((cx - x) / wx) ** 2 + ((cy - y) / wy) ** 2) / 2.0)


def _fit_gaussian(data, cx0, cy0, fit_radius=4.0):
    """Levenberg–Marquardt PSF fit within *fit_radius* of (cx0, cy0)."""
    yy, xx = np.indices(data.shape)
    mask = (xx - cx0) ** 2 + (yy - cy0) ** 2 > fit_radius**2
    mdata = ma.array(data, mask=mask)
    p0 = _moments(mdata)

    def err_fn(p):
        g = _gaussian2d(*p)(*np.indices(mdata.shape))
        return np.ravel(g - mdata)

    p, _ = leastsq(err_fn, p0)
    return p  # height, cx, cy, wx, wy


def _locate_source(img, guess=(60, 60)):
    """Find brightest source with DAOStarFinder; fall back to *guess*."""
    mean, med, std = sigma_clipped_stats(img, sigma=3.0)
    finder = DAOStarFinder(fwhm=3.0, threshold=20.0 * std)
    srcs = finder(img - med) if finder is not None else None
    if srcs is not None and len(srcs) > 0:
        idx = np.argmax(srcs["peak"])
        return float(srcs["xcentroid"][idx]), float(srcs["ycentroid"][idx])
    return guess

def get_gain():
    with fits.open(GAIN_FILE) as hdul:
        substrt_x = int(hdul[0].header["SUBSTRT1"]) - 1
        substrt_y = int(hdul[0].header["SUBSTRT2"]) - 1
        gain = np.asarray(np.rot90(hdul[1].data, ROTATE)[
            TOP-substrt_y : BOT-substrt_y,
            LEFT-substrt_x : RIGHT-substrt_x])
    return gain

def get_read_noise(gain):    
    with fits.open(RNOISE_FILE) as hdul:
        substrt_x = int(hdul[0].header["SUBSTRT1"]) - 1
        substrt_y = int(hdul[0].header["SUBSTRT2"]) - 1
        return gain * np.array(np.rot90(hdul[1].data, ROTATE)[
            TOP-substrt_y : BOT-substrt_y,
            LEFT-substrt_x : RIGHT-substrt_x], dtype=np.float64) / np.sqrt(2)
# ---------------------------------------------------------------
# ------------------ median‑profile constructor ------------------
# ---------------------------------------------------------------

def _build_median_profile(
    img_cube: np.ndarray,
    position: tuple[float, float],
    ap_r: float,
    saturate_neg: bool = True,
) -> np.ndarray:
    """Make a normalized, error‑weighted profile from a *sigma‑clipped* cube median."""
    # New: sigma‑clip the cube along the integration axis to suppress outliers
    clipped_cube = sigma_clip(img_cube, sigma=10, axis=0)
    median_img = np.nanmedian(clipped_cube, axis=0)

    # Circular mask (simple); can be adapted as needed
    yy, xx = np.indices(median_img.shape)
    rr = np.sqrt((xx - position[0]) ** 2 + (yy - position[1]) ** 2)
    weight_mask = rr <= ap_r

    profile = median_img.copy()
    if saturate_neg:
        profile[profile < 0] = 0  # enforce positivity
    profile *= weight_mask

    # Poisson error weighting: divide by sqrt(signal)
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = profile / np.sqrt(profile)
    profile = ma.masked_invalid(profile).filled(0)

    # Normalise
    if profile.sum() == 0:
        raise RuntimeError("Median profile sums to zero – check input data.")
    profile = profile / profile.sum()
    return profile

def _build_zcut_profile(med_frame, centroid, box_size=20, zcut=0.001):
    """
    Build a z-cut profile for optimal extraction.

    Parameters
    ----------
    med_frame : 2D ndarray
        Median (or representative) image.
    centroid : (float, float)
        (x, y) center of the source in full-frame coordinates.
    box_size : int
        Size of the square cutout (e.g., 20 → 20x20 pixels).
    zcut : float
        Fraction of the peak flux to keep (pixels below are discarded).

    Returns
    -------
    profile : 2D ndarray
        Normalized weight map (sum = 1).
    x0, y0 : ints
        Origin (lower-left indices) of the cutout in the full frame.
    """
    half = box_size // 2
    cx, cy = centroid
    x0 = int(cx) - half 
    y0 = int(cy) - half
    cut = med_frame[y0:y0+box_size, x0:x0+box_size].copy()
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(med_frame.shape[1], x0 + box_size)
    y1 = min(med_frame.shape[0], y0 + box_size)

    peak = np.nanmax(cut)
    mask = cut >= (zcut * peak)

    profile = np.where(mask, cut, 0.0)

    # Force positivity (should already be)
    profile[profile < 0] = 0

    # Normalize; add a tiny epsilon to prevent zero-div
    norm = profile.sum()
    if norm <= 0:
        raise ValueError("z-cut removed all pixels—lower zcut or check median frame.")
    profile /= norm


    profile_full = np.zeros_like(med_frame, dtype=float)
    profile_full[y0:y1, x0:x1] = profile
    return profile_full
# ---------------------------------------------------------------
# -------------- main optimal extraction sub‑routine -------------
# ---------------------------------------------------------------

def optimal_extract(
    filelist: list[str] | str,
    xwin: list[int],
    ywin: list[int],
    *,
    ap_r: float = 4.0,
    rin: float = 12.0,
    rout: float = 26.0,
    debug_every: int | None = None,
    profile_type: str = "gaussian",  # 'gaussian' | 'median'
    saved_profile: np.ndarray | None = None,
):
    """Optimal (inverse‑variance weighted) photometry for a list of cubes.

    Parameters
    ----------
    filelist : list[str] or str
        List of *rateints.fits* files, or a glob‑style wildcard string.
    xwin, ywin : [x0, x1], [y0, y1]
        Pixel window to cut from the full frame.
    ap_r : float
        Reference radius for the weight map (and the circular comparison
        aperture used in the simple‑sum flux).
    rin, rout : float
        Inner/outer radii of the sky annulus (same units as *ap_r*).
    rn_path : str or pathlib.Path, optional
        Reference read‑noise file for MIRI; if ``None``, read‑noise is ignored.
    debug_every : int, optional
        If given, save a PNG diagnostic every *N* integrations.
    profile_type : {'gaussian', 'median'}
        Weighting scheme.
    saved_profile : ndarray, optional
        Re‑use a pre‑built median profile (only relevant if
        ``profile_type='median'``).

    Returns
    -------
    dict of np.ndarray
        Keys: ``flux_opt``, ``error_opt``, ``flux_ap``, ``bkg``, ``xc``, ``yc``,
        ``sigx``, ``sigy``, ``time``.
    """
    if isinstance(filelist, str):
        filelist = sorted(glob.glob(filelist))
    else:
        filelist = list(filelist)
    if profile_type not in {"gaussian", "median", "zcut"}:
        raise ValueError("profile_type must be 'gaussian' or 'median' or 'zcut'")

    # ----------- read‑noise variance map --------------------
    var_rn = get_read_noise(get_gain())[Y_WINDOW[0] : Y_WINDOW[1], X_WINDOW[0] : X_WINDOW[1]]

    # -------------------- prepare profile template ------------------
    if profile_type == "gaussian":
        # Fit once to first file's median frame
        cube = None
        for f in filelist:
            if not Path(f).exists():
                raise FileNotFoundError(f"File {f} does not exist.")
            with fits.open(f) as hd:
                data = hd[1].data[:, ywin[0] : ywin[1], xwin[0] : xwin[1]]
            if cube is None:
                cube = data
            else:
                cube = np.concatenate((cube, data), axis=0)
        print("Cube shape is " , cube.shape)
        med0 = np.nanmedian(sigma_clip(cube, sigma=3, axis=0), axis=0)
        cx0, cy0 = _locate_source(med0)
        bkg0 = np.median(med0)
        h, cx_fit, cy_fit, sigx0, sigy0 = _fit_gaussian((med0 - bkg0).T, cx0, cy0)

        yy, xx = np.indices(med0.shape)
        gaussian_template = _gaussian2d(1.0, cx_fit, cy_fit, sigx0, sigy0)(yy, xx)
        gaussian_template /= gaussian_template.sum()
        plt.figure()
        plt.imshow(gaussian_template, origin="lower", vmin=np.percentile(gaussian_template, 0.05), vmax=np.percentile(gaussian_template, 99.95))
        plt.savefig('gaussian_template.png')
        prof_center = (cx_fit, cy_fit)
    elif profile_type == "median":  # median profile
        if saved_profile is None:
            cube = None
            for f in filelist:
                if not Path(f).exists():
                    raise FileNotFoundError(f"File {f} does not exist.")
                with fits.open(f) as hd:
                    data = hd[1].data[:, ywin[0] : ywin[1], xwin[0] : xwin[1]]
                if cube is None:
                    cube = data
                else:
                    cube = np.concatenate((cube, data), axis=0)
            print("Cube shape is " , cube.shape)
            cx0, cy0 = _locate_source(np.nanmedian(cube, axis=0))
            saved_profile = _build_median_profile(cube, (cx0, cy0), ap_r)
            prof_center = (cx0, cy0)
        else:
            # assume centre is geometric centre of profile array
            prof_center = tuple(np.array(saved_profile.shape)[::-1] / 2.0)
        plt.figure()
        plt.imshow(saved_profile, origin="lower", vmin=np.percentile(saved_profile, 0.05), vmax=np.percentile(saved_profile, 99.95))
        plt.savefig('median_profile.png')

    elif profile_type == "zcut":
        cube = None
        for f in filelist:
            if not Path(f).exists():
                raise FileNotFoundError(f"File {f} does not exist.")
            with fits.open(f) as hd:
                data = hd[1].data[:, ywin[0] : ywin[1], xwin[0] : xwin[1]]
            if cube is None:
                cube = data
            else:
                cube = np.concatenate((cube, data), axis=0)
        print("Cube shape is " , cube.shape)
        med0 = np.nanmedian(sigma_clip(cube, sigma=3, axis=0), axis=0)
        cx0, cy0 = _locate_source(med0)
        saved_profile = _build_zcut_profile(
                med0,
                (cx0, cy0),
                box_size=20,
                zcut=0.005,
            )
        plt.figure()
        plt.imshow(saved_profile, origin="lower", vmin=np.percentile(saved_profile, 0.05), vmax=np.percentile(saved_profile, 99.95))
        #plt.show()
        plt.savefig('zcut_profile.png')
        prof_center = (cx0, cy0)

    # -------------------- allocate outputs --------------------------
    nint = sum(fits.getdata(f, 1).shape[0] for f in filelist)
    res = {key: np.zeros(nint) for key in [
        "flux_opt",
        "error_opt",
        "flux_ap",
        "bkg",
        "xc",
        "yc",
        "sigx",
        "sigy",
        "time",
    ]}

    grid = None  # cached coordinate grid
    counter = 0

    # -------------------- main loop over cubes ----------------------
    for fname in filelist:
        with fits.open(fname) as hd:
            cube = hd[1].data[:, ywin[0] : ywin[1], xwin[0] : xwin[1]]
            times = hd["INT_TIMES"].data["int_mid_BJD_TDB"]

            for idx, frame in enumerate(cube):
                if grid is None:
                    grid = np.indices(frame.shape)

                # 1. centroid
                cx, cy = _locate_source(frame, guess=prof_center)

                # 2. build weight profile
                if profile_type == "gaussian":
                    dy, dx = cy - prof_center[1], cx - prof_center[0]
                    weight = ndi_shift(
                        gaussian_template, shift=(dy, dx), order=1, mode="nearest"
                    )
                    weight[weight < 0] = 0.0
                    weight /= weight.sum()
                    sigx_eff, sigy_eff = sigx0, sigy0
                    #plt.imshow(gaussian_template, origin="lower", vmin=np.percentile(gaussian_template, 0.05), vmax=np.percentile(gaussian_template, 99.95))
                    #plt.show()
                elif profile_type == "median":
                    dy, dx = cy - prof_center[1], cx - prof_center[0]
                    weight = ndi_shift(saved_profile, shift=(dy, dx), order=1, mode="nearest")
                    weight[weight < 0] = 0.0
                    weight /= weight.sum()
                    sigx_eff = sigy_eff = np.nan
                elif profile_type == "zcut":
                    dy, dx = cy - prof_center[1], cx - prof_center[0]
                    weight = ndi_shift(
                        saved_profile, shift=(dy, dx), order=1, mode="nearest"
                    )
                    weight[weight < 0] = 0.0
                    weight /= weight.sum()
                    sigx_eff = sigy_eff = np.nan
                    #plt.figure()
                    #plt.imshow(saved_profile, origin="lower", vmin=np.percentile(saved_profile, 0.05), vmax=np.percentile(saved_profile, 99.95))
                    #plt.show()
                pos = [(cx, cy)]
                aper = CircularAperture(pos, r=ap_r)
                ann  = CircularAnnulus(pos, r_in=rin, r_out=rout)
                bkg_level = ApertureStats(frame, ann, sigma_clip=SigmaClip(5)).mean[0]
                res["bkg"][counter] = bkg_level

                # 4. variance map (Poisson + RN)
                data_bs = frame - bkg_level
                var_pix = np.abs(data_bs) + var_rn
                minvar = var_pix[var_pix > 0].min()
                var_pix[var_pix <= 0] = minvar

                # 5. weighted flux
                num = np.sum(weight * data_bs / var_pix)
                den = np.sum(weight**2 / var_pix)
                flux_opt = num / den
                err_opt  = np.sqrt(1.0 / den)

                # 6. simple aperture flux (comparison)
                flux_ap = aperture_photometry(frame, aper)["aperture_sum"][0] - bkg_level * aper.area

                # 7. store
                res["flux_opt"][counter] = flux_opt
                res["error_opt"][counter] = err_opt
                res["flux_ap"][counter] = flux_ap
                res["xc"][counter], res["yc"][counter] = cx, cy
                res["sigx"][counter], res["sigy"][counter] = sigx_eff, sigy_eff
                res["time"][counter] = times[idx]

                # 8. debug plot
                if debug_every and counter % debug_every == 0:
                    _plot_image(
                        frame,
                        aper,
                        ann,
                        title=f"{Path(fname).name}  int={idx}  flux={flux_opt:.1f}",
                        savefile=f"debug/img_{counter:05d}.png",
                    )

                counter += 1

    return res



# ---------------------------------------------------------------
hdul_list = glob.glob("../../r1*/*rateints*")
X_WINDOW = [9,129]
Y_WINDOW = [1,121]

hdul_list.sort()
apsize_list = [10,15]
rin_list = [18,20,22,24,26,28,30,32]
rout_list = [40,42,44,46,48,50,52,54]
profile_type = "median"  # 'gaussian' | 'median' | 'zcut'
with open('mad_dict_optimal_'+profile_type+'.txt', 'a+') as f:
    f.write('ap_size annulus_r_in annulus_r_out mad profiletype\n')
for apsize in apsize_list:
    for rin in rin_list:
        for rout in rout_list:
            if rin >= rout:
                continue
            print(f"Processing ap_size={apsize}, rin={rin}, rout={rout}")
            out = optimal_extract(
             hdul_list,
             X_WINDOW,
             Y_WINDOW,
             ap_r=apsize,
             rin=rin,
             rout=rout,
             profile_type=profile_type,
             )
             
            normalized_flux_list = out["flux_opt"] / np.median(out["flux_opt"])
            trend = median_filter(normalized_flux_list, size=30)
            mad = cal_MAD(normalized_flux_list - trend)
            plt.figure()
            plt.errorbar(out["time"],  out["flux_opt"] / np.median(out["flux_opt"]), yerr = out["error_opt"] / np.median(out["flux_opt"]), zorder = 0, alpha = 0.3)
            plt.scatter(out["time"], trend, color = 'red', zorder = 1, s = 3)
            plt.savefig(f'./img/lc_opt_extract_ap{apsize}_in{rin}_out{rout}.png')
            with open('mad_dict_optimal_'+profile_type+'.txt', 'a+') as f:
                f.write(f'{apsize} {rin} {rout} {mad} {profile_type}\n')
            out = pd.DataFrame(out)
            out.to_csv(f'./opt_extract_ap{apsize}_in{rin}_out{rout}.csv', index=False)



