# Simple Planetary Atmosphere Reduction Tool for Anyone

Simple, fast, barebones, utilitarian.  Spartan.

How to use SPARTA to analyze MIRI/LRS or NIRCAM data of exoplanets:

1. Open constants.py.  Change INSTRUMENT to either MIRI or NIRCAM.  Change SUBARRAY and FILTER to the desired values (found in the file header).  The reference files should be correct and relatively up to date, but if you want to verify, go to https://jwst-crds.stsci.edu/.

2. Make a working directory and copy some useful files there.  For example:
```mkdir hd189733b
cd hd189733b
cp ../sparta/hd189733b/* .
```

3. Download all the uncalibrated files from MAST to the working directory.  The *uncal.fits should be in the directory directly, not wrapped in subdirectories. 

4. Calibrate the uncalibrated files and output rateint files, which contain slopes of the up-the-ramp samples:
```python ../sparta/calibrate.py jw*uncal.fits```

5. Remove the background, and for NIRCAM, also detect and remove outliers:
```python ../sparta/remove_bkd.py rateints_jw*.fits```

This outputs cleaned_rateints_jw*.fits files.

6. Calculate the x and y positions of the trace for each integration of each segment, outputting them into positions.txt:
```python ../sparta/get_positions.py cleaned_rateints_jw*.fits```

7. Extract the spectrum.

   - To do simple sum extraction:
     ```python ../sparta/simple_extract.py cleaned_rateints_jw*.fits```
     This outputs x1d_cleaned_rateints_jw*.fits files.

   - To do optimal extraction:
     ```
     python ../sparta/get_med_image.py cleaned_rateints_jw*.fits
     python ../sparta/optimal_extract.py cleaned_rateints_jw*.fits
     ```

     The first line computes a median image from all the integrations, outputting median_image.npy.  The second line performs the optimal extraction, outputting optx1d_cleaned_rateints_jw*.fits files.

8. Gather up the fluxes and positions into one data.pkl pickle file, while filtering out the bad integrations:
```python ../sparta/gather_and_filter.py optx1d_rateints_jw*.fits```

Replace optx1d with x1d, if using simple extraction.

9. Run white light fit (in this example, from 3.94 to 4.98 um):
```python ../sparta/extract_eclipse.py hd189733b.cfg 3940 4980 -e 580```

There are similar scripts for transits and phase curves, which additionally require a limb_dark.txt file to be present (see example in gj1214b directory).

This step produces some plots, in addition to a white_light_result.txt summarizing the retrieved parameters, and a white_lightcurve.txt containing the fluxes, the astrophysical model, the systematics model, and the residuals.  The chain is saved in chain/.  -e indicates the number of integrations to exclude from the beginning.  Excluding at least 30 minutes for NIRCam and 1 hour for MIRI is recommended.

9. Run spectral fit:
```python ../sparta/extract_eclipse_limited.py hd189733b.cfg 4000 4050 -e 580```

This runs emcee for the 4--4.05 um spectral bin.

For convenience, a script has been added to the hd189733b/ example directory to run the spectral fit in parallel on all spectral bins.  Please modify it according to your needs, and run it:

```./run_all_wavelengths.sh```

The retrieved parameters are in result.txt, the lightcurves are in lightcurves.txt, while the chains are stored in their own files for each bin (e.g. chain_5000_5500.npy stores the chain for 5–5.5 um).
