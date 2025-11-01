Configuration
==============
SPARTA requires two configuration files to be modified:

- ``constants.py``  
  Defines the instrument modes, required reduction steps, reference files, and image processing parameters.

- ``planet.cfg``  
  Contains the light curve fitting parameters.

Data reduction parameters
=========================
.. _constants:

constants.py
------------

- ``INSTRUMENT``  
  Specifies the instrument to be used (now supports ``NIRSpec``, ``NIRCam``, ``MIRI_IMAGE``, ``MIRI_LRS``). 

- ``SUBARRAY``  
  Defines the SUBARRAY in use (e.g., ``SUB2048``, ``SUB512``, ``SUB256``, ``SUB128``, ``FULL``). 

- ``FILTER``  
  Defines the filter in use (e.g., for ``NIRSpec/PRISM``, use ``CLEAR``; for ``NIRCam``, use ``F322W2`` or ``F444W``; for ``MIRI_IMAGE``, use ``F1500W`` etc.). 
  Additional filters are under development.


- ``REFERENCE_FILES``  
  Paths to the calibration reference files (e.g., flat fields, dark). 
  The master directory for all the reference files can be set up by ``REF_DIR``.

  Users **must** obtain suitable reference files for the selected instrument and subarray from the `JWST CRDS <https://jwst-crds.stsci.edu/>`_ archive and specify the desired file versions.

- ``SKIP_REF/FLAT/SUPERBIAS/EMICORR``  
  Flags to skip specific reference file corrections during processing. Set to ``True`` or ``False`` as needed.
  
  For ``MIRI_IMAGE``, these steps are skipped by default. We encourage users to experiment with the emicorr step and compare the resulting outputs.


Light curve fitting parameters
==============================
planet.cfg
----------

The configuration file specifies both the inference settings and the light curve fitting parameters.  
Each row defines a parameter with columns: 

- The first column is the parameter name.
- The second column is the parameter type: ``free``, ``fixed``, ``independent``.  
- ``PriorType`` can be:

  - ``U`` = Uniform (PriorPar1 = lower bound, PriorPar2 = upper bound)  
  - ``LU`` = Log-Uniform (PriorPar1 = lower log bound, PriorPar2 = upper log bound)  
  - ``N`` = Normal (PriorPar1 = mean, PriorPar2 = standard deviation)  


.. list-table:: 
   :header-rows: 1
   :widths: 24 76

   * - **Parameter**
     - **Description**
   * - ``fitting_method``
     - Sampling engine (e.g., ``emcee`` for MCMC or ``dynesty`` for nested sampling).
   * - ``mcmc_nwalkers``
     - Number of walkers for ``emcee``.
   * - ``mcmc_burnin``
     - Number of burn-in steps (discarded).
   * - ``mcmc_production``
     - Number of production steps (kept).
   * - ``dynesty_nlive``
     - Live points for nested sampling.
   * - ``dynesty_bound``
     - Bounding strategy (``none``, ``single``, ``multi``, ``balls``, ``cubes``).
   * - ``dynesty_sampling``
     - NS sampling method (``unit``, ``rwalk``, ``slice``, ``rslice``, ``hslice``).
   * - ``dynesty_dlogz``
     - Convergence threshold (smaller = stricter).
   * - ``joint_fit``
     - If ``True``, fit multiple datasets simultaneously.
   * - ``start_wave``
     - Lower wavelength bound (nm).
   * - ``end_wave``
     - Upper wavelength bound (nm).
   * - ``exclude``
     - Integrations to exclude. Should be a list of lists, e.g., [[start1, end1], [start2, end2]...].
   * - ``data_path``
     - Path to the light-curve data from ``ap_extract.py``.
   * - ``lc_savepath``
     - Output path for modeled light curve.
   * - ``best_fit_savepath``
     - Output path for best-fit parameters.
   * - ``rp``
     - Planet-to-star radius ratio (Rp/Rs)
   * - ``fp``
     - Planet-to-star flux ratio (secondary-eclipse depth).
   * - ``per``
     - Orbital period (days).
   * - ``t0``
     - Mid-transit epoch (BJD).
   * - ``inc``
     - Orbital inclination (deg).
   * - ``b``
     - Orbital impact parameter. Please ensure commenting out ``inc`` if ``b`` is used, and vice versa.
   * - ``a_star``
     - Scaled semi-major axis (a/Rs).
   * - ``sqrt_ecosw``
     - Eccentricity parameterization term. :math:`\sqrt{e}cos\omega`
   * - ``sqrt_esinw``
     - Eccentricity parameterization term. :math:`\sqrt{e}sin\omega`
   * - ``t_secondary``
     - Secondary-eclipse mid-time (BJD).
   * - ``Rs``
     - Stellar radius.
   * - ``limb_dark``
     - LD law (``uniform``, ``linear``, ``quadratic``, ``kipping2013``. Only those supported by ``batman`` are allowed).
   * - ``q1 and q2``
     - Limb-darkening coefficient
   * - ``Fstar``
     - Stellar baseline flux normalization.
   * - ``m``
     - Linear slope.
   * - ``A``
     - Exponential ramp amplitude term.
   * - ``tau``
     - Exponential ramp time scale.
   * - ``x_coeff``
     - Linear decorrelation vs. detector x-position.
   * - ``y_coeff``
     - Linear decorrelation vs. detector y-position.
   * - ``error_factor``
     - Multiplicative inflation of uncertainties.
