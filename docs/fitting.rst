Fitting the light curve
=======================
Photometry
----------

Update the planetary and fitting parameters for your run in the configuration file ``planet.cfg``. 

For tutorial purpose, there is a ready-made configuration file ``gj3929b.cfg`` in the ``example/gj3929b/eclipse_fit`` directory.

.. code-block:: console

   cd eclipse_fit

Then run the fitting script:

.. code-block:: console

   python ../sparta/extract_eclipse_photometry.py gj3929b.cfg

Normalized light curves used for fitting will be displayed in a GUI window.  
The fitting process begins automatically, and progress is printed to the terminal.

After the fit completes:

- The best-fit parameters are printed in the terminal.
- The normalized light curves overplotted with the best-fit model are shown and saved.
- Corner plots of the posterior distributions are generated and saved.

All outputs are written to the working directory.

Spectroscopy
------------
Under active construction...
