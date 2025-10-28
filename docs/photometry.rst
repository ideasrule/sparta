Photometric data reduction
==============================

Example
--------

In this example we will reduce a single JWST/MIRI imaging (F1500W) from a public program.

Go to ``/example/photometry``. The ``uncal`` folder already contains the raw JWST ``*_uncal.fits`` files needed for photometric reduction.

Once inside, create a new directory called ``reduced`` to store the reduced data.

.. code-block:: bash

    mkdir reduced
    cd reduced

Before running the pipeline, make sure :file:`../sparta/constants.py` has all the necessary :doc:`constants <configuration>` defined.

And now let's start reducing the data.
Step one is to run the calibration pipeline:

.. code-block:: bash

    python ../sparta/calibrate.py ../uncal/jw*fits

This will create ``rateint.fits`` in the output directory. 
The ``SCI`` layer contains either DN per slope or electron number per slope depending on whether the gain scale step is applied. 
When the script is running, steps performed are printed to your terminal, and diagnostic plots are also generated for the ``emicorr`` step.

Now run 

.. code-block:: bash

    python ../sparta/ap_extract.py ./rateints*

to do classic aperture photometry and extract the light curve.

or

.. code-block:: bash

    python ../sparta/opt_ap_extract.py ./rateints*

to do optimal aperture photometry and extract the light curve.


Before running ``ap_extract.py``, make sure to update the following parameters:

- ``apsize_list``  
  Aperture radius for photometric extraction.  

- ``annulus_r_in_list``  
  Inner radius of the sky background annulus.  

- ``annulus_r_out_list``  
  Outer radius of the sky background annulus.  


If your data were obtained in full-array mode, you may want to crop the image
to remove other illuminated regions. This can be done by setting:

- ``X_WINDOW``  
  X-axis cropping window for the region of interest.  

- ``Y_WINDOW``  
  Y-axis cropping window for the region of interest.  


This step generates light curves and saves them to a machine-readable CSV file:

- **CSV filename**:  
  ``ap_extract_ap{ap_size}_in{annulus_r_in}_out{annulus_r_out}.csv``

- **Light-curve plot**:  
  The light curve is also plotted and saved to:  
  ``./img/ap{ap_size}_in{annulus_r_in}_out{annulus_r_out}_lightcurve.png``
  
 For the subsequent light curve fitting, please go to  :doc:`Photometry <fitting>`
