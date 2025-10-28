.. SPARTA documentation master file, created by
   sphinx-quickstart on Wed Aug 27 16:29:39 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SPARTA's documentation!
=====================================

**SPARTA** (Simple Planetary Atmosphere Reduction Tool for Anyone) is a JWST data reduction and light curve fitting tool. 
It supports NIRSpec, NIRSpec/PRISM, NIRCam, MIRI/LRS and MIRI/Imaging. 

It aims for **Simple, fast, barebones, utilitarian. Spartan. data analysis**. 

It provides a stand-alone, end-to-end pipeline for reducing JWST data.  
It is designed to be instrument-agnostic, **customizable**, and **independent** of the official STScI pipeline.  
The reduction process converts uncalibrated FITS images into science-ready light curves, suitable for transit and eclipse modeling. 

**SPARTA** also provides a user-friendly interface for light curve fitting.

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   spectroscopy
   photometry
   fitting

.. toctree::
   :maxdepth: 1
   :caption: Configuration

   configuration
