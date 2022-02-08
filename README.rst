==================================
Irrad_Spectroscopy |test-status|
==================================

Introduction
============

```ìrrad_spectroscopy``` is a package which provides functions for gamma and X-ray spectroscopy, including isotope identification, activity determination and spectral dose calculations.
Furthermore, it offers functions for calculating the gamma equivalent dose for given isotopes as a function of their initial activity.
The package was developed for spectroscopic analysis or proton-irradiated semiconductoir detector devices but can be used to analyze various samples
from radioactive sources to activated machine parts. It consits of few independent methods which togehter allow for a complete spectroscopic analysis, including plotting, of
radioactive gamma-spectra. A step-by-step full spectroscopy of an example spectrum can be found in the ``examples`` folder.

Structure
=========

The general approach to perform a spectroscopy is divided into the following steps:

Energy calibartion with a known source
--------------------------------------

.. image:: static/figs/energy_calib_eu152.png
  
Efficiency calibration with a known source
------------------------------------------

.. image:: static/figs/efficiency_calib_eu152.png

Fitting of spectrum of unknown sample
-------------------------------------

.. image:: static/figs/sample_spectrum.png


Installation
============

You have to have Python 3 with the following packages installed:

- numpy
- scipy
- pyyaml
- matplotlib
- jupyter (examples)
- pandas (creating gamma library from the web)

It's recommended to use a Python environment like `Miniconda <https://conda.io/miniconda.html>`_. After installation you can use Minicondas package manager ``conda`` to install the required packages

.. code-block:: bash

   conda install numpy scipy pyyaml matplotlib jupyter pandas

To finally install ``irrad_spectroscopy`` run the setup file

.. code-block:: bash

   python setup.py develop

Example usage
=============

Full spectroscopy
-----------------

Check the ``examples`` folder for several measured data sets of different sources for calibration and analysis. A `Jupyter Notebook <http://jupyter.org/>`_
with a step-by-step analysis of an example spectrum of an irradiated chip is provided. Install jupyter and run

.. code-block:: bash
   
   jupyter notebook

in order to open the web interface.

Eqivalent dose calculation
--------------------------

TBD

Testing
=======

The code in this package has unit-tests. These tests contain a benchmark with actual gamma-spectroscopy data of
two calibrated, radioactive sources, namely 22-Na and 133-Ba. The activity reconstruction efficiencies for the 
tested data sets are tested to be above 90%.
 

.. |test-status| image:: https://github.com/Silab-Bonn/irrad_spectroscopy/actions/workflows/main.yml/badge.svg?branch=development
    :target: https://github.com/SiLab-Bonn/irrad_spectroscopy/actions
    :alt: Build status
