==================================
Irrad_Spectroscopy |test-status|
==================================

Introduction
============

``irrad_spectroscopy`` is a package intended to do gammma spectroscopy of (proton) irradiated samples such as chips, sensors,
PCBs, etc. but can be also used for general gamma spectroscopy e.g. of radioactive sources. It consits of few independent
methods which togehter allow for a complete spectroscopic analysis of radioactive gamma-spectra. A step-by-step full spectroscopy
of an example spectrum can be found in the ``examples`` folder.

Installation
============

You have to have Python 2/3 with the following packages installed:

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

Check the ``examples`` folder for several measured data sets of different sources for calibration and analysis. A `Jupyter Notebook <http://jupyter.org/>`_
with a step-by-step analysis of an example spectrum of an irradiated chip is provided. Install jupyter and run

.. code-block:: bash
   
   jupyter notebook

in order to open the web interface.

Testing
=======

The code in this package has unit-tests. These tests contain a benchmark with actual gamma-spectroscopy data of
two calibrated, radioactive sources, namely 22-Na and 133-Ba. The activity reconstruction efficiencies for the 
tested data sets are tested to be above 90%.
 

.. |test-status| image:: https://github.com/Silab-Bonn/irrad_spectroscopy/actions/workflows/main.yml/badge.svg?branch=development
    :target: https://github.com/SiLab-Bonn/irrad_spectroscopy/actions
    :alt: Build status
