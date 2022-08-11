==================================
Irrad_Spectroscopy |test-status|
==================================

Introduction
============

``ìrrad_spectroscopy`` is a package which provides functions for gamma and X-ray spectroscopy, including isotope identification, activity determination and spectral dose calculations.
Furthermore, it offers functions for calculating the gamma equivalent dose for given isotopes as a function of their initial activity.

The package was developed for spectroscopic analysis or proton-irradiated semiconductor devices but can be used to analyze various samples,
from radioactive sources to activated machine parts. It consits of few independent methods which togehter allow for a complete spectroscopic analysis, including plotting, of
radioactive gamma-spectra. A step-by-step full spectroscopy of an example spectrum can be found in the ``examples`` folder.

Installation
============

You have to have Python 3 with the following packages installed:

- numpy
- scipy
- pyyaml
- matplotlib
- jupyter (examples)
- pandas (creating gamma library from the web)
- pytest (run the tests)

It's recommended to use a Python environment like `Miniconda <https://conda.io/miniconda.html>`_. After installation you can use Minicondas package manager ``conda`` to install the required packages

.. code-block:: bash

   conda install -y numpy scipy pyyaml matplotlib jupyter pandas pytest

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

Equivalent dose calculation
---------------------------

The package implements dose rate calculations for individual gamma lines as well as full gamma spectra of isotopes
for various materials (search materials in `this table <https://github.com/SiLab-Bonn/irrad_spectroscopy/blob/development/irrad_spectroscopy/tables/xray_coefficient_table.yaml>`_)
Dose rate calculations are implemented, assuming a point-like source!

Calculating dose rate of an individual gamma line in air:

.. code-block:: python

   # Import 
   from irrad_spectroscopy.physics import gamma_dose_rate

   # Get dose rate of single gamma line in uSv/h
   # Zn65 line at 1115.564 keV, prob 50.60%, activity of 20 kBq at a distance of 100 cm in air
   res = gamma_dose_rate(energy=1115.546,
                         probability=0.506,
                         activity=20e3,
                         distance=100,
                         material='air')

   print(res)  # Prints 1.515e-3  # uSv/h

Calculating the (integrated) gamma dose rate of an isotope in air:

.. code-block:: python

   # Import 
   from irrad_spectroscopy.physics import isotope_dose_rate

   # Zn65 with activity of 20 kBq at a distance of 100 cm in air
   res = isotope_dose_rate(isotope='65_Zn',
                           activity=20e3,
                           distance=100,
                           material='air')
   
   print(res)  # Prints {'65_Zn': 1.515e-3}  # uSv/h

   # Zn65 with activity of 20 kBq at a distance of 100 cm in air
   # integrated over 2000 hours
   res = isotope_dose_rate(isotope='65_Zn',
                           activity=20e3,
                           distance=100,
                           material='air',
                           time=2000)
   
   print(res)  # Prints {'65_Zn': 2.66}  # uSv

Calculating the gamma dose rate of multiple isotopes in air:

.. code-block:: python

   # Import 
   from irrad_spectroscopy.physics import isotope_dose_rate

   # Multiple isotopes (Zn65 and Be7) with different activities
   # (20 kBq, 100kBq) at a distance of 100 cm in air
   res = isotope_dose_rate(isotope=('65_Zn', '7_Be'),
                           activity=(20e3, 100e3),
                           distance=100,
                           material='air')
   
   print(res)  # Prints {'65_Zn': 1.515e-3, '7_Be': 0.73e-3}  # uSv/h

Particle fluence calculation from isotope activity
--------------------------------------------------

It is possible to calculate the number of particles per unit area, which penetrated a given sample material,
by knowing their producion cross-section for activating an isotope in the material.
Given the activity of the isotope, its molar mass as well as the mass of the sample
, the ``irrad_spectroscopy.physics`` submodule provides a function for the calculation:

.. code-block:: python

    # Import
    from irrad_spectroscopy.physics import fluence_from_activity

    # Vanadium 48, generated with ~380 mb effective cross section from proton irradiation of Titanium foil, weighing 11 mg.
    res = fluence_from_activity(isotope='48_V',  # needed for half-life determination
                                activity=28e3,  # Bq
                                cross_section=380,  # mb Ti -> 48 V, effective cross section
                                molar_mass=47.952, # g/mol
                                sample_mass=11)  # mg
    
    print(res)  # Prints 1.062e15 protons/cm²

You can add a cooldown time to correct for the decay of isotope bewteen isotope activation and activity measurement.
Furthermore, if the production cross-section is not "effective" but rather resolved specifically for the specific isotope,
you can pass the abundance of the isotope's parent in the sample material to get the effective production:

.. code-block:: python

    # Import
    from irrad_spectroscopy.physics import fluence_from_activity

    # Vanadium 48, generated with ~380 mb effective cross section from proton irradiation of Titanium foil, weighing 11 mg.
    res = fluence_from_activity(isotope='48_V',  # needed for half-life determination
                                activity=28e3,  # Bq
                                cross_section=550,  # mb for 48 Ti -> (p,n) -> 48 V, dedicated cross section
                                molar_mass=47.952, # g/mol
                                sample_mass=11,  # mg
                                abundance=0.7372,  # % of stable Titanium
                                cooldown_time=48)  # hours between activation and measurement of activity
    
    print(res)  # Prints 9.1226e14 protons/cm²

Testing
=======

The code in this package has unit-tests. These tests contain a benchmark with actual gamma-spectroscopy data of
two calibrated, radioactive sources, namely 22-Na and 133-Ba. The activity reconstruction efficiencies for the 
tested data sets are tested to be above 90%.
Furthermore, the ``irrad_spectroscopy.physics.isotope_dose_rate`` function is cross-checked with results from
`RadCalculatorPro <http://www.radprocalculator.com/Gamma.aspx>`_ for a handful of isotopes to be in agreement,
with a maximum deviation of 20%.
 
.. |test-status| image:: https://github.com/Silab-Bonn/irrad_spectroscopy/actions/workflows/main.yml/badge.svg?branch=development
    :target: https://github.com/SiLab-Bonn/irrad_spectroscopy/actions
    :alt: Build status

Example spectrum
================

Generated spectrum, including background and identified peaks, of a radioactive sample after proton irradiation.
Multiple isotopes can be assigned to one peak due to the uncertaiunty of the energy calibration.

.. image:: static/figs/sample_spectrum.png