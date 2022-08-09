# collection of physics formulas used in dosimetry and spectroscopy

import logging
import numpy as np
from scipy.integrate import quad
from irrad_spectroscopy import xray_coefficient_table, xray_coefficient_table_file
from irrad_spectroscopy.spec_utils import get_isotope_info



def decay_constant(half_life):
    return np.log(2.) / half_life


def decay_law(t, x0, half_life):
    return x0 * np.exp(-decay_constant(half_life) * t)


def activity(n0, half_life):
    return decay_constant(half_life) * n0


def mean_lifetime(half_life):
    return 1. / decay_constant(half_life)


def gamma_dose_rate(energy, probability, activity, distance, material='air'):
    """
    Calculation of the per-gamma dose rate in air according to hps.org/publicinformation/ate/faqs/gammaandexposure.html

    Parameters
    ----------
    energy : float
        gamma energy
    probability : float from 0 to 1
        probability of emitting this gamma per disintegration
    activity : float
        disintegrations per second (Bq)
    distance : float
        distance in cm from gamma source the dose rate should be calculated at
    material : str
        string of material the dose is to be calculated in. Must be key in xray_coeffs

    Returns
    -------

    dose_rate: float
        dose rate from gamma in uSv/h
    """

    if material not in xray_coefficient_table['material'].keys():
        msg = 'No x-Ray coefficient table for material "{}". Please add table to {}.'.format(material, xray_coefficient_table_file)
        raise KeyError(msg)

    # load values for energy-absorption coefficients from package
    xray_energies = np.array(xray_coefficient_table['material'][material]['energy'])
    xray_en_absorption = np.array(xray_coefficient_table['material'][material]['energy_absorption'])

    # factor for conversion of intermedate result to uSv/h
    # 1st: link above; 2nd: Roentgen to Sievert; 3rd: combination of Sv to uSv and keV to MeV
    custom_factor = 5.263e-6 * 1. / 107.185 * 1e3

    # find energy-absorption coefficient from coefficients file through linear interpolation
    idx = np.where(xray_energies <= energy)[0][-1]

    if idx == len(xray_en_absorption) - 1:
        msg = '{} keV larger than largest energy in x-Ray coefficient table.' \
              ' Taking coefficient of largest energy available ({} keV) instead'.format(energy, xray_energies[-1])
        logging.warning(msg)
        tmp_xray_en_ab_interp = xray_en_absorption[-1]
    else:
        tmp_xray_en_ab_interp = np.interp(energy, xray_energies[idx:idx+2], xray_en_absorption[idx:idx+2])

    return custom_factor * energy * probability * tmp_xray_en_ab_interp * activity / distance**2.


def isotope_dose_rate(isotope, activity, distance, material='air', time=None):
    """
    Calculation of the per-isotope dose rate in *material* according to hps.org/publicinformation/ate/faqs/gammaandexposure.html

    Parameters
    ----------
    isotope : str
        identifier according to
    probability : float from 0 to 1
        probability of emitting this gamma per disintegration
    activity : float
        disintegrations per second (Bq)
    distance : float
        distance in cm from gamma source the dose rate should be calculated at
    material : str
        string of material the dose is to be calculated in. Must be key in xray_coeffs
    time : int, float
        time to integrate over in hours

    Returns
    -------

    dose_rate: float
        dose rate from gamma in uSv/h
    """

    if material not in xray_coefficient_table['material'].keys():
        msg = 'No x-Ray coefficient table for material "{}". Please add table to {}.'.format(material, xray_coefficient_table_file)
        raise KeyError(msg)

    if not isinstance(isotope, (list, tuple)):
        isotope = [isotope]
        if any(not isinstance(iso, str) for iso in isotope):
            raise TypeError('*isotope* must be str or list of strings with identifiers.')

    if not isinstance(activity, (list, tuple)):
        activity = [activity]
        if any(not isinstance(act, (float, int)) for act in activity) or len(activity) != len(isotope):
            raise TypeError('*activity* must be number or list of number with corresponding to *isotope*.')


    half_lifes = None if time is None else get_isotope_info(info='half_life')

    total_dose_rate = {}

    for iso, act in zip(isotope, activity):

        # Get intensity and energy of isotope lines
        iso_probabilities = get_isotope_info(info='probability', iso_filter=iso)
        iso_energies = get_isotope_info(info='lines', iso_filter=iso)

        total_dose_rate[iso] = 0  # uSv/h

        for line in iso_energies:

            total_dose_rate[iso] += gamma_dose_rate(energy=iso_energies[line],
                                                    probability=iso_probabilities[line],
                                                    activity=act,
                                                    distance=distance,
                                                    material=material)

        # Integrate over time returning absolut dose in uSv
        if half_lifes:
            total_dose_rate[iso], _ = quad(decay_law, 0, time, args=(total_dose_rate[iso], half_lifes[iso]/60.**2))

    return total_dose_rate


def fluence_from_activity(isotope, acticity, cross_section, molar_mass, sample_mass, abundance=1.0, cooldown_time=0.0):
    """
    Calculation of the theoretical particle fluence [# particles / cm^2] which produced a given *activity* of
    an *isotope* with a production *cross_section* in a given, thin (e.g. *cross_section* const.) *sample_mass*.
    The *isotope* has *molar_mass* and the *sample_mass* has an *abundance* of atoms that produce said *isotope* with
    *cross_section*.
    The return value is a scalar and contains no information about the distribution of the particles on the sample area.

    Parameters
    ----------
    isotope : str
        identifier in the form of *NNN_XX* where NNN is the mass number and XX the abbreviation of the element e.g. '65_Zn'
    activity : float
        disintegrations per second (Bq)
    cross_section : float
        Production cross-section for the process: particle -> sample => isotope in milli-barn (mb) 
    molar_mass : float
        Molar mass of the isotope in g/mol
    sample_mass : float
        Mass of the sample in milligram (mg)
    abundance : float, optional
        Abundance of the atoms in samples that produced *isotope* with *cross_section*, by default 1.0
        The default value of 1.0 assumes that either the samples atoms are 100% producing *isotope* with given *cross_section*
        or that the given *cross_section* is an effective cross-section.
    cooldown_time : float, optional
        Time in seconds elapsed since *activity* was generated; used to correct for decay

    Returns
    -------
    fluence : float
        Particle fluence in # particles / cm^2 which
    """

    # Get isotope half life
    half_life = get_isotope_info(info='half_life')[isotope]

    # Conversions
    cross_section_in_cm_square = cross_section * 1e-27  # Convert mb to cm^2
    sample_mass_in_grams = sample_mass * 1e-3  # Convert mg to g
    sample_mass_in_grams *= abundance  # Correct for abundance in material
    dc = decay_constant(half_life)

    fluence = acticity / cross_section_in_cm_square * molar_mass / (sample_mass_in_grams * 6.02214076e23) * 1 / dc
    fluence *= np.exp(-dc * cooldown_time)  # Correct for time passed since activity was produced

    return fluence
