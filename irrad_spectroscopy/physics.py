# collection of physics formulas used in dosimetry and spectroscopy

import numpy as np
from irrad_spectroscopy import xray_coeffs


def decay_constant(half_life):
    return np.log(2.) / half_life


def decay_law(t, x0, half_life):
    return x0 * np.exp(-decay_constant(half_life) * t)


def activity(n0, half_life):
    return decay_constant(half_life) * n0


def mean_lifetime(half_life):
    return 1. / decay_constant(half_life)


def gamma_dose_rate(energy, probability, activity, distance):
    """
    Calculation of the per-gamma dose rate in air according to hps.org/publicinformation/ate/faqs/gammaandexposure.html

    Parameters
    ----------
    energy: float
        gamma energy
    probability: float from 0 to 1
        probability of emitting this gamm per disintegration
    activity: float
        disintegrations per second (Bq)
    distance: float
        distance in cm from gamma source the dose rate should be calculated at

    Returns
    -------

    dose_rate: float
        dose rate from gamma in uSv/h
    """

    # load values for energy-absorption coefficients from package
    xray_energies = np.array(xray_coeffs['energy'])
    xray_en_absorption = np.array(xray_coeffs['energy_absorption'])

    # factor for conversion of intermedate result to uSv/h
    # 1st: link above; 2nd: Roentgen to Sievert; 3rd: combination of Sv to uSv and keV to MeV
    custom_factor = 5.263e-6 * 1. / 107.185 * 1e3

    # find energy-absorption coefficient from coefficients file through linear interpolation
    idx = np.where(xray_energies <= energy)[0][-1]

    if idx == len(xray_en_absorption) - 1:
        tmp_xray_en_ab_interp = xray_en_absorption[-1]
    else:
        tmp_xray_en_ab_interp = np.interp(energy, xray_energies[idx:idx+2], xray_en_absorption[idx:idx+2])

    return custom_factor * energy * probability * tmp_xray_en_ab_interp * activity / distance**2.
