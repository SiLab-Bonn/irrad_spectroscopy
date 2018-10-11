# Imports
import os
from collections import OrderedDict


def get_measurement_time(spectrum_file):
    """
    Reads time of measurement from mcd file of same name as spectrum file.
    """

    # get mcd file of respective spectrum file
    mcd_file = '%s.%s' % (spectrum_file.split('.')[0], 'mcd')

    # file does not exist
    if not os.path.isfile(mcd_file):
        raise IOError('%s does not exist!' % mcd_file)

    # init variable
    t_res = None
    # open file and loop through lines
    with open(mcd_file, 'r') as f_open:
        for line in f_open:
            # "LIVETIME" is measured time
            if 'livetime' in line.lower():
                tmp = line.replace('LIVETIME: ', '')
                t_res = float(tmp)
                break

    # if nothing was found
    if t_res is None:
        raise ValueError('Could not read measurement time from file.')

    return t_res


def isotopes_to_dict(lib, info='lines', fltr=None):
    """
    Method to return dict of isotope keys and info. Info
    can either be 'lines' or 'probability'
    """

    if not isinstance(lib, dict):
        raise TypeError('Isotope library must be dict.')
    if 'isotopes' not in lib:
        raise ValueError('Isotope library must contain isotopes.')
    else:
        isotopes = lib['isotopes']

        # init result dict and loop over different isotopes
        result = {}
        for symbol in isotopes:
            if info in isotopes[symbol]:
                if not isinstance(isotopes[symbol][info], dict):
                    result[symbol] = isotopes[symbol][info]
                else:
                    mass_nums = isotopes[symbol][info].keys()
                    result[symbol] = mass_nums if len(mass_nums) > 1 else mass_nums[0]

            else:
                mass_number = isotopes[symbol]['A']
                for A in mass_number:
                    identifier = '%s_%s' % (str(A), str(symbol))
                    if info in mass_number[A]:
                        if isinstance(mass_number[A][info], list):
                            for i, n in enumerate(mass_number[A][info]):
                                result[identifier + '_%i' % i] = n
                        else:
                            result[identifier] = mass_number[A][info]

        if not result:
            raise ValueError('Isotope library does not contain info %s.' % info)

        if fltr:
            sortout = [k for k in result if fltr not in k]
            for s in sortout:
                del result[s]

        return result


def source_to_dict(source, info='lines'):
    """
    Method to convert a source dict to a dict containing isotpe keys and info.
    """

    reqs = ('A', 'symbol', info)
    if not all(req in source for req in reqs):
        raise ValueError('Missing reuqired data in source dict: %s' % ', '.join(req for req in reqs if req not in source))
    return dict(('%i_%s_%i' % (source['A'], source['symbol'], i) , l) for i, l in enumerate(source[info]))


def calc_activity(observed_peaks, probability_peaks):
    """
    Method to calculate activity isotope-wise. The peak-wise activities of all peaks of
    each isotope are added and scaled with their respectively summed-up probability
    """
    activities = OrderedDict()
    
    for peak in observed_peaks:
        isotope = '_'.join(peak.split('_')[:-1])
        if isotope not in activities:
            activities[isotope] = OrderedDict([('nominal', 0), ('sigma', 0),
                                               ('probability', 0), ('unscaled', {'nominal': 0, 'sigma': 0})])
        
        if peak in probability_peaks:
            activities[isotope]['unscaled']['nominal'] += observed_peaks[peak]['activity']['nominal']
            activities[isotope]['unscaled']['sigma'] += observed_peaks[peak]['activity']['sigma']
            activities[isotope]['probability'] += probability_peaks[peak]
    
    for iso in activities:
        try:
            activities[iso]['nominal'] = activities[iso]['unscaled']['nominal'] * 1. / activities[iso]['probability']
            activities[iso]['sigma'] = activities[iso]['unscaled']['sigma'] * 1. / activities[iso]['probability']
        # when no probabilty given
        except ZeroDivisionError:
            pass

    return activities
