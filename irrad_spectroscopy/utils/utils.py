# Imports
import os
from collections import OrderedDict


def get_measurement_time(spectrum_file):

    # get mcd file of respective spectrum file
    mcd_file = '%s.%s' % (spectrum_file.split('.')[0], 'mcd')

    if not os.path.isfile(mcd_file):
        raise IOError('%s does not exist!' % mcd_file)
    t_res = None
    with open(mcd_file, 'r') as f_open:
        for line in f_open:
            if 'livetime' in line.lower():
                tmp = line.replace('LIVETIME: ', '')
                t_res = float(tmp)
                break
    if t_res is None:
        raise ValueError('Could not read measurement time from file.')
    return t_res


def isotopes_to_dict(lib, info='lines'):
    if not isinstance(lib, dict):
        raise TypeError('Isotope library must be dict.')
    if 'isotopes' not in lib:
        raise ValueError('Isotope library must contain isotopes.')
    else:
        isotopes = lib['isotopes']
    result = {}
    for symbol in isotopes:
        mass_number = isotopes[symbol]['A']
        for A in mass_number:
            identifier = '%s_%s' % (str(A), str(symbol))
            for i, n in enumerate(mass_number[A][info]):
                result[identifier + '_%i' % i] = n
    return result


def source_to_dict(source, info='lines'):
    reqs = ('A', 'symbol', info)
    if not all(req in source for req in reqs):
        raise ValueError('Missing reuqired data in source dict: %s' % ', '.join(req for req in reqs if req not in source))
    return dict(('%i_%s_%i' % (source['A'], source['symbol'], i) , l) for i, l in enumerate(source[info]))


def calc_activity(observed_peaks, probability_peaks):
    activities = OrderedDict()
    
    for peak in observed_peaks:
        isotope = ''.join(peak.split('_')[:-1])
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
        except ZeroDivisionError:
            pass

    return activities
    


def add_isotope_to_lib(lib_file, isotope_dict):
    # open lib file
    with open(lib_file, 'r') as lf:
        lib = yaml.safe_load(lf)

    reqs = ['symbol', 'Z', 'A', 'lines', 'probability']

    for req in reqs:
        missing_reqs = []
        if req not in isotope_dict:
            missing_reqs.append(req)
    if missing_reqs:
        raise ValueError('Required info of %s is missing!' % ', '.join(missing_reqs))

    if isotope_dict['symbol'] in lib['isotopes']:
        if isotope_dict['A'] in lib['isotopes'][isotope_dict['symbol']]['A']:
            tmp = list(lib['isotopes'][isotope_dict['symbol']]['A'][isotope_dict['A']]['probability'])
            tmp.append(isotope_dict['probability'])
            index = list(reversed(sorted(tmp))).index(isotope_dict['probability'])
            current_isotope = lib['isotopes'][isotope_dict['symbol']]['A'][isotope_dict['A']]
            current_isotope['probability'].insert(index, isotope_dict['probability'])
            current_isotope['lines'].insert(index, isotope_dict['lines'])
        else:
            mass_number = lib['isotopes'][isotope_dict['symbol']]['A']
            mass_number[isotope_dict['A']] = {'lines': list(isotope_dict['lines']),
                                              'probability': list(isotope_dict['probability'])}
    else:
        lib['isotopes'][isotope_dict['symbol']] = {'Z': isotope_dict['Z'],
                                                   'A': isotope_dict['A']}
        lib['isotopes'][isotope_dict['symbol']]['A'] = {}
        lib['isotopes'][isotope_dict['symbol']]['A'][isotope_dict['A']] = {'lines': [isotope_dict['lines']],
                                                                           'probability': [isotope_dict['probability']]}
