# Imports
import os
import yaml
import time
import datetime
import logging
import irrad_spectroscopy as isp
import numpy as np
from collections import OrderedDict
from copy import deepcopy

# try importing pandas
try:
    import pandas as pd
    _PANDAS_FLAG = False
except ImportError:
    _PANDAS_FLAG = True


# needed to dump OrderedDict into file, representer for OrderedDict (https://stackoverflow.com/a/8661021)
represent_dict_order = lambda self, data: self.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, represent_dict_order)


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


def date_to_posix(year, month, day, hour=0, minute=0, second=0):
    """ Returns posix timestamp from date and optionally time"""
    return time.mktime(datetime.datetime(year, month, day, hour, minute, second).timetuple())


def get_isotope_info(table=isp.gamma_table, info='lines', iso_filter=None):
    """
    Method to return dict of isotope info from gamma table. Info can either be 'lines', 'probability', 'half_life',
    'decay_mode', 'name', 'A', or 'Z'. Keys of result dict are element symbols.

    Parameters
    ----------
    table : dict
        gamma table of isotopes with additional info. Default is isp.gamma_table
    info : str
        information which is needed. Default is 'lines' which corresponds to gamma energies. Can be either of the ones
        listed above
    iso_filter : str
        string of certain isotope whichs info you want to filter e.g. '65_Zn' or '65' or 'Zn'
    """

    if not isinstance(table, dict):
        raise TypeError('Gamma table must be dict.')
    if 'isotopes' not in table:
        raise ValueError('Gamma table must contain isotopes.')
    else:
        isotopes = table['isotopes']

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
            raise ValueError('Gamma table does not contain info %s.' % info)

        if iso_filter:
            sortout = [k for k in result if iso_filter not in k]
            for s in sortout:
                del result[s]

        return result


def source_to_dict(source, info='lines'):
    """
    Method to convert a source dict to a dict containing isotope keys and info.
    """

    reqs = ('A', 'symbol', info)
    if not all(req in source for req in reqs):
        raise ValueError('Missing reuqired data in source dict: %s' % ', '.join(req for req in reqs if req not in source))
    return dict(('%i_%s_%i' % (source['A'], source['symbol'], i) , l) for i, l in enumerate(source[info]))


def select_peaks(selection, peaks):
    """
    Convenience function to remove certain lines from peaks dictionary. Returns copy in order to avoid mutating.

    Parameters
    ----------

    selection : iterable of keys
        list or iterable of keys which are in peaks which should be selected
    peaks : dict
        return value of irrad_spectroscopy.spectroscopy.fit_spectrum

    Returns
    -------

    selected_peaks : dict
        copy of peaks with every key removed except for the ones contained in selection
    """

    # sanity checks
    check = [s in k for s in selection for k in peaks]
    if not any(check):
        raise ValueError('None of the selection criteria matches any peaks!')

    selected_peaks = deepcopy(peaks)

    # remove
    for k in peaks:
        if not any(s in k for s in selection):
            del selected_peaks[k]

    return selected_peaks


def create_gamma_table(outfile=None, e_min=1.0, e_max=20000.0, half_life=1.0, n_lines=10, prob_lim=1e-2):
    """
    Method that creates a table of gammas from radiactive isotopes from http://atom.kaeri.re.kr:8080/gamrays.html.
    The data is structured in OrderedDicts and dumped into a yaml. Pandas needs to be installed.

    Parameters
    ----------

    outfile: str
        path to output yaml or None; if None only return table dict
    e_min: float
        minimum energy in keV to include into the table file
    e_max: float
        maximum energy in keV to include into the table file
    half_life: float
        minimum half life in days the isotopes need to have
    n_lines:
        max number of prominent lines per isotope
    prob_lim: float
        minimum probability the individual lines have to have

    Returns
    -------

    res: dict
        result dict with gammas lines and info
    """

    # check wheter pandas is installed
    if _PANDAS_FLAG:
        logging.error('Pandas could not be imported. Please make sure it is installed!')
        return

    # result to be dumped in yaml
    res = OrderedDict()

    # half life factors for conversion to seconds
    hf_factors = {'D': 24. * 60**2, 'H': 60.**2, 'Y': 365. * 24. * 60**2., 'M': 60., 'S': 1.}

    # read gamma info from http://atom.kaeri.re.kr:8080/gamrays.html
    url = r'http://atom.kaeri.re.kr:8080/cgi-bin/readgam?xmin={}&xmax={}&h={}&i={}&l=100000'.format(e_min, e_max,
                                                                                                    half_life, n_lines)
    logging.info('Reading gamma table from {}...'.format(url))

    # read html table
    gamma_table = np.array(pd.read_html(url)[0])

    logging.info('Finished reading gamma table. Restructure...')

    # extract data from gamma_table
    energies = gamma_table[1:, 0]
    probabilities = gamma_table[1:, 1]
    meta = gamma_table[1:, 2]

    for i in range(gamma_table.shape[0] - 1):
        try:
            # make tmp variables
            tmp_e = float(energies[i].split('(')[0])
            tmp_prob = float(probabilities[i]) / 100.0
            tmp_iso = meta[i].split(' ')[0]
            tmp_decay = meta[i].split(' ')[1][1:]
            tmp_symb = tmp_iso.split('-')[0]
            tmp_A = int(tmp_iso.split('-')[1])
            tmp_hf = float(meta[i].split(' ')[2]) * hf_factors[meta[i].split(' ')[3][0]]

            # only take lines with emission probability above prob_lim
            if tmp_prob < prob_lim or ',' in tmp_symb:
                continue

        # isomeric transitions are not included
        except ValueError:
            continue

        # make entries
        if tmp_symb not in res:
            res[tmp_symb] = OrderedDict()
            res[tmp_symb]['name'] = isp.element_table['names'][tmp_symb]
            res[tmp_symb]['Z'] = isp.element_table['Z'][tmp_symb]
            res[tmp_symb]['A'] = OrderedDict()

        # make entries for mass numbers
        if tmp_A not in res[tmp_symb]['A']:
            res[tmp_symb]['A'][tmp_A] = OrderedDict()
            res[tmp_symb]['A'][tmp_A]['reaction'] = None
            res[tmp_symb]['A'][tmp_A]['cross_section'] = None
            res[tmp_symb]['A'][tmp_A]['half_life'] = tmp_hf
            res[tmp_symb]['A'][tmp_A]['decay_mode'] = tmp_decay

        # make lists for lines and emission probabilities
        if 'lines' not in res[tmp_symb]['A'][tmp_A]:
            res[tmp_symb]['A'][tmp_A]['lines'] = []
        if 'probability' not in res[tmp_symb]['A'][tmp_A]:
            res[tmp_symb]['A'][tmp_A]['probability'] = []

        # sort emission lines by probability
        tmp_prob_sort = list(reversed(sorted(res[tmp_symb]['A'][tmp_A]['probability'])))
        ind = 0
        for p_sort in tmp_prob_sort:
            if tmp_prob <= p_sort:
                ind += 1
            else:
                break

        # write to result
        res[tmp_symb]['A'][tmp_A]['probability'].insert(ind, tmp_prob)
        res[tmp_symb]['A'][tmp_A]['lines'].insert(ind, tmp_e)

    # sort by atomic number Z
    res_sort = OrderedDict()

    # add meta data
    meta_data = 'Automated gamma table created on {}. '.format(time.asctime())
    meta_data += 'Contains gammas with energies between {} keV and {} keV with half life greater than {} days.' \
                 'The {} most prominent lines with probabilities above {} % are included.'.format(e_min, e_max,
                                                                                                  half_life, n_lines,
                                                                                                  prob_lim * 100)
    res_sort['meta_data'] = meta_data
    res_sort['isotopes'] = OrderedDict()

    # loop over all existing Z
    for j in range(1, 118):
        # loop over symbols
        for sym in res:
            if res[sym]['Z'] == j:
                res_sort['isotopes'][sym] = res[sym]

    if outfile is not None:
        with open(outfile, 'w') as out:
            yaml.dump(res_sort, out, default_flow_style=False)

    return res_sort
