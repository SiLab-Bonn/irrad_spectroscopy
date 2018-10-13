#######################################################################
# Collection of functions for general gamma spectroscopy, written to  #
# analyse spectrum of samples irradiated at the HISKP Bonn cyclotron. #
# Main function is fit_spectrum which identifies peaks in given data. #
#######################################################################

# Imports
import logging
import inspect
import warnings
import numpy as np
from collections import OrderedDict, Iterable
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import quad
from scipy.special import erfc


# set logging level when doing import
logging.getLogger().setLevel(logging.INFO)

# ignore scipy throwing warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


### Functions

### general fit functions

def gauss(x, mu, sigma, h):
    return h * np.exp(-0.5 * np.power((x - mu) / sigma, 2.))


def gauss_general(x, mu, sigma, n, h):
    return h * np.exp(-np.power((x - mu) / sigma, n))


def gaux(x, mu, sigma, tau, h):
    """
    Convolution of Gauss and exponential under assumption of small tau (e.g. small exponential contribution)
    https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    """

    term1 = np.exp(-0.5 * np.power((x - mu) / sigma, 2.))
    term2 = 1 - (x - mu) * tau / np.power(sigma, 2.)
    return h * term1 * term2


def lin(x, a, b):
    return a * x + b

### spectral background model fit functions


def plateau_linear(x, a, b, mu, sigma, h):
    return lin(x, a, b) + gauss_general(x, mu, sigma, 4., h)  # flat gauss + straight line


def gauss_linear(x, a, b, mu, sigma, h):
    return lin(x, a, b) + gauss(x, mu, sigma, h)  # gauss + straight line


def gauss_exp_tail(x, mu, sigma, _lambda, h):
    amplitude = h * _lambda / 2.
    term_1 = np.exp(_lambda / 2. * (2. * mu + _lambda * np.power(sigma, 2.) - 2. * x))
    term_2 = erfc((mu + _lambda * np.power(sigma, 2.) - x) / (np.sqrt(2.) * sigma))
    return amplitude * term_1 * term_2


def modified_gauss_exp_tail(x, mu, sigma, _lambda, h, mu_2, sigma_2, h_2):
    return gauss_exp_tail(x, mu, sigma, _lambda, h) + gauss(x, mu_2, sigma_2, h_2)


def get_background_mask(spectrum, low_lim=0.0, high_lim=1e-3):

    # get abs slopes; n - 1 array; slopes from i to i + 1
    slopes = np.abs(spectrum[:-1] - spectrum[1:])

    # get max values
    max_slope = np.max(slopes)

    # bool mask
    mask = np.logical_and(low_lim * max_slope <= slopes, slopes <= high_lim * max_slope)

    # Add final zero in order to correct shape
    mask = np.append(mask, np.array([0], dtype=np.bool))

    return mask

### physics


def get_activity(n0, half_life, t_0, t_1):
    return n0 * np.exp(-(np.log(2)/half_life) * (t_1 - t_0))


def get_release_time(a, a_limit, half_life):
    return -np.log(a_limit / a) / (np.log(2.) / half_life)

### calibrations


def do_energy_calibration(observed_peaks, peak_energies, cal_func=lin):
    """
    Method to do a calibration between expected and obeserved values.

    Parameters
    ----------

    observed_peaks : dict
        dict of observed peaks; return value of fit_spectrum
    peak_energies : dict
        dict of corresponing peaks; same keys as observed_peaks, values are the corresponding values to same key in observed_peaks

    Returns
    -------
    energy_calibration : func
        function describing calibration
    popt : np.array
        array with optimized parameters from curve_fit
    perr : np.array
        array with errors on values in popt
    """

    cal = {}

    # make arrays for fitting
    x_calib, y_calib = np.zeros(shape=len(observed_peaks)), np.zeros(shape=len(observed_peaks))

    # fill calibration arrays
    for i, peak in enumerate(observed_peaks):
        x_calib[i] = observed_peaks[peak]['peak_fit']['popt'][0]
        y_calib[i] = peak_energies[peak]

    # do fit and calculate error
    popt, pcov = curve_fit(cal_func, x_calib, y_calib, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    msg = 'Parameters: '+ ', '.join('%.3f' % par for par in popt), 'Errors: ' + ', '.join('%.3f' % err for err in perr)
    logging.info(msg)

    # define energy calibration
    def energy_calibration(x):
        return lin(x, *popt)

    cal['func'] = energy_calibration
    cal['popt'] = popt
    cal['perr'] = perr
    cal['accuracy'] = 5 * np.std(y_calib / energy_calibration(x_calib))  # include 99.9999 % of the data
    cal['type'] = 'energy'

    return cal


def do_efficiency_calibration(observed_peaks, source_specs, cal_func=lin):
    """
    Method to do a calibration between expected and obeserved values.

    Parameters
    ----------

    observed_peaks : dict
        dict of observed peaks; return value of fit_spectrum
    source_specs : dict
        dict of corresponing calibrated source specs; same keys as observed_peaks, values are the corresponding values to same key in observed_peaks

    Returns
    -------
    efficiency_calibration : func
        function describing calibration
    popt : np.array
        array with optimized parameters from curve_fit
    perr : np.array
        array with errors on values in popt
    """

    cal = {}

    # make arrays for fitting
    x_calib, y_calib, y_error = np.zeros(shape=len(observed_peaks)), np.zeros(shape=len(observed_peaks)), np.zeros(shape=len(observed_peaks))

    # get activity of at the day of measurement
    activity_now = get_activity(n0=source_specs['activity'][0],
                                half_life=source_specs['half_life'],
                                t_0=source_specs['timestamp_calibration'],
                                t_1=source_specs['timestamp_measurement'])

    # get activity uncertainty
    activity_error_now = get_activity(n0=source_specs['activity'][1],
                                      half_life=source_specs['half_life'],
                                      t_0=source_specs['timestamp_calibration'],
                                      t_1=source_specs['timestamp_measurement'])

    # dict with energies as keys and prob as values
    energy_probs = dict(('%i_%s_%i' % (source_specs['A'], source_specs['symbol'], i) , l) for i, l in enumerate(source_specs['probability']))

    # fill calibration arrays
    for i, peak in enumerate(observed_peaks):
        activity_meas = observed_peaks[peak]['activity']['nominal'] / source_specs['t_measurement']
        activity_theo = energy_probs[peak] * activity_now
        del_activity_meas = observed_peaks[peak]['activity']['sigma'] / source_specs['t_measurement']
        del_activity_theo = energy_probs[peak] * activity_error_now
        x_calib[i] = observed_peaks[peak]['peak_fit']['popt'][0]
        y_calib[i] = activity_theo / activity_meas
        y_error[i] = np.sqrt((del_activity_theo / activity_meas)**2.0 + ((activity_theo * del_activity_meas) / activity_meas**2.0)**2.0)

    popt, pcov = curve_fit(cal_func, x_calib, y_calib, sigma=y_error, absolute_sigma=True, maxfev=5000)
    perr = np.sqrt(np.diag(pcov))

    msg = 'Parameters: '+ ', '.join('%.3f' % par for par in popt), 'Errors: ' + ', '.join('%.3f' % err for err in perr)
    logging.info(msg)

    # define efficiency calibration
    def efficiency_calibration(x):
        return lin(x, *popt)

    cal['func'] = efficiency_calibration
    cal['popt'] = popt
    cal['perr'] = perr
    cal['accuracy'] = 5 * np.std(y_calib / efficiency_calibration(x_calib))  # include 99.9999 % of the data
    cal['type'] = 'efficiency'

    return cal

### fitting


def fit_background(x, y, model=None, low_lim=0.0, high_lim=1e-3, p0=None, calibration=None):
    """
    Method to identify the background of a spectrum by looking at absolute values of the slopes of spectrum.
    According to the slopes between low_lim * max(slope) and high_lim * max(slope) the spectrum is masked with an array
    and this masked part then fitted to a function describing the background (model). If a calibration is provided,
    the channels are translated into energies beforehand.

    Parameters
    ----------

    x : array
        array of channels
    y : array
        array of counts
    model : func
        function describing background. If None try all models and see which has lowest red. chi2
    low_lim / high_lim : float
        floats of fraction of max. slope to mask background in
    p0 : iterable
        starting parameters for background model fit. If None take predefined values
    calibration : func
        function that translates channels to energy

    Returns
    -------

    background_model : func
        function describing background
    background mask : np.array
        masked array of background points in spectrum
    """

    # make tmp variables of spectrum to avoid altering input
    _x, _y = x, y

    # apply calibration if not None
    _x = _x if calibration is None else calibration(_x)

    # get background mask; see function definition
    background_mask = get_background_mask(_y, low_lim=low_lim, high_lim=high_lim)

    # mask spectrum
    y_background = _y[background_mask]
    x_background = _x[background_mask]

    # get background peak position
    x_peak_bkg = x_background[np.where(y_background == np.max(y_background))[0][0]]

    # make fixed start parameters for each background model; typical values FIXME: find non-static solution
    _P0 = {gauss_linear: (3e-03,  4., x_peak_bkg, 1.5e2, 1e1),
           gauss_exp_tail: (x_peak_bkg, 1e1, 7e-3, 3e5),
           modified_gauss_exp_tail: (x_peak_bkg, 8e1, 1e-3, 2.5e6, x_peak_bkg*10, 1.5e3, 1e2),
           plateau_linear: (3e-03,  4., x_peak_bkg, 1.5e2, 1e1)}

    # no background model provided, try fitting all pre-defined models and select best fit
    if model is None:
        # logging to user
        models = ', '.join([str(model.__name__) for model in _P0])
        logging.info('No background model provided. Finding best model from %s. This may take a moment.' % models)

        # dicts for optimized parameters of each model and relative errors
        popts = {}
        rel_errs = {}

        # loop over models and fit
        for fit_model in _P0:
            try:
                # do fit; allow high number of maxfev since we come from staic p0
                bkg_opt, bkg_cov = curve_fit(fit_model, x_background, y_background, p0=_P0[fit_model], maxfev=50000, absolute_sigma=True)
                bkg_err = np.sqrt(np.diag(bkg_cov))
            except RuntimeError:
                continue

            # if fit reulted in non-nan errors
            if not any(np.isnan(bkg_err)):
                popts[fit_model] = bkg_opt
                rel_errs[fit_model] = np.sum(np.abs(bkg_err / bkg_opt))

        # select model with smallest sum of relative errors on fit parameters
        best_fit = min(rel_errs, key=rel_errs.get)

        # logging to user
        logging.info('Selected %s as best model for given background.' % str(best_fit.__name__))

        # define model function describing background
        def best_model(x):
            return best_fit(x, *popts[best_fit])

    # model is provided
    else:
        # if no start parameters are provided but model is from pre-defined models
        _p0 = p0 if p0 is not None else _P0[model] if model in _P0 else None

        # do fit; if fitting fails user gets RuntimeError; select better model or starting parameters
        bkg_opt, bkg_cov = curve_fit(model, x_background, y_background, p0=_p0, maxfev=10000, absolute_sigma=True)
        bkg_err = np.sqrt(np.diag(bkg_cov))

        # define model function describing background
        def best_model(x):
            return model(x, *bkg_opt)

    return best_model, background_mask


def fit_spectrum(x, y, background=None, local_background=True, n_peaks=None, channel_sigma=5, energy_cal=None, efficiency_cal=None, t_spectrum=None, expected_peaks=None, expected_accuracy=5e-3, peak_fit=gauss, energy_range=None, reliable_only=True, full_output=True):
    """
    Method that identifies the first n_peaks peaks in a spectrum. They are identified in descending order from highest
    to lowest peak. A Gaussian is fitted to each peak within a fit region of +- channel_sigma of its peak center.
    If a calibration is provided, the channels are translated into energies. The integral of each peak within its fitting
    range is calculated as well as the error. If a dict of expected peaks is given, only these peaks are looked for
    within a accuracy of expected_accuracy.

    Parameters
    ----------

    x : np.array
        array of channels
    y : np.array
        array of counts
    background : func
        function describing background; if local_background is True, background is only needed to find peaks and not for activity calculation
    local_background: bool
        if True, every peaks background will be determined by a linear fit right below the peak
    n_peaks : int
        number of peaks to identify
    channel_sigma : int
        defines fit region around center of peak in channels
    energy_cal : func
        calibration function that translates channels to energy
    efficiency_cal : func
        calibration function that scales activity as function of energy (compensation of detector inefficiencies)
    t_spectrum : float
        integrated time of measured spectrum y in seconds
    expected_peaks : dict
        dict of expected peaks with names (e.g. 40K_1 for 1 kalium peak) as keys and values either in channels or energies
    expected_accuracy : float from 0 to 1
        accuracy with which the expected peak has to be found
    peak_fit : func
        function to fit peaks in spectrum with. This is only fitted to peaks
    energy_range : iterable or iterable of iterables
        iterable (of iterables) with two values (per element) one for lower and upper limit. Is only applied when energy calibration is provided.
    reliable_only: bool
        whether to accept peaks with unreliable fits
    full_output : bool
        whether to return dict with all info

    Returns
    -------

    peaks : dict
        dictionary with fit parameters of gauss as well as errors and integrated counts (signal) of each peak with error
    """

    # define tmp fit function of peak: either just gauss or gauss plus background
    def tmp_fit(x, *args):
        if background is not None:
            return peak_fit(x, *args) + background(x) if not local_background else peak_fit(x, *args)
        else:
            return peak_fit(x, *args)

    # make tmp variables of spectrum to avoid altering input
    _x, _y = x, y

    # peak counter
    counter = 0

    # counter to break fitting when runtime is too large
    runtime_counter = 0

    # maximum peaks in spectrum that are tried to be foundd
    _MAX_PEAKS = 100

    # result dict
    peaks = OrderedDict()

    # boolean masks
    # masking regions due to failing general conditions (peak_mask)
    # masking successfully fitted regions (peak_mask_fitted)
    peak_mask, peak_mask_fitted = np.ones_like(y, dtype=np.bool), np.ones_like(y, dtype=np.bool)

    # flag whether expected peaks have been checked
    checked_expected = False

    # calibrate channels if a calibration is given
    _x = _x if energy_cal is None else energy_cal(_x)

    # correct y by background to find peaks
    y_find_peaks = _y if background is None else _y - background(_x)

    # make tmp variable for n_peaks in order to not change input
    if n_peaks is None and expected_peaks is None:
        raise ValueError('Either n_peaks or expected_peaks has to be given!')
    else:
        # expected peaks are checked first
        tmp_n_peaks = n_peaks if expected_peaks is None else len(expected_peaks)

        # total peaks which are checked
        total_n_peaks = n_peaks if expected_peaks is None else len(expected_peaks) if n_peaks is None else n_peaks + len(expected_peaks)

    # make channel sigma iterable if not already iterable
    if not isinstance(channel_sigma, Iterable):
        channel_sigma = [channel_sigma] * total_n_peaks
    else:
        if len(channel_sigma) != total_n_peaks:
            raise IndexError('Not enough channel sigmas for number of peaks')

    # check whether energy range cut should be applied for finding peaks
    if energy_range:
        _e_msg = None
        if energy_cal is None:
            _e_msg = 'No energy calibration provided. Setting an energy range has no effect.'
        else:
            # one range specified
            if isinstance(energy_range, Iterable) and not isinstance(energy_range[0], Iterable):
                if len(energy_range) == 2:
                    _e_range_mask = (_x <= energy_range[0]) | (_x >= energy_range[1])
                    peak_mask[_e_range_mask] = peak_mask_fitted[_e_range_mask] = False
                else:
                    _e_msg = 'Energy range {} must contain two elements of lower/upper limit.'.format(energy_range)
            # multiple energy sections specified; invert bool masks and set all specified ranges True
            elif isinstance(energy_range, Iterable) and isinstance(energy_range[0], Iterable):
                if all(len(er) == 2 for er in energy_range):
                    peak_mask, peak_mask_fitted = ~peak_mask, ~peak_mask_fitted
                    for e_section in energy_range:
                        _e_sec_mask = (e_section[0] <= _x) & (_x <= e_section[1])
                        peak_mask[_e_sec_mask] = peak_mask_fitted[_e_sec_mask] = True
                else:
                    _e_msg = 'Each element of {} must contain two elements of lower/upper limit.'.format(energy_range)
            else:
                _e_msg = 'Energy range {} must be an iterable of size two. No range set.'.format(energy_range)
        if _e_msg:
            logging.warning(_e_msg)

    logging.info('Start fitting...')

    # loop over tmp_n_peaks
    while counter < tmp_n_peaks:

        # try to find the expected peaks by going through _MAX_PEAKS peaks in spectrum
        if runtime_counter > _MAX_PEAKS:
            runtime_msg = 'Not all peaks could be found! '
            if expected_peaks is not None:
                missing_peaks = [str(p) for p in expected_peaks if p in expected_peaks and p not in peaks]
                runtime_msg += ', '.join(missing_peaks) + ' missing!'
                logging.warning(runtime_msg)

                # check expected peaks first; then reset and look for another n_peaks
                if n_peaks is not None:
                    msg='Finding additional %i peaks ...' % n_peaks
                    counter = 0
                    runtime_counter = 0
                    tmp_n_peaks = n_peaks
                    checked_expected = True
                    peak_mask = peak_mask_fitted
                    logging.info(msg)
                    continue
            else:
                logging.warning(runtime_msg)
            break

        runtime_counter += 1

        # get peak aka maximum
        try:
            y_peak = np.max(y_find_peaks[peak_mask])
        # this happens for small energy ranges with all entries being masked after a couple fits
        except ValueError:
            msg = 'All data masked. '
            if expected_peaks is not None:
                missing_peaks = [str(p) for p in expected_peaks if p in expected_peaks and p not in peaks]
                if missing_peaks:
                    msg += ', '.join(missing_peaks) + ' missing! '
            if n_peaks is not None:
                msg += '' if counter == tmp_n_peaks else '{} of {} peaks found.'.format(counter, tmp_n_peaks)
            logging.info(msg)
            break

        # get corresponding channel number
        x_peak = np.where(y_peak == y_find_peaks)[0][0]

        # make fit environment; fit around x_peak +- some channel_sigma
        # if we're fitting channels
        if energy_cal is None:
            low = x_peak - channel_sigma[counter] if x_peak - channel_sigma[counter] > 0 else 0
            high = x_peak + channel_sigma[counter] if x_peak + channel_sigma[counter] < len(_x) else len(_x) - 1

        # if we're fitting already calibrated channels
        else:
            tmp_peak = np.where(_x == energy_cal(x_peak))[0][0]
            low = tmp_peak - channel_sigma[counter] if tmp_peak - channel_sigma[counter] > 0 else 0
            high = tmp_peak + channel_sigma[counter] if tmp_peak + channel_sigma[counter] < len(_x) else len(_x) - 1

        # make fit regions in x and y; a little confusing to look at but we need the double indexing to
        # obtain the same shapes
        x_fit, y_fit = _x[low:high][peak_mask[low:high]], _y[low:high][peak_mask[low:high]]

        # check whether we have enough points to fit to
        if len(x_fit) < 5:  # skip less than 5 data points
            logging.debug('Only %i data points in fit region. Skipping' % len(x_fit))
            peak_mask[low:high] = False
            continue

        # start fitting
        try:  # get fit parameters and errors
            # starting parameters
            _mu = x_peak if energy_cal is None else energy_cal(x_peak)
            _sigma = np.abs(x_fit[y_fit >= y_peak / 2.0][-1] - x_fit[y_fit >= y_peak / 2.0][0]) / 2.3548
            _p0 = {'mu': _mu, 'sigma': _sigma, 'h': y_peak}
            fit_args = inspect.getargspec(peak_fit)[0][1:]
            p0 = tuple(_p0[arg] if arg in _p0 else 1 for arg in fit_args)
            popt, pcov = curve_fit(tmp_fit, x_fit, y_fit, p0=p0, sigma=np.sqrt(y_fit), absolute_sigma=True, maxfev=5000)
            perr = np.sqrt(np.diag(pcov))  # get std deviation

            # update
            _mu, _sigma = [popt[fit_args.index(par)] for par in ('mu', 'sigma')]

            # if fitting resulted in nan errors
            if any(np.isnan(perr)):
                peak_mask[low:high] = False
                continue

            if any(_mu == peaks[p]['peak_fit']['popt'][0] for p in peaks):
                logging.debug('Peak at %.2f already fitted. Skipping' % _mu)
                peak_mask[low:high] = False
                continue

            # if fit is unreliable
            if any(np.abs(perr / popt) > 1.0):
                if not reliable_only:
                    logging.warning('Unreliable fit for peak at %.2f. Uncertainties larger than 100 percent.' % _mu)
                else:
                    logging.debug('Skipping fit for peak at %.2f. Uncertainties larger than 100 percent.' % _mu)
                    peak_mask[low:high] = False
                    continue

        # fitting did not succeed
        except RuntimeError:  # disable failed region for next iteration
            logging.debug('Fitting failed. Skipping peak at %.2f' % _mu)
            peak_mask[low:high] = False
            continue

        # check if our fitted peak is expected
        if expected_peaks is not None and not checked_expected:

            # make list for potential candidates
            candidates = []

            # loop over all expected peaks and check which check out as expected within the accuracy
            for ep in expected_peaks:

                # get upper and lower estimates
                lower_est, upper_est = [(1 + sgn * expected_accuracy) * expected_peaks[ep] for sgn in (-1, 1)]

                # if current peak checks out set peak name and break
                if lower_est <= _mu <= upper_est:
                    candidates.append(ep)

            # if no candidates are found, current peak was not expected
            if not candidates:
                logging.debug('Peak at %.2f not expected. Skipping' % _mu)
                peak_mask[low:high] = False
                continue

            # if all candidates are already found
            if all(c in peaks for c in candidates):
                logging.debug('Peak at %.2f already fitted. Skipping' % _mu)
                peak_mask[x_peak] = False
                continue
        else:
            candidates = ['peak_%i' % counter]

        ### FROM HERE ON THE FITTED PEAK WILL BE IN THE RESULT DICT ###

        # get integration limits within 3 sigma for non local background
        low_lim, high_lim = _mu - 3 * _sigma, _mu + 3 * _sigma # integrate within 3 sigma

        # get background via integration of previously fitted background model
        if not local_background and background is not None:
            bkg, bkg_err = quad(background, low_lim, high_lim) # background integration

        # get local background and update limits
        else:
            # find local background bounds; start looking at bkg from (+-3 to +-6) sigma
            # increase bkg to left/right of peak until to avoid influence of nearby peaks
            _i_dev = 6
            _deviation = None
            while _i_dev < int(_MAX_PEAKS / 2):
                # Make tmp array of mean bkg values left and right of peak
                _tmp_dev_array = [np.mean(_y[(_mu - _i_dev * _sigma <= _x) & (_x <= _mu - 3 * _sigma)]),
                                  np.mean(_y[(_mu + 3 * _sigma <= _x) & (_x <= _mu + _i_dev * _sigma)])]
                # look at std. deviation; as long as it decreases for increasing bkg area update
                if np.std(_tmp_dev_array) < _deviation or _deviation is None:
                    _deviation = np.std(_tmp_dev_array)
                # if std. deviation increases again, break
                elif np.std(_tmp_dev_array) >= _deviation:
                    _i_dev -= 1
                    break
                # increment
                _i_dev += 1

            # get background from 3 to _i_dev sigma left of peak
            lower_bkg = np.logical_and(_mu - _i_dev * _sigma <= _x, _x <= _mu - 3 * _sigma)
            # get background from 3 to _i_dev sigma right of peak
            upper_bkg = np.logical_and(_mu + 3 * _sigma <= _x, _x <= _mu + _i_dev * _sigma)
            # combine bool mask
            bkg_mask = np.logical_or(lower_bkg, upper_bkg)
            # mask other peaks in bkg so local background fit will not be influenced by nearby peak
            bkg_mask[~peak_mask] = False
            # do fit
            bkg_opt, bkg_cov = curve_fit(lin, _x[bkg_mask], _y[bkg_mask])
            # _x values of current peak
            _peak_x = _x[(low_lim <= _x) & (_x <= high_lim)]
            # estimate intersections of background and peak from data
            x0_low = _peak_x[np.where(_peak_x >= np.mean(_x[lower_bkg]))[0][0]]
            x0_high = _peak_x[np.where(_peak_x <= np.mean(_x[upper_bkg]))[0][-1]]
            # find intersections of line and gauss; should be in 3-sigma environment since background is not 0
            # increase environment to 5 sigma to be sure
            low_lim, high_lim = _mu - 5 * _sigma, _mu + 5 * _sigma
            # fsolve heavily relies on correct start parameters; estimate from data and loop
            try:
                _i_tries = 0
                found = False
                while _i_tries < _MAX_PEAKS:
                    diff = np.abs(high_lim - low_lim) / _MAX_PEAKS * _i_tries

                    _x0_low = low_lim + diff / 2.
                    _x0_high = high_lim - diff / 2.

                    # find intersections; needs to be sorted since sometimes higher intersection is found first
                    _low, _high = sorted(fsolve(lambda k: tmp_fit(k, *popt) - lin(k, *bkg_opt), x0=[_x0_low, _x0_high]))

                    # if intersections have been found
                    if not np.isclose(_low, _high) and np.abs(_high - _low) <= 7 * _sigma:
                        low_lim, high_lim = _low, _high
                        found = True
                        break

                    # increment
                    _i_tries += 1

                # raise error
                if not found:
                    raise ValueError

            except (TypeError, ValueError):
                msg = 'Intersections between peak and local background for peak(s) %s could not be found. ' \
                      'Use estimates from data instead.' % ', '.join(candidates)
                logging.info(msg)
                low_lim, high_lim = x0_low, x0_high

            # do background integration
            bkg, bkg_err = quad(lin, low_lim, high_lim, args=tuple(bkg_opt))

        # get counts via integration of fit
        counts, _ = quad(tmp_fit, low_lim, high_lim, args=tuple(popt)) # count integration

        # estimate lower uncertainty limit
        counts_low, _ = quad(tmp_fit, low_lim, high_lim, args=tuple(popt - perr)) # lower counts limit

        # estimate lower uncertainty limit
        counts_high, _ = quad(tmp_fit, low_lim, high_lim, args=tuple(popt + perr)) # lower counts limit

        low_count_err, high_count_err = np.abs(counts - counts_low), np.abs(counts_high - counts)

        max_count_err = high_count_err if high_count_err >= low_count_err else low_count_err

        # calc activity and error
        activity, activity_err = counts - bkg, np.sqrt(np.power(max_count_err,2.) + np.power(bkg_err,2.))

        # scale activity to compensate for dectector inefficiency at given energy
        if efficiency_cal is not None:
            activity, activity_err = (efficiency_cal(popt[0]) * x for x in [activity, activity_err])

        # normalize to counts / s == Bq
        if t_spectrum is not None:
            activity, activity_err = activity / t_spectrum, activity_err / t_spectrum

        # write current results to dict for every candidate
        for peak_name in candidates:
            # make entry for current peak
            peaks[peak_name] = OrderedDict()

            # entries for data
            peaks[peak_name]['background'] = OrderedDict()
            peaks[peak_name]['peak_fit'] = OrderedDict()
            peaks[peak_name]['activity'] = OrderedDict()

            # write background to result dict
            peaks[peak_name]['background']['popt'] = bkg_opt.tolist()
            peaks[peak_name]['background']['perr'] = np.sqrt(np.diag(bkg_cov)).tolist()
            peaks[peak_name]['background']['type'] = 'local' if local_background or background is None else 'global'

            # write optimal fit parameters/erros for every peak to result dict
            peaks[peak_name]['peak_fit']['popt'] = popt.tolist()
            peaks[peak_name]['peak_fit']['perr'] = perr.tolist()
            peaks[peak_name]['peak_fit']['int_lims'] = [float(low_lim), float(high_lim)]
            peaks[peak_name]['peak_fit']['type'] = peak_fit.__name__

            # write activity data to output dict
            peaks[peak_name]['activity']['nominal'] = float(activity)
            peaks[peak_name]['activity']['sigma'] = float(activity_err)
            peaks[peak_name]['activity']['type'] = 'integrated' if t_spectrum is None else 'normalized'
            peaks[peak_name]['activity']['unit'] = 'becquerel' if t_spectrum is not None else 'counts / t_spectrum'
            peaks[peak_name]['activity']['calibrated'] = efficiency_cal is not None

            counter += 1  # increase counter

        runtime_counter = 0  # peak(s) were found; reset runtime counter

        # disable fitted region for next iteration
        current_mask = (low_lim <= _x) & (_x <= high_lim)
        peak_mask[current_mask] = peak_mask_fitted[current_mask] = False

        # check whether we have found all expected peaks and there's still n_peaks to look after
        if counter == tmp_n_peaks and expected_peaks is not None:
            msg = 'All %s have been found!' % ('expected peaks' if not checked_expected else 'peaks')
            # expected peaks are all found if they're not None
            if n_peaks is not None and not checked_expected:
                msg += 'Finding additional %i peaks ...' % n_peaks
                counter = 0
                runtime_counter = 0
                tmp_n_peaks = n_peaks
                checked_expected = True
                peak_mask = peak_mask_fitted
            logging.info(msg)

    logging.info('Finished fitting.')

    # remove inof from result dict
    if not full_output:
        for iso in peaks:
            del peaks[iso]['background']
            del peaks[iso]['peak_fit']

    return peaks
