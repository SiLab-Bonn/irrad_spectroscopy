# Imports
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from irrad_spectroscopy.spectroscopy import lin, gauss

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

    
def plot_calibration(x, popt, perr, axis=None, calib_type=None, func=lin):
    
    # define calibration
    def calib(x):
        return func(x, *popt)
        
    y_labels = {'energy': 'Energy / keV', 'efficiency': r'Ratio: $A_\mathrm{theo}\ /\ A_\mathrm{meas}$'}
        
    # plot in same graph
    if axis is None:
        ax1 = plt.gca() 
    else:
        ax = axis
        ax1 = ax.twinx()
    
    ax1.plot(x, calib(x), c='g', marker='x', lw=1, ls='None', label='Calibration data')
    calib_label = r'Linear fit: $f(x)=ax+b$' \
                  + '\n\t' + r'$a=(%.2E \pm %.2E)\ $ keV/channel' % (popt[0], perr[0]) \
                  + '\n\t' +  r'$b=(%.2f \pm %.2f)\ $ keV' % (popt[1], perr[1]) if func is lin else 'Calibration fit'
    ax1.plot(x, calib(x), c='g', lw=1, ls='--', label=calib_label)
    ax1.set_ylabel(y_labels[calib_type] if calib_type in y_labels else '')
    ax1.set_ylim(0, np.max(calib(x))*1.5)
    ax1.tick_params('y', colors='g')
    h1, l1 = ax1.get_legend_handles_labels()
    
    if axis is not None:
        h, l = ax.get_legend_handles_labels()
        h += h1
        l += l1
        ax.legend(h ,l, loc='upper center')
        align_yaxis(ax, 0, ax1, 0)
    else:
        ax1.legend(h1 ,l1, loc='upper center')
        ax1.set_xlabel('Channel')
        ax1.set_title('%s calibration to background spectrum' % str(calib_type).capitalize())
        ax1.grid()
    
    
def plot_spectrum(counts, channels=None, peaks=None, bkg=None, plot_calib=False, energy_cal=None, title=None, output_plot=None, peak_fit=gauss):

    # some sanity checks for input data
    # check if input is np.array
    try:
        _ = counts.shape
        _cnts = counts[:]
    except AttributeError:
        _cnts = np.array(counts)

    # check for correct shape
    if len(_cnts.shape) != 1:
        raise ValueError('Counts must be 1-dimensional array')

    _chnnls = np.arange(_cnts.shape[0]) if channels is None else channels[:]

    # calibrate channels if a calibration is given
    _chnnls = _chnnls if energy_cal is None else energy_cal(_chnnls)
    
    x_label = 'Channel' if energy_cal is None else 'Energy / keV'
    y_label = 'Counts'
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title if title is not None else '')
    plt.grid()
    
    # plot spectrum first
    plt.errorbar(_chnnls, _cnts, yerr=np.sqrt(_cnts), marker='.',lw=1, ls='None', label='Spectrum')
    
    ### plot rest

    # plot background
    if bkg is not None:
        plt.plot(_chnnls, bkg(_chnnls), c='y',lw=1, ls='--', zorder=7, label='Global background')
    
    # plot fitted peaks
    if peaks is not None and (bkg is not None or all('local' == peaks[p]['background']['type'] for p in peaks)):
        
        local_flag = False   
        if all('local'== peaks[p]['background']['type'] for p in peaks):
            plt.plot([], [], ls='--', lw=1, c='k', label='Local background fits')
            local_flag = True

        # define tmp fit function of peak plus background or just gauss if local backgrounds are selected
        def tmp_fit(x, *args):
            return peak_fit(x, *args) + bkg(x) if not local_flag else peak_fit(x, *args)
        
        plt.plot([], [], lw=1, ls='--', zorder=10, c='r', label='Peak fits')

        # store multiple mu positions in order to stack labels on plot
        _y_texts = {}

        # loop over fitted peaks
        for p in peaks:
            
            # get integration limits
            low_lim, high_lim = peaks[p]['peak_fit']['int_lims']
            
            # make x values for plotting
            _tmp_x = np.linspace(low_lim, high_lim, 100)

            # add peak fitting label
            plt.plot(_tmp_x, tmp_fit(_tmp_x, *peaks[p]['peak_fit']['popt']), lw=1, ls='--', zorder=10, c='r')
            
            # plot signal and background
            if local_flag:
                plt.plot(_tmp_x, lin(_tmp_x, *peaks[p]['background']['popt']), lw=1, ls='--', zorder=10, c='k')
                plt.fill_between(_tmp_x, tmp_fit(_tmp_x, *peaks[p]['peak_fit']['popt']), lin(_tmp_x, *peaks[p]['background']['popt']), color='r', alpha=0.3)
                plt.fill_between(_tmp_x, lin(_tmp_x, *peaks[p]['background']['popt']), np.zeros_like(_tmp_x), color='k', alpha=0.3)
            else:
                plt.fill_between(_tmp_x, tmp_fit(_tmp_x, *peaks[p]['peak_fit']['popt']), bkg(_tmp_x), color='r', alpha=0.3)
                plt.fill_between(_tmp_x, bkg(_tmp_x), np.zeros_like(_tmp_x), color='k', alpha=0.3)
            
            # set text in plot
            _mu = peaks[p]['peak_fit']['popt'][0]
            _h = peaks[p]['peak_fit']['popt'][-1]

            if _mu not in _y_texts:
                text_flag = False
            else:
                text_flag = True

            text = str(p) + ': %.2f +- %.2f' % (_mu, peaks[p]['peak_fit']['perr'][0]) if 'peak' in str(p) else str(p)

            if text_flag:
                prev_text = _y_texts[_mu].get_text()
                new_text = prev_text + ', ' + text
                _y_texts[_mu].set_text(new_text)
                _y_texts[_mu].set_fontweight('semibold')
            else:
                y_text = (_h + bkg(_mu)) * 1.05 if not local_flag else _h * 1.05
                _y_texts[_mu] = plt.text(_mu, y_text, text, fontsize=8, bbox=dict(edgecolor='k', facecolor='none'))
        
        # plot also calibration
        if plot_calib:
            # make legend entry for local background fits
            plot_calibration(x=np.array([peaks[pp]['peak_fit']['popt'][0] for pp in peaks]),
                             popt=plot_calib['popt'],
                             perr=plot_calib['perr'],
                             axis=plt.gca(),
                             calib_type=plot_calib['type'])
    
    # if not calib handle legend and grid; if plot_calib, legend is handeled in plot_calibration
    if not plot_calib:
        plt.legend(loc='upper center')
    
    # show
    plt.show()
    
    if output_plot is not None:
        plt.savefig(output_plot, bbox_inches='tight')

