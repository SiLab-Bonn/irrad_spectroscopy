import sys
import os
import logging
import yaml
import numpy as np
import unittest
import irrad_spectroscopy

from irrad_spectroscopy.spectroscopy import fit_spectrum, fit_background, do_energy_calibration, do_efficiency_calibration, get_activity
from irrad_spectroscopy.utils.utils import get_measurement_time, isotopes_to_dict, calc_activity, source_to_dict

# Get the absolute path of the irrad_spectroscopy installation
package_path = os.path.dirname(irrad_spectroscopy.__file__)

testing_path = os.path.dirname(os.path.abspath(__file__))

test_data_path = os.path.join(testing_path, 'test_data')


class TestSpectroscopy(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        # detector channels
        cls.n_channels = 8192
        
        # load spectra; all spectra have been recorded with the same detector setup
        cls.sample_spectrum = np.loadtxt(os.path.join(test_data_path, 'example_sample.txt'), unpack=True)
        cls.Ba133_spectrum = np.loadtxt(os.path.join(test_data_path, 'Ba133_point.txt'), unpack=True)
        cls.Na22_spectrum = np.loadtxt(os.path.join(test_data_path, 'Na22_point.txt'), unpack=True)
        cls.Eu152_spectrum = np.loadtxt(os.path.join(test_data_path, 'Eu152_point.txt'), unpack=True)
        
        # calibration related
        with open(os.path.join(test_data_path, '152Eu_point_source_2.yaml'), 'r') as source_specs:
            cls.Eu152_source_specs = yaml.safe_load(source_specs) 
        
        with open(os.path.join(test_data_path, '152Eu_peaks.yaml'), 'r') as energy_peaks:
            cls.energy_calib_peaks = yaml.safe_load(energy_peaks)
            
        # library yaml
        with open(os.path.join(package_path, 'isotope_lib.yaml'), 'r') as iso_lib:
            cls.isotope_lib = yaml.safe_load(iso_lib)
        
        # benchmark source specs of Na22 and Ba133
        with open(os.path.join(test_data_path, '22Na_point_source.yaml'), 'r') as na22:
            cls.Na22_source_specs = yaml.safe_load(na22)
        
        with open(os.path.join(test_data_path, '133Ba_point_source.yaml'), 'r') as ba133:
            cls.Ba133_source_specs = yaml.safe_load(ba133)
            
        # times
        cls.t_sample = get_measurement_time(os.path.join(test_data_path, 'example_sample.mcd'))
        cls.t_Na22 = get_measurement_time(os.path.join(test_data_path, 'Na22_point.mcd'))
        cls.t_Ba133 = get_measurement_time(os.path.join(test_data_path, 'Ba133_point.mcd'))
        
        # variables
        cls.energy_calibration = None
        cls.efficiency_calibration = None
                
    @classmethod
    def tearDownClass(cls):
        pass
    
    def test_data(self):
        """Test some random things in the loaded data"""
        
        # integrated measurement time of example sample
        self.assertEqual(self.t_sample, 67497.603)
        
        # correct shape
        self.assertEqual(self.sample_spectrum.shape[1], self.n_channels)
        
        # half lifes of all sources in seconds
        self.assertEqual(self.Eu152_source_specs['half_life'], 427075200.0)
        self.assertEqual(self.Ba133_source_specs['half_life'], 331443360.0)
        self.assertEqual(self.Na22_source_specs['half_life'], 82053518.4)        
        
    def test_energy_calibration(self):
        """Do energy calibration of detector channels to spectrum of 152-Eu source"""
        
        energy_calib_bkg, _ = fit_background(x=self.Eu152_spectrum[0],
                                             y=self.Eu152_spectrum[1])
        
        energy_calib_peaks = fit_spectrum(x=self.Eu152_spectrum[0],
                                          y=self.Eu152_spectrum[1],
                                          expected_peaks=self.energy_calib_peaks['channel'],
                                          background=energy_calib_bkg)
        
        # check whether all peaks have been found
        self.assertListEqual(sorted(energy_calib_peaks.keys()), sorted(self.energy_calib_peaks['channel'].keys()))
        
        self.energy_calibration= do_energy_calibration(observed_peaks=energy_calib_peaks,
                                                       peak_energies=self.energy_calib_peaks['energy'])
        
        # 5 % error on fit parameters
        self.assertTrue(all(self.energy_calibration['perr'] / self.energy_calibration['popt'] <= 0.5))
            
    def test_efficiency_calibration(self):
        """Do efficiency calibration of detector to spectrum of calibrated 152-Eu source"""
        
        # test are run sorted by the string representations of the test methods; we need this to be run first
        if self.energy_calibration is None:
            self.test_energy_calibration()
        
        # generate expected peaks from source specs
        Eu152_expected = source_to_dict(self.Eu152_source_specs, info='lines')
        
        efficiency_calib_bkg, _ = fit_background(x=self.Eu152_spectrum[0],
                                                 y=self.Eu152_spectrum[1])
        
        efficiency_calib_peaks = fit_spectrum(x=self.Eu152_spectrum[0],
                                              y=self.Eu152_spectrum[1],
                                              energy_cal=self.energy_calibration['func'],
                                              expected_accuracy=self.energy_calibration['accuracy'],
                                              expected_peaks=Eu152_expected,
                                              background=efficiency_calib_bkg)
        
        # check whether all peaks have been found
        self.assertListEqual(sorted(efficiency_calib_peaks.keys()), sorted(Eu152_expected.keys()))
        
        self.efficiency_calibration = do_efficiency_calibration(observed_peaks=efficiency_calib_peaks,
                                                                source_specs=self.Eu152_source_specs)
        # 5 % error on fit parameters
        self.assertTrue(all(self.efficiency_calibration['perr'] / self.efficiency_calibration['popt'] <= 0.5))
        
    def test_benchmark(self):
        """
        Do a benchmark of the software with two, calibrated sources; 22-Na and 133-Ba. The expected peaks of both spectra
        are fitted to first benchmark the energy calibration with the peak positions. Then the integrated and scaled activity
        is compared to the theoretically expected activity to benchmark the efficiency calibration as well as the fitting and
        background subtraction.
        Overall, a reconstruction accuracy of above 90 % is achieved for both sources (after approx. 10 minute measurement)
        
        The 22-Na source has one line/peak => 22-Na activity is (93.4 +- 2.7)% accurately measured
        The 133-Ba source has 5 lines/peaks => 133-Ba activity is (97.0 +- 1.2)% accurately measured
        """
        
        # test are run sorted by the string representations of the test methods; we need this to be run first
        if self.energy_calibration is None:
            self.test_energy_calibration()
        
        # test are run sorted by the string representations of the test methods; we need this to be run first
        if self.efficiency_calibration is None:
            self.test_efficiency_calibration()
        
        # test energy calibration by applying to Na22 and Ba133 spectra
        
        # generate expected peaks and probabilities from source specs
        Na22_expected = source_to_dict(self.Na22_source_specs, info='lines')
        Na22_probs = source_to_dict(self.Na22_source_specs, info='probability')
        
        # fit spectrum of source
        Na22_peaks = fit_spectrum(x=self.Na22_spectrum[0],
                                  y=self.Na22_spectrum[1],
                                  energy_cal=self.energy_calibration['func'],
                                  efficiency_cal=self.efficiency_calibration['func'],
                                  t_spectrum=self.t_Na22,
                                  expected_peaks=Na22_expected,
                                  expected_accuracy=self.energy_calibration['accuracy'])
        
        # check whether all peaks have been found
        self.assertListEqual(sorted(Na22_peaks.keys()), sorted(Na22_expected.keys()))
        
        # check whether all activities are in Bq
        self.assertTrue(all('normalized' == Na22_peaks[p]['activity']['type'] for p in Na22_peaks))
        
        # check whether all activities are calibrated for efficiency
        self.assertTrue(all(Na22_peaks[p]['activity']['calibrated'] for p in Na22_peaks))
        
        # check whether all peaks are at correct energies within 1 per mille accuracy
        for na22_peak in Na22_peaks:
            low, high = (x * Na22_expected[na22_peak] for x in (1 - self.energy_calibration['accuracy'], 1 + self.energy_calibration['accuracy']))
            self.assertTrue(low <= Na22_peaks[na22_peak]['peak_fit']['popt'][0] <= high)
            
        # check for correct activity
        Na22_activity_meas = calc_activity(Na22_peaks, Na22_probs)
        Na22_activity_theo = get_activity(n0=np.array(self.Na22_source_specs['activity']),
                                          half_life=self.Na22_source_specs['half_life'],
                                          t_0=self.Na22_source_specs['timestamp_calibration'],
                                          t_1=self.Na22_source_specs['timestamp_measurement'])
        
        # check to see at least 93% of the expected activity; this is the percentage at the moment of writing this test
        self.assertTrue(Na22_activity_meas['22Na']['nominal'] / Na22_activity_theo[0] >= 0.93)
        
        # generate expected peaks and probabilities from source specs
        Ba133_expected = dict(('%i_%s_%i' % (self.Ba133_source_specs['A'],
                                             self.Ba133_source_specs['symbol'], i) , l) for i, l in enumerate(self.Ba133_source_specs['lines']))
        
        Ba133_probs = dict(('%i_%s_%i' % (self.Ba133_source_specs['A'],
                                          self.Ba133_source_specs['symbol'], i) , l) for i, l in enumerate(self.Ba133_source_specs['probability']))
        # fit spectrum of source
        Ba133_peaks = fit_spectrum(x=self.Ba133_spectrum[0],
                                   y=self.Ba133_spectrum[1],
                                   energy_cal=self.energy_calibration['func'],
                                   efficiency_cal=self.efficiency_calibration['func'],
                                   t_spectrum=self.t_Ba133,
                                   expected_peaks=Ba133_expected,
                                   expected_accuracy=self.energy_calibration['accuracy'])
        
        # check whether all peaks ghave been found
        self.assertListEqual(sorted(Ba133_peaks.keys()), sorted(Ba133_expected.keys()))
        
        # check whether all activities are in Bq
        self.assertTrue(all('normalized' == Ba133_peaks[p]['activity']['type'] for p in Ba133_peaks))
        
        # check whether all activities are calibrated for efficiency
        self.assertTrue(all(Ba133_peaks[p]['activity']['calibrated'] for p in Ba133_peaks))
        
        # check whether all peaks are at correct energies within 1 per mill accuracy
        for ba133_peak in Ba133_peaks:
            low, high = (x * Ba133_expected[ba133_peak] for x in (1 - self.energy_calibration['accuracy'], 1 + self.energy_calibration['accuracy']))
            self.assertTrue(low <= Ba133_peaks[ba133_peak]['peak_fit']['popt'][0] <= high)
            
        # check for correct activity
        Ba133_activity_meas = calc_activity(Ba133_peaks, Ba133_probs)
        Ba133_activity_theo = get_activity(n0=np.array(self.Ba133_source_specs['activity']),
                                          half_life=self.Ba133_source_specs['half_life'],
                                          t_0=self.Ba133_source_specs['timestamp_calibration'],
                                          t_1=self.Ba133_source_specs['timestamp_measurement'])
        
        # check to see at least 97% of the expected activity; this is the percentage at the moment of writing this test
        self.assertTrue(Ba133_activity_meas['133Ba']['nominal'] / Ba133_activity_theo[0] >= 0.969)
        
            
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpectroscopy)
    unittest.TextTestRunner(verbosity=2).run(suite)
    