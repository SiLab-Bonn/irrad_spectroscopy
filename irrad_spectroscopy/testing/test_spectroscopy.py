import os
import logging
import yaml
import numpy as np
import unittest
import irrad_spectroscopy.spectroscopy as sp
from irrad_spectroscopy.spec_utils import get_measurement_time, source_to_dict, select_peaks
from irrad_spectroscopy.physics import decay_law
from irrad_spectroscopy import testing_path, gamma_table


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
        cls.gamma_table = gamma_table
        cls.accuracy = 1e-3
                
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
        
        energy_calib_bkg = sp.interpolate_bkg(counts=self.Eu152_spectrum[1])
        
        energy_calib_peaks = sp.fit_spectrum(counts=self.Eu152_spectrum[1],
                                             expected_peaks=self.energy_calib_peaks['channel'],
                                             bkg=energy_calib_bkg)
        
        # check whether all peaks have been found
        self.assertListEqual(sorted(energy_calib_peaks.keys()), sorted(self.energy_calib_peaks['channel'].keys()))
        
        self.energy_calibration= sp.do_energy_calibration(observed_peaks=energy_calib_peaks,
                                                          peak_energies=self.energy_calib_peaks['energy'])
        
        # 50 % error on fit parameters
        self.assertTrue(all(self.energy_calibration['perr'] / self.energy_calibration['popt'] <= 0.5))
            
    def test_efficiency_calibration(self):
        """Do efficiency calibration of detector to spectrum of calibrated 152-Eu source"""
        
        # test are run sorted by the string representations of the test methods; we need this to be run first
        if self.energy_calibration is None:
            self.test_energy_calibration()
        
        # generate expected peaks from source specs
        Eu152_expected = source_to_dict(self.Eu152_source_specs, info='lines')
        
        efficiency_calib_bkg = sp.interpolate_bkg(counts=self.Eu152_spectrum[1])
        
        efficiency_calib_peaks = sp.fit_spectrum(counts=self.Eu152_spectrum[1],
                                                 energy_cal=self.energy_calibration['func'],
                                                 expected_accuracy=self.accuracy,  # self.energy_calibration['accuracy']
                                                 expected_peaks=Eu152_expected,
                                                 bkg=efficiency_calib_bkg)
        
        # check whether all peaks have been found
        self.assertListEqual(sorted(efficiency_calib_peaks.keys()), sorted(Eu152_expected.keys()))
        
        self.efficiency_calibration = sp.do_efficiency_calibration(observed_peaks=efficiency_calib_peaks,
                                                                source_specs=self.Eu152_source_specs)
        # 50 % error on fit parameters
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
        
        # generate expected peaks from source specs
        Na22_expected = source_to_dict(self.Na22_source_specs, info='lines')

        # fit spectrum of source
        Na22_peaks, Na22_bkg = sp.fit_spectrum(counts=self.Na22_spectrum[1],
                                               energy_cal=self.energy_calibration['func'],
                                               efficiency_cal=self.efficiency_calibration['func'],
                                               t_spec=self.t_Na22,
                                               expected_accuracy=self.accuracy)  # self.energy_calibration['accuracy']
        
        # check whether all expected peaks have been found from library
        self.assertTrue(all(ep in Na22_peaks for ep in Na22_expected.keys()))

        # check whether all activities are in Bq
        self.assertTrue(all('normalized' == Na22_peaks[p]['activity']['type'] for p in Na22_peaks))
        
        # check whether all activities are calibrated for efficiency
        self.assertTrue(all(Na22_peaks[p]['activity']['calibrated'] for p in Na22_peaks))
        
        # check whether all peaks are at correct energies within 1 per mille accuracy
        for na22_peak in Na22_peaks:
            if na22_peak in Na22_expected:
                low, high = (x * Na22_expected[na22_peak] for x in (1 - self.accuracy, 1 + self.accuracy))
                self.assertTrue(low <= Na22_peaks[na22_peak]['peak_fit']['popt'][0] <= high)
            
        # check for correct activity from library
        Na22_activity_meas = sp.get_activity(Na22_peaks)
        Na22_activity_theo = decay_law(t=self.Na22_source_specs['timestamp_measurement']-self.Na22_source_specs['timestamp_calibration'],
                                       x0=np.array(self.Na22_source_specs['activity']), half_life=self.Na22_source_specs['half_life'])

        # check to see at least 90% of the expected activity; only order of magnitude relevant
        self.assertTrue(0.9 <= Na22_activity_meas['22_Na']['nominal'] / Na22_activity_theo[0] <= 1.0)

        # generate expected peaks and probabilities from source specs
        Ba133_expected = source_to_dict(self.Ba133_source_specs, info='lines')

        # fit spectrum of source
        Ba133_peaks, Ba133_bkg = sp.fit_spectrum(counts=self.Ba133_spectrum[1],
                                                 energy_cal=self.energy_calibration['func'],
                                                 efficiency_cal=self.efficiency_calibration['func'],
                                                 t_spec=self.t_Ba133,
                                                 expected_accuracy=self.accuracy)  # self.energy_calibration['accuracy']
        
        # check whether all expected peaks have been found from library
        self.assertTrue(all(ep in Ba133_peaks for ep in Ba133_expected.keys()))
        
        # check whether all activities are in Bq
        self.assertTrue(all('normalized' == Ba133_peaks[p]['activity']['type'] for p in Ba133_peaks))
        
        # check whether all activities are calibrated for efficiency
        self.assertTrue(all(Ba133_peaks[p]['activity']['calibrated'] for p in Ba133_peaks))
        
        # check whether all peaks are at correct energies within 1 per mill accuracy
        for ba133_peak in Ba133_peaks:
            if ba133_peak in Ba133_expected:
                low, high = (x * Ba133_expected[ba133_peak] for x in (1 - self.accuracy, 1 + self.accuracy))
                self.assertTrue(low <= Ba133_peaks[ba133_peak]['peak_fit']['popt'][0] <= high, msg=str(self.energy_calibration['accuracy']))
            
        # check for correct activity
        Ba133_activity_meas = sp.get_activity(Ba133_peaks)
        Ba133_activity_theo = decay_law(t=self.Ba133_source_specs['timestamp_measurement']-self.Ba133_source_specs['timestamp_calibration'],
                                        x0=np.array(self.Ba133_source_specs['activity']),
                                        half_life=self.Ba133_source_specs['half_life'])
        
        # check to see at least 90% of the expected activity; only order of magnitude relevant
        self.assertTrue(0.9 <= Ba133_activity_meas['133_Ba']['nominal'] / Ba133_activity_theo[0] <= 1.0)

    def test_dose(self):
        """Testing the dose calculations on the example spectrum"""

        # test are run sorted by the string representations of the test methods; we need this to be run first
        if self.energy_calibration is None:
            self.test_energy_calibration()

        # test are run sorted by the string representations of the test methods; we need this to be run first
        if self.efficiency_calibration is None:
            self.test_efficiency_calibration()

        peaks_sample, bkg_sample = sp.fit_spectrum(counts=self.sample_spectrum[1],
                                                   energy_cal=self.energy_calibration['func'],
                                                   efficiency_cal=self.efficiency_calibration['func'],
                                                   t_spec=self.t_sample)

        dose = sp.get_dose(peaks_sample, distance=50, time=2000)

        self.assertTrue(np.isclose(dose['nominal'], 1.1330667863561383))
        self.assertTrue(np.isclose(dose['sigma'], 0.03766586624214202))
        self.assertTrue(dose['unit']=='uSv')

        dose = sp.get_dose(peaks_sample, distance=1)

        self.assertTrue(np.isclose(dose['nominal'], 1.8437455480798448))
        self.assertTrue(np.isclose(dose['sigma'], 0.05874337083851746))
        self.assertTrue(dose['unit'] == 'uSv/h')

        selected_peaks = select_peaks(['65_Zn', '48_V', '7_Be'], peaks_sample)

        self.assertEqual(len(selected_peaks), 4)

        selected_dose = sp.get_dose(selected_peaks, distance=50, time=2000)
        self.assertTrue(np.isclose(selected_dose['nominal'], 0.5343614796430098))

        selected_dose_time_1 = sp.get_dose(selected_peaks, distance=50, time=1e6)['nominal']
        selected_dose_time_2 = sp.get_dose(selected_peaks, distance=50, time=1e7)['nominal']

        self.assertTrue(np.isclose(selected_dose_time_1,selected_dose_time_2))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSpectroscopy)
    unittest.TextTestRunner(verbosity=2).run(suite)
