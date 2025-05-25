import unittest
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import math

# Assuming your functions are in snr_analyzer.snr_analyzer
from snr_analyzer.snr_analyzer import generate_sine_wave, calculate_rms, measure_snr, list_audio_devices
from rich.table import Table # For isinstance check

# Mock the Rich Console to prevent actual printing during tests
# This mock_console will be used by the @patch decorator at the class or method level
mock_console_instance = MagicMock()

# This is the mock for the 'sd' module itself. Specific functions on it will be further mocked.
mock_sd_module = MagicMock()
# Make sd.PortAudioError an actual exception type for 'isinstance' checks if needed in SUT
mock_sd_module.PortAudioError = type('PortAudioError', (Exception,), {})


@patch('snr_analyzer.snr_analyzer.console', mock_console_instance) # Global console mock for all tests
@patch('snr_analyzer.snr_analyzer.sd', mock_sd_module) # Global sd module mock for all tests
class TestSNRAnalyzer(unittest.TestCase):

    def setUp(self):
        # Reset mocks before each test to ensure test isolation
        mock_console_instance.reset_mock()
        # Reset the main sd mock, also clearing side_effect and return_value of its children if they were set.
        mock_sd_module.reset_mock(return_value=True, side_effect=True)
        # Redefine PortAudioError on the mock for each test, as reset_mock might clear it.
        # Attributes that are not mocks themselves (like this direct assignment) need to be re-added.
        mock_sd_module.PortAudioError = type('PortAudioError', (Exception,), {})


    def test_generate_sine_wave_properties(self):
        samplerate = 48000
        duration = 1.0
        frequency = 100.0
        amplitude = 0.5
        wave = generate_sine_wave(frequency, duration, amplitude, samplerate)

        self.assertIsInstance(wave, np.ndarray)
        self.assertEqual(wave.ndim, 1)
        self.assertEqual(len(wave), int(samplerate * duration))
        self.assertAlmostEqual(np.max(np.abs(wave)), amplitude, places=3)
        self.assertTrue(np.any(wave > 0) and np.any(wave < 0))
        if duration * frequency == int(duration * frequency):
             self.assertAlmostEqual(np.mean(wave), 0.0, places=3)

    def test_calculate_rms(self):
        self.assertAlmostEqual(calculate_rms(np.ones(1000) * 0.5), 0.5, places=5)
        self.assertAlmostEqual(calculate_rms(np.array([1, -1, 1, -1])), 1.0, places=5)
        amplitude = 0.8
        wave = amplitude * np.sin(np.linspace(0, 2 * np.pi * 10, 1000)) 
        self.assertAlmostEqual(calculate_rms(wave), amplitude / np.sqrt(2), places=3)
        self.assertAlmostEqual(calculate_rms(np.zeros(1000)), 1e-9, places=9)
        
        # Test with empty array
        self.assertEqual(calculate_rms(np.array([])), 0.0)
        mock_console_instance.print.assert_any_call("[yellow]Warning: Audio segment is empty or None. RMS treated as 0.[/yellow]")
        mock_console_instance.reset_mock() # Reset after checking specific call
        
        # Test with None
        self.assertEqual(calculate_rms(None), 0.0)
        mock_console_instance.print.assert_any_call("[yellow]Warning: Audio segment is empty or None. RMS treated as 0.[/yellow]")

    def test_list_audio_devices_prints_table(self):
        mock_console_instance.reset_mock() # Ensure console is clean for this test's assertions

        # Define the successful return value for sd.query_devices()
        mock_devices_data = [
            {'name': 'Mock Device A', 'max_input_channels': 2, 'max_output_channels': 2, 'default_samplerate': 44100.0, 'hostapi': 0, 'index':0},
            {'name': 'Mock Device B', 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': 48000.0, 'hostapi': 0, 'index':1},
        ]

        # Use patch.object to control mock_sd_module.query_devices for this test
        # mock_sd_module is the globally patched 'sd' module from the class decorator
        with patch.object(mock_sd_module, 'query_devices', return_value=mock_devices_data) as mock_specific_query_devices:
            list_audio_devices() # Call the function that uses sd.query_devices()

            printed_table = False
            table_title_found = None
            # Check calls made to the globally mocked console instance
            for call_obj in mock_console_instance.print.call_args_list:
                if call_obj.args and isinstance(call_obj.args[0], Table):
                    table_title_found = call_obj.args[0].title # This is a string if Table was initialized with a string title
                    if table_title_found == "Available Audio Devices": # Direct string comparison
                        printed_table = True
                        self.assertEqual(len(call_obj.args[0].rows), 2) # Check number of device rows
                        self.assertEqual(call_obj.args[0].columns[0].header, "ID") # Check a column header
                        break
            self.assertTrue(printed_table, f"The 'Available Audio Devices' table was not printed or title mismatch. Title found: '{table_title_found}'. Console calls: {mock_console_instance.print.call_args_list}")
            mock_specific_query_devices.assert_called_once_with() # Verify sd.query_devices() was called once with no arguments

    def test_list_audio_devices_exception_handling(self):
        # Setup mock for sd.query_devices() to raise an exception
        test_exception_message = "Simulated query_devices error"
        mock_sd_module.query_devices.side_effect = Exception(test_exception_message)

        list_audio_devices()

        expected_error_print = f"[bold red]Error listing audio devices: {test_exception_message}[/bold red]"
        found_error_print = False
        for call_obj in mock_console_instance.print.call_args_list:
            if call_obj.args and call_obj.args[0] == expected_error_print:
                found_error_print = True
                break
        self.assertTrue(found_error_print, f"Expected error message not found. Calls: {mock_console_instance.print.call_args_list}")


    def run_measure_snr_test_case(self, mock_rms_signal_plus_noise_val, mock_rms_noise_val,
                                  expected_snr_db, expected_rms_signal_only,
                                  samplerate=48000, device_id_in=0, device_id_out=1):
        """
        Helper function to run common test logic for measure_snr.
        It sets up mocks for sounddevice functions.
        """
        mock_console_instance.reset_mock() # Reset console mock for SNR specific prints

        # Configure sd.query_devices mock (used by measure_snr)
        def mock_query_devices_side_effect(device=None, kind=None): # Changed 'device_id' to 'device' to match sounddevice
            if device is None: # sd.query_devices()
                return [ # List of all devices
                    {'name': 'MockInput', 'index': device_id_in, 'hostapi': 0, 'max_input_channels': 2, 'max_output_channels': 0, 'default_samplerate': samplerate},
                    {'name': 'MockOutput', 'index': device_id_out, 'hostapi': 0, 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': samplerate},
                    {'name': 'OtherDevice', 'index': 2, 'hostapi': 0, 'max_input_channels': 1, 'max_output_channels': 1, 'default_samplerate': samplerate}
                ]
            # sd.query_devices(device_id)
            if device == device_id_in:
                return {'name': 'MockInput', 'index': device_id_in, 'hostapi': 0, 'max_input_channels': 2, 'max_output_channels': 0, 'default_samplerate': samplerate}
            elif device == device_id_out:
                return {'name': 'MockOutput', 'index': device_id_out, 'hostapi': 0, 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': samplerate}
            raise ValueError(f"Unmocked device query for device ID: {device}")
        mock_sd_module.query_devices.side_effect = mock_query_devices_side_effect
        
        # Prepare dummy audio data
        duration = 1.0 
        num_samples = int(duration * samplerate)

        # Data for signal + noise path (DC signal whose RMS is its value)
        signal_plus_noise_data = np.full(num_samples, mock_rms_signal_plus_noise_val, dtype=np.float32)
        mock_sd_module.playrec.return_value = signal_plus_noise_data.reshape(-1, 1)

        # Data for noise path (DC signal whose RMS is its value)
        noise_data = np.full(num_samples, mock_rms_noise_val, dtype=np.float32)
        mock_sd_module.rec.return_value = noise_data.reshape(-1, 1)

        # Ensure sd.wait and sd.sleep are simple mocks that do nothing
        mock_sd_module.wait = MagicMock()
        mock_sd_module.sleep = MagicMock()

        # Call measure_snr
        snr_db, rms_signal, rms_noise_actual = measure_snr(
            output_device_id=device_id_out, # Use the ID configured in mock_query_devices_side_effect
            input_device_id=device_id_in,   # Use the ID configured in mock_query_devices_side_effect
            output_channel=1, # Assuming valid channel for mock device
            input_channel=1,  # Assuming valid channel for mock device
            samplerate=samplerate,
            signal_freq=1000, 
            signal_amp=0.5, # Amplitude of generated signal if not using file
            signal_duration=duration,
            noise_duration=duration
        )
        
        # Assert that sd.playrec was called with data of the correct shape (N, 1)
        self.assertTrue(mock_sd_module.playrec.called, "sd.playrec was not called.")
        if mock_sd_module.playrec.called: # Ensure it was called before trying to access call_args
            call_args_playrec, _ = mock_sd_module.playrec.call_args
            played_signal_data = call_args_playrec[0] # First positional argument is 'data'
            self.assertIsInstance(played_signal_data, np.ndarray, "Data passed to playrec should be a NumPy array.")
            self.assertEqual(played_signal_data.ndim, 2, "Data passed to playrec should be 2D (N, 1).")
            self.assertEqual(played_signal_data.shape[1], 1, "Data passed to playrec should have 1 column.")
            # Check that the number of samples is as expected
            expected_num_samples = int(duration * samplerate)
            self.assertEqual(played_signal_data.shape[0], expected_num_samples, "Number of samples in data passed to playrec is incorrect.")

        # Assertions for SNR results
        # Check for unexpected error prints from measure_snr
        unexpected_error_prints = [
            str(call_args.args[0]) for call_args in mock_console_instance.print.call_args_list
            if isinstance(call_args.args[0], str) and "unexpected error occurred" in call_args.args[0].lower()
        ]
        self.assertEqual(len(unexpected_error_prints), 0, f"measure_snr printed unexpected errors: {unexpected_error_prints}")


        if expected_snr_db == float('inf'):
            self.assertEqual(snr_db, float('inf'), f"SNR should be float('inf'). Got {snr_db}.")
        else:
            self.assertIsNotNone(snr_db, "SNR DB is None, expected a float value.")
            self.assertAlmostEqual(snr_db, expected_snr_db, places=2, msg=f"Calculated SNR dB ({snr_db}) does not match expected ({expected_snr_db}).")
        
        self.assertIsNotNone(rms_signal, "RMS Signal is None.")
        self.assertAlmostEqual(rms_signal, expected_rms_signal_only, places=5, msg=f"Calculated RMS of signal only ({rms_signal}) does not match expected ({expected_rms_signal_only}).")
        
        self.assertIsNotNone(rms_noise_actual, "RMS Noise Actual is None.")
        # calculate_rms returns 1e-9 for effectively zero noise input
        expected_mock_rms_noise_val_for_calc = mock_rms_noise_val if mock_rms_noise_val >= 1e-9 else 1e-9
        self.assertAlmostEqual(rms_noise_actual, expected_mock_rms_noise_val_for_calc, places=5, msg=f"Calculated RMS of noise ({rms_noise_actual}) does not match mock input ({expected_mock_rms_noise_val_for_calc}).")

    def test_snr_calculation_ideal_signal(self):
        rms_s_p_n = 1.0
        rms_n = 0.1
        expected_rms_s_only = np.sqrt(max(0, rms_s_p_n**2 - rms_n**2))
        expected_snr = 20 * np.log10(expected_rms_s_only / rms_n) if rms_n > 1e-9 and expected_rms_s_only > 1e-9 else float('inf')
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)

    def test_snr_calculation_noise_dominates_or_no_signal(self):
        rms_s_p_n = 0.1
        rms_n = 0.1
        expected_rms_s_only = 0.0 
        # measure_snr logic: if rms_signal_only < 1e-9, snr_db = 20 * math.log10(1e-9 / rms_noise)
        expected_snr = 20 * math.log10(1e-9 / rms_n) if rms_n > 1e-9 else float('inf')
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)

    def test_snr_calculation_signal_less_than_noise(self):
        # This case means (S+N) energy is less than N energy, which is unphysical if signal has energy.
        # measure_snr treats rms_signal_only as 0.0 in this case.
        rms_s_p_n = 0.05 
        rms_n = 0.1
        expected_rms_s_only = 0.0
        expected_snr = 20 * math.log10(1e-9 / rms_n) if rms_n > 1e-9 else float('inf')
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)
        # Check for warning print
        found_warning = any("[bold red]Warning: Measured RMS of (Signal+Noise) is less than RMS of Noise" in call.args[0] 
                            for call in mock_console_instance.print.call_args_list if call.args)
        self.assertTrue(found_warning, "Warning for S+N < N not printed.")


    def test_snr_calculation_zero_noise(self):
        rms_s_p_n = 1.0
        rms_n = 0.0 # Will be treated as 1e-9 by calculate_rms
        expected_rms_s_only = rms_s_p_n # Since rms_n^2 is effectively 0
        expected_snr = float('inf') # measure_snr specific check for rms_noise < 1e-9
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)
        # Check for specific print
        found_message = any("[green]Noise level is extremely low. SNR is considered very high (Infinity).[/green]" in call.args[0]
                            for call in mock_console_instance.print.call_args_list if call.args)
        self.assertTrue(found_message, "Zero noise message not printed.")
        

if __name__ == '__main__':
    unittest.main()
