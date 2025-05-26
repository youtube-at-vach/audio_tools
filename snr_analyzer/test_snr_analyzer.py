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
# This will be set in setUp for each test.


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
        # Also, ensure check_input_settings is a fresh mock for each test that might configure it.
        mock_sd_module.PortAudioError = type('PortAudioError', (Exception,), {})
        mock_sd_module.check_input_settings = MagicMock() # Ensure it's a fresh mock for each test run via helper

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

        mock_devices_data = [
            {'name': 'Mock Device A', 'max_input_channels': 2, 'max_output_channels': 2, 'default_samplerate': 44100.0, 'hostapi': 0, 'index':0},
            {'name': 'Mock Device B', 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': 48000.0, 'hostapi': 0, 'index':1},
        ]
        with patch.object(mock_sd_module, 'query_devices', return_value=mock_devices_data) as mock_specific_query_devices:
            list_audio_devices() 

            printed_table = False
            table_title_found = None
            for call_obj in mock_console_instance.print.call_args_list:
                if call_obj.args and isinstance(call_obj.args[0], Table):
                    table_title_found = call_obj.args[0].title 
                    if table_title_found == "Available Audio Devices": 
                        printed_table = True
                        self.assertEqual(len(call_obj.args[0].rows), 2) 
                        self.assertEqual(call_obj.args[0].columns[0].header, "ID") 
                        break
            self.assertTrue(printed_table, f"The 'Available Audio Devices' table was not printed or title mismatch. Title found: '{table_title_found}'. Console calls: {mock_console_instance.print.call_args_list}")
            mock_specific_query_devices.assert_called_once_with() 

    def test_list_audio_devices_exception_handling(self):
        test_exception_message = "Simulated query_devices error"
        # Configure query_devices directly on mock_sd_module for this test
        mock_sd_module.query_devices.side_effect = Exception(test_exception_message)

        list_audio_devices()

        expected_error_print = f"[bold red]Error listing audio devices: {test_exception_message}[/bold red]"
        found_error_print = any(call.args and call.args[0] == expected_error_print for call in mock_console_instance.print.call_args_list)
        self.assertTrue(found_error_print, f"Expected error message not found. Calls: {mock_console_instance.print.call_args_list}")


    def run_measure_snr_test_case(self, mock_rms_signal_plus_noise_val, mock_rms_noise_val,
                                  expected_snr_db, expected_rms_signal_only,
                                  samplerate=48000, device_id_in=0, device_id_out=1,
                                  input_channel_to_test=1, output_channel_to_test=1): # Added channel params for flexibility
        """
        Helper function to run common test logic for measure_snr.
        It sets up mocks for sounddevice functions.
        """
        mock_console_instance.reset_mock() 

        def mock_query_devices_side_effect(device=None, kind=None):
            if device is None: 
                return [ 
                    {'name': 'MockInput', 'index': device_id_in, 'hostapi': 0, 'max_input_channels': 2, 'max_output_channels': 0, 'default_samplerate': samplerate},
                    {'name': 'MockOutput', 'index': device_id_out, 'hostapi': 0, 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': samplerate},
                    {'name': 'OtherDevice', 'index': 2, 'hostapi': 0, 'max_input_channels': 1, 'max_output_channels': 1, 'default_samplerate': samplerate}
                ]
            if device == device_id_in:
                return {'name': 'MockInput', 'index': device_id_in, 'hostapi': 0, 'max_input_channels': 2, 'max_output_channels': 0, 'default_samplerate': samplerate}
            elif device == device_id_out:
                return {'name': 'MockOutput', 'index': device_id_out, 'hostapi': 0, 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': samplerate}
            raise ValueError(f"Unmocked device query for device ID: {device}")
        mock_sd_module.query_devices.side_effect = mock_query_devices_side_effect
        
        duration = 1.0 
        num_samples = int(duration * samplerate)

        signal_plus_noise_data = np.full(num_samples, mock_rms_signal_plus_noise_val, dtype=np.float32)
        mock_sd_module.playrec.return_value = signal_plus_noise_data.reshape(-1, 1)

        noise_data = np.full(num_samples, mock_rms_noise_val, dtype=np.float32)
        mock_sd_module.rec.return_value = noise_data.reshape(-1, 1)

        mock_sd_module.wait = MagicMock()
        mock_sd_module.sleep = MagicMock()
        # Ensure check_input_settings is a mock that doesn't raise error by default for standard tests
        # This is now handled in setUp by assigning MagicMock() to it.
        # If a specific test needs it to fail, it will configure side_effect.

        snr_db, rms_signal, rms_noise_actual = measure_snr(
            output_device_id=device_id_out, 
            input_device_id=device_id_in,   
            output_channel=output_channel_to_test, 
            input_channel=input_channel_to_test,  
            samplerate=samplerate,
            signal_freq=1000, 
            signal_amp=0.5, 
            signal_duration=duration,
            noise_duration=duration
        )
        
        # Assert sd.playrec call
        self.assertTrue(mock_sd_module.playrec.called, "sd.playrec was not called.")
        if mock_sd_module.playrec.called: 
            call_args_playrec, _ = mock_sd_module.playrec.call_args
            played_signal_data = call_args_playrec[0] 
            self.assertIsInstance(played_signal_data, np.ndarray)
            self.assertEqual(played_signal_data.ndim, 2)
            self.assertEqual(played_signal_data.shape[1], 1)
            expected_num_samples = int(duration * samplerate)
            self.assertEqual(played_signal_data.shape[0], expected_num_samples)

        # Assert sd.check_input_settings call (for successful cases)
        # This assertion will be added here, assuming for most tests it passes.
        # For tests where it's expected to fail, this might need adjustment or be checked separately.
        if mock_sd_module.check_input_settings.side_effect is None: # Only assert if not configured to fail
            mock_sd_module.check_input_settings.assert_called_once_with(
                device=device_id_in,
                channels=1,
                # mapping argument removed as it's not used by the corrected check_input_settings call
                samplerate=samplerate
            )
        
        # Assert sd.rec call
        self.assertTrue(mock_sd_module.rec.called, "sd.rec was not called.")
        if mock_sd_module.rec.called:
            mock_sd_module.rec.assert_called_once_with(
                int(duration * samplerate), 
                samplerate=samplerate,
                mapping=[input_channel_to_test],                
                channels=1,
                device=device_id_in,        
                blocking=True
            )

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
        expected_mock_rms_noise_val_for_calc = mock_rms_noise_val if mock_rms_noise_val >= 1e-9 else 1e-9
        self.assertAlmostEqual(rms_noise_actual, expected_mock_rms_noise_val_for_calc, places=5, msg=f"Calculated RMS of noise ({rms_noise_actual}) does not match mock input ({expected_mock_rms_noise_val_for_calc}).")
        return snr_db, rms_signal, rms_noise_actual # Return values for specific checks in new tests

    def test_snr_calculation_ideal_signal(self):
        rms_s_p_n = 1.0; rms_n = 0.1
        expected_rms_s_only = np.sqrt(max(0, rms_s_p_n**2 - rms_n**2))
        expected_snr = 20 * np.log10(expected_rms_s_only / rms_n) if rms_n > 1e-9 and expected_rms_s_only > 1e-9 else float('inf')
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)

    def test_snr_calculation_noise_dominates_or_no_signal(self):
        rms_s_p_n = 0.1; rms_n = 0.1
        expected_rms_s_only = 0.0 
        expected_snr = 20 * math.log10(1e-9 / rms_n) if rms_n > 1e-9 else float('inf')
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)

    def test_snr_calculation_signal_less_than_noise(self):
        rms_s_p_n = 0.05; rms_n = 0.1
        expected_rms_s_only = 0.0
        expected_snr = 20 * math.log10(1e-9 / rms_n) if rms_n > 1e-9 else float('inf')
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)
        found_warning = any("[bold red]Warning: Measured RMS of (Signal+Noise) is less than RMS of Noise" in call.args[0] 
                            for call in mock_console_instance.print.call_args_list if call.args)
        self.assertTrue(found_warning, "Warning for S+N < N not printed.")

    def test_snr_calculation_zero_noise(self):
        rms_s_p_n = 1.0; rms_n = 0.0
        expected_rms_s_only = rms_s_p_n 
        expected_snr = float('inf') 
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)
        found_message = any("[green]Noise level is extremely low. SNR is considered very high (Infinity).[/green]" in call.args[0]
                            for call in mock_console_instance.print.call_args_list if call.args)
        self.assertTrue(found_message, "Zero noise message not printed.")

    def test_measure_snr_check_input_settings_fails(self):
        mock_console_instance.reset_mock()
        device_id_in_test = 0
        input_channel_test = 1
        samplerate_test = 48000
        
        # Configure sd.query_devices mock (minimal setup for this test)
        # This is needed because measure_snr calls query_devices before check_input_settings
        mock_sd_module.query_devices.side_effect = lambda device=None, kind=None: {
            'name': 'MockInput', 'index': device_id_in_test, 'hostapi': 0, 'max_input_channels': 2, 'max_output_channels': 0, 'default_samplerate': samplerate_test
        } if device == device_id_in_test else {
            'name': 'MockOutput', 'index': 1, 'hostapi': 0, 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': samplerate_test
        }

        # Configure check_input_settings to fail
        simulated_error_msg = "Simulated check_input_settings failure"
        # Need to use the actual PortAudioError type defined in setUp
        mock_sd_module.check_input_settings.side_effect = mock_sd_module.PortAudioError(simulated_error_msg)

        results = measure_snr(
            output_device_id=1, input_device_id=device_id_in_test, 
            output_channel=1, input_channel=input_channel_test, 
            samplerate=samplerate_test, signal_freq=1000, signal_amp=0.5, 
            signal_duration=1.0, noise_duration=1.0
        )

        self.assertEqual(results, (None, None, None), "measure_snr should return (None, None, None) on check_input_settings failure.")
        
        mock_sd_module.check_input_settings.assert_called_once_with(
            device=device_id_in_test, channels=1, samplerate=samplerate_test
            # mapping argument removed
        )
        
        # Check for specific console prints related to check_input_settings failure
        prints = [str(call.args[0]) for call in mock_console_instance.print.call_args_list if call.args]
        # This error message was updated in snr_analyzer.py to reflect that check_input_settings doesn't know about specific channels from mapping.
        expected_error_msg_main = f"[bold red]Error: Input device {device_id_in_test} does not support the required settings (samplerate: {samplerate_test}Hz, channels: 1) for noise recording.[/bold red]"
        self.assertIn(expected_error_msg_main, prints)
        self.assertIn(f"[bold red]PortAudio Error details: {simulated_error_msg}[/bold red]", prints)
        
        mock_sd_module.rec.assert_not_called() # sd.rec should not be called if check fails

    def test_measure_snr_rec_fails_after_check_passes(self):
        mock_console_instance.reset_mock()
        device_id_in_test = 0
        input_channel_test = 1
        samplerate_test = 48000
        duration_test = 1.0

        # Configure sd.query_devices (minimal setup)
        mock_sd_module.query_devices.side_effect = lambda device=None, kind=None: {
            'name': 'MockInput', 'index': device_id_in_test, 'hostapi': 0, 'max_input_channels': 2, 'max_output_channels': 0, 'default_samplerate': samplerate_test
        } if device == device_id_in_test else {
            'name': 'MockOutput', 'index': 1, 'hostapi': 0, 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': samplerate_test
        }
        
        # Configure check_input_settings to pass (default behavior of MagicMock is fine, or set side_effect = None)
        mock_sd_module.check_input_settings.side_effect = None 
        
        # Configure playrec to return some dummy data
        num_samples = int(duration_test * samplerate_test)
        mock_sd_module.playrec.return_value = np.zeros((num_samples, 1), dtype=np.float32)

        # Configure rec to fail
        simulated_rec_error_msg = "Simulated rec failure"
        mock_sd_module.rec.side_effect = mock_sd_module.PortAudioError(simulated_rec_error_msg)

        results = measure_snr(
            output_device_id=1, input_device_id=device_id_in_test, 
            output_channel=1, input_channel=input_channel_test, 
            samplerate=samplerate_test, signal_freq=1000, signal_amp=0.5, 
            signal_duration=duration_test, noise_duration=duration_test
        )

        self.assertEqual(results, (None, None, None), "measure_snr should return (None, None, None) on rec failure.")
        
        mock_sd_module.check_input_settings.assert_called_once_with(
            device=device_id_in_test, channels=1, samplerate=samplerate_test
            # mapping argument removed
        )
        mock_sd_module.rec.assert_called_once() # Check it was called (even if it failed)
        
        prints = [str(call.args[0]) for call in mock_console_instance.print.call_args_list if call.args]
        self.assertIn(f"[bold red]Error during noise recording with device ID {device_id_in_test} (channel {input_channel_test}):[/bold red]", prints)
        self.assertIn(f"[bold red]PortAudio Error: {simulated_rec_error_msg}[/bold red]", prints)

if __name__ == '__main__':
    unittest.main()
