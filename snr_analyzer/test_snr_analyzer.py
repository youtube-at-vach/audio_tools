import unittest
from unittest.mock import patch, MagicMock, ANY, call # Ensure call is imported
import numpy as np
import math

# Assuming your functions are in snr_analyzer.snr_analyzer
from snr_analyzer.snr_analyzer import generate_sine_wave, calculate_rms, measure_snr, list_audio_devices
from rich.table import Table # For isinstance check
import sounddevice as sd # Import for spec in MagicMock

# Mock the Rich Console to prevent actual printing during tests
mock_console_instance = MagicMock()

# This is the mock for the 'sd' module itself. Specific functions on it will be further mocked.
mock_sd_module = MagicMock()
# Make sd.PortAudioError an actual exception type for 'isinstance' checks if needed in SUT
# This will be set in setUp for each test.


@patch('snr_analyzer.snr_analyzer.console', mock_console_instance) # Global console mock for all tests
@patch('snr_analyzer.snr_analyzer.sd', mock_sd_module) # Global sd module mock for all tests
class TestSNRAnalyzer(unittest.TestCase):

    def setUp(self):
        mock_console_instance.reset_mock()
        mock_sd_module.reset_mock(return_value=True, side_effect=True)
        
        # Define mock exception types on the mock_sd_module for consistent use
        mock_sd_module.PortAudioError = type('PortAudioError', (Exception,), {})
        
        # Explicitly create/reset MagicMock attributes on the shared mock_sd_module for each test
        mock_sd_module.query_devices = MagicMock()
        mock_sd_module.playrec = MagicMock()
        # sd.rec is no longer used for noise, sd.InputStream will be.
        mock_sd_module.rec = MagicMock() # Keep for other potential uses or ensure it's not called
        mock_sd_module.InputStream = MagicMock(spec=sd.InputStream) # Mock for sd.InputStream constructor
        mock_sd_module.wait = MagicMock()
        mock_sd_module.sleep = MagicMock()
        mock_sd_module.stop = MagicMock() 
        # sd.check_input_settings will be patched directly using patch('snr_analyzer.snr_analyzer.sd.check_input_settings')

    def test_generate_sine_wave_properties(self):
        samplerate = 48000; duration = 1.0; frequency = 100.0; amplitude = 0.5
        wave = generate_sine_wave(frequency, duration, amplitude, samplerate)
        self.assertIsInstance(wave, np.ndarray); self.assertEqual(wave.ndim, 1)
        self.assertEqual(len(wave), int(samplerate * duration))
        self.assertAlmostEqual(np.max(np.abs(wave)), amplitude, places=3)
        self.assertTrue(np.any(wave > 0) and np.any(wave < 0))
        if duration * frequency == int(duration * frequency): self.assertAlmostEqual(np.mean(wave), 0.0, places=3)

    def test_calculate_rms(self):
        self.assertAlmostEqual(calculate_rms(np.ones(1000) * 0.5), 0.5, places=5)
        self.assertAlmostEqual(calculate_rms(np.array([1, -1, 1, -1])), 1.0, places=5)
        amplitude = 0.8; wave = amplitude * np.sin(np.linspace(0, 2 * np.pi * 10, 1000)) 
        self.assertAlmostEqual(calculate_rms(wave), amplitude / np.sqrt(2), places=3)
        self.assertAlmostEqual(calculate_rms(np.zeros(1000)), 1e-9, places=9)
        self.assertEqual(calculate_rms(np.array([])), 0.0)
        mock_console_instance.print.assert_any_call("[yellow]Warning: Audio segment is empty or None. RMS treated as 0.[/yellow]")
        mock_console_instance.reset_mock()
        self.assertEqual(calculate_rms(None), 0.0)
        mock_console_instance.print.assert_any_call("[yellow]Warning: Audio segment is empty or None. RMS treated as 0.[/yellow]")

    def test_list_audio_devices_prints_table(self):
        mock_console_instance.reset_mock(); mock_sd_module.query_devices.reset_mock()
        mock_devices_data = [
            {'name': 'Mock Device A', 'max_input_channels': 2, 'max_output_channels': 2, 'default_samplerate': 44100.0, 'hostapi': 0, 'index':0},
            {'name': 'Mock Device B', 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': 48000.0, 'hostapi': 0, 'index':1},
        ]
        mock_sd_module.query_devices.return_value = mock_devices_data
            
        list_audio_devices() 
        printed_table = any(isinstance(call.args[0], Table) and call.args[0].title == "Available Audio Devices" 
                            for call in mock_console_instance.print.call_args_list if call.args)
        self.assertTrue(printed_table, "Table not printed or title mismatch.")
        mock_sd_module.query_devices.assert_called_once_with() 

    def test_list_audio_devices_exception_handling(self):
        mock_console_instance.reset_mock(); mock_sd_module.query_devices.reset_mock()
        test_exception_message = "Simulated query_devices error"
        mock_sd_module.query_devices.side_effect = Exception(test_exception_message)
        
        list_audio_devices()
        expected_error_print = f"[bold red]Error listing audio devices: {test_exception_message}[/bold red]"
        self.assertTrue(any(call.args and call.args[0] == expected_error_print for call in mock_console_instance.print.call_args_list))

    def run_measure_snr_test_case(self, mock_rms_signal_plus_noise_val, mock_rms_noise_val,
                                  expected_snr_db, expected_rms_signal_only,
                                  samplerate=48000, device_id_in=0, device_id_out=1,
                                  input_channel_to_test=1, output_channel_to_test=1):
        mock_console_instance.reset_mock() 
        mock_sd_module.query_devices.reset_mock()
        mock_sd_module.playrec.reset_mock()
        mock_sd_module.InputStream.reset_mock() # Reset InputStream constructor mock
        mock_sd_module.wait.reset_mock()
        mock_sd_module.sleep.reset_mock()
        mock_sd_module.stop.reset_mock() 

        def mock_query_devices_side_effect(device=None, kind=None):
            if device is None: return [{'name': 'MockInput', 'index': device_id_in, 'hostapi': 0, 'max_input_channels': 2, 'max_output_channels': 0, 'default_samplerate': samplerate},
                                      {'name': 'MockOutput', 'index': device_id_out, 'hostapi': 0, 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': samplerate}]
            if device == device_id_in: return {'name': 'MockInput', 'index': device_id_in, 'hostapi': 0, 'max_input_channels': 2, 'max_output_channels': 0, 'default_samplerate': samplerate}
            if device == device_id_out: return {'name': 'MockOutput', 'index': device_id_out, 'hostapi': 0, 'max_input_channels': 0, 'max_output_channels': 2, 'default_samplerate': samplerate}
            raise ValueError(f"Unmocked device query: {device}")
        mock_sd_module.query_devices.side_effect = mock_query_devices_side_effect
        
        duration = 1.0 
        num_samples = int(duration * samplerate)
        num_noise_frames = int(duration * samplerate)

        signal_plus_noise_data = np.full(num_samples, mock_rms_signal_plus_noise_val, dtype=np.float32)
        mock_sd_module.playrec.return_value = signal_plus_noise_data.reshape(-1, 1)

        # Mock for sd.InputStream (noise recording)
        mock_stream_instance_noise = MagicMock(spec=sd.InputStream)
        mock_noise_data_array = np.full(num_noise_frames, mock_rms_noise_val, dtype=np.float32).reshape(-1,1)
        mock_stream_instance_noise.read.return_value = (mock_noise_data_array, False) # data, overflow_status
        mock_sd_module.InputStream.return_value = mock_stream_instance_noise # Configure constructor to return our instance

        with patch('snr_analyzer.snr_analyzer.sd.check_input_settings') as mock_direct_check_input_settings:
            mock_direct_check_input_settings.side_effect = None 

            snr_db, rms_signal, rms_noise_actual = measure_snr(
                output_device_id=device_id_out, input_device_id=device_id_in,   
                output_channel=output_channel_to_test, input_channel=input_channel_to_test,  
                samplerate=samplerate, signal_freq=1000, signal_amp=0.5, 
                signal_duration=duration, noise_duration=duration
            )
        
            self.assertTrue(mock_sd_module.playrec.called)
            mock_sd_module.wait.assert_any_call() 
            mock_sd_module.stop.assert_called_once_with(ignore_errors=True)
            
            expected_call_args = call(device=device_id_in, channels=1, samplerate=samplerate)
            self.assertEqual(mock_direct_check_input_settings.call_count, 2, 
                             f"Expected check_input_settings to be called twice (anomaly). Actual calls: {mock_direct_check_input_settings.call_count}")
            self.assertEqual(mock_direct_check_input_settings.mock_calls, [expected_call_args, expected_call_args],
                             f"Expected two identical calls to check_input_settings. Calls: {mock_direct_check_input_settings.mock_calls}")
        
        # Assert sd.InputStream workflow for noise recording
        mock_sd_module.InputStream.assert_called_once_with(
            device=device_id_in, mapping=[input_channel_to_test], channels=1, 
            samplerate=samplerate, dtype='float32'
        )
        mock_stream_instance_noise.__enter__.assert_called_once()
        mock_stream_instance_noise.read.assert_called_once_with(num_noise_frames)
        # stream.stop() is called by the context manager via __exit__ which calls close() which calls stop()
        # For a blocking read, __exit__ directly calls close(), which calls abort() then close().
        # Direct check of stop might be tricky depending on exact sd.InputStream mock spec behavior.
        # Let's assume close() implies stop for now or rely on __exit__.
        mock_stream_instance_noise.__exit__.assert_called_once() 
        
        self.assertEqual(mock_sd_module.wait.call_count, 1) # Only one wait after playrec, no wait after InputStream.read

        self.assertIsNotNone(snr_db) 
        if expected_snr_db == float('inf'): self.assertEqual(snr_db, float('inf'))
        else: self.assertAlmostEqual(snr_db, expected_snr_db, places=2)
        self.assertAlmostEqual(rms_signal, expected_rms_signal_only, places=5)
        expected_mock_rms_noise_val_for_calc = mock_rms_noise_val if mock_rms_noise_val >= 1e-9 else 1e-9
        self.assertAlmostEqual(rms_noise_actual, expected_mock_rms_noise_val_for_calc, places=5)
        return snr_db, rms_signal, rms_noise_actual

    def test_snr_calculation_ideal_signal(self):
        rms_s_p_n = 1.0; rms_n = 0.1
        expected_rms_s_only = np.sqrt(max(0, rms_s_p_n**2 - rms_n**2))
        expected_snr = 20 * np.log10(expected_rms_s_only / rms_n) if rms_n > 1e-9 and expected_rms_s_only > 1e-9 else float('inf')
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)

    def test_snr_calculation_noise_dominates_or_no_signal(self):
        rms_s_p_n = 0.1; rms_n = 0.1; expected_rms_s_only = 0.0 
        expected_snr = 20 * math.log10(1e-9 / rms_n) if rms_n > 1e-9 else float('inf')
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)

    def test_snr_calculation_signal_less_than_noise(self):
        rms_s_p_n = 0.05; rms_n = 0.1; expected_rms_s_only = 0.0
        expected_snr = 20 * math.log10(1e-9 / rms_n) if rms_n > 1e-9 else float('inf')
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)
        self.assertTrue(any("[bold red]Warning: Measured RMS of (Signal+Noise) is less than RMS of Noise" in call.args[0] 
                            for call in mock_console_instance.print.call_args_list if call.args))

    def test_snr_calculation_zero_noise(self):
        rms_s_p_n = 1.0; rms_n = 0.0; expected_rms_s_only = rms_s_p_n 
        expected_snr = float('inf') 
        self.run_measure_snr_test_case(rms_s_p_n, rms_n, expected_snr, expected_rms_s_only)
        self.assertTrue(any("[green]Noise level is extremely low. SNR is considered very high (Infinity).[/green]" in call.args[0]
                            for call in mock_console_instance.print.call_args_list if call.args))

    def test_measure_snr_check_input_settings_fails(self):
        mock_console_instance.reset_mock(); device_id_in_test = 0; input_channel_test = 1; samplerate_test = 48000
        
        mock_sd_module.query_devices.reset_mock()
        mock_sd_module.playrec.reset_mock() 
        mock_sd_module.wait.reset_mock()    
        mock_sd_module.stop.reset_mock()    
        mock_sd_module.InputStream.reset_mock() # Ensure InputStream constructor not called

        mock_sd_module.query_devices.side_effect = lambda device=None, kind=None: {
            'name': 'MockInput', 'index': device_id_in_test, 'max_input_channels': 2} if device == device_id_in_test else {'name': 'MockOutput', 'max_output_channels': 2}
        simulated_error_msg = "Simulated check_input_settings failure"
        
        with patch('snr_analyzer.snr_analyzer.sd.check_input_settings', side_effect=mock_sd_module.PortAudioError(simulated_error_msg)) as mock_direct_check_input_settings:
            results = measure_snr(1, device_id_in_test, 1, input_channel_test, samplerate_test, 1000, 0.5, 1.0, 1.0)
            
            self.assertEqual(results, (None, None, None))
            mock_direct_check_input_settings.assert_called_once_with(device=device_id_in_test, channels=1, samplerate=samplerate_test)
            prints = [str(call.args[0]) for call in mock_console_instance.print.call_args_list if call.args]
            self.assertIn(f"[bold red]Error: Input device {device_id_in_test} does not support the required settings (samplerate: {samplerate_test}Hz, channels: 1) for noise recording.[/bold red]", prints)
            mock_sd_module.InputStream.assert_not_called() 
            self.assertTrue(mock_sd_module.playrec.called) 
            mock_sd_module.wait.assert_called_once() 
            mock_sd_module.stop.assert_called_once_with(ignore_errors=True) 

    def test_measure_snr_input_stream_read_fails(self): # Renamed
        mock_console_instance.reset_mock(); device_id_in_test = 0; input_channel_test = 1; samplerate_test = 48000; duration_test = 1.0
        
        mock_sd_module.query_devices.reset_mock()
        mock_sd_module.playrec.reset_mock()
        mock_sd_module.InputStream.reset_mock()
        mock_sd_module.wait.reset_mock()    
        mock_sd_module.stop.reset_mock()    

        mock_sd_module.query_devices.side_effect = lambda device=None, kind=None: {
            'name': 'MockInput', 'max_input_channels': 2} if device == device_id_in_test else {'name': 'MockOutput', 'max_output_channels': 2}
        
        num_samples = int(duration_test * samplerate_test)
        mock_sd_module.playrec.return_value = np.zeros((num_samples, 1), dtype=np.float32)

        simulated_stream_error_msg = "Simulated InputStream read failure"
        
        mock_stream_instance_noise = MagicMock(spec=sd.InputStream)
        mock_stream_instance_noise.read.side_effect = mock_sd_module.PortAudioError(simulated_stream_error_msg)
        # Configure the main InputStream mock to return this instance
        mock_sd_module.InputStream.return_value = mock_stream_instance_noise
        
        with patch('snr_analyzer.snr_analyzer.sd.check_input_settings') as mock_direct_check_input_settings:
            mock_direct_check_input_settings.side_effect = None 

            results = measure_snr(1, device_id_in_test, 1, input_channel_test, samplerate_test, 1000, 0.5, duration_test, duration_test)

            self.assertEqual(results, (None, None, None))
            expected_call_args = call(device=device_id_in_test, channels=1, samplerate=samplerate_test)
            self.assertEqual(mock_direct_check_input_settings.call_count, 2,
                             f"Expected check_input_settings to be called twice (anomaly). Actual calls: {mock_direct_check_input_settings.call_count}")
            self.assertEqual(mock_direct_check_input_settings.mock_calls, [expected_call_args, expected_call_args],
                             f"Expected two identical calls to check_input_settings. Calls: {mock_direct_check_input_settings.mock_calls}")
            
            mock_sd_module.InputStream.assert_called_once_with(
                device=device_id_in_test, mapping=[input_channel_test], channels=1, 
                samplerate=samplerate_test, dtype='float32'
            )
            mock_stream_instance_noise.__enter__.assert_called_once()
            mock_stream_instance_noise.read.assert_called_once() 
            mock_stream_instance_noise.__exit__.assert_called_once() # Should be called due to 'with' statement

            mock_sd_module.wait.assert_called_once() # Only the one after playrec
            mock_sd_module.stop.assert_called_once_with(ignore_errors=True)

            prints = [str(call.args[0]) for call in mock_console_instance.print.call_args_list if call.args]
            self.assertTrue(any(f"Error during noise recording with device ID {device_id_in_test}" in p for p in prints))
            self.assertTrue(any(simulated_stream_error_msg in p for p in prints))

if __name__ == '__main__':
    unittest.main()
