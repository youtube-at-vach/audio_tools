
import unittest
from unittest.mock import patch
import numpy as np
from rich.table import Table
from audio_loopback_finder.audio_loopback_finder import find_loopback

class TestAudioLoopbackFinder(unittest.TestCase):

    @patch('sounddevice.query_devices')
    @patch('sounddevice.playrec')
    @patch('sounddevice.wait')
    def test_find_loopback_path_found(self, mock_wait, mock_playrec, mock_query_devices):
        """Test that a loopback path is correctly identified."""
        mock_device_info = {
            'name': 'Test Device',
            'max_output_channels': 2,
            'max_input_channels': 2,
        }
        mock_query_devices.return_value = mock_device_info

        sample_rate = 48000
        duration = 0.1
        test_freq = 440
        
        def playrec_side_effect(output_signal, samplerate, channels, device):
            recorded = np.zeros((len(output_signal), 2), dtype=np.float32)
            # Simulate loopback from output channel 1 to input channel 2
            if np.any(output_signal[:, 0]):
                recorded[:, 1] = output_signal[:, 0]
            return recorded

        mock_playrec.side_effect = playrec_side_effect

        with patch('rich.console.Console.print') as mock_print:
            find_loopback(device_id=0, sample_rate=sample_rate, test_freq=test_freq, duration=duration, threshold=0.1)
            
            # Check if a table with the correct title was printed
            found_table = False
            for call in mock_print.call_args_list:
                arg = call.args[0]
                if isinstance(arg, Table) and arg.title == "Found Loopback Paths":
                    found_table = True
                    # Further check if the row content is as expected
                    self.assertEqual(len(arg.rows), 1)
                    self.assertEqual(arg.columns[0]._cells, ['1'])
                    self.assertEqual(arg.columns[1]._cells, ['2'])
                    break
            self.assertTrue(found_table, "The 'Found Loopback Paths' table was not printed.")

    @patch('sounddevice.query_devices')
    @patch('sounddevice.playrec')
    @patch('sounddevice.wait')
    def test_find_loopback_no_path_found(self, mock_wait, mock_playrec, mock_query_devices):
        """Test that no loopback path is reported when none exists."""
        mock_device_info = {
            'name': 'Test Device',
            'max_output_channels': 2,
            'max_input_channels': 2,
        }
        mock_query_devices.return_value = mock_device_info

        mock_playrec.return_value = np.zeros((int(48000 * 0.1), 2), dtype=np.float32)

        with patch('rich.console.Console.print') as mock_print:
            find_loopback(device_id=0)
            
            found_message = False
            for call in mock_print.call_args_list:
                if "No loopback paths found" in str(call.args[0]):
                    found_message = True
                    break
            self.assertTrue(found_message, "The 'No loopback paths found' message was not printed.")

if __name__ == '__main__':
    unittest.main()
