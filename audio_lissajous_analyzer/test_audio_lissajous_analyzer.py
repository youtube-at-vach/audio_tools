# test_audio_lissajous_analyzer.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# By placing tests in the same package, we can use relative imports
from .audio_lissajous_analyzer import list_audio_devices, main

class TestLissajousAnalyzer(unittest.TestCase):

    @patch('audio_lissajous_analyzer.audio_lissajous_analyzer.sd')
    @patch('audio_lissajous_analyzer.audio_lissajous_analyzer.console')
    def test_list_devices_handles_no_devices(self, mock_console, mock_sd):
        """
        Test that list_audio_devices() correctly handles the case with no devices.
        """
        mock_sd.query_devices.return_value = []
        list_audio_devices()
        # Check that the table title is printed, even if there are no rows.
        mock_console.print.assert_called()
        self.assertIn("Available Audio Devices", mock_console.print.call_args[0][0].title)

    @patch('audio_lissajous_analyzer.audio_lissajous_analyzer.sd')
    @patch('audio_lissajous_analyzer.audio_lissajous_analyzer.console')
    def test_list_devices_handles_api_error(self, mock_console, mock_sd):
        """
        Test that list_audio_devices() prints an error if sd.query_devices() fails.
        """
        mock_sd.query_devices.side_effect = Exception("Test error")
        list_audio_devices()
        # Check that an error message was printed by iterating through all calls.
        self.assertTrue(mock_console.print.called)
        call_texts = " ".join(str(call) for call in mock_console.print.call_args_list)
        self.assertIn("Error querying audio devices", call_texts)
        self.assertIn("Test error", call_texts)

    @patch('sys.argv', ['audio_lissajous_analyzer.py', '--list-devices'])
    @patch('audio_lissajous_analyzer.audio_lissajous_analyzer.list_audio_devices')
    def test_main_calls_list_devices(self, mock_list_devices):
        """
        Test that main() calls list_audio_devices() when --list-devices is passed.
        """
        main()
        mock_list_devices.assert_called_once()

    @patch('sys.argv', ['audio_lissajous_analyzer.py', '-d', '0'])
    @patch('audio_lissajous_analyzer.audio_lissajous_analyzer.sd')
    @patch('audio_lissajous_analyzer.audio_lissajous_analyzer.plt.show')
    @patch('audio_lissajous_analyzer.audio_lissajous_analyzer.FuncAnimation')
    def test_main_runs_with_valid_args(self, mock_animation, mock_plt_show, mock_sd):
        """
        Test that the main function attempts to start the stream with valid arguments.
        This is a high-level integration test.
        """
        # Mock device query to return a valid device
        mock_device = {
            'name': 'Mock Device',
            'max_input_channels': 2,
            'max_output_channels': 2,
            'default_samplerate': 48000
        }
        mock_sd.query_devices.return_value = mock_device
        mock_sd.InputStream.return_value.__enter__.return_value = None # Mock the stream context

        main()

        # Check that we tried to create an InputStream
        mock_sd.InputStream.assert_called_once()
        # Check that we tried to show the plot
        mock_plt_show.assert_called_once()


if __name__ == '__main__':
    unittest.main()
