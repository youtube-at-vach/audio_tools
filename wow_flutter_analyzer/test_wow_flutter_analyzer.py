import unittest
import numpy as np
import sys
import os

# Adjust path to import module from parent directory
# This assumes the test script is run from the 'wow_flutter_analyzer' directory
# or that the 'wow_flutter_analyzer' parent directory is in PYTHONPATH
# For robust execution, it's often better to run tests using `python -m unittest discover`
# from the project root, and ensure the project is structured as a package.
# However, for this specific environment, we might need a direct path adjustment.

# Get the absolute path to the directory containing this test file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path to the project root (one level up)
project_root = os.path.dirname(current_dir)
# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wow_flutter_analyzer import wow_flutter_analyzer

# Suppress console output from the module during tests
# by redirecting its console object's print method
class SuppressRichOutput:
    def __enter__(self):
        self.original_print = wow_flutter_analyzer.console.print
        wow_flutter_analyzer.console.print = lambda *args, **kwargs: None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        wow_flutter_analyzer.console.print = self.original_print

def generate_sine_wave(frequency: float, duration: float, sample_rate: int, amplitude: float = 0.8) -> np.ndarray:
    """Generates a simple sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def generate_fm_sine_wave(carrier_freq: float, mod_freq: float, deviation: float, duration: float, sample_rate: int, amplitude: float = 0.8) -> np.ndarray:
    """
    Generates a sine wave with frequency modulation.
    instantaneous_freq(t) = carrier_freq + deviation * np.cos(2 * np.pi * mod_freq * t)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Integral of (deviation * cos(2*pi*mod_freq*t)) is (deviation / (2*pi*mod_freq)) * sin(2*pi*mod_freq*t)
    modulating_signal_phase = (deviation / mod_freq) * np.sin(2 * np.pi * mod_freq * t)
    return amplitude * np.sin(2 * np.pi * carrier_freq * t + modulating_signal_phase)


class TestWowFlutterAnalyzer(unittest.TestCase):
    def setUp(self):
        """Suppress console output for all tests."""
        self.suppressor = SuppressRichOutput()
        self.suppressor.__enter__()

    def tearDown(self):
        """Restore console output after all tests."""
        self.suppressor.__exit__(None, None, None)

    def test_generate_sine_wave_properties(self):
        sample_rate = 44100
        duration = 1.0
        frequency = 1000.0
        amplitude = 0.5
        signal = generate_sine_wave(frequency, duration, sample_rate, amplitude)
        
        self.assertEqual(len(signal), int(sample_rate * duration))
        self.assertAlmostEqual(np.max(np.abs(signal)), amplitude, delta=0.01) # Check amplitude
        
        # Check frequency (approximate)
        fft_result = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), d=1/sample_rate)
        dominant_freq_idx = np.argmax(np.abs(fft_result[:len(signal)//2]))
        self.assertAlmostEqual(np.abs(fft_freq[dominant_freq_idx]), frequency, delta=1)

    def test_generate_fm_sine_wave_properties(self):
        sample_rate = 48000
        duration = 2.0
        carrier_freq = 3150.0
        mod_freq = 4.0 # Wow frequency
        deviation = 10.0 # 10 Hz deviation
        amplitude = 0.7
        
        signal = generate_fm_sine_wave(carrier_freq, mod_freq, deviation, duration, sample_rate, amplitude)
        
        self.assertEqual(len(signal), int(sample_rate * duration))
        self.assertTrue(np.max(np.abs(signal)) <= amplitude + 0.01) # Max amplitude should be around the specified one

    def test_load_non_existent_file(self):
        data, rate = wow_flutter_analyzer.load_audio_file("non_existent_file.wav")
        self.assertIsNone(data)
        self.assertIsNone(rate)

    def test_demodulate_pure_sine(self):
        sample_rate = 48000
        duration = 2.0
        ref_freq = 3150.0
        
        audio_signal = generate_sine_wave(ref_freq, duration, sample_rate)
        
        demodulated_freq = wow_flutter_analyzer.demodulate_frequency(audio_signal, sample_rate, ref_freq)
        
        self.assertIsNotNone(demodulated_freq)
        # Check mean frequency, skip first and last few samples due to filter/hilbert edge effects
        stable_region = demodulated_freq[int(0.1*sample_rate) : -int(0.1*sample_rate)]
        if len(stable_region) > 0:
             self.assertAlmostEqual(np.mean(stable_region), ref_freq, delta=ref_freq*0.005) # within 0.5%
        else:
            self.fail("Stable region for demodulation check is empty. Check signal duration or demodulation process.")


    def test_demodulate_fm_signal(self):
        sample_rate = 48000
        duration = 3.0 # Longer duration for better resolution of low mod_freq
        carrier_freq = 3000.0
        mod_freq = 4.0  # 4 Hz wow
        deviation = 15.0  # 15 Hz peak deviation
        
        fm_signal = generate_fm_sine_wave(carrier_freq, mod_freq, deviation, duration, sample_rate)
        demodulated_freq = wow_flutter_analyzer.demodulate_frequency(fm_signal, sample_rate, carrier_freq)
        
        self.assertIsNotNone(demodulated_freq)
        
        # Mean should be around carrier frequency
        # Skipping edges due to potential filter warm-up/Hilbert transform artifacts
        stable_demod_freq = demodulated_freq[int(0.2*sample_rate) : -int(0.2*sample_rate)]
        if len(stable_demod_freq) == 0:
            self.fail("Stable region for demodulation check is empty.")

        self.assertAlmostEqual(np.mean(stable_demod_freq), carrier_freq, delta=carrier_freq * 0.01) # within 1%

        # The deviation signal is demodulated_freq - carrier_freq
        deviation_signal = stable_demod_freq - carrier_freq
        
        # The peak deviation of this signal should be close to the input 'deviation'
        # This is a simplified check. A full check would involve FFT.
        self.assertAlmostEqual(np.max(deviation_signal), deviation, delta=deviation * 0.20) # within 20% (generous for simple check)
        self.assertAlmostEqual(np.min(deviation_signal), -deviation, delta=deviation * 0.20)

        # Check that the modulation frequency is present in the deviation signal
        fft_deviation = np.fft.fft(deviation_signal)
        fft_deviation_freq = np.fft.fftfreq(len(deviation_signal), d=1/sample_rate)
        
        # Find the peak frequency in the positive spectrum (excluding DC)
        positive_freq_indices = np.where(fft_deviation_freq > 0.1)[0] # Start search above 0.1 Hz
        if len(positive_freq_indices) > 0:
            dominant_mod_freq_idx = positive_freq_indices[np.argmax(np.abs(fft_deviation[positive_freq_indices]))]
            detected_mod_freq = fft_deviation_freq[dominant_mod_freq_idx]
            self.assertAlmostEqual(detected_mod_freq, mod_freq, delta=mod_freq * 0.15) # within 15%
        else:
            self.fail("Could not find dominant modulation frequency in deviation signal.")


    def test_apply_bandpass_filter_output(self):
        sample_rate = 1000
        duration = 2.0
        
        # Signal with components at 1Hz (out), 4Hz (in), 50Hz (in, edge), 150Hz (out)
        s1 = generate_sine_wave(1, duration, sample_rate)
        s4 = generate_sine_wave(4, duration, sample_rate) # Target, inside passband
        s50 = generate_sine_wave(50, duration, sample_rate) # Target, inside passband
        s150 = generate_sine_wave(150, duration, sample_rate)
        test_signal = s1 + s4 + s50 + s150
        
        # Filter to pass 3 Hz to 60 Hz
        lowcut = 3.0
        highcut = 60.0
        filtered_signal = wow_flutter_analyzer.apply_bandpass_filter(test_signal, lowcut, highcut, sample_rate, order=4)
        
        fft_filtered = np.fft.fft(filtered_signal)
        fft_freq = np.fft.fftfreq(len(filtered_signal), d=1/sample_rate)
        
        # Get magnitudes for positive frequencies
        half_point = len(fft_freq) // 2
        positive_freqs = fft_freq[:half_point]
        positive_mags = np.abs(fft_filtered[:half_point])

        freq_mag_map = {round(f,1): m for f,m in zip(positive_freqs, positive_mags)}

        # Assert that 4Hz and 50Hz components are strong
        self.assertTrue(freq_mag_map.get(4.0, 0) > np.max(positive_mags)*0.5) # 4Hz has significant energy
        self.assertTrue(freq_mag_map.get(50.0, 0) > np.max(positive_mags)*0.5) # 50Hz has significant energy
        
        # Assert that 1Hz and 150Hz components are attenuated
        # Attenuation depends on filter order and distance from cutoffs.
        # For a 4th order filter, expect significant attenuation.
        self.assertTrue(freq_mag_map.get(1.0, 0) < np.max(positive_mags)*0.1) # 1Hz attenuated
        # 150Hz is further out, should be more attenuated.
        # Need to find the magnitude at 150Hz. The key might not be exactly 150.0.
        idx_150 = np.argmin(np.abs(positive_freqs - 150.0))
        self.assertTrue(positive_mags[idx_150] < np.max(positive_mags)*0.05)


    def test_calculate_metrics_known_signal(self):
        sample_rate = 1000
        duration = 2.0
        ref_freq_for_calc = 3150.0
        
        # Create a synthetic frequency deviation signal: 5 Hz peak deviation at 4 Hz
        deviation_amplitude = 5.0 # Hz
        deviation_frequency = 4.0 # Hz
        # This is already a _deviation_ signal, so it's centered around 0.
        synthetic_deviation_hz = generate_sine_wave(deviation_frequency, duration, sample_rate, amplitude=deviation_amplitude)
        
        metrics = wow_flutter_analyzer.calculate_metrics(synthetic_deviation_hz, ref_freq_for_calc, "TestLabel")
        
        self.assertAlmostEqual(metrics["TestLabel Peak (Hz)"], deviation_amplitude, delta=0.01)
        self.assertAlmostEqual(metrics["TestLabel RMS (Hz)"], deviation_amplitude / np.sqrt(2), delta=0.01)
        
        expected_peak_percent = (deviation_amplitude / ref_freq_for_calc) * 100
        self.assertAlmostEqual(metrics["TestLabel Peak (%)"], expected_peak_percent, delta=0.01)
        
        expected_rms_percent = ( (deviation_amplitude / np.sqrt(2)) / ref_freq_for_calc) * 100
        self.assertAlmostEqual(metrics["TestLabel RMS (%)"], expected_rms_percent, delta=0.01)


if __name__ == '__main__':
    unittest.main()
