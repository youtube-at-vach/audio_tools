import unittest
import numpy as np
import scipy.signal
import soundfile as sf
import os
from wow_flutter_analyzer import wow_flutter_analyzer # Import the module to be tested

# Helper function to generate a sine wave
def generate_sine_wave(frequency, duration, sample_rate, amplitude=0.5):
    """Generates a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)

class TestAudioLoading(unittest.TestCase):
    """Tests for the load_audio function."""
    def setUp(self):
        """Set up for test methods."""
        self.test_dir = "test_audio_files"
        os.makedirs(self.test_dir, exist_ok=True)
        self.sample_rate = 44100
        self.duration = 1.0
        self.frequency = 1000.0

    def tearDown(self):
        """Tear down after test methods."""
        for f in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f))
        os.rmdir(self.test_dir)

    def test_load_mono_wav(self):
        """Test loading a mono WAV file."""
        filepath = os.path.join(self.test_dir, "mono.wav")
        data = generate_sine_wave(self.frequency, self.duration, self.sample_rate)
        sf.write(filepath, data, self.sample_rate, format='WAV', subtype='PCM_16')

        loaded_data, loaded_sr = wow_flutter_analyzer.load_audio(filepath)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_sr, self.sample_rate)
        self.assertEqual(loaded_data.ndim, 1)
        self.assertEqual(len(loaded_data), int(self.sample_rate * self.duration))
        np.testing.assert_allclose(loaded_data, data, atol=1e-4) # sf may slightly alter data

    def test_load_stereo_wav_to_mono(self):
        """Test loading a stereo WAV file, expecting conversion to mono."""
        filepath = os.path.join(self.test_dir, "stereo.wav")
        data_ch1 = generate_sine_wave(self.frequency, self.duration, self.sample_rate)
        data_ch2 = generate_sine_wave(self.frequency / 2, self.duration, self.sample_rate, amplitude=0.3)
        stereo_data = np.vstack((data_ch1, data_ch2)).T # Create a 2-channel array
        sf.write(filepath, stereo_data, self.sample_rate, format='WAV', subtype='PCM_16')

        loaded_data, loaded_sr = wow_flutter_analyzer.load_audio(filepath)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_sr, self.sample_rate)
        self.assertEqual(loaded_data.ndim, 1) # Should be mixed down to mono
        self.assertEqual(len(loaded_data), int(self.sample_rate * self.duration))
        
        # Expected mono data is the average of the two channels
        expected_mono_data = np.mean(stereo_data, axis=1)
        np.testing.assert_allclose(loaded_data, expected_mono_data, atol=1e-4)

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        loaded_data, loaded_sr = wow_flutter_analyzer.load_audio("non_existent_file.wav")
        self.assertIsNone(loaded_data)
        self.assertIsNone(loaded_sr)


class TestFrequencyDetection(unittest.TestCase):
    """Tests for the find_fundamental_frequency function."""
    def setUp(self):
        """Set up basic parameters for tests."""
        self.sample_rate = 48000
        self.duration = 2.0 # seconds
        self.test_freq_3150 = 3150.0

    def test_find_pure_tone(self):
        """Test finding a pure sine wave frequency."""
        audio_data = generate_sine_wave(self.test_freq_3150, self.duration, self.sample_rate)
        detected_freq = wow_flutter_analyzer.find_fundamental_frequency(audio_data, self.sample_rate, self.test_freq_3150)
        self.assertIsNotNone(detected_freq)
        np.testing.assert_allclose(detected_freq, self.test_freq_3150, rtol=1e-2) # Allow for some deviation due to Welch windowing

    def test_find_tone_with_noise(self):
        """Test finding a sine wave frequency in the presence of noise."""
        pure_signal = generate_sine_wave(self.test_freq_3150, self.duration, self.sample_rate, amplitude=0.5)
        noise = np.random.normal(0, 0.1, int(self.sample_rate * self.duration)) # Gaussian noise
        audio_data = pure_signal + noise
        
        detected_freq = wow_flutter_analyzer.find_fundamental_frequency(audio_data, self.sample_rate, self.test_freq_3150)
        self.assertIsNotNone(detected_freq)
        np.testing.assert_allclose(detected_freq, self.test_freq_3150, rtol=1e-2)

    def test_no_clear_peak_in_window(self):
        """Test behavior when no clear peak is in the expected window (should find overall peak)."""
        # Signal with a frequency far from the expected one
        actual_freq = 100.0
        audio_data = generate_sine_wave(actual_freq, self.duration, self.sample_rate)
        # Expecting 3150 Hz, but signal is 100 Hz
        detected_freq = wow_flutter_analyzer.find_fundamental_frequency(audio_data, self.sample_rate, self.test_freq_3150)
        self.assertIsNotNone(detected_freq)
        # It should find the actual peak (100 Hz) as it's outside the narrow search for 3150 Hz
        np.testing.assert_allclose(detected_freq, actual_freq, rtol=1e-2)

    def test_frequency_at_edge_of_window(self):
        """Test finding a frequency that's at the edge of the Welch analysis window."""
        # Use a frequency that might be tricky for some FFT bin resolutions
        edge_freq = 3150.0 * 0.905 # Just inside the 0.9 to 1.1 window
        audio_data = generate_sine_wave(edge_freq, self.duration, self.sample_rate)
        detected_freq = wow_flutter_analyzer.find_fundamental_frequency(audio_data, self.sample_rate, self.test_freq_3150)
        self.assertIsNotNone(detected_freq)
        np.testing.assert_allclose(detected_freq, edge_freq, rtol=1e-2)


class TestFrequencyVariationTracking(unittest.TestCase):
    """Tests for the track_frequency_variation function."""
    def setUp(self):
        self.sample_rate = 48000
        self.duration = 2.0  # seconds
        self.nominal_freq = 3150.0
        # STFT parameters from the main script (can be adjusted if needed for tests)
        self.block_size_ms = 50 
        self.hop_size_ms = 10
        self.stft_window_nperseg = int(self.sample_rate * self.block_size_ms / 1000)
        self.stft_hop_samples = int(self.sample_rate * self.hop_size_ms / 1000)
        # Expected number of time points in STFT output
        self.expected_num_t_points = ( (len(generate_sine_wave(100, self.duration, self.sample_rate)) - self.stft_window_nperseg) // self.stft_hop_samples ) + 1


    def test_no_frequency_variation(self):
        """Test with a pure sine wave (no actual frequency variation)."""
        audio_data = generate_sine_wave(self.nominal_freq, self.duration, self.sample_rate)
        
        deviations, time_axis = wow_flutter_analyzer.track_frequency_variation(
            audio_data, self.sample_rate, self.nominal_freq, 
            block_size_ms=self.block_size_ms, hop_size_ms=self.hop_size_ms
        )
        
        self.assertIsNotNone(deviations)
        self.assertIsNotNone(time_axis)
        self.assertEqual(len(deviations), len(time_axis))
        # With a pure tone, deviations should be very close to zero.
        # STFT peak finding might have small errors.
        np.testing.assert_allclose(deviations, 0, atol=1.0) # Allow up to 1 Hz error due to STFT resolution/estimation

    def test_slow_sine_fm_variation(self):
        """Test with a slow sinusoidal frequency modulation (Wow-like)."""
        mod_freq = 2.0  # Hz (slow variation)
        mod_depth = 10.0 # Hz (peak deviation)
        
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        # Instantaneous frequency: f_nominal + mod_depth * sin(2*pi*mod_freq*t)
        # To generate the signal, we need the integral of the instantaneous frequency (phase)
        phase = 2 * np.pi * self.nominal_freq * t + (mod_depth / mod_freq) * (1 - np.cos(2 * np.pi * mod_freq * t))
        audio_data = 0.5 * np.sin(phase)
        
        # Expected deviation pattern (simplified - actual STFT output is windowed)
        expected_deviation_pattern = mod_depth * np.sin(2 * np.pi * mod_freq * np.linspace(0, self.duration, self.expected_num_t_points))

        deviations, time_axis = wow_flutter_analyzer.track_frequency_variation(
            audio_data, self.sample_rate, self.nominal_freq,
            block_size_ms=self.block_size_ms, hop_size_ms=self.hop_size_ms
        )
        
        self.assertIsNotNone(deviations)
        self.assertEqual(len(deviations), len(time_axis))
        # Check if the shape of the deviation matches the slow sine wave
        # This is a qualitative check; exact values are hard to match perfectly due to STFT windowing and peak finding.
        # We expect the mean to be near zero for a symmetric FM.
        self.assertAlmostEqual(np.mean(deviations), 0, delta=mod_depth * 0.2) # Mean should be close to 0
        # Peak deviation should be in the ballpark of mod_depth
        self.assertLessEqual(np.max(np.abs(deviations)), mod_depth * 1.5) # Allow some overshoot/undershoot
        self.assertGreaterEqual(np.max(np.abs(deviations)), mod_depth * 0.5) # Should capture a good portion

    def test_linear_chirp_variation(self):
        """Test with a linear frequency chirp."""
        start_freq = self.nominal_freq - 50 # Start 50 Hz below nominal
        end_freq = self.nominal_freq + 50   # End 50 Hz above nominal
        
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        # Instantaneous frequency f(t) = start_freq + (end_freq - start_freq) * t / duration
        # Phase phi(t) = 2*pi * (start_freq * t + (end_freq - start_freq) * t^2 / (2 * duration))
        phase = 2 * np.pi * (start_freq * t + (end_freq - start_freq) * (t**2) / (2 * self.duration))
        audio_data = 0.5 * np.sin(phase)

        deviations, time_axis = wow_flutter_analyzer.track_frequency_variation(
            audio_data, self.sample_rate, self.nominal_freq,
            block_size_ms=self.block_size_ms, hop_size_ms=self.hop_size_ms
        )

        self.assertIsNotNone(deviations)
        self.assertEqual(len(deviations), len(time_axis))
        
        # Expected deviations should roughly go from -50 Hz to +50 Hz linearly.
        # Check start, middle, and end points (approximate due to STFT windowing)
        self.assertAlmostEqual(deviations[len(deviations)//10], start_freq - self.nominal_freq, delta=20) # Near start
        self.assertAlmostEqual(deviations[len(deviations)//2], 0, delta=20) # Near middle (should cross nominal_freq)
        self.assertAlmostEqual(deviations[-len(deviations)//10], end_freq - self.nominal_freq, delta=20) # Near end


class TestCalculateWowFlutter(unittest.TestCase):
    """Tests for the calculate_wow_flutter function."""
    def setUp(self):
        self.nominal_freq = 3150.0
        self.deviation_sample_rate = 100.0  # Hz
        self.duration = 2.0 # seconds
        self.num_deviation_points = int(self.deviation_sample_rate * self.duration)
        self.t_deviations = np.linspace(0, self.duration, self.num_deviation_points, endpoint=False)

    def test_zero_deviation(self):
        """Test with zero frequency deviation."""
        deviations = np.zeros(self.num_deviation_points)
        wow_pct, flutter_rms_pct = wow_flutter_analyzer.calculate_wow_flutter(
            deviations, self.nominal_freq, self.deviation_sample_rate
        )
        self.assertAlmostEqual(wow_pct, 0.0, places=5)
        self.assertAlmostEqual(flutter_rms_pct, 0.0, places=5)

    def test_pure_wow_signal(self):
        """Test with a pure 'wow' signal (slow sinusoidal deviation)."""
        wow_freq = 1.0  # 1 Hz, clearly in wow range
        peak_deviation_hz = self.nominal_freq * 0.005 # 0.5% peak deviation
        
        # Create a deviation signal that is purely 'wow'
        deviations = peak_deviation_hz * np.sin(2 * np.pi * wow_freq * self.t_deviations)
        
        # Manually calculate expected Wow (peak-to-peak)
        # The low-pass filter in calculate_wow_flutter should pass this frequency with minimal attenuation
        expected_wow_peak_to_peak_hz = 2 * peak_deviation_hz
        expected_wow_pct = (expected_wow_peak_to_peak_hz / self.nominal_freq) * 100
        
        wow_pct, flutter_rms_pct = wow_flutter_analyzer.calculate_wow_flutter(
            deviations, self.nominal_freq, self.deviation_sample_rate, wow_cutoff_hz=4.0
        )
        
        self.assertAlmostEqual(wow_pct, expected_wow_pct, delta=expected_wow_pct*0.1) # Allow 10% tolerance due to filter
        # Flutter should be very low as the signal is mostly below the 4Hz cutoff for flutter.
        # Some energy might leak through the high-pass filter if its rolloff is not steep enough.
        self.assertLess(flutter_rms_pct, expected_wow_pct * 0.1) # Expect flutter to be significantly less than wow

    def test_pure_flutter_signal(self):
        """Test with a pure 'flutter' signal (faster sinusoidal deviation)."""
        flutter_freq = 10.0  # 10 Hz, clearly in flutter range (above 4Hz default wow cutoff)
        peak_deviation_hz = self.nominal_freq * 0.001 # 0.1% peak deviation
        
        deviations = peak_deviation_hz * np.sin(2 * np.pi * flutter_freq * self.t_deviations)
        
        # Manually calculate expected RMS Flutter
        # The high-pass filter (implicitly, by being > wow_cutoff) should pass this.
        # RMS of a sine wave is Amplitude / sqrt(2)
        expected_flutter_rms_hz = peak_deviation_hz / np.sqrt(2)
        expected_flutter_rms_pct = (expected_flutter_rms_hz / self.nominal_freq) * 100
        
        wow_pct, flutter_rms_pct = wow_flutter_analyzer.calculate_wow_flutter(
            deviations, self.nominal_freq, self.deviation_sample_rate, wow_cutoff_hz=4.0
        )
        
        self.assertAlmostEqual(flutter_rms_pct, expected_flutter_rms_pct, delta=expected_flutter_rms_pct*0.15) # Allow 15% tol
        # Wow should be very low
        self.assertLess(wow_pct, expected_flutter_rms_pct * 0.5) # Wow should be less than flutter for this signal

    def test_wow_cutoff_effect(self):
        """Test how changing wow_cutoff_hz affects results."""
        # A signal at 3 Hz.
        test_freq = 3.0
        peak_deviation_hz = self.nominal_freq * 0.002 # 0.2% peak deviation
        deviations = peak_deviation_hz * np.sin(2 * np.pi * test_freq * self.t_deviations)

        # Case 1: wow_cutoff_hz = 2.0 Hz (test_freq should be mostly flutter)
        wow1, flutter1 = wow_flutter_analyzer.calculate_wow_flutter(
            deviations, self.nominal_freq, self.deviation_sample_rate, wow_cutoff_hz=2.0
        )
        
        # Case 2: wow_cutoff_hz = 4.0 Hz (test_freq should be mostly wow)
        wow2, flutter2 = wow_flutter_analyzer.calculate_wow_flutter(
            deviations, self.nominal_freq, self.deviation_sample_rate, wow_cutoff_hz=4.0
        )
        
        self.assertGreater(flutter1, wow1) # At 2Hz cutoff, 3Hz signal is flutter
        self.assertGreater(wow2, flutter2) # At 4Hz cutoff, 3Hz signal is wow
        # Also check that total "energy" (not strictly, but magnitude) is somewhat preserved
        self.assertAlmostEqual(wow1 + flutter1, wow2 + flutter2, delta=(wow1+flutter1)*0.2)


class TestCalculateWRMSFlutter(unittest.TestCase):
    """Tests for the calculate_wrms_flutter (placeholder) function."""
    def setUp(self):
        self.nominal_freq = 3150.0
        # Adjusted deviation_sample_rate to be more appropriate for a 200Hz high cutoff
        # Nyquist for deviations will be 500Hz / 2 = 250Hz.
        # This allows the 0.5Hz-200Hz bandpass filter to operate without cutoff adjustments.
        self.deviation_sample_rate = 500.0  
        self.duration = 2.0 # seconds
        self.num_deviation_points = int(self.deviation_sample_rate * self.duration)
        self.t_deviations = np.linspace(0, self.duration, self.num_deviation_points, endpoint=False)

    def test_zero_deviation_wrms(self):
        """Test WRMS with zero frequency deviation."""
        deviations = np.zeros(self.num_deviation_points)
        wrms_pct = wow_flutter_analyzer.calculate_wrms_flutter(
            deviations, self.nominal_freq, self.deviation_sample_rate
        )
        self.assertAlmostEqual(wrms_pct, 0.0, places=5)

    def test_sine_deviation_in_passband_wrms(self):
        """Test WRMS with a sine deviation that should be in the placeholder filter's passband."""
        # Placeholder filter is 0.5 Hz to 200 Hz. Let's use a 10 Hz signal.
        test_flutter_freq = 10.0 # Hz
        peak_deviation_hz = self.nominal_freq * 0.001 # 0.1% peak deviation
        
        deviations = peak_deviation_hz * np.sin(2 * np.pi * test_flutter_freq * self.t_deviations)
        
        # Expected RMS of this signal (if filter passes it perfectly)
        expected_rms_hz = peak_deviation_hz / np.sqrt(2)
        expected_wrms_pct = (expected_rms_hz / self.nominal_freq) * 100
        
        wrms_pct = wow_flutter_analyzer.calculate_wrms_flutter(
            deviations, self.nominal_freq, self.deviation_sample_rate
        )
        
        # The placeholder filter is a simple Butterworth. It won't have a perfectly flat passband.
        # We expect the result to be close to the input RMS, but allow some tolerance.
        # Increased delta slightly to account for real filter behavior.
        self.assertAlmostEqual(wrms_pct, expected_wrms_pct, delta=expected_wrms_pct * 0.30) 
        self.assertGreater(wrms_pct, 0) # Should be non-zero

    def test_sine_deviation_outside_passband_wrms(self):
        """Test WRMS with sine deviations mostly outside the placeholder filter's passband."""
        peak_deviation_hz = self.nominal_freq * 0.001 

        # Low frequency (e.g., 0.1 Hz, below 0.5 Hz cutoff of the placeholder)
        low_flutter_freq = 0.1 # Hz
        deviations_low = peak_deviation_hz * np.sin(2 * np.pi * low_flutter_freq * self.t_deviations)
        wrms_pct_low = wow_flutter_analyzer.calculate_wrms_flutter(
            deviations_low, self.nominal_freq, self.deviation_sample_rate
        )
        
        # High frequency (e.g., 220 Hz, above 200 Hz cutoff of the placeholder)
        # Nyquist is 250 Hz, so 220 Hz is valid.
        high_flutter_freq = 220.0 # Hz
        deviations_high = peak_deviation_hz * np.sin(2 * np.pi * high_flutter_freq * self.t_deviations)
        wrms_pct_high = wow_flutter_analyzer.calculate_wrms_flutter(
            deviations_high, self.nominal_freq, self.deviation_sample_rate
        )

        # A signal well within the passband (e.g., 10Hz from previous test)
        mid_flutter_freq = 10.0 # Hz
        deviations_mid = peak_deviation_hz * np.sin(2 * np.pi * mid_flutter_freq * self.t_deviations)
        wrms_pct_mid = wow_flutter_analyzer.calculate_wrms_flutter(
            deviations_mid, self.nominal_freq, self.deviation_sample_rate
        )
        
        # Expect significant attenuation for signals outside the 0.5-200Hz passband
        self.assertLess(wrms_pct_low, wrms_pct_mid * 0.2, "Low frequency not attenuated enough.") 
        self.assertLess(wrms_pct_high, wrms_pct_mid * 0.2, "High frequency not attenuated enough.")
        self.assertGreater(wrms_pct_low, 0) # May not be zero due to filter skirts
        self.assertGreater(wrms_pct_high, 0)


if __name__ == "__main__":
    unittest.main()
