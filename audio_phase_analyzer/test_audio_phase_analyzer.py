import unittest
import numpy as np
# Assuming audio_phase_analyzer.py is in the same directory and __init__.py makes the package importable.
# If audio_phase_analyzer/ is a package, this import style is correct.
from audio_phase_analyzer.audio_phase_analyzer import calculate_phase_difference, generate_sine_wave

class TestAudioPhaseAnalyzer(unittest.TestCase):
    SAMPLE_RATE = 48000
    DURATION = 1.0  # seconds
    FREQUENCY = 1000  # Hz
    AMPLITUDE = 0.5 # Default amplitude for tests

    def _generate_test_stereo_signal(self, phase_diff_deg_ch2_minus_ch1, 
                                     sample_rate=None, duration=None, frequency=None, 
                                     amplitude=None, noise_level=0.0):
        """
        Helper function to generate a 2-channel stereo signal with a defined phase difference
        between channel 2 and channel 1.
        A positive phase_diff_deg_ch2_minus_ch1 means channel 2 leads channel 1.
        """
        sr = sample_rate if sample_rate is not None else self.SAMPLE_RATE
        dur = duration if duration is not None else self.DURATION
        freq = frequency if frequency is not None else self.FREQUENCY
        amp = amplitude if amplitude is not None else self.AMPLITUDE

        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        
        # Channel 1: Reference sine wave (phase = 0)
        ch1_signal = amp * np.sin(freq * t * 2 * np.pi)
        
        # Channel 2: Sine wave with specified phase difference relative to Channel 1
        # The phase_diff_rad is the phase of Ch2 - phase of Ch1.
        # So, Ch2 = sin(omega*t + phase_diff_rad)
        phase_diff_rad = np.deg2rad(phase_diff_deg_ch2_minus_ch1)
        ch2_signal = amp * np.sin(freq * t * 2 * np.pi + phase_diff_rad)

        if noise_level > 0:
            # Ensure noise is scaled by amplitude to be relative if amp is not 1.0
            # If amp is, say, 0.1, noise_level=0.05 means 0.005 absolute.
            actual_noise_std_dev_ch1 = noise_level * (amp if amp != 0 else 1.0) 
            actual_noise_std_dev_ch2 = noise_level * (amp if amp != 0 else 1.0)
            
            noise_ch1 = np.random.normal(0, actual_noise_std_dev_ch1, len(t))
            noise_ch2 = np.random.normal(0, actual_noise_std_dev_ch2, len(t))
            ch1_signal += noise_ch1
            ch2_signal += noise_ch2
            
        return np.column_stack((ch1_signal, ch2_signal))

    def test_zero_phase_difference(self):
        """Test with two identical signals (0 degrees phase difference)."""
        stereo_signal = self._generate_test_stereo_signal(phase_diff_deg_ch2_minus_ch1=0.0)
        phase = calculate_phase_difference(stereo_signal, self.SAMPLE_RATE, self.FREQUENCY)
        self.assertIsNotNone(phase, "Phase calculation returned None for zero phase difference.")
        self.assertAlmostEqual(phase, 0.0, places=1, msg="Phase should be close to 0.0 degrees.")

    def test_90_degree_phase_difference(self):
        """Test with Ch2 leading Ch1 by 90 degrees. Expect function to return Ch1-Ch2 = -90."""
        stereo_signal = self._generate_test_stereo_signal(phase_diff_deg_ch2_minus_ch1=90.0)
        phase = calculate_phase_difference(stereo_signal, self.SAMPLE_RATE, self.FREQUENCY)
        self.assertIsNotNone(phase, "Phase calculation returned None for +90 deg Ch2 lead.")
        self.assertAlmostEqual(phase, -90.0, places=1, msg="Phase should be close to -90.0 degrees for Ch2 leading Ch1 by 90.")

    def test_minus_90_degree_phase_difference(self):
        """Test with Ch2 lagging Ch1 by 90 degrees. Expect function to return Ch1-Ch2 = +90."""
        stereo_signal = self._generate_test_stereo_signal(phase_diff_deg_ch2_minus_ch1=-90.0)
        phase = calculate_phase_difference(stereo_signal, self.SAMPLE_RATE, self.FREQUENCY)
        self.assertIsNotNone(phase, "Phase calculation returned None for -90 deg Ch2 lag.")
        self.assertAlmostEqual(phase, 90.0, places=1, msg="Phase should be close to 90.0 degrees for Ch2 lagging Ch1 by 90.")
        
    def test_180_degree_phase_difference(self):
        """Test with signals 180 degrees out of phase (inverted)."""
        stereo_signal = self._generate_test_stereo_signal(phase_diff_deg_ch2_minus_ch1=180.0)
        phase = calculate_phase_difference(stereo_signal, self.SAMPLE_RATE, self.FREQUENCY)
        self.assertIsNotNone(phase, "Phase calculation returned None for 180 deg difference.")
        # Normalization might make it -180.0 or 180.0. abs() handles this.
        self.assertAlmostEqual(abs(phase), 180.0, places=1, msg="Absolute phase should be close to 180.0 degrees.")

    def test_minus_180_degree_phase_difference_explicit(self):
        """Test with signals -180 degrees out of phase (inverted)."""
        stereo_signal = self._generate_test_stereo_signal(phase_diff_deg_ch2_minus_ch1=-180.0)
        phase = calculate_phase_difference(stereo_signal, self.SAMPLE_RATE, self.FREQUENCY)
        self.assertIsNotNone(phase, "Phase calculation returned None for -180 deg difference.")
        self.assertAlmostEqual(abs(phase), 180.0, places=1, msg="Absolute phase should be close to 180.0 degrees for -180 input.")

    def test_phase_with_noise_zero_deg(self):
        """Test zero phase difference with added noise."""
        noise_level_fraction = 0.05 # 5% noise relative to amplitude
        stereo_signal = self._generate_test_stereo_signal(
            phase_diff_deg_ch2_minus_ch1=0.0, 
            noise_level=noise_level_fraction
        )
        phase = calculate_phase_difference(stereo_signal, self.SAMPLE_RATE, self.FREQUENCY)
        self.assertIsNotNone(phase, "Phase calculation returned None for noisy signal (0 deg target).")
        # Looser tolerance due to noise. Delta of 5 degrees.
        self.assertAlmostEqual(phase, 0.0, delta=5, msg="Phase should be close to 0.0 degrees even with 5% noise.")

    def test_phase_with_noise_90_deg(self):
        """Test with Ch2 leading Ch1 by 90 degrees, with added noise. Expect function to return Ch1-Ch2 = -90."""
        noise_level_fraction = 0.05 # 5% noise relative to amplitude
        stereo_signal = self._generate_test_stereo_signal(
            phase_diff_deg_ch2_minus_ch1=90.0, 
            noise_level=noise_level_fraction
        )
        phase = calculate_phase_difference(stereo_signal, self.SAMPLE_RATE, self.FREQUENCY)
        self.assertIsNotNone(phase, "Phase calculation returned None for noisy signal (+90 deg Ch2 lead target).")
        self.assertAlmostEqual(phase, -90.0, delta=5, msg="Phase should be close to -90.0 degrees even with 5% noise.")

    def test_short_signal_returns_none(self):
        """Test that short signals (too short for correlation) return None."""
        # Duration that gives < 10 samples (default min_length_for_correlation)
        # e.g., 5 samples: 5 / 48000 s
        short_duration = 5 / self.SAMPLE_RATE 
        stereo_signal = self._generate_test_stereo_signal(
            phase_diff_deg_ch2_minus_ch1=0.0, 
            duration=short_duration
        )
        # calculate_phase_difference prints a warning via Rich console and returns None.
        phase = calculate_phase_difference(stereo_signal, self.SAMPLE_RATE, self.FREQUENCY)
        self.assertIsNone(phase, "Phase should be None for very short signals.")

    def test_silent_signal_warning_and_result(self):
        """Test silent signals: expects a warning and a 0.0 phase result."""
        stereo_signal = self._generate_test_stereo_signal(
            phase_diff_deg_ch2_minus_ch1=0.0,
            amplitude=0.0 # Silent signal
        )
        phase = calculate_phase_difference(stereo_signal, self.SAMPLE_RATE, self.FREQUENCY)
        # With the change to return 0.0 for silent signals:
        self.assertIsNotNone(phase, "Phase calculation returned None for silent signal (expected 0.0).")
        self.assertAlmostEqual(phase, 0.0, places=1, 
                               msg="Phase for silent signal should be 0.0 after warning.")


if __name__ == '__main__':
    # This allows running the tests directly from the command line
    # `python audio_phase_analyzer/test_audio_phase_analyzer.py`
    # or more commonly with `python -m unittest audio_phase_analyzer.test_audio_phase_analyzer`
    # or `python -m unittest discover audio_phase_analyzer` from parent directory
    unittest.main()
