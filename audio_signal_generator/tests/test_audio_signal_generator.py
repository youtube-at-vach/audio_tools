import unittest
import os
import soundfile as sf
import numpy as np
import sys
import scipy.signal
from scipy.optimize import curve_fit

# Add the parent directory to the Python path to import the main script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from audio_signal_generator.audio_signal_generator import (
    generate_tone, save_tone_to_wav, get_wav_subtype,
    generate_square_wave, generate_triangle_wave, generate_sawtooth_wave,
    generate_sweep, generate_noise,
    apply_fade, generate_volume_sweep, apply_weighting # Added these
)

class TestAudioSignalGenerator(unittest.TestCase):

    def setUp(self):
        self.test_output_dir = "test_outputs"
        os.makedirs(self.test_output_dir, exist_ok=True)
        self.sample_rate = 48000
        self.duration = 1.0  # seconds
        self.frequency = 440  # Hz

    def tearDown(self):
        # Clean up created files
        for item in os.listdir(self.test_output_dir):
            os.remove(os.path.join(self.test_output_dir, item))
        os.rmdir(self.test_output_dir)

    def test_generate_simple_tone(self):
        filename = os.path.join(self.test_output_dir, "test_tone.wav")

        # 1. Test generate_tone function
        tone_signal = generate_tone(self.frequency, self.duration, self.sample_rate)

        # Check type and length
        self.assertIsInstance(tone_signal, np.ndarray)
        expected_length = int(self.sample_rate * self.duration)
        self.assertEqual(len(tone_signal), expected_length)

        # Check basic characteristics of a sine wave (e.g., values are within -1 and 1)
        self.assertTrue(np.all(tone_signal >= -1.0) and np.all(tone_signal <= 1.0))

        # Check a few points for approximate correctness (optional, can be more detailed)
        # For a 440Hz sine wave at 48000Hz, period is 48000/440 samples.
        # t = np.linspace(0, self.duration, expected_length, endpoint=False)
        # expected_signal = np.sin(2 * np.pi * self.frequency * t)
        # self.assertTrue(np.allclose(tone_signal, expected_signal, atol=1e-6)) # Potentially too strict due to linspace differences

        # 2. Test save_tone_to_wav function
        bit_depth = '16'
        dbfs = -6
        save_tone_to_wav(filename, tone_signal, self.sample_rate, bit_depth, dbfs=dbfs)

        # 3. Verify output file creation
        self.assertTrue(os.path.exists(filename))

        # 4. Verify WAV file properties
        info = sf.info(filename)
        self.assertEqual(info.samplerate, self.sample_rate)
        self.assertAlmostEqual(info.duration, self.duration, delta=0.01) # Allow small delta for duration
        self.assertEqual(info.channels, 1) # Assuming mono output for now
        self.assertEqual(info.format, 'WAV')

        # Check subtype based on bit_depth used in save_tone_to_wav
        expected_subtype = get_wav_subtype(bit_depth).split('_')[0] # e.g. PCM_16 -> PCM
        self.assertTrue(info.subtype.startswith(expected_subtype))


    def test_get_wav_subtype(self):
        self.assertEqual(get_wav_subtype('16'), 'PCM_16')
        self.assertEqual(get_wav_subtype('24'), 'PCM_24')
        self.assertEqual(get_wav_subtype('32'), 'PCM_32')
        self.assertEqual(get_wav_subtype('float'), 'FLOAT')
        with self.assertRaises(ValueError):
            get_wav_subtype('invalid')

    def test_generate_square_wave(self):
        filename = os.path.join(self.test_output_dir, "test_square.wav")
        wave_signal = generate_square_wave(self.frequency, self.duration, self.sample_rate)

        self.assertIsInstance(wave_signal, np.ndarray)
        expected_length = int(self.sample_rate * self.duration)
        self.assertEqual(len(wave_signal), expected_length)

        # Square wave values should be -1, 0, or 1
        self.assertTrue(np.all(np.isin(wave_signal, [-1.0, 0.0, 1.0])))

        save_tone_to_wav(filename, wave_signal, self.sample_rate, '16', dbfs=-6)
        self.assertTrue(os.path.exists(filename))
        info = sf.info(filename)
        self.assertEqual(info.samplerate, self.sample_rate)
        self.assertAlmostEqual(info.duration, self.duration, delta=0.01)

    def test_generate_triangle_wave(self):
        filename = os.path.join(self.test_output_dir, "test_triangle.wav")
        wave_signal = generate_triangle_wave(self.frequency, self.duration, self.sample_rate)

        self.assertIsInstance(wave_signal, np.ndarray)
        expected_length = int(self.sample_rate * self.duration)
        self.assertEqual(len(wave_signal), expected_length)

        # Triangle wave values should be between -1 and 1
        self.assertTrue(np.all(wave_signal >= -1.0) and np.all(wave_signal <= 1.0))

        save_tone_to_wav(filename, wave_signal, self.sample_rate, '16', dbfs=-6)
        self.assertTrue(os.path.exists(filename))
        info = sf.info(filename)
        self.assertEqual(info.samplerate, self.sample_rate)
        self.assertAlmostEqual(info.duration, self.duration, delta=0.01)

    def test_generate_sawtooth_wave_ramp_plus(self):
        filename = os.path.join(self.test_output_dir, "test_sawtooth_plus.wav")
        wave_signal = generate_sawtooth_wave(self.frequency, self.duration, self.sample_rate, ramp_type='ramp+')

        self.assertIsInstance(wave_signal, np.ndarray)
        expected_length = int(self.sample_rate * self.duration)
        self.assertEqual(len(wave_signal), expected_length)

        # Sawtooth wave values should be between -1 and 1
        self.assertTrue(np.all(wave_signal >= -1.0) and np.all(wave_signal <= 1.0))

        save_tone_to_wav(filename, wave_signal, self.sample_rate, '16', dbfs=-6)
        self.assertTrue(os.path.exists(filename))
        info = sf.info(filename)
        self.assertEqual(info.samplerate, self.sample_rate)
        self.assertAlmostEqual(info.duration, self.duration, delta=0.01)

    def test_generate_sawtooth_wave_ramp_minus(self):
        filename = os.path.join(self.test_output_dir, "test_sawtooth_minus.wav")
        wave_signal = generate_sawtooth_wave(self.frequency, self.duration, self.sample_rate, ramp_type='ramp-')

        self.assertIsInstance(wave_signal, np.ndarray)
        expected_length = int(self.sample_rate * self.duration)
        self.assertEqual(len(wave_signal), expected_length)

        # Sawtooth wave values should be between -1 and 1
        self.assertTrue(np.all(wave_signal >= -1.0) and np.all(wave_signal <= 1.0))

        save_tone_to_wav(filename, wave_signal, self.sample_rate, '16', dbfs=-6)
        self.assertTrue(os.path.exists(filename))
        info = sf.info(filename)
        self.assertEqual(info.samplerate, self.sample_rate)
        self.assertAlmostEqual(info.duration, self.duration, delta=0.01)

    def test_generate_sawtooth_wave_invalid_ramp(self):
        with self.assertRaises(ValueError):
            generate_sawtooth_wave(self.frequency, self.duration, self.sample_rate, ramp_type='invalid')

    def test_generate_linear_sine_sweep(self):
        filename = os.path.join(self.test_output_dir, "test_linear_sine_sweep.wav")
        start_freq = 20
        end_freq = 2000
        wave_signal = generate_sweep(start_freq, end_freq, self.duration, self.sample_rate, logarithmic=False, wave_type='sine')

        self.assertIsInstance(wave_signal, np.ndarray)
        expected_length = int(self.sample_rate * self.duration)
        self.assertEqual(len(wave_signal), expected_length)
        self.assertTrue(np.all(wave_signal >= -1.0) and np.all(wave_signal <= 1.0)) # Sine values

        save_tone_to_wav(filename, wave_signal, self.sample_rate, '16', dbfs=-6)
        self.assertTrue(os.path.exists(filename))
        info = sf.info(filename)
        self.assertEqual(info.samplerate, self.sample_rate)
        self.assertAlmostEqual(info.duration, self.duration, delta=0.01)

    def test_generate_logarithmic_sine_sweep(self):
        filename = os.path.join(self.test_output_dir, "test_log_sine_sweep.wav")
        start_freq = 20
        end_freq = 2000
        wave_signal = generate_sweep(start_freq, end_freq, self.duration, self.sample_rate, logarithmic=True, wave_type='sine')

        self.assertIsInstance(wave_signal, np.ndarray)
        expected_length = int(self.sample_rate * self.duration)
        self.assertEqual(len(wave_signal), expected_length)
        self.assertTrue(np.all(wave_signal >= -1.0) and np.all(wave_signal <= 1.0)) # Sine values

        save_tone_to_wav(filename, wave_signal, self.sample_rate, '16', dbfs=-6)
        self.assertTrue(os.path.exists(filename))
        info = sf.info(filename)
        self.assertEqual(info.samplerate, self.sample_rate)
        self.assertAlmostEqual(info.duration, self.duration, delta=0.01)

    def test_generate_linear_square_sweep(self):
        filename = os.path.join(self.test_output_dir, "test_linear_square_sweep.wav")
        start_freq = 20
        end_freq = 2000
        wave_signal = generate_sweep(start_freq, end_freq, self.duration, self.sample_rate, logarithmic=False, wave_type='square')

        self.assertIsInstance(wave_signal, np.ndarray)
        expected_length = int(self.sample_rate * self.duration)
        self.assertEqual(len(wave_signal), expected_length)
        # Square wave values should be -1, 0, or 1
        self.assertTrue(np.all(np.isin(wave_signal, [-1.0, 0.0, 1.0])))

        save_tone_to_wav(filename, wave_signal, self.sample_rate, '16', dbfs=-6)
        self.assertTrue(os.path.exists(filename))
        info = sf.info(filename)
        self.assertEqual(info.samplerate, self.sample_rate)
        self.assertAlmostEqual(info.duration, self.duration, delta=0.01)

    def test_generate_logarithmic_square_sweep(self):
        filename = os.path.join(self.test_output_dir, "test_log_square_sweep.wav")
        start_freq = 20
        end_freq = 2000
        wave_signal = generate_sweep(start_freq, end_freq, self.duration, self.sample_rate, logarithmic=True, wave_type='square')

        self.assertIsInstance(wave_signal, np.ndarray)
        expected_length = int(self.sample_rate * self.duration)
        self.assertEqual(len(wave_signal), expected_length)
        # Square wave values should be -1, 0, or 1
        self.assertTrue(np.all(np.isin(wave_signal, [-1.0, 0.0, 1.0])))

        save_tone_to_wav(filename, wave_signal, self.sample_rate, '16', dbfs=-6)
        self.assertTrue(os.path.exists(filename))
        info = sf.info(filename)
        self.assertEqual(info.samplerate, self.sample_rate)
        self.assertAlmostEqual(info.duration, self.duration, delta=0.01)

    def _test_noise_type(self, color, filename_suffix):
        filename = os.path.join(self.test_output_dir, f"test_noise_{filename_suffix}.wav")

        # Temporarily reduce duration for faster noise tests if needed, but use self.duration for consistency
        noise_signal = generate_noise(self.duration, self.sample_rate, color=color)

        self.assertIsInstance(noise_signal, np.ndarray)
        expected_length = int(self.sample_rate * self.duration)
        # For noise, length can sometimes be off by one due to FFT processing, allow small tolerance
        self.assertAlmostEqual(len(noise_signal), expected_length, delta=2)

        # Noise is normalized to -1 to 1
        self.assertTrue(np.all(noise_signal >= -1.0) and np.all(noise_signal <= 1.0), f"{color} noise not in range")
        # Ensure not all values are zero or constant
        self.assertTrue(np.std(noise_signal) > 1e-6, f"{color} noise has no variation")


        save_tone_to_wav(filename, noise_signal, self.sample_rate, '16', dbfs=-6)
        self.assertTrue(os.path.exists(filename))
        info = sf.info(filename)
        self.assertEqual(info.samplerate, self.sample_rate)
        self.assertAlmostEqual(info.duration, self.duration, delta=0.02) # Increased delta slightly for noise

    def test_generate_white_noise(self):
        self._test_noise_type('white', 'white')

    def test_generate_pink_noise(self):
        self._test_noise_type('pink', 'pink')

    def test_generate_grey_noise(self):
        self._test_noise_type('grey', 'grey')

    def test_generate_brown_noise(self):
        self._test_noise_type('brown', 'brown')

    def test_generate_red_noise(self):
        # Red noise is an alias for brown noise in the generator
        self._test_noise_type('red', 'red')

    def test_generate_blue_noise(self):
        self._test_noise_type('blue', 'blue')

    def test_generate_purple_noise(self):
        self._test_noise_type('purple', 'purple')

    def test_generate_violet_noise(self):
        # Violet noise is an alias for purple noise in the generator
        self._test_noise_type('violet', 'violet')

    def test_apply_fade(self):
        fade_duration = 0.1 # 100ms
        tone_signal = generate_tone(self.frequency, self.duration, self.sample_rate)
        original_max_abs = np.max(np.abs(tone_signal))

        faded_signal = apply_fade(tone_signal.copy(), self.sample_rate, fade_duration)

        self.assertIsInstance(faded_signal, np.ndarray)
        self.assertEqual(len(faded_signal), len(tone_signal))

        fade_samples = int(self.sample_rate * fade_duration)

        # Check fade-in: first few samples should be close to 0
        self.assertTrue(np.all(np.abs(faded_signal[:fade_samples//2]) < original_max_abs * 0.5)) # rough check
        self.assertAlmostEqual(faded_signal[0], 0.0, delta=1e-6)
        if fade_samples > 0 : # Ensure there are samples to check for fade-in increase
             self.assertTrue(np.abs(faded_signal[fade_samples-1]) > np.abs(faded_signal[0]))


        # Check fade-out: last few samples should be close to 0
        self.assertTrue(np.all(np.abs(faded_signal[-fade_samples//2:]) < original_max_abs * 0.5)) # rough check
        self.assertAlmostEqual(faded_signal[-1], 0.0, delta=1e-6)
        if fade_samples > 0: # Ensure there are samples to check for fade-out decrease
            self.assertTrue(np.abs(faded_signal[-fade_samples]) > np.abs(faded_signal[-1]))

        # Middle part should be largely unaffected if duration > 2*fade_duration
        if self.duration > 2 * fade_duration:
            middle_part_original = tone_signal[fade_samples:-fade_samples]
            middle_part_faded = faded_signal[fade_samples:-fade_samples]
            if len(middle_part_original) > 0: # Ensure there is a middle part
                 self.assertTrue(np.allclose(middle_part_faded, middle_part_original, atol=1e-5))

        # Test too short duration for fade
        short_duration_signal = generate_tone(self.frequency, fade_duration / 2, self.sample_rate)
        with self.assertRaises(ValueError):
            apply_fade(short_duration_signal, self.sample_rate, fade_duration)

    def test_generate_volume_sweep_linear(self):
        base_signal = np.ones(int(self.sample_rate * self.duration)) # Constant signal for easy envelope check
        swept_signal = generate_volume_sweep(base_signal.copy(), self.sample_rate, self.duration, logarithmic=False)

        self.assertIsInstance(swept_signal, np.ndarray)
        self.assertEqual(len(swept_signal), len(base_signal))

        # Linear sweep should go from approx 0 to 1 (or base_signal max)
        self.assertAlmostEqual(swept_signal[0], 0.0, delta=1e-6)
        self.assertAlmostEqual(swept_signal[-1], np.max(base_signal), delta=1e-2) # allow some tolerance for linspace end
        # Check if it's monotonically increasing
        # Differences between consecutive elements should be non-negative (allowing for small float inaccuracies)
        diffs = np.diff(swept_signal)
        self.assertTrue(np.all(diffs >= -1e-6))


    def test_generate_volume_sweep_logarithmic(self):
        base_signal = np.ones(int(self.sample_rate * self.duration)) # Constant signal
        swept_signal = generate_volume_sweep(base_signal.copy(), self.sample_rate, self.duration, logarithmic=True)

        self.assertIsInstance(swept_signal, np.ndarray)
        self.assertEqual(len(swept_signal), len(base_signal))

        # Log sweep starts near 0.001 (as per np.logspace(-3...)) and ends near 1
        self.assertAlmostEqual(swept_signal[0], 0.001 * np.max(base_signal), delta=1e-3)
        self.assertAlmostEqual(swept_signal[-1], np.max(base_signal), delta=1e-2)
        # Check if it's monotonically increasing
        diffs = np.diff(swept_signal)
        self.assertTrue(np.all(diffs >= -1e-6))


    def test_apply_weighting(self):
        # Using white noise as it has a broad spectrum
        input_signal = generate_noise(self.duration, self.sample_rate, color='white')

        for weight_type in ['A', 'B', 'C']:
            weighted_signal = apply_weighting(input_signal.copy(), self.sample_rate, weight_type)
            self.assertIsInstance(weighted_signal, np.ndarray)
            self.assertEqual(len(weighted_signal), len(input_signal))
            # Check that the signal has changed (RMS might be different, or at least values)
            # This is a basic check; true spectral shaping is harder to verify here
            self.assertFalse(np.allclose(input_signal, weighted_signal, atol=1e-5), f"Signal did not change for {weight_type}-weighting")
            # Check values are still reasonable (e.g. not all zero or NaN)
            self.assertTrue(np.std(weighted_signal) > 1e-9)


    def test_apply_invalid_weighting(self):
        input_signal = generate_tone(self.frequency, self.duration, self.sample_rate)
        with self.assertRaises(ValueError):
            apply_weighting(input_signal, self.sample_rate, 'X')

    def test_save_different_bit_depths(self):
        base_signal = generate_tone(self.frequency, self.duration, self.sample_rate)
        bit_depths_to_test = ['16', '24', '32', 'float']

        for bit_depth in bit_depths_to_test:
            filename = os.path.join(self.test_output_dir, f"test_bit_depth_{bit_depth}.wav")
            save_tone_to_wav(filename, base_signal.copy(), self.sample_rate, bit_depth, dbfs=-6)

            self.assertTrue(os.path.exists(filename))
            info = sf.info(filename)

            self.assertEqual(info.samplerate, self.sample_rate)
            self.assertAlmostEqual(info.duration, self.duration, delta=0.01)
            self.assertEqual(info.channels, 1) # Assuming mono
            self.assertEqual(info.format, 'WAV')

            expected_subtype_prefix = get_wav_subtype(bit_depth).split('_')[0]
            if bit_depth == 'float': # soundfile reports 'FLOAT' or 'DOUBLE' based on precision
                self.assertTrue(info.subtype == 'FLOAT' or info.subtype == 'DOUBLE')
            else:
                self.assertTrue(info.subtype.startswith(expected_subtype_prefix))

    def test_save_with_dbfs_scaling(self):
        base_signal_unscaled = generate_tone(self.frequency, self.duration, self.sample_rate)
        # Ensure base_signal_unscaled is normalized to peak at 1.0 for predictable scaling
        base_signal_normalized = base_signal_unscaled / np.max(np.abs(base_signal_unscaled))

        levels_to_test = [-3.0, -6.0, -12.0, 0.0] # 0.0 dBFS might clip if not careful, but generator normalizes then scales

        for dbfs_level in levels_to_test:
            filename = os.path.join(self.test_output_dir, f"test_dbfs_{abs(dbfs_level)}.wav")
            # Use 'float' bit_depth for easier verification of amplitude
            save_tone_to_wav(filename, base_signal_normalized.copy(), self.sample_rate, 'float', dbfs=dbfs_level)

            self.assertTrue(os.path.exists(filename))
            info = sf.info(filename)
            self.assertEqual(info.subtype, 'FLOAT') # or DOUBLE, check as above
            self.assertTrue(info.subtype == 'FLOAT' or info.subtype == 'DOUBLE')


            # Read back the signal to check its peak amplitude
            saved_signal, read_sr = sf.read(filename, dtype='float32')
            self.assertEqual(read_sr, self.sample_rate)

            expected_amplitude = 10**(dbfs_level / 20.0)
            actual_max_amplitude = np.max(np.abs(saved_signal))

            # The save_tone_to_wav internally normalizes before applying dBFS,
            # so the peak of base_signal_normalized (which is 1.0) should scale to expected_amplitude.
            self.assertAlmostEqual(actual_max_amplitude, expected_amplitude, delta=0.01,
                                 msg=f"Amplitude mismatch for {dbfs_level} dBFS: expected {expected_amplitude}, got {actual_max_amplitude}")

    def test_save_tone_invalid_bit_depth(self):
        base_signal = generate_tone(self.frequency, self.duration, self.sample_rate)
        filename = os.path.join(self.test_output_dir, "test_invalid_bit_depth.wav")
        with self.assertRaises(ValueError):
            save_tone_to_wav(filename, base_signal, self.sample_rate, '8', dbfs=-6)

    def _calculate_psd(self, signal, sample_rate):
        frequencies, psd = scipy.signal.welch(signal, fs=sample_rate, nperseg=min(len(signal), 4096), window='hann') # Increased nperseg for better resolution
        valid_indices = (frequencies > 1e-1) & (psd > 1e-15) # Avoid log(0) and filter noise in PSD, start freq slightly above DC
        if not np.any(valid_indices): # Handle case where no valid frequencies are found
            return np.array([]), np.array([])
        return frequencies[valid_indices], psd[valid_indices]

    def _fit_psd_slope(self, frequencies, psd):
        if len(frequencies) < 5 or len(psd) < 5: # Need a few points for a meaningful fit
            return np.nan
        try:
            log_f = np.log10(frequencies)
            log_psd = np.log10(psd)

            coeffs, _ = curve_fit(lambda x, m, c: m*x + c, log_f, log_psd)
            m = coeffs[0] # m is the slope in log10(PSD) vs log10(f) units
            # slope_db_octave = m * 10 * log10(2)
            # Power ratio for octave: P2/P1 = (f2/f1)^m = 2^m
            # dB for octave: 10 * log10(2^m) = 10 * m * log10(2)
            slope_db_octave = m * 10 * np.log10(2.0)
            return slope_db_octave
        except Exception as e:
            # print(f"Curve fit failed: {e}") # Optional: for debugging
            return np.nan

    def test_generate_pink_noise_spectrum(self):
        color = 'pink'
        # Using a longer duration and specific sample rate for more reliable spectral analysis
        test_duration = 3.0
        test_sample_rate = 48000
        noise_signal = generate_noise(test_duration, test_sample_rate, color=color)

        self.assertIsInstance(noise_signal, np.ndarray)
        expected_length = int(test_sample_rate * test_duration)
        self.assertAlmostEqual(len(noise_signal), expected_length, delta=2)

        frequencies, psd = self._calculate_psd(noise_signal, test_sample_rate)
        self.assertTrue(len(frequencies) > 10, f"PSD calculation yielded too few points for {color} noise ({len(frequencies)} points).")

        slope_db_octave = self._fit_psd_slope(frequencies, psd)
        self.assertFalse(np.isnan(slope_db_octave), f"Slope calculation failed for {color} noise.")
        # Pink noise: -3dB/octave. Tolerance: -2 to -4 dB/octave.
        self.assertTrue(-4.0 < slope_db_octave < -2.0,
                        f"{color} noise slope {slope_db_octave:.2f} dB/octave out of range (-4 to -2).")

    def test_generate_brown_noise_spectrum(self):
        color = 'brown'
        test_duration = 3.0
        test_sample_rate = 48000
        noise_signal = generate_noise(test_duration, test_sample_rate, color=color)

        self.assertIsInstance(noise_signal, np.ndarray)
        expected_length = int(test_sample_rate * test_duration)
        self.assertAlmostEqual(len(noise_signal), expected_length, delta=2)

        frequencies, psd = self._calculate_psd(noise_signal, test_sample_rate)
        self.assertTrue(len(frequencies) > 10, f"PSD calculation yielded too few points for {color} noise ({len(frequencies)} points).")

        slope_db_octave = self._fit_psd_slope(frequencies, psd)
        self.assertFalse(np.isnan(slope_db_octave), f"Slope calculation failed for {color} noise.")
        # Brown noise (1/f^2): -6dB/octave. Tolerance: -5 to -7 dB/octave.
        self.assertTrue(-7.0 < slope_db_octave < -5.0,
                        f"{color} noise slope {slope_db_octave:.2f} dB/octave out of range (-7 to -5).")

    def test_generate_blue_noise_spectrum(self):
        color = 'blue'
        test_duration = 3.0
        test_sample_rate = 48000
        noise_signal = generate_noise(test_duration, test_sample_rate, color=color)

        self.assertIsInstance(noise_signal, np.ndarray)
        expected_length = int(test_sample_rate * test_duration)
        self.assertAlmostEqual(len(noise_signal), expected_length, delta=2)

        frequencies, psd = self._calculate_psd(noise_signal, test_sample_rate)
        self.assertTrue(len(frequencies) > 10, f"PSD calculation yielded too few points for {color} noise ({len(frequencies)} points).")

        slope_db_octave = self._fit_psd_slope(frequencies, psd)
        self.assertFalse(np.isnan(slope_db_octave), f"Slope calculation failed for {color} noise.")
        # Blue noise (f^1): +3dB/octave. Tolerance: +2 to +4 dB/octave.
        self.assertTrue(2.0 < slope_db_octave < 4.0,
                        f"{color} noise slope {slope_db_octave:.2f} dB/octave out of range (+2 to +4).")

    def test_generate_invalid_noise_color(self):
        with self.assertRaisesRegex(ValueError, "Unknown noise color: invalid_color"):
            generate_noise(self.duration, self.sample_rate, color='invalid_color')

if __name__ == '__main__':
    unittest.main()
