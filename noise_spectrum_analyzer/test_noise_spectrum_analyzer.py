#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from scipy.io import wavfile
import os
import io
from contextlib import redirect_stdout

from noise_spectrum_analyzer.noise_spectrum_analyzer import analyze_noise_spectrum

class TestNoiseSpectrumAnalyzer(unittest.TestCase):

    def setUp(self):
        self.fs = 48000
        self.duration = 10
        self.samples = int(self.fs * self.duration)
        self.test_wav_file = 'test_noise_mix.wav'

    def tearDown(self):
        if os.path.exists(self.test_wav_file):
            os.remove(self.test_wav_file)

    def generate_pink_noise(self, n_samples):
        white_noise = np.random.normal(0, 1, n_samples)
        fft_white = np.fft.rfft(white_noise)
        frequencies = np.fft.rfftfreq(n_samples, 1/self.fs)
        # Scale FFT by 1/sqrt(f) for 1/f power spectrum (pink noise)
        with np.errstate(divide='ignore', invalid='ignore'):
            fft_pink = fft_white / np.sqrt(frequencies)
        fft_pink[0] = 0 # Set DC to zero
        pink_noise = np.fft.irfft(fft_pink)
        return pink_noise / np.sqrt(np.mean(pink_noise**2)) # Normalize to RMS=1

    def test_full_noise_mix_analysis(self):
        # 1. Generate component signals
        white_noise_component = np.random.normal(0, 0.2, self.samples) # White noise at RMS ~0.2
        pink_noise_component = self.generate_pink_noise(self.samples) * 0.5 # Pink noise at RMS ~0.5
        
        t = np.linspace(0, self.duration, self.samples, False)
        hum_component = 0.1 * np.sin(2 * np.pi * 60 * t) # 60Hz hum at RMS ~0.07
        hum_component += 0.05 * np.sin(2 * np.pi * 180 * t) # 3rd harmonic

        # 2. Combine them into a single signal
        mixed_noise = white_noise_component + pink_noise_component + hum_component
        wavfile.write(self.test_wav_file, self.fs, mixed_noise.astype(np.float32))

        # 3. Analyze the file and capture output
        f = io.StringIO()
        with redirect_stdout(f):
            analyze_noise_spectrum(self.test_wav_file)
        output = f.getvalue()

        # 4. Assert that the key components are identified in the output
        self.assertIn("1/f Noise", output)
        self.assertIn("White Noise", output)
        self.assertIn("Power Line Noise (60Hz)", output)

        # Check that the total voltage is reasonable (sum of squares of RMS values)
        # Expected total RMS = sqrt(0.2^2 + 0.5^2 + 0.07^2) ~= sqrt(0.04 + 0.25 + 0.005) ~= 0.54
        # We extract the reported total voltage to check if it's in the right ballpark.
        try:
            reported_voltage_uv = float(output.split('uV RMS')[0].split('Total Noise Voltage:')[1].strip())
            reported_voltage = reported_voltage_uv / 1e6
            expected_rms_approx = np.sqrt(0.2**2 + 0.5**2 + (0.1/np.sqrt(2))**2 + (0.05/np.sqrt(2))**2)
            # Check if the reported total RMS is within 30% of the expected value
            self.assertAlmostEqual(reported_voltage, expected_rms_approx, delta=expected_rms_approx * 0.3)
        except (ValueError, IndexError) as e:
            self.fail(f"Could not parse total voltage from output. Error: {e}\nOutput was:\n{output}")

if __name__ == '__main__':
    unittest.main()
