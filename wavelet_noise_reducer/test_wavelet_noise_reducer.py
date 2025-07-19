
import unittest
import numpy as np
from wavelet_noise_reducer import WaveletNoiseReducer

class TestWaveletNoiseReducer(unittest.TestCase):

    def test_noise_reduction(self):
        samplerate = 44100
        duration = 1
        t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)

        # Clean signal: low frequency sine wave
        clean_signal = 0.5 * np.sin(2. * np.pi * 100 * t)

        # Noise: high frequency sine wave
        noise = 0.1 * np.sin(2. * np.pi * 5000 * t)
        noisy_signal = clean_signal + noise

        # Apply noise reduction
        reducer = WaveletNoiseReducer(wavelet_name='db8', levels=8)
        # Assuming high frequency noise is in higher detail levels
        # Mask: [A, D1, D2, D3, D4, D5, D6, D7, D8]
        # We want to keep low frequency (A) and some lower detail levels, zero out higher detail levels
        # This mask needs to be adjusted based on the actual wavelet decomposition and noise characteristics
        threshold_mask = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0] # Example: keep first 4 detail levels, zero out rest
        denoised_signal = reducer.reduce_noise(noisy_signal, threshold_mask)

        # Ensure the denoised signal has the same length as the original clean signal
        self.assertEqual(len(denoised_signal), len(clean_signal))

        # Assert that the denoised signal is closer to the clean signal
        rmse_noisy = np.sqrt(np.mean((noisy_signal - clean_signal) ** 2))
        rmse_denoised = np.sqrt(np.mean((denoised_signal - clean_signal) ** 2))

        self.assertLess(rmse_denoised, rmse_noisy, "Denoised signal should be closer to the clean signal")

if __name__ == '__main__':
    unittest.main()
