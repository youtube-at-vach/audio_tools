import sys
import os
import numpy as np
import unittest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.analysis import AudioCalc

class TestFrequencyCounter(unittest.TestCase):
    def test_optimize_frequency(self):
        sr = 48000
        duration = 0.1
        t = np.arange(int(sr * duration)) / sr
        
        # Test Case 1: Exact bin frequency
        target_freq = 1000.0
        signal = np.sin(2 * np.pi * target_freq * t)
        
        estimated = AudioCalc.optimize_frequency(signal, sr, 1000.0)
        print(f"Target: {target_freq}, Estimated: {estimated}")
        self.assertAlmostEqual(estimated, target_freq, places=4)
        
        # Test Case 2: Off-bin frequency
        target_freq = 1234.5678
        signal = np.sin(2 * np.pi * target_freq * t)
        
        # Initial guess from FFT would be close
        guess = 1230.0 
        estimated = AudioCalc.optimize_frequency(signal, sr, guess)
        print(f"Target: {target_freq}, Estimated: {estimated}")
        self.assertAlmostEqual(estimated, target_freq, places=4)
        
        # Test Case 3: With Noise
        noise = np.random.normal(0, 0.01, len(signal)) # -40dB noise
        signal_noisy = signal + noise
        
        estimated = AudioCalc.optimize_frequency(signal_noisy, sr, guess)
        print(f"Target: {target_freq} (Noisy), Estimated: {estimated}")
        self.assertAlmostEqual(estimated, target_freq, places=2) # Lower precision with noise

if __name__ == '__main__':
    unittest.main()
