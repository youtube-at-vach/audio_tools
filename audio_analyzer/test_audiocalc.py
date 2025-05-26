import unittest
import numpy as np
from audio_analyzer.audiocalc import AudioCalc # Assuming audiocalc.py is in the same directory or accessible

class TestAudioCalc(unittest.TestCase):

    def test_analyze_harmonics_with_sinad(self):
        # 1. Generate a simple test signal
        sample_rate = 48000
        duration = 1.0
        fundamental_freq = 1000
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Pure sine wave for basic signal component
        signal_component = 0.5 * np.sin(2 * np.pi * fundamental_freq * t)
        
        # Add some noise
        noise_component = 0.05 * np.random.randn(len(t))
        
        # Add a harmonic (e.g., 2nd harmonic at 2000 Hz)
        harmonic_component = 0.1 * np.sin(2 * np.pi * 2 * fundamental_freq * t)
        
        audio_data = signal_component + noise_component + harmonic_component
        
        # 2. Call AudioCalc.analyze_harmonics
        analysis_results = AudioCalc.analyze_harmonics(
            audio_data=audio_data,
            fundamental_freq=fundamental_freq,
            window_name='hann', # Using hann window as it's common
            sampling_rate=sample_rate,
            min_db=-140.0
        )
        
        # 3. Assertions
        self.assertIn('sinad_db', analysis_results, "SINAD(dB) should be in analysis results")
        self.assertIn('thdn_db', analysis_results, "THD+N(dB) should be in analysis results")
        
        # Key assertion: SINAD(dB) = -THD+N(dB)
        if analysis_results['thdn_db'] is not None and analysis_results['sinad_db'] is not None:
            self.assertAlmostEqual(analysis_results['sinad_db'], -analysis_results['thdn_db'], places=5,
                                 msg="SINAD(dB) should be the negative of THD+N(dB)")
        else:
            # If one is None, the other should ideally also reflect a similar state or this test might need adjustment
            # For now, fail if they are not both valid numbers for the core check.
            self.fail("THD+N(dB) or SINAD(dB) is None, cannot perform core assertion.")

        # Basic check for other values (not exhaustive, but good for sanity)
        self.assertIn('thd_percent', analysis_results)
        self.assertIsNotNone(analysis_results['thd_percent'])
        self.assertIn('thdn_percent', analysis_results)
        self.assertIsNotNone(analysis_results['thdn_percent'])
        self.assertIn('basic_wave', analysis_results)
        self.assertIsNotNone(analysis_results['basic_wave']['amplitude_dbfs'])

    # You can add more test methods here for other functionalities of AudioCalc if needed.

if __name__ == '__main__':
    unittest.main()
