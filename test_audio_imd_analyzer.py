import unittest # Using unittest for structure and some assertions like assertRaises
import numpy as np
import scipy.signal
import sys

# Assuming audio_imd_analyzer.py is in the same directory or accessible via PYTHONPATH
try:
    from audio_imd_analyzer import generate_dual_tone, analyze_imd_smpte, dbfs_to_linear
except ImportError:
    # If running from a different context, or if the file isn't directly accessible,
    # this provides a way to inform the user. For the sandbox, it should be fine.
    sys.path.append('.') # Add current dir to path
    from audio_imd_analyzer import generate_dual_tone, analyze_imd_smpte, dbfs_to_linear


# Test Runner Storage
test_results = {}

class TestAudioImdAnalyzer(unittest.TestCase):

    def test_gdt_amplitudes_and_clipping(self):
        print("Running test_gdt_amplitudes_and_clipping...")
        sample_rate = 48000
        duration = 1.0
        f1, f2 = 60.0, 7000.0
        
        # Test 1: Normal operation
        amp1_dbfs = -12.0
        ratio_f1_f2 = 4.0
        signal = generate_dual_tone(f1, amp1_dbfs, f2, ratio_f1_f2, duration, sample_rate)
        
        expected_linear_amp1 = dbfs_to_linear(amp1_dbfs)
        expected_linear_amp2 = expected_linear_amp1 / ratio_f1_f2
        
        # Max possible amplitude is sum of individual amplitudes. Actual max might be less due to phase.
        self.assertTrue(np.max(np.abs(signal)) <= (expected_linear_amp1 + expected_linear_amp2 + 1e-9)) # Add tolerance
        self.assertTrue(np.max(np.abs(signal)) > 0) # Ensure signal is not all zeros
        self.assertTrue(np.max(np.abs(signal)) <= 1.0)

        # Test 2: Clipping scenario (sum of linear amplitudes > 1.0)
        amp1_dbfs_clip = 0.0 # amp1_linear = 1.0
        ratio_f1_f2_clip = 1.0 # amp2_linear = 1.0
        # sum = 2.0, should raise ValueError
        with self.assertRaises(ValueError):
            generate_dual_tone(f1, amp1_dbfs_clip, f2, ratio_f1_f2_clip, duration, sample_rate)

        # Test 3: Another clipping scenario (amp1_linear + amp2_linear slightly > 1.0)
        amp1_dbfs_clip_2 = -7.0 # approx 0.447 linear
        ratio_f1_f2_clip_2 = 1.0 # amp2_linear approx 0.447 linear
        # sum = 0.894, should pass
        try:
            generate_dual_tone(f1, amp1_dbfs_clip_2, f2, ratio_f1_f2_clip_2, duration, sample_rate)
        except ValueError:
            self.fail("generate_dual_tone raised ValueError unexpectedly for -7dBFS tones")

        amp1_dbfs_clip_3 = -5.0 # approx 0.562 linear
        ratio_f1_f2_clip_3 = 1.0 # amp2_linear approx 0.562 linear
        # sum = 1.124, should fail
        with self.assertRaises(ValueError):
            generate_dual_tone(f1, amp1_dbfs_clip_3, f2, ratio_f1_f2_clip_3, duration, sample_rate)
        
        test_results['test_gdt_amplitudes_and_clipping'] = "PASS"


    def test_gdt_frequencies(self):
        print("Running test_gdt_frequencies...")
        sample_rate = 48000
        duration = 1.0
        # Use frequencies that are not harmonically related and easy to distinguish in FFT
        f1_test, f2_test = 600.0, 1500.0 
        amp1_dbfs = -20.0 # Low enough to avoid complex interactions for this simple test
        ratio_f1_f2 = 1.0
        
        signal = generate_dual_tone(f1_test, amp1_dbfs, f2_test, ratio_f1_f2, duration, sample_rate)
        
        N = len(signal)
        fft_result = np.fft.rfft(signal)
        fft_magnitudes = np.abs(fft_result)
        fft_frequencies = np.fft.rfftfreq(N, d=1/sample_rate)
        
        # Find indices of the two largest peaks
        # We ignore DC component by starting search from index 1 (or a few bins away)
        min_idx_search = int(100 / (sample_rate/N)) # Ignore below 100 Hz for robustness
        
        # Find the two largest peaks in the spectrum (excluding DC)
        # A simple way: sort magnitudes and pick top 2 indices
        # More robust: find peaks using scipy.signal.find_peaks, but for simplicity:
        peak_indices = np.argsort(fft_magnitudes[min_idx_search:])[-2:] + min_idx_search
        
        detected_freqs = sorted(fft_frequencies[peak_indices])
        expected_freqs = sorted([f1_test, f2_test])
        
        self.assertTrue(np.allclose(detected_freqs, expected_freqs, atol=sample_rate/N * 2)) # Tolerance of +/- 2 bins
        test_results['test_gdt_frequencies'] = "PASS"


    def _create_time_array(self, duration, sample_rate):
        return np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    def test_analyze_smpte_clean_signal(self):
        print("Running test_analyze_smpte_clean_signal...")
        sr = 48000
        duration = 1.0
        f1, f2 = 60.0, 7000.0
        
        amp1_dbfs = -12.0
        ratio = 4.0
        amp1_lin = dbfs_to_linear(amp1_dbfs)
        amp2_lin = amp1_lin / ratio
        
        time_array = self._create_time_array(duration, sr)
        signal = amp1_lin * np.sin(2 * np.pi * f1 * time_array) + \
                 amp2_lin * np.sin(2 * np.pi * f2 * time_array)
        
        # Ensure signal does not exceed [-1, 1] before passing to analysis
        signal = np.clip(signal, -1.0, 1.0)

        results = analyze_imd_smpte(signal, sr, f1, f2)
        
        self.assertIsNotNone(results)
        # For a clean signal, IMD should be very low (numerical noise)
        self.assertTrue(np.isclose(results['imd_percentage'], 0.0, atol=1e-2)) # increased atol slightly
        self.assertTrue(np.isclose(results['amp_f2_linear'], amp2_lin, rtol=0.15)) # FFT peak estimation tolerance
        test_results['test_analyze_smpte_clean_signal'] = "PASS"

    def test_analyze_smpte_with_known_sidebands(self):
        print("Running test_analyze_smpte_with_known_sidebands...")
        sr = 48000
        duration = 1.0
        f1, f2 = 60.0, 7000.0 # SMPTE standard frequencies
        
        amp1_dbfs = -12.0 
        ratio = 4.0 # f1 is 4x f2 in amplitude (12 dB difference)
        
        amp1_lin = dbfs_to_linear(amp1_dbfs) # This is V1 (low freq tone)
        amp2_lin = amp1_lin / ratio          # This is V2 (high freq tone, reference for IMD)

        time_array = self._create_time_array(duration, sr)
        signal = amp1_lin * np.sin(2 * np.pi * f1 * time_array) + \
                 amp2_lin * np.sin(2 * np.pi * f2 * time_array)
        
        # Add known sidebands
        # Sidebands are at f2 +/- f1, f2 +/- 2*f1, etc.
        # For SMPTE, IMD % = (RMS sum of sidebands / Amplitude of f2) * 100
        # Let's add first order sidebands (f2 +/- f1)
        sb1_freq_minus = f2 - f1
        sb1_freq_plus = f2 + f1
        
        # Each sideband is 1% of V2 (amp2_lin), which is -40dB relative to V2
        sb1_amp_lin = amp2_lin * 0.01 
        
        signal += sb1_amp_lin * np.sin(2 * np.pi * sb1_freq_minus * time_array)
        signal += sb1_amp_lin * np.sin(2 * np.pi * sb1_freq_plus * time_array)

        # Ensure signal does not exceed [-1, 1] before passing to analysis
        signal = np.clip(signal, -1.0, 1.0)

        results = analyze_imd_smpte(signal, sr, f1, f2, num_sideband_pairs=1)
        
        self.assertIsNotNone(results)
        self.assertTrue(np.isclose(results['amp_f2_linear'], amp2_lin, rtol=0.15)) # Check V2 amplitude
        
        # Expected IMD calculation:
        # Sum of squares of sideband amplitudes
        sum_sq_sidebands = sb1_amp_lin**2 + sb1_amp_lin**2
        expected_rms_sidebands = np.sqrt(sum_sq_sidebands)
        expected_imd_percentage = (expected_rms_sidebands / amp2_lin) * 100
        
        # We expect IMD % to be sqrt(0.01^2 + 0.01^2) * 100 = sqrt(2 * 0.0001) * 100 = 0.01 * sqrt(2) * 100 = 1.414... %
        self.assertTrue(np.isclose(results['imd_percentage'], expected_imd_percentage, rtol=0.15)) # rtol 15%
        test_results['test_analyze_smpte_with_known_sidebands'] = "PASS"

    def test_analyze_smpte_low_f2_signal(self):
        print("Running test_analyze_smpte_low_f2_signal...")
        sr = 48000
        duration = 1.0
        f1, f2 = 60.0, 7000.0
        
        amp1_lin = 0.1 # Some amplitude for f1
        amp2_lin_very_low = 1e-12 # -240 dBFS, effectively noise floor for analysis

        time_array = self._create_time_array(duration, sr)
        signal = amp1_lin * np.sin(2 * np.pi * f1 * time_array) + \
                 amp2_lin_very_low * np.sin(2 * np.pi * f2 * time_array)
        
        results = analyze_imd_smpte(signal, sr, f1, f2)
        
        self.assertIsNotNone(results)
        # Check if amp_f2_linear is reported as very low
        self.assertTrue(results['amp_f2_linear'] < 1e-9) # As per analyze_imd_smpte logic
        # Check if IMD values indicate effectively no result or error due to low f2
        self.assertTrue(results['imd_percentage'] == 0.0 or np.isinf(results['imd_percentage']))
        self.assertTrue(results['imd_db'] == -np.inf)
        test_results['test_analyze_smpte_low_f2_signal'] = "PASS"


def run_tests():
    suite = unittest.TestSuite()
    # Manually add tests if not using unittest.main() discovery
    suite.addTest(TestAudioImdAnalyzer('test_gdt_amplitudes_and_clipping'))
    suite.addTest(TestAudioImdAnalyzer('test_gdt_frequencies'))
    suite.addTest(TestAudioImdAnalyzer('test_analyze_smpte_clean_signal'))
    suite.addTest(TestAudioImdAnalyzer('test_analyze_smpte_with_known_sidebands'))
    suite.addTest(TestAudioImdAnalyzer('test_analyze_smpte_low_f2_signal'))
    
    # Can use TextTestRunner for more detailed unittest output
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

    # Custom simple runner as per prompt
    test_instance = TestAudioImdAnalyzer()
    test_methods = [method_name for method_name in dir(test_instance) if method_name.startswith('test_')]
    
    all_passed = True
    for method_name in test_methods:
        try:
            # For each test method, create a new instance to ensure test isolation
            # if tests had complex state, though these are mostly functional.
            current_test_instance = TestAudioImdAnalyzer(methodName=method_name)
            current_test_instance.setUp() # In case setUp is defined later
            getattr(current_test_instance, method_name)()
            # If using the global test_results dict, it's already updated inside the test.
            # Otherwise, could set it here.
            # test_results[method_name] = "PASS" # This is now done inside each test
        except AssertionError as e:
            test_results[method_name] = f"FAIL: {e}"
            all_passed = False
        except Exception as e:
            test_results[method_name] = f"ERROR: {e}" # For unexpected errors
            all_passed = False
        finally:
            current_test_instance.tearDown() # In case tearDown is defined later


    print("\n--- Test Summary ---")
    for test_name, result in test_results.items():
        status_indicator = "[PASS]" if "PASS" in result else "[FAIL]" if "FAIL" in result else "[ERROR]"
        print(f"{status_indicator} {test_name}")
        if "FAIL" in result or "ERROR" in result:
             # The result string already contains the error message from the custom runner.
             # For unittest runner, this would be different.
             print(f"    {result.split(':', 1)[1].strip()}")


    if all_passed:
        print("\nAll tests passed successfully!")
        return True
    else:
        print("\nSome tests FAILED or ERRORED.")
        return False

if __name__ == '__main__':
    if not run_tests():
        sys.exit(1) # Exit with error code if any test fails
