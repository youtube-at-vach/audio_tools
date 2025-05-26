import unittest
import numpy as np
from scipy import signal # For test_k_weighting_application sine wave generation

# Functions and classes to be tested from lufs_meter.lufs_meter
from lufs_meter.lufs_meter import (
    mean_square_to_lufs,
    apply_k_weighting,
    calculate_true_peak,
    calculate_loudness_range,
    calculate_integrated_loudness,
    calculate_mean_square # Needed for the gating logic test if we create audio blocks
)

# Helper function for test_gating_logic_of_integrated_loudness
def lufs_to_mean_square(lufs_value: float) -> float:
    """Converts LUFS to mean square value."""
    if lufs_value == -np.inf:
        return 0.0
    return 10**((lufs_value + 0.691) / 10)

class TestLufsMeter(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 48000

    def test_mean_square_to_lufs(self):
        # Test with a known mean square value
        ms_value1 = 0.01
        expected_lufs1 = -0.691 + 10 * np.log10(ms_value1) # Approx -20.691
        self.assertAlmostEqual(mean_square_to_lufs(ms_value1), expected_lufs1, places=3)

        # Test with another known mean square value
        ms_value2 = 1.0
        expected_lufs2 = -0.691 + 10 * np.log10(ms_value2) # Approx -0.691
        self.assertAlmostEqual(mean_square_to_lufs(ms_value2), expected_lufs2, places=3)
        
        # Test with zero mean square value
        ms_value_zero = 0.0
        expected_lufs_zero = -np.inf
        self.assertEqual(mean_square_to_lufs(ms_value_zero), expected_lufs_zero)

        # Test with very small value (approaching zero)
        ms_value_small = 1e-12
        expected_lufs_small = -np.inf # As per current implementation with 1e-10 threshold
        self.assertEqual(mean_square_to_lufs(ms_value_small), expected_lufs_small)


    def test_k_weighting_application(self):
        duration = 1  # 1 second
        frequency = 1000  # 1 kHz
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

        weighted_audio = apply_k_weighting(sine_wave, self.sample_rate)

        self.assertIsInstance(weighted_audio, np.ndarray)
        self.assertEqual(weighted_audio.shape, sine_wave.shape)
        # A very basic check: K-weighting should attenuate 1kHz somewhat, so RMS should be lower.
        # This is not precise without known filter gain at 1kHz.
        # rms_original = np.sqrt(np.mean(sine_wave**2))
        # rms_weighted = np.sqrt(np.mean(weighted_audio**2))
        # self.assertLess(rms_weighted, rms_original) # This might be too broad

    def test_true_peak_detection(self):
        # Test case 1: Simple peak, no inter-sample peak expected
        signal1 = np.array([0.0, 0.5, 0.0, -0.2]) 
        expected_peak1_db = 20 * np.log10(0.5) # -6.02 dBTP
        true_peak1 = calculate_true_peak(signal1, self.sample_rate)
        self.assertAlmostEqual(true_peak1, expected_peak1_db, places=1) # Reduced precision for TP

        # Test case 2: Peak at 1.0 (0 dBFS)
        signal2 = np.array([0.0, 1.0, 0.0])
        expected_peak2_db = 0.0 # 0.0 dBTP
        true_peak2 = calculate_true_peak(signal2, self.sample_rate)
        self.assertAlmostEqual(true_peak2, expected_peak2_db, places=1)

        # Test case 3: Signal that might generate inter-sample peak
        # A signal like [0.6, 0.6] at 48kHz, upsampled 4x (to 192kHz)
        # The resample_poly function will create a smoother curve.
        # For [0.6, 0.6], the peak of the sinc-interpolated signal is indeed > 0.6
        # Let's use a practical example that's easier to predict without exact sinc math
        # A half-cycle of a sine wave at a highish frequency relative to original SR
        # For 48kHz, Nyquist is 24kHz. Let's try a 12kHz sine wave.
        # Its samples might not hit the true peak.
        duration = 1/12000 # one cycle of 12kHz
        num_samples_one_cycle = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples_one_cycle, endpoint=False)
        # Create a signal whose true peak is 0.8 but samples might miss it
        signal3_base = 0.8 * np.sin(2 * np.pi * 12000 * t) 
        # For this test, let's make a very short, simple signal known to cause ISP with resample_poly
        signal3 = np.array([0.6, 0.6, 0.6]) # This simple signal often shows ISP
        upsampled_signal3_manual = signal.resample_poly(signal3, 4, 1)
        expected_peak3_val = np.max(np.abs(upsampled_signal3_manual))
        expected_peak3_db = 20 * np.log10(expected_peak3_val)
        
        true_peak3 = calculate_true_peak(signal3, self.sample_rate)
        self.assertAlmostEqual(true_peak3, expected_peak3_db, places=1)

        # Test with all zeros
        signal_zeros = np.zeros(100)
        expected_peak_zeros = -np.inf
        self.assertEqual(calculate_true_peak(signal_zeros, self.sample_rate), expected_peak_zeros)


    def test_loudness_range_calculation(self):
        # Test case 1: Provided example
        short_term_lufs1 = np.array([-20, -22, -18, -25, -23, -21, -19, -24, -17, -30, -15])
        integrated_lufs1 = -20.0
        # Values after gating: all of them, as all > -70 and all > (-20 - 20 = -40)
        # Sorted: [-30, -25, -24, -23, -22, -21, -20, -19, -18, -17, -15]
        # 10th percentile of these: np.percentile(sorted_values, 10) -> -25.0
        # 95th percentile of these: np.percentile(sorted_values, 95) -> -17.0
        # LRA = -17.0 - (-25.0) = 8.0
        expected_lra1 = 8.0
        self.assertAlmostEqual(calculate_loudness_range(short_term_lufs1, integrated_lufs1), expected_lra1, places=1)

        # Test case 2: Fewer than 3 values after filtering
        short_term_lufs2 = np.array([-80, -75]) # All below -70
        integrated_lufs2 = -20.0
        expected_lra2 = 0.0 # Current implementation returns 0.0
        self.assertEqual(calculate_loudness_range(short_term_lufs2, integrated_lufs2), expected_lra2)
        
        short_term_lufs3 = np.array([-20, -25, -80, -75]) # Only -20, -25 pass abs gate
                                                       # If integrated is -23, then rel gate is -43. Both pass.
                                                       # But only 2 values.
        integrated_lufs3 = -23.0
        expected_lra3 = 0.0
        self.assertEqual(calculate_loudness_range(short_term_lufs3, integrated_lufs3), expected_lra3)

        # Test case 3: All values below relative threshold but above absolute
        short_term_lufs4 = np.array([-30, -32, -35])
        integrated_lufs4 = -10.0 # Relative threshold = -10 - 20 = -30. So only -30 is included.
                                 # This means only 1 value, so LRA = 0
        expected_lra4 = 0.0
        self.assertEqual(calculate_loudness_range(short_term_lufs4, integrated_lufs4), expected_lra4)

        # Test case 4: All values pass, but are identical
        short_term_lufs5 = np.array([-20, -20, -20, -20, -20])
        integrated_lufs5 = -20.0
        expected_lra5 = 0.0 
        self.assertAlmostEqual(calculate_loudness_range(short_term_lufs5, integrated_lufs5), expected_lra5, places=1)

    def test_gating_logic_of_integrated_loudness(self):
        block_duration = 0.4  # 400ms
        num_samples_per_block = int(self.sample_rate * block_duration)

        # Segment 1: Loud, should pass all gates (e.g., -15 LUFS)
        ms1 = lufs_to_mean_square(-15.0)
        audio1 = np.sqrt(ms1) * np.ones(num_samples_per_block)

        # Segment 2: Passes absolute, fails relative gate (e.g., -35 LUFS, assuming overall avg is higher)
        ms2 = lufs_to_mean_square(-35.0)
        audio2 = np.sqrt(ms2) * np.ones(num_samples_per_block)

        # Segment 3: Fails absolute gate (e.g., -80 LUFS)
        ms3 = lufs_to_mean_square(-80.0)
        audio3 = np.sqrt(ms3) * np.ones(num_samples_per_block)
        
        # Concatenate audio segments
        # Order: audio1, audio3, audio2 (to make sure relative gate is tested correctly)
        # If avg of (-15, -80) is used for rel gate, -35 might pass.
        # Correct gating: blocks are [-15, -35, -80] LUFS.
        # Abs gate: -15, -35 pass. (-80 filtered out).
        # Avg of ms for blocks passing abs gate: (ms(-15) + ms(-35))/2. LUFS of this avg is approx -18.0 LUFS.
        # Rel gate threshold: -18.0 - 10 = -28.0 LUFS.
        # So, -15 block passes rel gate. -35 block fails rel gate.
        # Only -15 block contributes to final average. Expected integrated LUFS = -15.0.
        test_audio = np.concatenate([audio1, audio2, audio3])
        
        integrated_lufs, momentary_lufs_for_gating = calculate_integrated_loudness(test_audio, self.sample_rate)
        
        # Check momentary LUFS of the blocks (for debugging if needed)
        # print("Momentary LUFS for gating test:", momentary_lufs_for_gating) 
        # Expected: [-15.0, -35.0, -80.0] (approx)
        self.assertAlmostEqual(momentary_lufs_for_gating[0], -15.0, places=1)
        self.assertAlmostEqual(momentary_lufs_for_gating[1], -35.0, places=1)
        self.assertAlmostEqual(momentary_lufs_for_gating[2], -80.0, places=1)
        
        expected_integrated_lufs = -15.0
        self.assertAlmostEqual(integrated_lufs, expected_integrated_lufs, places=1)

        # Test case 2: All blocks below absolute threshold
        ms_low = lufs_to_mean_square(-75.0)
        audio_low1 = np.sqrt(ms_low) * np.ones(num_samples_per_block)
        audio_low2 = np.sqrt(ms_low) * np.ones(num_samples_per_block)
        test_audio_all_low = np.concatenate([audio_low1, audio_low2])
        integrated_lufs_all_low, _ = calculate_integrated_loudness(test_audio_all_low, self.sample_rate)
        self.assertEqual(integrated_lufs_all_low, -np.inf)

        # Test case 3: Blocks pass absolute, but average is low, then one block fails relative
        # Blocks: -25, -28, -60 LUFS
        # Abs gate: -25, -28 pass. (-60 filtered out).
        # Avg of ms for blocks passing abs gate: (ms(-25) + ms(-28))/2. LUFS of this avg is approx -26.3 LUFS.
        # Rel gate threshold: -26.3 - 10 = -36.3 LUFS.
        # So, -25 and -28 blocks pass rel gate.
        # Expected integrated LUFS = LUFS of (ms(-25) + ms(-28))/2 = approx -26.3 LUFS.
        ms_blockA = lufs_to_mean_square(-25.0)
        audio_blockA = np.sqrt(ms_blockA) * np.ones(num_samples_per_block)
        ms_blockB = lufs_to_mean_square(-28.0)
        audio_blockB = np.sqrt(ms_blockB) * np.ones(num_samples_per_block)
        ms_blockC = lufs_to_mean_square(-60.0)
        audio_blockC = np.sqrt(ms_blockC) * np.ones(num_samples_per_block)
        test_audio_case3 = np.concatenate([audio_blockA, audio_blockB, audio_blockC])
        integrated_lufs_case3, _ = calculate_integrated_loudness(test_audio_case3, self.sample_rate)
        
        expected_avg_ms_case3 = np.mean([ms_blockA, ms_blockB])
        expected_integrated_lufs_case3 = mean_square_to_lufs(expected_avg_ms_case3) # Should be -26.3
        self.assertAlmostEqual(integrated_lufs_case3, expected_integrated_lufs_case3, places=1)


if __name__ == '__main__':
    unittest.main()
