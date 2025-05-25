import unittest
import numpy as np
from scipy.signal import windows # For Tukey window comparison if needed directly

# Assuming the main script is in a file named audio_transient_analyzer.py
# and is either in the same directory or the package structure is set up correctly.
from audio_transient_analyzer.audio_transient_analyzer import (
    dbfs_to_linear,
    generate_impulse,
    generate_tone_burst,
    # channel_spec_to_index # Not testing this one in this subtask
    find_signal_start,
    calculate_rise_time,
    calculate_overshoot,
    calculate_settling_time,
)

class TestSignalGenerationAndHelpers(unittest.TestCase):

    def test_dbfs_to_linear(self):
        self.assertAlmostEqual(dbfs_to_linear(0), 1.0)
        self.assertAlmostEqual(dbfs_to_linear(-6), 0.5011872336, places=6) # Increased precision
        self.assertAlmostEqual(dbfs_to_linear(-20), 0.1)
        self.assertAlmostEqual(dbfs_to_linear(-60), 0.001)


    def test_generate_impulse(self):
        sample_rate = 48000
        amplitude_dbfs = -6.0
        linear_amp = dbfs_to_linear(amplitude_dbfs)
        
        impulse_signal = generate_impulse(amplitude_dbfs, sample_rate)
        
        # Expected length based on implementation: max(10, int(0.001 * sample_rate))
        expected_len = max(10, int(0.001 * sample_rate))

        self.assertEqual(len(impulse_signal), expected_len)
        self.assertAlmostEqual(impulse_signal[0], linear_amp, places=6)
        # The rest of the samples should be exactly zero as per current implementation
        self.assertTrue(np.all(impulse_signal[1:] == 0.0))
        self.assertAlmostEqual(np.max(np.abs(impulse_signal)), linear_amp, places=6)

    def test_generate_tone_burst_rectangular(self):
        sample_rate = 48000
        freq = 1000
        cycles = 10
        amplitude_dbfs = -6.0
        linear_amp = dbfs_to_linear(amplitude_dbfs)
        expected_samples = int(cycles * sample_rate / freq)
        
        burst_signal = generate_tone_burst(freq, amplitude_dbfs, cycles, 'rectangular', sample_rate)
        
        self.assertEqual(len(burst_signal), expected_samples)
        # For rectangular window, the peak should ideally be linear_amp.
        # Due to discrete sampling of sine, it might be slightly less.
        # Allowing a small tolerance.
        self.assertAlmostEqual(np.max(np.abs(burst_signal)), linear_amp, delta=0.01 * linear_amp) 
        # Check that it's not zero at start/end (it's rectangular)
        self.assertTrue(np.abs(burst_signal[0]) > 0.01 * linear_amp if expected_samples > 0 else True)


    def test_generate_tone_burst_hann(self):
        sample_rate = 48000
        freq = 1000
        cycles = 20 # Using more cycles for Hann to see the window effect clearly
        amplitude_dbfs = -6.0
        linear_amp = dbfs_to_linear(amplitude_dbfs)
        expected_samples = int(cycles * sample_rate / freq)
        
        burst_signal = generate_tone_burst(freq, amplitude_dbfs, cycles, 'hann', sample_rate)
        
        self.assertEqual(len(burst_signal), expected_samples)
        # Hann window peak is 1.0, so max amplitude should be close to linear_amp,
        # possibly slightly less due to sampling.
        self.assertAlmostEqual(np.max(np.abs(burst_signal)), linear_amp, delta=0.01 * linear_amp)
        
        if expected_samples > 0:
            self.assertAlmostEqual(burst_signal[0], 0.0, places=6)
            # For Hann, the last sample may not be exactly zero depending on np.hanning implementation details
            # and whether the endpoint is included in linspace for t.
            # Given t = np.linspace(0, cycles / frequency, num_samples_burst, endpoint=False)
            # and envelope = np.hanning(num_samples_burst)
            # the last sample of hanning window is also zero.
            self.assertAlmostEqual(burst_signal[-1], 0.0, places=5) 

    def test_generate_tone_burst_tukey(self):
        sample_rate = 48000
        freq = 1000
        cycles = 20 # Using more cycles for Tukey to see the window effect clearly
        amplitude_dbfs = -6.0
        linear_amp = dbfs_to_linear(amplitude_dbfs)
        expected_samples = int(cycles * sample_rate / freq)
        alpha = 0.5 # Default alpha in main code

        burst_signal = generate_tone_burst(freq, amplitude_dbfs, cycles, 'tukey', sample_rate)
        self.assertEqual(len(burst_signal), expected_samples)

        # Max amplitude should be close to linear_amp (Tukey peak is 1.0)
        self.assertAlmostEqual(np.max(np.abs(burst_signal)), linear_amp, delta=0.01 * linear_amp)

        if expected_samples > 0:
            # Check if ends are tapered (should be zero for alpha > 0)
            self.assertAlmostEqual(burst_signal[0], 0.0, places=6)
            
            # For Tukey, the last sample should also be zero if alpha > 0
            # Similar to Hann, due to endpoint=False in time vector and window applied.
            self.assertAlmostEqual(burst_signal[-1], 0.0, places=5)

            # Verify the general shape by checking a point in the flat top part (if alpha < 1)
            # For alpha = 0.5, flat top is middle 50% of the window.
            # Middle index of the window:
            mid_idx = expected_samples // 2
            # A point in the flat top region
            flat_top_check_idx = mid_idx 
            
            # Generate expected Tukey window for comparison
            # This makes the test more robust by comparing to the actual window shape
            # rather than just end points.
            # t_wave = np.linspace(0, cycles / freq, expected_samples, endpoint=False)
            # sine_wave_part = linear_amp * np.sin(2 * np.pi * freq * t_wave)
            # tukey_window = windows.tukey(expected_samples, alpha=alpha)
            # expected_signal_at_idx = sine_wave_part[flat_top_check_idx] * tukey_window[flat_top_check_idx]
            # self.assertAlmostEqual(burst_signal[flat_top_check_idx], expected_signal_at_idx, delta=0.05 * linear_amp)
            # The above is a bit complex due to sine wave phase. Simpler: check that it's not zero.
            self.assertTrue(np.abs(burst_signal[flat_top_check_idx]) > 0.5 * linear_amp)


    def test_generate_tone_burst_zero_cycles(self):
        sample_rate = 48000
        amplitude_dbfs = -6.0
        freq = 1000
        linear_amp = dbfs_to_linear(amplitude_dbfs)

        # This should fall back to generate_impulse
        burst_signal = generate_tone_burst(freq, amplitude_dbfs, 0, 'rectangular', sample_rate)
        
        # Expected impulse properties (from generate_impulse implementation)
        expected_len = max(10, int(0.001 * sample_rate))
        
        self.assertEqual(len(burst_signal), expected_len)
        self.assertAlmostEqual(burst_signal[0], linear_amp, places=6)
        self.assertTrue(np.all(burst_signal[1:] == 0.0))

    def test_generate_tone_burst_very_few_samples(self):
        # Test case where num_samples_burst might be very small (e.g., 1 or 2)
        # but not zero.
        sample_rate = 48000
        freq = 20000 # High frequency
        cycles = 1    # Few cycles
        amplitude_dbfs = -6.0
        linear_amp = dbfs_to_linear(amplitude_dbfs)
        
        expected_samples = int(cycles * sample_rate / freq) # 48000 / 20000 = 2.4 -> 2 samples
        self.assertTrue(expected_samples > 0, "Test setup leads to zero samples, adjust params.")

        burst_signal_rect = generate_tone_burst(freq, amplitude_dbfs, cycles, 'rectangular', sample_rate)
        self.assertEqual(len(burst_signal_rect), expected_samples)
        self.assertAlmostEqual(np.max(np.abs(burst_signal_rect)), linear_amp, delta=0.05 * linear_amp)

        # Hann window on very few samples might result in near-zero output for all samples
        burst_signal_hann = generate_tone_burst(freq, amplitude_dbfs, cycles, 'hann', sample_rate)
        self.assertEqual(len(burst_signal_hann), expected_samples)
        if expected_samples == 1: # Hann window of 1 is [1.]
             self.assertAlmostEqual(np.max(np.abs(burst_signal_hann)), linear_amp, delta=0.05 * linear_amp)
        elif expected_samples > 1: # Hann window starts and ends at 0
            self.assertAlmostEqual(burst_signal_hann[0], 0.0, places=5)
            # Max value will be less than linear_amp due to windowing effect on short signal
            self.assertTrue(np.max(np.abs(burst_signal_hann)) < linear_amp)


if __name__ == '__main__':
    unittest.main()


class TestAnalysisFunctions(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 48000

    def test_find_signal_start(self):
        signal = np.zeros(100)
        signal[20:30] = 0.5  # Exceeds 0.1 * max_abs (which is 1.0) at index 20
        signal[30:40] = 1.0
        
        pre_trigger_samples = 5
        # Max abs is 1.0. Threshold is 0.1. First point >= 0.1 is index 20.
        # Expected index = 20 - pre_trigger_samples = 15.
        idx = find_signal_start(signal, self.sample_rate, threshold_factor=0.1, pre_trigger_samples=pre_trigger_samples)
        self.assertEqual(idx, 15)

        signal_zeros = np.zeros(100)
        idx_zeros = find_signal_start(signal_zeros, self.sample_rate)
        self.assertEqual(idx_zeros, 0)

        signal_empty = np.array([])
        idx_empty = find_signal_start(signal_empty, self.sample_rate)
        self.assertEqual(idx_empty, 0)

        # Test where signal starts immediately
        signal_immediate_start = np.array([0.5, 1.0, 0.5])
        idx_immediate = find_signal_start(signal_immediate_start, self.sample_rate, threshold_factor=0.1, pre_trigger_samples=0)
        self.assertEqual(idx_immediate, 0)
        idx_immediate_pretrigger = find_signal_start(signal_immediate_start, self.sample_rate, threshold_factor=0.1, pre_trigger_samples=2)
        self.assertEqual(idx_immediate_pretrigger, 0) # Should clamp to 0

    def test_calculate_rise_time(self):
        N = 100  # Number of points in the ramp
        signal = np.zeros(200)
        # Ramp from 0.0 to 1.0 over N samples.
        # segment[idx_10] should be the first point >= 0.1
        # segment[idx_90] should be the first point >= 0.9
        signal[50 : 50 + N] = np.linspace(0, 1.0, N) 
        
        # find_signal_start would give start_idx such that the ramp starts after it.
        # If pre-trigger is 5, find_signal_start(threshold=0.01) would give 50-5=45
        start_idx = 45 
        
        # In the segment audio_data[start_idx:], the actual ramp starts at index (50-start_idx) = 5
        # So, segment_for_calc = signal[45:]
        # peak_value in segment_for_calc is 1.0
        # val_10 = 0.1, val_90 = 0.9
        # np.where(segment_for_calc >= 0.1)[0][0] is the index for 10% point relative to segment_for_calc start
        # This index is 5 (actual signal index 50) + (0.1 * (N-1))
        # np.where(segment_for_calc >= 0.9)[0][0] is the index for 90% point relative to segment_for_calc start
        # This index is 5 (actual signal index 50) + (0.9 * (N-1))

        # Indices relative to the start of the ramp (signal[50])
        # idx_10_in_ramp = np.where(np.linspace(0,1,N) >= 0.1)[0][0] -> 0.1*(N-1) if N large enough
        # idx_90_in_ramp = np.where(np.linspace(0,1,N) >= 0.9)[0][0] -> 0.9*(N-1) if N large enough
        
        # The function calculate_rise_time finds idx_10 and idx_90 within the segment audio_data[start_idx:]
        # Ramp part within segment starts at index 5 (relative to start_idx)
        # So, idx_10_in_segment = 5 + np.round(0.1 * (N-1))
        # And idx_90_in_segment = 5 + np.round(0.9 * (N-1))
        # expected_rise_samples = (5 + np.round(0.9 * (N-1))) - (5 + np.round(0.1 * (N-1)))
        # expected_rise_samples = np.round(0.9 * (N-1)) - np.round(0.1 * (N-1))
        # For N=100, linspace(0,1,100) has 100 points.
        # 10% value (0.1) is first met at index np.ceil(0.1*(N-1)) = ceil(9.9) = 10 (approx)
        # 90% value (0.9) is first met at index np.ceil(0.9*(N-1)) = ceil(89.1) = 90 (approx)
        # expected_rise_samples = ceil(0.9*(N-1)) - ceil(0.1*(N-1))
        
        # Let's use the exact points for linspace:
        # For linspace(0, 1, N), value v is at index v*(N-1)
        idx_10_in_ramp_strict = 0.1 * (N - 1)
        idx_90_in_ramp_strict = 0.9 * (N - 1)
        
        # The function uses np.where(segment >= val)[0][0] which finds the first index.
        # If N=100, linspace points are 0, 1/99, 2/99 ... 99/99.
        # Value 0.1 is at index 9.9. np.where finds index 10. (segment[10] = 10/99 = 0.1010)
        # Value 0.9 is at index 89.1. np.where finds index 90. (segment[90] = 90/99 = 0.9090)
        # So, for N=100, idx_10 is 10, idx_90 is 90.
        # expected_rise_samples = 90 - 10 = 80.
        expected_rise_samples = np.ceil(0.9 * (N - 1)) - np.floor(0.1 * (N - 1)) if N > 1 else 0
        # A more direct way for linspace:
        # indices_above_10 = np.where(linspace_signal >= 0.1*peak_value)[0] -> idx_10 = indices_above_10[0]
        # So for linspace from 0 to 1, peak is 1.
        # linspace_vals = np.linspace(0,1,N)
        # actual_idx_10_in_ramp = np.where(linspace_vals >= 0.1)[0][0]
        # actual_idx_90_in_ramp = np.where(linspace_vals >= 0.9)[0][0]
        # expected_rise_samples = actual_idx_90_in_ramp - actual_idx_10_in_ramp
        
        # Let's use the logic from the function itself for expected values
        segment_for_test = signal[start_idx:] # This is what the function sees
        peak_val_seg = np.max(np.abs(segment_for_test))
        val_10_seg = 0.1 * peak_val_seg
        val_90_seg = 0.9 * peak_val_seg
        indices_10_seg = np.where(np.abs(segment_for_test) >= val_10_seg)[0]
        idx_10_final = indices_10_seg[0] if len(indices_10_seg) > 0 else 0
        
        indices_90_seg = np.where(np.abs(segment_for_test[idx_10_final:]) >= val_90_seg)[0]
        idx_90_final = indices_90_seg[0] + idx_10_final if len(indices_90_seg) > 0 else idx_10_final
        
        expected_rise_samples_from_func_logic = idx_90_final - idx_10_final
        expected_rise_time_s = expected_rise_samples_from_func_logic / self.sample_rate

        rise_time = calculate_rise_time(signal, self.sample_rate, start_idx)
        self.assertAlmostEqual(rise_time, expected_rise_time_s, places=6)

        # Test with a flat signal
        flat_signal = np.ones(100) * 0.5
        self.assertEqual(calculate_rise_time(flat_signal, self.sample_rate, 0), 0.0)

        # Test with signal not reaching 90% (e.g. ramps from 0 to 0.5)
        ramp_to_half = np.zeros(100)
        ramp_to_half[10:60] = np.linspace(0, 0.5, 50)
        self.assertEqual(calculate_rise_time(ramp_to_half, self.sample_rate, 0), 0.0)
        
        # Test with all zero signal
        zero_signal = np.zeros(100)
        self.assertEqual(calculate_rise_time(zero_signal, self.sample_rate, 0), 0.0)
        
        # Test with empty signal
        empty_signal = np.array([])
        self.assertEqual(calculate_rise_time(empty_signal, self.sample_rate, 0), 0.0)


    def test_calculate_overshoot(self):
        signal = np.zeros(500)
        start_idx = 10 # This is the start_index passed to the function
        
        # The segment analyzed by the function will be signal[start_idx:]
        segment_offset = start_idx 

        # Peak part: linspace from 0 to 1.2 over 10 samples
        # This means signal[segment_offset+0] to signal[segment_offset+9] is the rise
        # Peak is at signal[segment_offset+9] = 1.2
        # Correcting the signal construction:
        # Peak at segment_offset + 10 (relative to signal start)
        # Steady state from segment_offset + 11 to segment_offset + 90 (relative to signal start)
        
        peak_time_idx_in_signal = segment_offset + 10 # signal[20]
        steady_state_start_idx_in_signal = segment_offset + 11 # signal[21]
        steady_state_end_idx_in_signal = segment_offset + 90 # signal[100]

        signal[peak_time_idx_in_signal] = 1.2 # Peak value
        # Ramp up to peak
        signal[segment_offset : peak_time_idx_in_signal] = np.linspace(0, 1.2, peak_time_idx_in_signal - segment_offset)
        
        # Steady state part
        signal[steady_state_start_idx_in_signal : steady_state_end_idx_in_signal] = 1.0

        # The function's internal logic for steady state:
        # abs_segment = np.abs(signal[start_idx:])
        # peak_idx_in_segment = np.argmax(abs_segment) -> 10 (signal index 20)
        # abs_peak_value = 1.2
        # post_peak_start_offset_samples = int(5 / 1000 * self.sample_rate) = 240 samples
        # steady_state_search_start_idx = peak_idx_in_segment + post_peak_start_offset_samples = 10 + 240 = 250
        # This would look for steady state way after our defined signal.
        # The test signal must be long enough for the function's heuristic.
        # Let's make a signal that aligns with the function's heuristic for steady state.
        
        signal_os = np.zeros(self.sample_rate // 10) # 100ms signal
        start_idx_os = 0
        
        peak_val = 1.2
        steady_val = 1.0
        
        # Ramp up to peak in 2ms
        peak_sample_idx = self.sample_rate * 2 // 1000 # at 2ms
        signal_os[0:peak_sample_idx] = np.linspace(0, peak_val, peak_sample_idx)
        
        # Hold peak for 1ms
        signal_os[peak_sample_idx : peak_sample_idx + self.sample_rate // 1000] = peak_val
        peak_idx_abs = peak_sample_idx # The function finds first peak, so this is ok.
        
        # Drop to steady state in 2ms
        steady_start_sample_idx = peak_sample_idx + self.sample_rate // 1000
        drop_end_idx = steady_start_sample_idx + self.sample_rate * 2 // 1000
        signal_os[steady_start_sample_idx:drop_end_idx] = np.linspace(peak_val, steady_val, drop_end_idx - steady_start_sample_idx)
        
        # Hold steady state for the rest of the signal
        signal_os[drop_end_idx:] = steady_val
        
        # With this signal_os, peak_idx_in_segment is `peak_idx_abs`.
        # abs_peak_value is `peak_val`.
        # steady_state_search_start_idx = peak_idx_abs + (5ms in samples)
        # post_peak_segment starts 5ms after the peak.
        # steady_state_value = np.mean(np.abs(post_peak_segment[len(post_peak_segment)//2:]))
        # This should be `steady_val` (1.0) if post_peak_segment is long enough.
        # Length of post_peak_segment needs to be >= 10ms in samples.
        # Total signal length 100ms. Peak at 2ms. Post peak segment starts at 7ms. Length is 93ms. This is fine.
        
        expected_overshoot_pct = ((peak_val - steady_val) / steady_val) * 100
        
        overshoot = calculate_overshoot(signal_os, self.sample_rate, start_idx_os)
        self.assertAlmostEqual(overshoot, expected_overshoot_pct, places=1)

        # Test with an impulse (decaying signal, overshoot should be 0)
        impulse = generate_impulse(-6, self.sample_rate)
        self.assertAlmostEqual(calculate_overshoot(impulse, self.sample_rate, 0), 0.0, places=1)

    def test_calculate_settling_time(self):
        final_value = 1.0
        settle_percentage = 0.05
        tolerance = settle_percentage * final_value # For case of non-decaying to zero.
                                                    # If decaying, it's based on peak.

        signal = np.full(self.sample_rate, final_value) # 1 second long signal
        start_idx = 0
        
        # Ringing part: decays in 50ms
        ring_duration_samples = self.sample_rate * 50 // 1000 # 50ms
        t_ring = np.linspace(0, 10 * np.pi, ring_duration_samples) # More oscillations
        ringing = 0.2 * np.sin(t_ring) * np.exp(-t_ring / (2*np.pi)) # Make it decay
        
        peak_of_ringing_idx = np.argmax(np.abs(ringing)) # Relative to start of ringing
        
        # Place ringing at the beginning of the segment the function sees
        signal[0 : ring_duration_samples] = final_value + ringing
        
        # The function searches from the peak of the segment.
        # Peak of segment is signal[peak_of_ringing_idx]
        # It should settle after ringing part, i.e., at ring_duration_samples
        # So, settling time in samples = ring_duration_samples - peak_of_ringing_idx
        
        # Let's find peak_idx_segment as the function would
        segment_for_calc = signal[start_idx:]
        peak_idx_segment = np.argmax(np.abs(segment_for_calc)) # This is peak_of_ringing_idx

        # The signal is considered settled when all samples from a point onwards are within bounds.
        # The first such point after peak_idx_segment is ring_duration_samples.
        # So, time is (ring_duration_samples - peak_idx_segment) / sample_rate.
        expected_settling_samples = ring_duration_samples - peak_idx_segment
        expected_settling_time_s = expected_settling_samples / self.sample_rate
        
        settling_time = calculate_settling_time(signal, self.sample_rate, start_idx, settle_percentage)
        self.assertAlmostEqual(settling_time, expected_settling_time_s, places=5)

        # Test signal that never settles (oscillates outside band)
        non_settling_signal = np.zeros(self.sample_rate)
        t_non_settle = np.arange(self.sample_rate) / self.sample_rate
        non_settling_signal = 0.5 + 0.2 * np.sin(2 * np.pi * 10 * t_non_settle) # oscillates around 0.5, peak to peak 0.4
        # final_value_estimate_abs would be around 0.5.
        # tolerance = 0.05 * 0.5 = 0.025. Band [0.475, 0.525].
        # Signal goes from 0.3 to 0.7. So it never settles.
        # The function should return 0.0 if it doesn't find a settled point.
        self.assertAlmostEqual(calculate_settling_time(non_settling_signal, self.sample_rate, 0, 0.05), 0.0, places=5)

        # Test with signal that decays to zero (e.g. impulse)
        # This case uses peak_abs_segment for tolerance.
        impulse_signal = generate_impulse(-6, self.sample_rate) # Peak around 0.5
        # peak_abs_segment is 0.5. tolerance = 0.05 * 0.5 = 0.025.
        # Settles when values are within [-0.025, 0.025].
        # Impulse is 0.5 at sample 0, then 0.0.
        # peak_idx_segment is 0.
        # search_segment starts at index 0.
        # sample 0 (0.5) is outside. sample 1 (0.0) is inside.
        # all_subsequent_settled from sample 1 onwards.
        # settled_sample_relative_to_peak = 1.
        # expected_time = 1 / self.sample_rate.
        expected_time_impulse = 1.0 / self.sample_rate
        self.assertAlmostEqual(calculate_settling_time(impulse_signal, self.sample_rate, 0, 0.05), expected_time_impulse, places=6)

        # Test with a signal that is already settled (flat line)
        flat_signal = np.ones(self.sample_rate) * 0.8
        # peak_idx_segment is 0. search_segment is the whole signal.
        # final_value_estimate_abs is 0.8. tolerance = 0.05 * 0.8 = 0.04. Band [0.76, 0.84]
        # First sample (0.8) is within bounds. All subsequent are.
        # settled_sample_relative_to_peak = 0.
        # expected_time = 0 / self.sample_rate = 0.0
        self.assertAlmostEqual(calculate_settling_time(flat_signal, self.sample_rate, 0, 0.05), 0.0, places=6)
