import os
import sys

import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.gui.widgets.lock_in_amplifier import LockInAmplifier


class MockAudioEngine:
    def __init__(self):
        self.sample_rate = 48000
        self.calibration = MockCalibration()

    def register_callback(self, cb):
        return 1

    def unregister_callback(self, cb_id):
        pass

class MockCalibration:
    def __init__(self):
        self.output_gain = 1.0
        self.input_sensitivity = 1.0

def test_harmonic_mode():
    engine = MockAudioEngine()
    lia = LockInAmplifier(engine)

    # Setup LIA
    lia.buffer_size = 4096
    lia.input_data = np.zeros((lia.buffer_size, 2))
    lia.signal_channel = 0
    lia.ref_channel = 1
    lia.averaging_count = 1

    # Generate Test Signal
    # Ref: 1kHz Sine
    # Sig: 1kHz Sine + 3kHz Sine (3rd Harmonic)
    fs = 48000
    t = np.arange(lia.buffer_size) / fs
    f_fund = 1000

    ref_sig = np.cos(2 * np.pi * f_fund * t)

    # Signal:
    # Fundamental: Amplitude 0.5, Phase 0
    # 3rd Harmonic: Amplitude 0.2, Phase 45 deg
    sig_fund = 0.5 * np.cos(2 * np.pi * f_fund * t)
    sig_3rd = 0.2 * np.cos(2 * np.pi * 3 * f_fund * t + np.radians(45))

    sig_combined = sig_fund + sig_3rd

    # Fill Buffer
    lia.input_data[:, 0] = sig_combined # Signal
    lia.input_data[:, 1] = ref_sig      # Reference

    print("--- Test 1: Fundamental Mode (Order=1) ---")
    lia.harmonic_order = 1
    lia.history.clear()
    lia.process_data()

    mag = lia.current_magnitude
    phase = lia.current_phase
    print(f"Magnitude: {mag:.4f} (Expected ~0.5)")
    print(f"Phase: {phase:.2f} (Expected ~0.0)")

    if abs(mag - 0.5) > 0.05:
        print("FAIL: Fundamental Magnitude incorrect")
        return False

    print("PASS")

    print("\n--- Test 2: 3rd Harmonic Mode (Order=3) ---")
    lia.harmonic_order = 3
    lia.history.clear()
    lia.process_data()

    mag = lia.current_magnitude
    phase = lia.current_phase
    print(f"Magnitude: {mag:.4f} (Expected ~0.2)")
    print(f"Phase: {phase:.2f} (Expected ~45.0)")

    if abs(mag - 0.2) > 0.02:
        print("FAIL: 3rd Harmonic Magnitude incorrect")
        return False

    # Phase might wrap, check error
    phase_err = abs(phase - 45.0)
    if phase_err > 360: phase_err -= 360
    if phase_err > 5.0:
        print("FAIL: 3rd Harmonic Phase incorrect")
        return False

    print("PASS")

    print("\n--- Test 3: 2nd Harmonic Mode (Order=2) ---")
    # Should be near zero
    lia.harmonic_order = 2
    lia.history.clear()
    lia.process_data()

    mag = lia.current_magnitude
    print(f"Magnitude: {mag:.4f} (Expected ~0.0)")

    if mag > 0.01:
        print("FAIL: 2nd Harmonic should be zero")
        return False

    print("PASS")

    return True

if __name__ == "__main__":
    if test_harmonic_mode():
        print("\nAll Tests Passed!")
        sys.exit(0)
    else:
        print("\nTests Failed!")
        sys.exit(1)
