import unittest
import numpy as np
from tests.mock_audio import MockAudio

class TestNetworkAnalyzer(unittest.TestCase):
    def setUp(self):
        self.mock = MockAudio()
        self.sample_rate = 48000

    def calculate_transfer_function(self, ref_sig, dut_sig):
        # Windowing
        window = np.hanning(len(ref_sig))
        ref_w = ref_sig * window
        dut_w = dut_sig * window
        
        # FFT
        ref_fft = np.fft.rfft(ref_w)
        dut_fft = np.fft.rfft(dut_w)
        
        # Find peak freq bin from Reference
        peak_idx = np.argmax(np.abs(ref_fft))
        
        # Transfer Function H = DUT / Ref
        # Avoid division by zero
        if np.abs(ref_fft[peak_idx]) < 1e-9:
            return 0, 0
            
        H = dut_fft[peak_idx] / ref_fft[peak_idx]
        
        gain_db = 20 * np.log10(np.abs(H))
        phase_deg = np.degrees(np.angle(H))
        
        return gain_db, phase_deg

    def test_gain_phase(self):
        print("\nTesting Gain and Phase Measurement...")
        
        # Test Case 1: -6dB Gain, 0 deg Phase, 0 Latency
        self.mock.set_characteristics(latency_ms=0, gain_db=-6, phase_shift_deg=0)
        
        # Generate Sine Wave
        f = 1000
        duration = 0.1
        t = np.linspace(0, duration, int(self.sample_rate*duration), False)
        sig = np.sin(2*np.pi*f*t)
        outdata = np.column_stack((sig, sig)) # Stereo output
        
        indata = self.mock.playrec(outdata, self.sample_rate, 2)
        
        gain, phase = self.calculate_transfer_function(indata[:, 0], indata[:, 1])
        print(f"Case 1 (-6dB, 0deg): Measured {gain:.2f} dB, {phase:.2f} deg")
        
        self.assertAlmostEqual(gain, -6.0, delta=0.1)
        self.assertAlmostEqual(phase, 0.0, delta=1.0)

        # Test Case 2: 0dB Gain, 90 deg Phase, 0 Latency
        self.mock.set_characteristics(latency_ms=0, gain_db=0, phase_shift_deg=90)
        indata = self.mock.playrec(outdata, self.sample_rate, 2)
        gain, phase = self.calculate_transfer_function(indata[:, 0], indata[:, 1])
        print(f"Case 2 (0dB, 90deg): Measured {gain:.2f} dB, {phase:.2f} deg")
        
        self.assertAlmostEqual(gain, 0.0, delta=0.1)
        self.assertAlmostEqual(phase, 90.0, delta=1.0)
        
        # Test Case 3: 0dB Gain, 0 deg Phase, 10ms Latency
        # Latency creates a phase shift proportional to frequency: phi = -2*pi*f*t
        # At 1000Hz, 10ms (0.01s) -> 10 cycles -> 0 phase shift (wrapped)? 
        # Wait, 10ms = 1/100s. 1000Hz period is 1ms. So 10ms is exactly 10 periods.
        # Phase shift should be 0 (modulo 360).
        # Let's try 0.25ms latency (1/4 period) -> -90 deg.
        
        self.mock.set_characteristics(latency_ms=0.25, gain_db=0, phase_shift_deg=0)
        indata = self.mock.playrec(outdata, self.sample_rate, 2)
        gain, phase = self.calculate_transfer_function(indata[:, 0], indata[:, 1])
        print(f"Case 3 (Latency 0.25ms -> -90deg): Measured {gain:.2f} dB, {phase:.2f} deg")
        
        self.assertAlmostEqual(gain, 0.0, delta=0.1)
        # Phase might be -90 or 270
        diff = (phase - (-90)) % 360
        if diff > 180: diff -= 360
        self.assertAlmostEqual(diff, 0.0, delta=1.0)

if __name__ == '__main__':
    unittest.main()
