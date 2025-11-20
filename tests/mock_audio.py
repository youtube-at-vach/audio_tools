import numpy as np
import time

class MockAudio:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.latency_samples = 0
        self.gain_db = 0
        self.phase_shift_deg = 0
        self.noise_level = 0.0

    def set_characteristics(self, latency_ms, gain_db, phase_shift_deg, noise_level=-100):
        self.latency_samples = int(latency_ms * self.sample_rate / 1000)
        self.gain_db = gain_db
        self.phase_shift_deg = phase_shift_deg
        self.noise_level = 10**(noise_level/20)

    def playrec(self, outdata, samplerate, channels, device=None, blocking=True):
        """
        Simulates playing and recording.
        Input: outdata (N, out_ch)
        Output: indata (N, in_ch)
        
        We assume:
        - Channel 0 is Reference (Loopback) -> No modification (or minimal latency)
        - Channel 1 is DUT -> Applied Gain, Phase, Latency
        """
        n_samples = outdata.shape[0]
        indata = np.zeros((n_samples, channels), dtype=np.float32)
        
        # Input signal (usually sine wave)
        # We assume outdata[:, 0] or outdata[:, 1] is the source.
        # Let's take the first active channel as source.
        source_sig = outdata[:, 0] if outdata.shape[1] > 0 else np.zeros(n_samples)
        if np.max(np.abs(source_sig)) == 0 and outdata.shape[1] > 1:
             source_sig = outdata[:, 1]

        # Channel 0: Reference (Direct Loopback)
        # Perfect copy
        if channels > 0:
            indata[:, 0] = source_sig

        # Channel 1: DUT (Modified)
        if channels > 1:
            # Apply Gain
            gain_lin = 10**(self.gain_db/20)
            dut_sig = source_sig * gain_lin
            
            # Apply Phase Shift (Constant phase shift across all freqs is non-causal/Hilbert, 
            # but for single tone we can just shift start phase if we knew freq.
            # Since we don't know freq here easily without FFT, we can simulate Delay.
            # But user asked for Phase Shift. 
            # A constant phase shift is hard to simulate in time domain for arbitrary signal without Hilbert transform.
            # For simplicity, let's assume the input is a sine wave and we just delay it?
            # No, let's do it properly with FFT if we want to simulate "Phase Shift" distinct from "Delay".
            # Or just simulate Delay which causes Phase Shift = -2*pi*f*delay.
            
            # Let's simulate Delay (Latency)
            if self.latency_samples > 0:
                dut_sig = np.pad(dut_sig, (self.latency_samples, 0))[:n_samples]
            
            # If we really want to simulate constant phase shift (e.g. 90 deg) for testing:
            # We can use Hilbert transform.
            if self.phase_shift_deg != 0:
                # Analytic signal
                analytic = scipy.signal.hilbert(dut_sig)
                # Rotate
                phase_rad = np.radians(self.phase_shift_deg)
                dut_sig = np.real(analytic * np.exp(1j * phase_rad))

            # Add Noise
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level, n_samples)
                dut_sig += noise

            indata[:, 1] = dut_sig

        # Simulate blocking time
        duration = n_samples / samplerate
        # time.sleep(duration) 
        
        return indata

import scipy.signal
