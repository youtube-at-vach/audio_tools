import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.signal_generator import SignalGenerator
from src.core.audio_engine import AudioEngine

def test_noise_generation():
    print("Initializing SignalGenerator...")
    # Mock AudioEngine
    class MockEngine:
        def __init__(self):
            self.sample_rate = 48000
            self.register_callback = lambda x: 0
            self.unregister_callback = lambda x: None
            
    engine = MockEngine()
    sg = SignalGenerator(engine)
    
    noise_types = ['white', 'pink', 'brown', 'blue', 'violet', 'grey']
    
    for color in noise_types:
        print(f"Testing {color} noise...")
        sg.noise_color = color
        
        # Generate 1 second
        noise = sg._generate_noise_buffer(48000, duration=1.0)
        
        # Check for validity
        if np.any(np.isnan(noise)) or np.any(np.isinf(noise)):
            print(f"FAILED: {color} noise contains NaNs or Infs")
            continue
            
        rms = np.sqrt(np.mean(noise**2))
        peak = np.max(np.abs(noise))
        
        print(f"  RMS: {rms:.4f}, Peak: {peak:.4f}")
        
        if peak == 0:
            print(f"FAILED: {color} noise is silent")
            continue
            
        # Rough Slope Check
        # FFT
        fft = np.abs(np.fft.rfft(noise))
        freqs = np.fft.rfftfreq(len(noise), d=1/48000)
        
        # Ignore DC and very low freq
        idx = np.where((freqs > 100) & (freqs < 10000))[0]
        
        if len(idx) > 0:
            f_log = np.log10(freqs[idx])
            mag_log = 20 * np.log10(fft[idx] + 1e-12)
            
            # Linear fit
            slope, intercept = np.polyfit(f_log, mag_log, 1)
            print(f"  Estimated Slope: {slope:.2f} dB/decade")
            
            # Expected slopes (approx)
            # White: 0
            # Pink: -10 (3dB/oct * 3.32 oct/dec = 10)
            # Brown: -20
            # Blue: +10
            # Violet: +20
            
    print("Test Complete.")

if __name__ == "__main__":
    test_noise_generation()
