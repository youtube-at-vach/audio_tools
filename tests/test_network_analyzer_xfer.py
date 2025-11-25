import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.network_analyzer import NetworkAnalyzer

def test_xfer_mode():
    # Mock AudioEngine
    mock_engine = MagicMock()
    mock_engine.sample_rate = 48000
    mock_engine.calibration.input_sensitivity = 1.0
    
    na = NetworkAnalyzer(mock_engine)
    na.input_mode = 'XFER'
    na.output_channel = 'STEREO'
    
    # Mock PlayRecSession
    # We need to simulate a session where:
    # Ch0 (Ref) = Tone
    # Ch1 (Meas) = Tone * 0.5 (6dB attenuation)
    
    def run_play_rec_mock(output_data, input_channels=1):
        frames = len(output_data)
        input_data = np.zeros((frames, input_channels))
        
        # Simulate Loopback
        # Ch0 = Output Ch0
        input_data[:, 0] = output_data[:, 0]
        
        # Ch1 = Output Ch0 * 0.5 (Simulate DUT with -6dB gain)
        if input_channels > 1:
            input_data[:, 1] = output_data[:, 0] * 0.5
            
        return input_data
        
    na.run_play_rec = run_play_rec_mock
    
    # Run Stepped Sine Sweep (Single Point for speed)
    na.start_freq = 1000
    na.end_freq = 1000
    na.steps_per_octave = 1
    na.num_averages = 1
    
    # Capture emitted signals
    results = []
    na.signals.update_plot.connect(lambda f, m, p: results.append((f, m, p)))
    
    # Run synchronously (bypass thread for test)
    worker = MagicMock()
    worker.is_running = True
    na._execute_sweep(worker)
    
    if not results:
        print("No results emitted!")
        return
        
    freq, mag, phase = results[0]
    print(f"Freq: {freq} Hz, Mag: {mag:.2f} dB, Phase: {phase:.2f} deg")
    
    # Expected: Mag = 20*log10(0.5 / 1.0) = -6.02 dB
    if abs(mag - (-6.02)) < 0.1:
        print("XFER Magnitude: PASS")
    else:
        print(f"XFER Magnitude: FAIL (Expected -6.02, Got {mag:.2f})")
        
    # Expected: Phase = 0 (since no delay/phase shift simulated)
    if abs(phase) < 1.0:
        print("XFER Phase: PASS")
    else:
        print(f"XFER Phase: FAIL (Expected 0, Got {phase:.2f})")

if __name__ == "__main__":
    test_xfer_mode()
