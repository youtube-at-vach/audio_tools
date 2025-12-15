import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.signal_generator import SignalGenerator

def test_independent_channels():
    # Mock AudioEngine
    mock_engine = MagicMock()
    mock_engine.sample_rate = 48000
    mock_engine.calibration.output_gain = 1.0
    
    sg = SignalGenerator(mock_engine)
    
    # Configure L: Sine 1000Hz
    sg.params_L.waveform = 'sine'
    sg.params_L.frequency = 1000.0
    sg.params_L.amplitude = 1.0
    
    # Configure R: Square 500Hz
    sg.params_R.waveform = 'square'
    sg.params_R.frequency = 500.0
    sg.params_R.amplitude = 0.5
    
    # Start generation
    sg.start_generation()
    
    # Simulate callback
    frames = 480
    outdata = np.zeros((frames, 2))
    
    # We need to access the callback that was registered
    # In the code: self.callback_id = self.audio_engine.register_callback(callback)
    # We can inspect the mock to get the callback
    args, _ = mock_engine.register_callback.call_args
    callback = args[0]
    
    callback(None, outdata, frames, None, None)
    
    # Analyze output
    sig_l = outdata[:, 0]
    sig_r = outdata[:, 1]
    
    # Check L (Sine)
    t = np.arange(frames) / 48000
    expected_l = np.sin(2 * np.pi * 1000 * t)
    
    # Check R (Square)
    expected_r = 0.5 * np.sign(np.sin(2 * np.pi * 500 * t))
    
    # Verify
    if np.allclose(sig_l, expected_l, atol=1e-5):
        print("Left Channel: PASS (Sine 1000Hz)")
    else:
        print("Left Channel: FAIL")
        print("Max Diff:", np.max(np.abs(sig_l - expected_l)))
        
    if np.allclose(sig_r, expected_r, atol=1e-5):
        print("Right Channel: PASS (Square 500Hz)")
    else:
        print("Right Channel: FAIL")
        print("Max Diff:", np.max(np.abs(sig_r - expected_r)))

    # Test Output Routing
    print("\nTesting Routing: Left Only")
    sg.output_mode = 'L'
    outdata.fill(0)
    callback(None, outdata, frames, None, None)
    if np.all(outdata[:, 1] == 0) and not np.all(outdata[:, 0] == 0):
        print("Routing L: PASS")
    else:
        print("Routing L: FAIL")
        
    print("\nTesting Routing: Right Only")
    sg.output_mode = 'R'
    outdata.fill(0)
    callback(None, outdata, frames, None, None)
    if np.all(outdata[:, 0] == 0) and not np.all(outdata[:, 1] == 0):
        print("Routing R: PASS")
    else:
        print("Routing R: FAIL")

if __name__ == "__main__":
    test_independent_channels()
