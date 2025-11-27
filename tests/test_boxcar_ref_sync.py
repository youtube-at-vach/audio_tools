import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.boxcar_averager import BoxcarAverager

class MockAudioEngine:
    def __init__(self):
        self.sample_rate = 48000
    def register_callback(self, cb):
        return 1
    def unregister_callback(self, id):
        pass

def test_boxcar_ref_sync():
    engine = MockAudioEngine()
    boxcar = BoxcarAverager(engine)
    
    print("Testing Boxcar Averager External Sync...")
    
    # Setup
    boxcar.mode = 'External Reference'
    boxcar.ref_channel = 1 # Right
    boxcar.input_channel = 'Left'
    boxcar.period_samples = 100
    boxcar.trigger_level = 0.0
    boxcar.trigger_edge = 'Rising'
    
    boxcar.start_analysis()
    
    # Create Test Data
    # Ref: Square wave, period 200 samples
    # Sig: Sine wave, period 200 samples, aligned
    
    frames = 1000
    t = np.arange(frames)
    
    # Ref: Rising edge at 50, 250, 450...
    ref = np.zeros(frames)
    ref[50:150] = 1.0
    ref[250:350] = 1.0
    ref[450:550] = 1.0
    ref[650:750] = 1.0
    ref[850:950] = 1.0
    ref -= 0.5 # Center at 0
    
    # Sig: Ramp 0 to 1 over 100 samples starting at trigger
    sig = np.zeros(frames)
    
    # Fill signal segments corresponding to triggers
    for start in [50, 250, 450, 650, 850]:
        sig[start:start+100] = np.linspace(0, 1, 100)
        
    data = np.column_stack((sig, ref))
    
    # Feed data to callback
    outdata = np.zeros_like(data)
    boxcar._callback(data, outdata, frames, 0, None)
    
    # Process
    boxcar.process()
    
    print(f"Count: {boxcar.count}")
    assert boxcar.count == 5 # Should have captured 5 windows
    
    # Check Accumulator
    # Should be sum of 5 ramps -> 5 * linspace(0, 1, 100)
    expected = np.linspace(0, 1, 100) * 5
    
    # Accumulator is (period, 2)
    # We recorded Left channel
    acc_l = boxcar.accumulator[:, 0]
    
    # Verify
    diff = np.abs(acc_l - expected)
    max_diff = np.max(diff)
    print(f"Max Difference: {max_diff:.4f}")
    assert max_diff < 1e-5
    
    print("All tests passed!")

if __name__ == "__main__":
    test_boxcar_ref_sync()
