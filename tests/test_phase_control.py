import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.signal_generator import SignalGenerator, SignalParameters

class MockAudioEngine:
    def __init__(self):
        self.sample_rate = 48000
    def register_callback(self, cb):
        return 1
    def unregister_callback(self, id):
        pass

def test_phase_control():
    engine = MockAudioEngine()
    gen = SignalGenerator(engine)
    
    print("Testing Phase Control Logic...")
    
    sample_rate = 48000
    frames = 100
    t = np.arange(frames) / sample_rate
    
    # 1. Sine Wave Phase Offset
    gen.params_L.waveform = 'sine'
    gen.params_L.frequency = 1000
    gen.params_L.amplitude = 1.0
    gen.params_L.phase_offset = 0.0
    
    # Generate 0 deg
    gen.start_generation()
    # Access private method for testing logic directly or simulate callback
    # Since _generate_channel_signal is internal, let's use the callback logic simulation
    # But _generate_channel_signal is defined inside start_generation.
    # We can't easily access it.
    # However, we can use the callback if we mock it properly.
    
    # Let's just re-implement the logic snippet to verify math correctness
    # or rely on the fact that we modified the code.
    
    # Actually, we can just instantiate SignalParameters and call the logic if we extract it?
    # No, it's inside a closure.
    
    # Let's run the callback and capture output.
    outdata = np.zeros((frames, 2))
    
    # 0 deg
    gen.params_L.phase_offset = 0.0
    gen.params_L._phase = 0 # Reset
    
    # We need to trigger the callback. 
    # The callback is defined inside start_generation and registered.
    # But we can't access the local function `callback`.
    # Wait, `self.callback_id = self.audio_engine.register_callback(callback)`
    # The engine mock mock just returns 1. It doesn't store the callback.
    
    # Let's modify the MockAudioEngine to store the callback.
    pass

class BetterMockAudioEngine:
    def __init__(self):
        self.sample_rate = 48000
        self.cb = None
    def register_callback(self, cb):
        self.cb = cb
        return 1
    def unregister_callback(self, id):
        self.cb = None

def test_phase_control_real():
    engine = BetterMockAudioEngine()
    gen = SignalGenerator(engine)
    gen.start_generation()
    
    callback = engine.cb
    assert callback is not None
    
    frames = 48 # 1ms at 48k
    outdata = np.zeros((frames, 2))
    
    # Test 1: 0 vs 90 degrees
    gen.params_L.waveform = 'sine'
    gen.params_L.frequency = 1000
    gen.params_L.amplitude = 1.0
    gen.params_L.phase_offset = 0.0
    gen.params_L._phase = 0
    
    gen.params_R.waveform = 'sine'
    gen.params_R.frequency = 1000
    gen.params_R.amplitude = 1.0
    gen.params_R.phase_offset = 90.0
    gen.params_R._phase = 0
    
    gen.output_mode = 'STEREO'
    
    callback(None, outdata, frames, 0, None)
    
    l = outdata[:, 0]
    r = outdata[:, 1]
    
    # Check correlation
    # Sine and Cosine (90 deg shift) should have 0 correlation
    corr = np.sum(l * r) / (np.sqrt(np.sum(l**2)) * np.sqrt(np.sum(r**2)))
    print(f"Correlation (0 vs 90): {corr:.4f} (Expected 0.0)")
    assert abs(corr) < 0.01
    
    # Test 2: 0 vs 180 degrees
    gen.params_R.phase_offset = 180.0
    gen.params_R._phase = 0
    gen.params_L._phase = 0
    
    callback(None, outdata, frames, 0, None)
    l = outdata[:, 0]
    r = outdata[:, 1]
    
    corr = np.sum(l * r) / (np.sqrt(np.sum(l**2)) * np.sqrt(np.sum(r**2)))
    print(f"Correlation (0 vs 180): {corr:.4f} (Expected -1.0)")
    assert np.isclose(corr, -1.0, atol=0.01)
    
    print("All tests passed!")

if __name__ == "__main__":
    test_phase_control_real()
