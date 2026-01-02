import numpy as np
import pytest
from src.gui.widgets.bnim_meter import BNIMMeter

class MockAudioEngine:
    def __init__(self):
        self.sample_rate = 48000
    def register_callback(self, cb):
        return 1
    def unregister_callback(self, cid):
        pass

def test_bnim_processing():
    engine = MockAudioEngine()
    bnim = BNIMMeter(engine)
    bnim.start_analysis()
    
    # Generate a stereo signal with 0.4ms delay (ITD)
    fs = 48000
    t = np.arange(bnim.fft_size) / fs
    freq_test = 1000.0
    
    L = np.sin(2 * np.pi * freq_test * t)
    # Delay by 0.4ms
    R = np.sin(2 * np.pi * freq_test * (t - 0.0004))
    
    # Fill buffer
    bnim.audio_buffer = np.zeros((bnim.fft_size, 2))
    bnim.audio_buffer[:, 0] = L
    bnim.audio_buffer[:, 1] = R
    
    # Process
    bnim.process_buffer()
    
    # Check neural map
    neural_map = bnim.neural_map
    assert neural_map is not None
    
    # Find peak
    freq_idx = np.argmin(np.abs(bnim.frequencies - 1000.0))
    itd_pattern = neural_map[freq_idx]
    
    peak_itd_idx = np.argmax(itd_pattern)
    peak_itd_ms = bnim.itd_axis[peak_itd_idx]
    
    # Alignment (peak) occurs when tau = -delay
    assert abs(peak_itd_ms - (-0.4)) < 0.05
    
    bnim.stop_analysis()

def test_bnim_mono():
    engine = MockAudioEngine()
    bnim = BNIMMeter(engine)
    bnim.start_analysis()
    
    # Mono signal (L=R)
    fs = 48000
    t = np.arange(bnim.fft_size) / fs
    L = np.sin(2 * np.pi * 1000.0 * t)
    R = L
    
    bnim.audio_buffer = np.zeros((bnim.fft_size, 2))
    bnim.audio_buffer[:, 0] = L
    bnim.audio_buffer[:, 1] = R
    
    bnim.process_buffer()
    
    freq_idx = np.argmin(np.abs(bnim.frequencies - 1000.0))
    itd_pattern = bnim.neural_map[freq_idx]
    peak_itd_ms = bnim.itd_axis[np.argmax(itd_pattern)]
    
    # Peak should be at 0ms
    print(f"ITD axis: {bnim.itd_axis}")
    print(f"Pattern: {itd_pattern}")
    print(f"Peak ITD: {peak_itd_ms}")
    assert abs(peak_itd_ms) < 0.1 # Relaxed slightly for resolution
    
    bnim.stop_analysis()
