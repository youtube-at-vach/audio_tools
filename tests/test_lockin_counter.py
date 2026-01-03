
import numpy as np

from src.gui.widgets.lock_in_frequency_counter import LockInFrequencyCounter


# Mock AudioEngine
class MockAudioEngine:
    def __init__(self):
        self.sample_rate = 48000

    def register_callback(self, cb):
        return 1

    def unregister_callback(self, id):
        pass

def test_frequency_calculation():
    engine = MockAudioEngine()
    counter = LockInFrequencyCounter(engine)
    counter.gen_frequency = 1000.0 # NCO

    # Create a signal at 1001.0 Hz (Delta = +1.0 Hz)
    sr = 48000
    duration = 4096 / sr
    t = np.arange(4096) / sr
    sig = np.cos(2 * np.pi * 1001.0 * t) # 1001 Hz

    # Fill input data
    counter.input_data[:, 0] = sig
    counter.is_running = True

    # Process
    counter.process_data()

    print(f"Measured Delta F: {counter.current_freq_dev}")

    # Allow small error due to windowing/segmentation approximate method
    assert abs(counter.current_freq_dev - 1.0) < 0.05

def test_negative_frequency_deviation():
    engine = MockAudioEngine()
    counter = LockInFrequencyCounter(engine)
    counter.gen_frequency = 1000.0

    # Create a signal at 999.0 Hz (Delta = -1.0 Hz)
    sr = 48000
    t = np.arange(4096) / sr
    sig = np.cos(2 * np.pi * 999.0 * t)

    counter.input_data[:, 0] = sig
    counter.is_running = True

    counter.process_data()

    print(f"Measured Delta F: {counter.current_freq_dev}")
    assert abs(counter.current_freq_dev - (-1.0)) < 0.05
