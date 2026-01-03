from src.gui.widgets.distortion_analyzer import DistortionAnalyzer


class MockAudioEngine:
    def __init__(self):
        self.sample_rate = 48000
        self.calibration = type('obj', (object,), {'output_gain': 1.0})
    def register_callback(self, cb): return 1
    def unregister_callback(self, cid): pass

def test_amplitude_clamping():
    print("Testing gen_amplitude clamping...")
    engine = MockAudioEngine()
    da = DistortionAnalyzer(engine)

    # Test Normal
    da.gen_amplitude = 0.5
    print(f"Set 0.5 -> {da.gen_amplitude}")
    assert da.gen_amplitude == 0.5

    # Test Huge (Overflow candidate)
    huge_val = 1e100 # Fits in float64, but huge for amplitude
    da.gen_amplitude = huge_val
    print(f"Set 1e100 -> {da.gen_amplitude}")
    assert da.gen_amplitude <= 10.0

    # Test Negative
    da.gen_amplitude = -1.0
    print(f"Set -1.0 -> {da.gen_amplitude}")
    assert da.gen_amplitude == 0.0

    print("Clamping test passed.")

if __name__ == "__main__":
    test_amplitude_clamping()
