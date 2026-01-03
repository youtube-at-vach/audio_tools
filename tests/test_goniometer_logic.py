import os
import sys

import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.goniometer import Goniometer


class MockAudioEngine:
    def register_callback(self, cb):
        return 1
    def unregister_callback(self, id):
        pass

def test_goniometer():
    engine = MockAudioEngine()
    gonio = Goniometer(engine)

    print("Testing Goniometer Logic...")

    # 1. Mono (In-Phase)
    frames = 1024
    l = np.sin(np.linspace(0, 100, frames))
    r = l.copy()
    data = np.column_stack((l, r))

    # Call callback manually
    outdata = np.zeros_like(data)
    gonio._callback(data, outdata, frames, 0, None)

    print(f"Mono Correlation: {gonio.correlation:.4f} (Expected 1.0)")
    assert np.isclose(gonio.correlation, 1.0, atol=0.01)

    # 2. Inverted (Anti-Phase)
    r = -l
    data = np.column_stack((l, r))
    gonio._callback(data, outdata, frames, 0, None)

    print(f"Inverted Correlation: {gonio.correlation:.4f} (Expected -1.0)")
    assert np.isclose(gonio.correlation, -1.0, atol=0.01)

    # 3. Left Only
    r = np.zeros_like(l)
    data = np.column_stack((l, r))
    gonio._callback(data, outdata, frames, 0, None)

    print(f"Left Only Correlation: {gonio.correlation:.4f} (Expected 0.0)")
    assert np.isclose(gonio.correlation, 0.0, atol=0.01)

    # 4. Stereo (Random)
    # Random noise should be close to 0 correlation on average
    np.random.seed(42)
    l = np.random.randn(frames)
    r = np.random.randn(frames)
    data = np.column_stack((l, r))
    gonio._callback(data, outdata, frames, 0, None)

    print(f"Random Stereo Correlation: {gonio.correlation:.4f} (Expected ~0.0)")
    assert abs(gonio.correlation) < 0.1 # Should be low

    print("All tests passed!")

if __name__ == "__main__":
    test_goniometer()
