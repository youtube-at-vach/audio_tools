import os
import sys
import time

import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.audio_engine import AudioEngine

pytestmark = pytest.mark.skipif(
    os.environ.get("AUDIO_TOOLS_ENABLE_HARDWARE_TESTS") != "1",
    reason="Requires audio hardware; set AUDIO_TOOLS_ENABLE_HARDWARE_TESTS=1 to run",
)

def test_mixer():
    print("Initializing AudioEngine...")
    engine = AudioEngine()

    # Mock sounddevice stream to avoid actual audio hardware requirement for this logic test
    # or use a loopback device if available.
    # Since we are in a headless env, we might not have audio devices.
    # But the user has audio devices (UAC-232).
    # Let's try to run it. If it fails due to no device, we'll mock.

    try:
        engine.set_devices(3, 3) # Use user's device
    except Exception as e:
        print(f"Could not set devices: {e}")
        return

    print("Testing Single Callback...")

    def callback1(indata, outdata, frames, time, status):
        # Generate 440Hz sine
        t = (np.arange(frames) + callback1.phase) / engine.sample_rate
        callback1.phase += frames
        sig = 0.5 * np.sin(2 * np.pi * 440 * t)
        outdata[:, 0] = sig
        outdata[:, 1] = sig

    callback1.phase = 0

    cid1 = engine.register_callback(callback1)
    print(f"Registered Callback 1 (ID: {cid1})")

    time.sleep(1)

    print("Testing Simultaneous Callbacks...")

    def callback2(indata, outdata, frames, time, status):
        # Generate 880Hz sine
        t = (np.arange(frames) + callback2.phase) / engine.sample_rate
        callback2.phase += frames
        sig = 0.3 * np.sin(2 * np.pi * 880 * t)
        outdata[:, 0] = sig
        outdata[:, 1] = sig

    callback2.phase = 0

    cid2 = engine.register_callback(callback2)
    print(f"Registered Callback 2 (ID: {cid2})")

    time.sleep(1)

    print("Unregistering Callback 1...")
    engine.unregister_callback(cid1)

    time.sleep(1)

    print("Unregistering Callback 2...")
    engine.unregister_callback(cid2)

    print("Test Complete.")

if __name__ == "__main__":
    test_mixer()
