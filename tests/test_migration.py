import os
import sys
import time

import pytest
from PyQt6.QtWidgets import QApplication

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.audio_engine import AudioEngine
from src.gui.widgets.distortion_analyzer import DistortionAnalyzer
from src.gui.widgets.lock_in_amplifier import LockInAmplifier

pytestmark = pytest.mark.skipif(
    os.environ.get("AUDIO_TOOLS_ENABLE_HARDWARE_TESTS") != "1",
    reason="Requires audio hardware/loopback; set AUDIO_TOOLS_ENABLE_HARDWARE_TESTS=1 to run",
)

def test_simultaneous_usage():
    QApplication(sys.argv)

    print("Initializing AudioEngine...")
    engine = AudioEngine()

    print("Initializing Modules...")
    lockin = LockInAmplifier(engine)
    dist = DistortionAnalyzer(engine)

    print("Starting Lock-in Amplifier...")
    lockin.start_analysis()
    time.sleep(0.5)

    print("Starting Distortion Analyzer...")
    dist.start_analysis()
    time.sleep(0.5)

    status = engine.get_status()
    print(f"Engine Status: {status}")

    if status['active_clients'] != 2:
        print(f"FAILED: Expected 2 clients, got {status['active_clients']}")
    else:
        print("SUCCESS: 2 clients active simultaneously.")

    print("Stopping all...")
    lockin.stop_analysis()
    dist.stop_analysis()

    time.sleep(0.5)
    status = engine.get_status()
    print(f"Final Status: {status}")

    if status['active_clients'] != 0:
        print(f"FAILED: Expected 0 clients, got {status['active_clients']}")

    print("Test Complete.")

if __name__ == "__main__":
    test_simultaneous_usage()
