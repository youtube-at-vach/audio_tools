import sys
import os
import time
import numpy as np
from PyQt6.QtWidgets import QApplication

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.audio_engine import AudioEngine
from src.gui.widgets.network_analyzer import NetworkAnalyzer

def test_network_analyzer_migration():
    app = QApplication(sys.argv)
    
    print("Initializing AudioEngine...")
    engine = AudioEngine()
    
    print("Initializing NetworkAnalyzer...")
    na = NetworkAnalyzer(engine)
    
    print("Testing Latency Calibration (Mock)...")
    # We can't easily test full audio loopback without hardware, 
    # but we can verify it runs without crashing and uses the callback.
    
    # Mocking the run_play_rec to avoid actual long wait if no audio device or silence
    # But we want to test the actual callback logic if possible.
    # If we run it, it should finish in ~0.5s + overhead.
    
    try:
        na.calibrate_latency()
        print("Latency Calibration finished.")
    except Exception as e:
        print(f"Latency Calibration failed: {e}")

    # Check status
    status = engine.get_status()
    print(f"Engine Status: {status}")
    
    if status['active_clients'] == 0:
        print("SUCCESS: Client unregistered after calibration.")
    else:
        print(f"FAILED: Client still active: {status['active_clients']}")

    print("Test Complete.")

if __name__ == "__main__":
    test_network_analyzer_migration()
