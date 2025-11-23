import sys
import os
import time
from PyQt6.QtWidgets import QApplication

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.audio_engine import AudioEngine
from src.gui.widgets.imd_analyzer import IMDAnalyzer
from src.gui.widgets.lock_in_amplifier import LockInAmplifier
from src.gui.widgets.distortion_analyzer import DistortionAnalyzer

def test_simultaneous_usage():
    app = QApplication(sys.argv)
    
    print("Initializing AudioEngine...")
    engine = AudioEngine()
    
    print("Initializing Modules...")
    imd = IMDAnalyzer(engine)
    lockin = LockInAmplifier(engine)
    dist = DistortionAnalyzer(engine)
    
    print("Starting IMD Analyzer...")
    imd.start_analysis()
    time.sleep(0.5)
    
    print("Starting Lock-in Amplifier...")
    lockin.start_analysis()
    time.sleep(0.5)
    
    print("Starting Distortion Analyzer...")
    dist.start_analysis()
    time.sleep(0.5)
    
    status = engine.get_status()
    print(f"Engine Status: {status}")
    
    if status['active_clients'] != 3:
        print(f"FAILED: Expected 3 clients, got {status['active_clients']}")
    else:
        print("SUCCESS: 3 clients active simultaneously.")
        
    print("Stopping all...")
    imd.stop_analysis()
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
