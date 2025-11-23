import sys
import os
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config_manager import ConfigManager

def test_config_persistence():
    config_path = "test_config.json"
    
    # Clean up previous test
    if os.path.exists(config_path):
        os.remove(config_path)
        
    print("Initializing ConfigManager...")
    cm = ConfigManager(config_path)
    
    print("Checking default values...")
    in_dev, out_dev = cm.get_last_devices()
    if in_dev is not None or out_dev is not None:
        print("FAILED: Default devices should be None")
        return
        
    print("Setting devices...")
    cm.set_last_devices("My Mic", "My Speaker")
    
    print("Verifying in-memory update...")
    in_dev, out_dev = cm.get_last_devices()
    if in_dev != "My Mic" or out_dev != "My Speaker":
        print(f"FAILED: In-memory update failed. Got {in_dev}, {out_dev}")
        return
        
    print("Verifying file persistence...")
    # Create new instance to load from file
    cm2 = ConfigManager(config_path)
    in_dev, out_dev = cm2.get_last_devices()
    if in_dev != "My Mic" or out_dev != "My Speaker":
        print(f"FAILED: File persistence failed. Got {in_dev}, {out_dev}")
        return
        
    print("Cleaning up...")
    if os.path.exists(config_path):
        os.remove(config_path)
        
    print("Test Complete.")

if __name__ == "__main__":
    test_config_persistence()
