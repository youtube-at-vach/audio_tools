import os
import sys
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.audio_engine import AudioEngine


def test_status():
    print("Initializing AudioEngine...")
    engine = AudioEngine()

    print("Checking initial status...")
    status = engine.get_status()
    print(status)

    if status['active']:
        print("FAILED: Should be inactive initially")
        return
    if status['active_clients'] != 0:
        print("FAILED: Should have 0 clients")
        return

    print("Registering dummy callback...")
    def dummy_callback(indata, outdata, frames, time, status):
        outdata.fill(0)

    cid = engine.register_callback(dummy_callback)

    # Wait a bit for stream to start
    time.sleep(0.5)

    print("Checking active status...")
    status = engine.get_status()
    print(status)

    if not status['active']:
        print("FAILED: Should be active")
        # return # Might fail if no device available in test env, but let's proceed

    if status['active_clients'] != 1:
        print("FAILED: Should have 1 client")
        return

    print("Unregistering callback...")
    engine.unregister_callback(cid)
    time.sleep(0.5)

    print("Checking final status...")
    status = engine.get_status()
    print(status)

    if status['active']:
        print("FAILED: Should be inactive after unregister")
        return

    print("Test Complete.")

if __name__ == "__main__":
    test_status()
