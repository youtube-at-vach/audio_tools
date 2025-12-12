import unittest
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.audio_engine import AudioEngine

class TestAudioEngine(unittest.TestCase):
    def setUp(self):
        # This test requires a specific physical loopback wiring/device.
        # By default we skip it to keep CI/dev machines without hardware green.
        if os.environ.get("AUDIO_TOOLS_ENABLE_HARDWARE_TESTS") != "1":
            self.skipTest("Requires audio hardware + loopback wiring. Set AUDIO_TOOLS_ENABLE_HARDWARE_TESTS=1 to run.")

        self.engine = AudioEngine()
        # Use UAC-232 (Device 3) as per user instructions
        # Connected Output R (Channel 2) -> Input L (Channel 1)
        self.device_id = 3
        self.engine.set_devices(self.device_id, self.device_id)

    def test_stream_loopback(self):
        """Test that we can send and receive audio through the loopback device."""
        
        # Generate a sine wave
        frequency = 1000
        duration = 1.0
        sample_rate = 48000
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Buffer to store recorded data
        recorded_data = []
        
        # Index to keep track of playback position
        playback_index = 0
        
        def callback(indata, outdata, frames, time, status):
            nonlocal playback_index
            if status:
                print(status)
            
            # Record input (Channel 1 / Index 0)
            recorded_data.append(indata[:, 0].copy())
            
            # Play output (Channel 2 / Index 1)
            chunk = sine_wave[playback_index:playback_index + frames]
            
            # Clear output buffer first
            outdata.fill(0)
            
            if len(chunk) < frames:
                # Pad with zeros if end of buffer
                outdata[:len(chunk), 1] = chunk
                playback_index = 0 
            else:
                outdata[:, 1] = chunk
                playback_index += frames
            
            if playback_index % (sample_rate // 2) < frames: 
                print(f"Callback: Out Max={np.max(np.abs(outdata))}, In Max={np.max(np.abs(indata))}")

        # Start stream with 2 channels
        self.engine.start_stream(callback, channels=2)
        time.sleep(1.0) # Run for 1 second
        self.engine.stop_stream()
        
        # Analyze recorded data
        full_recording = np.concatenate(recorded_data)
        
        # Check if we recorded something significant (not silence)
        rms = np.sqrt(np.mean(full_recording**2))
        print(f"Recorded RMS: {rms}")
        
        self.assertTrue(rms > 0.01, "Recorded signal is too quiet, loopback might not be working")

if __name__ == '__main__':
    unittest.main()
