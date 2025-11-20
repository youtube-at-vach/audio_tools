import sounddevice as sd
import numpy as np
import time

OUTPUT_DEVICE = 17
TEST_FREQ = 1000
DURATION = 0.5
SAMPLE_RATE = 48000

def test_input_device(input_device_id):
    print(f"Testing input device {input_device_id}...")
    try:
        # Generate sine wave
        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        sine_wave = 0.5 * np.sin(2 * np.pi * TEST_FREQ * t)
        sine_wave = sine_wave.reshape(-1, 1)

        # Record and play
        # We need to play to OUTPUT_DEVICE and record from input_device_id
        # Since they are different devices, we can't use sd.playrec easily if they are not synchronized?
        # Actually sd.Stream allows different devices for input and output.
        
        recorded_data = []
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            recorded_data.append(indata.copy())
            # We need to fill outdata with our sine wave
            # But wait, the callback is called for BOTH input and output if we use a duplex stream.
            # If we use different devices, sounddevice handles it.
            
            # However, we need to manage the buffer state.
            # For simplicity, let's just play silence to output in the callback if we run out of data
            # But wait, we want to play the sine wave.
            pass

        # Let's use a simpler approach: start an output stream and an input stream separately?
        # No, sd.Stream with device=(in, out) is better.
        
        # We need to pass the data to the callback.
        playback_index = 0
        
        def callback_rw(indata, outdata, frames, time, status):
            nonlocal playback_index
            recorded_data.append(indata.copy())
            
            chunk_len = len(outdata)
            if playback_index + chunk_len <= len(sine_wave):
                outdata[:] = sine_wave[playback_index:playback_index+chunk_len]
                playback_index += chunk_len
            else:
                # Fill remainder with zeros
                remaining = len(sine_wave) - playback_index
                if remaining > 0:
                    outdata[:remaining] = sine_wave[playback_index:]
                    outdata[remaining:] = 0
                    playback_index += remaining
                else:
                    outdata[:] = 0

        with sd.Stream(device=(input_device_id, OUTPUT_DEVICE),
                       samplerate=SAMPLE_RATE, blocksize=1024,
                       channels=1, callback=callback_rw):
            time.sleep(DURATION + 0.1)
            
        # Analyze recording
        full_recording = np.concatenate(recorded_data)
        rms = np.sqrt(np.mean(full_recording**2))
        print(f"  RMS: {rms}")
        
        if rms > 0.01:
            print(f"  *** FOUND SIGNAL ON DEVICE {input_device_id} ***")
            return True
            
    except Exception as e:
        print(f"  Error: {e}")
        return False

print("Scanning input devices...")
devices = sd.query_devices()
for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        if test_input_device(i):
            break
