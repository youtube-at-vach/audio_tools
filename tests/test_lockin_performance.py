import sys
import os
import time
import numpy as np
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.audio_engine import AudioEngine
from src.gui.widgets.lock_in_amplifier import LockInAmplifier

def run_sweep(start_freq, end_freq, steps, duration_per_step, use_software_loopback=False):
    print("Initializing Audio Engine...")
    engine = AudioEngine()
    
    # List devices if needed, but for now we use default or software loopback
    if use_software_loopback:
        engine.set_loopback(True)
        print("Software Loopback Enabled. This tests the algorithm logic without hardware latency.")
    else:
        print("Using Hardware I/O. Ensure a loopback cable is connected from Output to Input.")
        # You might want to select specific devices here if defaults are not correct
        # engine.set_devices(in_id, out_id)

    print("Initializing Lock-in Amplifier...")
    lockin = LockInAmplifier(engine)
    
    # Configure Lock-in
    lockin.gen_amplitude = 0.5
    lockin.averaging_count = 10 # Average over 10 buffers
    lockin.buffer_size = 4096
    lockin.output_channel = 2 # Stereo output (Signal on both L and R)
    
    # Start Analysis
    lockin.start_analysis()
    
    # Give it a moment to start the stream
    time.sleep(1.0)
    
    freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), steps)
    
    results = []
    
    print(f"{'Freq (Hz)':<12} | {'Mag (Linear)':<12} | {'Phase (Deg)':<12} | {'Gain Err(dB)':<12}")
    print("-" * 60)
    
    try:
        for f in freqs:
            lockin.gen_frequency = f
            
            # Wait for settling
            # We need to wait at least (buffer_size / sample_rate) * averaging_count
            # Plus some margin for the filter to settle
            buffer_duration = lockin.buffer_size / engine.sample_rate
            min_wait = buffer_duration * 2 # Initial settle
            wait_time = max(duration_per_step, min_wait)
            
            time.sleep(wait_time)
            
            # Clear history to ensure we don't average old data
            lockin.history.clear()
            
            # Collect data
            for _ in range(lockin.averaging_count):
                 time.sleep(buffer_duration * 1.1) # Wait slightly more than buffer duration
                 lockin.process_data()
            
            # Read measurement
            mag = lockin.current_magnitude
            phase = lockin.current_phase
            
            # Calculate Errors
            # Gain Error (dB)
            expected_mag = lockin.gen_amplitude
            if mag > 1e-9:
                gain_err_db = 20 * np.log10(mag / expected_mag)
            else:
                gain_err_db = -999.0
            
            print(f"{f:<12.2f} | {mag:<12.4f} | {phase:<12.2f} | {gain_err_db:<12.3f}")
            
            results.append({
                'freq': f,
                'mag': mag,
                'phase': phase,
                'gain_err_db': gain_err_db
            })
            
    except KeyboardInterrupt:
        print("\nSweep interrupted.")
    finally:
        lockin.stop_analysis()
        engine.stop_stream()
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lock-in Amplifier Performance Test")
    parser.add_argument("--start", type=float, default=20.0, help="Start Frequency (Hz)")
    parser.add_argument("--end", type=float, default=20000.0, help="End Frequency (Hz)")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps")
    parser.add_argument("--time", type=float, default=0.5, help="Time per step (s)")
    parser.add_argument("--software-loopback", action="store_true", help="Use internal software loopback")
    
    args = parser.parse_args()
    
    run_sweep(args.start, args.end, args.steps, args.time, args.software_loopback)
