
import os
import sys

import numpy as np
import soundfile as sf
from PyQt6.QtCore import QCoreApplication

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.inverse_filter import ProcessingWorker


def test_inverse_filter():
    QCoreApplication([])

    print("Testing Inverse Filter Logic...")

    # 1. Create Dummy Input
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr*duration))
    # Input: Sine wave at 1kHz, Amplitude 1.0 (0 dBFS)
    input_sig = np.sin(2 * np.pi * 1000 * t) * 1.0

    input_path = "test_input.wav"
    output_path = "test_output.wav"

    sf.write(input_path, input_sig, sr)

    # 2. Calibration Map
    # Assume System adds +6dB gain at 1kHz.
    # Calibration Map says: at 1kHz, correction is +6dB (meaning measured was +6 relative to reality? No wait.)
    # In LockInAmplifier: Corrected = Measured - Correction.
    # If System Gain is +6dB. Measured = +6dB (for 0dB input).
    # We want Corrected = 0dB.
    # So Correction = +6dB.
    # So Calibration Map should have +6dB at 1kHz.

    # Inverse Filter should perform -6dB (x0.5).

    cal_map = [
        [20, 6.0, 0.0],
        [1000, 6.0, 0.0],
        [20000, 6.0, 0.0]
    ]

    # 3. Setup Worker
    # max_gain_db = 20, taps = 1024, smoothing = 0, normalize = False
    worker = ProcessingWorker(input_path, output_path, cal_map, 20.0, 4096, 0.0, False)

    # run() is blocking if called directly? No, run() is the method executed by start().
    # But QThread.run() is just a method. We can call it directly to test logic synchronously
    # (though signals won't be emitted in a thread-safe way loop, but here single thread is fine).
    # Actually, signals might need an event loop if connected?
    # I won't connect signals. I just check the output file.

    try:
        worker.run()
    except Exception as e:
        print(f"Worker run failed: {e}")
        return

    # 4. Check Output
    data, r_sr = sf.read(output_path)

    # Expected: Amplitude ~0.5
    # Skip start/end (filter warm up / transient)
    mid_data = data[int(sr*0.4):int(sr*0.6)]
    np.sqrt(np.mean(mid_data**2))
    peak = np.max(np.abs(mid_data))

    print("Input Peak: 1.0")
    print(f"Output Peak: {peak:.4f}")

    expected = 0.5
    error = abs(peak - expected)

    if error < 0.05: # Allow some tolerance (ripple, windowing)
        print("PASS: Output amplitude is approximately 0.5 (-6dB).")
    else:
        print(f"FAIL: Output amplitude {peak} is not close to {expected}.")

    # Cleanup
    if os.path.exists(input_path): os.remove(input_path)
    if os.path.exists(output_path): os.remove(output_path)

if __name__ == "__main__":
    test_inverse_filter()
