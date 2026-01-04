
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gui.widgets.timecode_monitor import LTCDecoder, TimecodeMonitor


# Mock AudioEngine
class MockAudioEngine:
    def __init__(self):
        self.sample_rate = 48000
        self.callbacks = {}
        self.next_id = 0

    def register_callback(self, cb):
        cid = self.next_id
        self.callbacks[cid] = cb
        self.next_id += 1
        return cid

    def unregister_callback(self, cid):
        if cid in self.callbacks:
            del self.callbacks[cid]

def test_ltc_issues():
    engine = MockAudioEngine()
    monitor = TimecodeMonitor(engine)
    monitor.fps = 30.0
    monitor.generator_enabled = True
    monitor.generator_mode = 'tod'

    print("--- Testing Issue 1: Timezone (Target: UTC) ---")
    monitor.start_analysis()

    # Run callback to generate some frames
    frames_per_block = 1024
    indata = np.zeros((frames_per_block, 2), dtype=np.float32)
    outdata = np.zeros((frames_per_block, 2), dtype=np.float32)

    # We need to simulate enough frames to decode a full LTC frame (80 bits)
    # 30 FPS, 48kHz -> 1600 samples per frame.
    # We need at least 1 frame duration + sync to decode.

    generated_audio = []

    cb = engine.callbacks[monitor.callback_id]

    # Generate ~2 frames worth of audio
    for _ in range(4): # 4 * 1024 = 4096 samples => > 2 frames (3200 samples)
        cb(indata, outdata, frames_per_block, None, None)
        # Capture channel 0 output
        generated_audio.extend(outdata[:, 0])

    generated_audio = np.array(generated_audio)

    # Decode
    decoder = LTCDecoder(48000, 30.0)
    # Use a chunk based process
    chunk_size = 1024
    decoded_tcs = []

    for i in range(0, len(generated_audio), chunk_size):
        chunk = generated_audio[i:i+chunk_size]
        if decoder.process_samples(chunk):
            if decoder.locked:
                decoded_tcs.append(decoder.decoded_tc)

    if not decoded_tcs:
        print("FAIL: No LTC decoded from generator output.")
        sys.exit(1)

    last_tc = decoded_tcs[-1]
    print(f"Decoded TC: {last_tc}")

    # Check against UTC and Local
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now() # System local

    def tc_to_dt(tc, ref_dt):
        try:
            hh, mm, ss, ff = map(int, tc.split(':'))
            # Construct dt using ref_dt date
            return ref_dt.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        except Exception:
            return None

    dt_from_tc_utc = tc_to_dt(last_tc, now_utc)
    #dt_from_tc_local = tc_to_dt(last_tc, now_local) # local comparison is harder due to naive nature, let's stick to UTC check primarily.

    # Make constructed dt aware (UTC) for fair comparison
    if dt_from_tc_utc:
        dt_from_tc_utc = dt_from_tc_utc.replace(tzinfo=timezone.utc)

    # Check deviation
    # We allow some slack purely because of execution time, but UTC vs Local (JST) is 9 hours difference.

    diff_utc = abs((dt_from_tc_utc - now_utc).total_seconds())

    # For local check, we construct naive local time
    # But since we confirmed generator uses astimezone() which implies local time usually.
    # To check if it is explicitly NOT UTC, we can check if diff_utc is large.

    print(f"Current UTC: {now_utc.strftime('%H:%M:%S')}")
    # print(f"Current Local: {now_local.strftime('%H:%M:%S')}")

    if diff_utc < 60.0:
        print("PASS: Generator output is UTC.")
    else:
        print(f"FAIL: Generator output is likely Local Time (Expected UTC). Diff from UTC: {diff_utc:.2f}s")

    monitor.stop_analysis()

    print("\n--- Testing Issue 2: Restart Drift ---")

    # Start again
    # If state is not reset, 'frames_generated' will continue from where it left off,
    # but 'start_analysis' calculates time based on Time.time() (if reset properly?)
    # Wait a bit to ensure a clear time gap
    time.sleep(1.1)

    monitor.start_analysis()
    cb = engine.callbacks[monitor.callback_id]

    # Generate immediate frames
    outdata.fill(0)
    cb(indata, outdata, frames_per_block, None, None) # First block

    # We need to poke internal state to see if valid
    # Check the target variable inside
    # In 'tod' mode:
    # t_target = self._tod_epoch_base + (self.frames_generated / fps)
    # self._tod_epoch_base is set on start_analysis() to time.time()
    # If self.frames_generated is NOT reset to 0, t_target will be:
    # Now + (Old_Frames / FPS) -> Future time!


    # Access private members for white-box testing
    frames_gen = monitor.channels['L'].gen.frames_generated
    print(f"Frames Generated after restart (should be ~1): {frames_gen}")

    # If reset works, it should be 1 (generated 1 frame to fill buffer).
    # If not reset, it would be continued from previous (e.g. 3 + 1 = 4).

    if frames_gen > 2:
        print(f"FAIL: Frames generated is high ({frames_gen}). State was not reset!")
    else:
        print("PASS: Frames generated reset correctly.")

if __name__ == "__main__":
    test_ltc_issues()
