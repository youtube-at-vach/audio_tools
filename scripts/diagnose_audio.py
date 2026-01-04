import argparse

import numpy as np
import sounddevice as sd

from src.core.audio_engine import AudioEngine
from src.gui.widgets.freq_response import FreqResponseAnalyzer
from src.gui.widgets.loopback_finder import LoopbackFinder


def main():
    parser = argparse.ArgumentParser(description="Audio Tools Diagnostic Script")
    parser.add_argument("--device", type=int, help="Audio Device ID")
    parser.add_argument("--list-devices", action="store_true", help="List available devices")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # Initialize Audio Engine
    engine = AudioEngine()

    # Select Device
    if args.device is not None:
        device_id = args.device
    else:
        # Default to UAC-232 (Device 3 in previous context, but safer to find by name or use default)
        # Let's try to find "UAC-232" or default to 3 if not found, or ask user.
        # For automation, we prefer explicit.
        print("No device specified. Listing devices:")
        print(sd.query_devices())
        try:
            device_id = int(input("Enter Device ID: "))
        except Exception:
            print("Invalid ID")
            return

    print(f"Selected Device ID: {device_id}")

    # Configure Engine
    # We assume Loopback test: Input and Output on same device
    engine.set_devices(device_id, device_id)
    engine.set_sample_rate(48000)
    engine.set_block_size(1024)
    engine.set_channel_mode('left', 'left') # Default to Left/Left for mono tests if needed, but modules override?
    # Actually, modules use engine.input_channel_mode.
    # Let's set to Stereo to allow modules to pick what they want, or specific modes.
    # Loopback Finder scans ALL channels, so mode doesn't matter much as it uses raw playrec with max channels.
    # Freq Response uses engine mode.

    engine.set_channel_mode('left', 'left') # Start with Left-Left loopback test

    print("\n--- Running Loopback Finder ---")
    finder = LoopbackFinder(engine)
    try:
        paths = finder.perform_scan(device_id, 48000, progress_callback=lambda p, m: print(f"\r{p}%: {m}", end=""))
        print("\nFound Loopback Paths:")
        for p in paths:
            print(f"  Out: {p[0]} -> In: {p[1]} (Level: {20*np.log10(p[2]):.1f} dB)")
    except Exception as e:
        print(f"\nError in Loopback Finder: {e}")

    print("\n--- Running Frequency Response Analyzer (Left -> Left) ---")
    analyzer = FreqResponseAnalyzer(engine)

    # Configure for Left -> Left
    engine.set_channel_mode('left', 'left')

    try:
        results = analyzer.perform_sweep(
            start_f=100, end_f=1000, steps_per_oct=1, amp_db=-20, duration=0.2,
            progress_callback=lambda p, m: print(f"\r{p}%: {m}", end=""),
            result_callback=None
        )
        print("\nSweep Result (First 5 points):")
        for r in results[:5]:
            print(f"  {r[0]:.1f} Hz: {r[1]:.1f} dB, {r[2]:.1f} deg")

        # Check for silence
        avg_level = np.mean([r[1] for r in results])
        print(f"Average Level: {avg_level:.1f} dB")
        if avg_level < -60:
            print("WARNING: Signal level is very low. Check connections or device settings.")

    except Exception as e:
        print(f"\nError in Freq Response Analyzer: {e}")

if __name__ == "__main__":
    main()
