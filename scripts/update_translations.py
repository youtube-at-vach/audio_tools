#!/usr/bin/env python3
"""
Script to automatically update translation files with missing keys.
This script adds missing keys from the check_trn_keys.py output to all language files.
"""

import json
import os

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANG_DIR = os.path.join(PROJECT_ROOT, "src", "assets", "lang")

# Missing keys in en.json (from check_trn_keys.py output)
MISSING_EN_KEYS = [
    "",
    "-- %",
    "-- dB",
    "-INF",
    "-inf dBFS",
    "-inf dBV",
    "-inf dBu",
    "0 dB",
    "0.0 s",
    "0.000 V",
    "0.000 deg",
    "1. Calibration Data",
    "20 Hz - 12.5 kHz",
    "20 Hz - 20 kHz (Wide)",
    "20 Hz - 8 kHz (Normal)",
    "ACTIVE",
    "Absolute Gain Calibrated.\nOffset adjusted by {0:+.3f} dB.\nNew Offset: {1:.3f} dB",
    "Audio Buffer Error: {0}",
    "Auto",
    "Avg",
    "Avg: {0}%",
    "BPF",
    "CF: {0:.1f}",
    "CPU Load of Audio Thread",
    "CPU: 0%",
    "CPU: {0:.1f}%",
    "CPU: {0:.1f}% [{1}]",
    "Calibration is currently disabled. To calibrate absolute gain correctly with the frequency map, we should enable calibration first.\n\nEnable and proceed?",
    "Clients: 0",
    "Clients: {0}",
    "Colormap:",
    "Current",
    "Cursors: Off",
    "Custom",
    "Display Settings",
    "ENOB:",
    "Edge:",
    "FFT size:",
    "Failed to load Settings: {0}",
    "Failed to load module {0}: {1}",
    "Filter",
    "Frequency Deviation Δf (Hz)",
    "Gain:",
    "Gate",
    "Generator",
    "Global output destination for all modules.",
    "Graph",
    "HPF",
    "I-Q Phase Space",
    "I/O",
    "IDLE",
    "IMD (dB):",
    "Idle",
    "In: - | Out: -",
    "In: {0} | Out: {1}",
    "Input level:",
    "Input:",
    "Integrated Phase φ (deg)",
    "Internal Impulse",
    "Internal PRBS/MLS",
    "Invalid measurement bandwidth: upper cutoff must be higher than lower cutoff.",
    "L",
    "L: Vrms: 0.000 V  Vpp: 0.000 V",
    "L: Vrms: {0:.3f} V  Vpp: {1:.3f} V",
    "LPF",
    "LUFS",
    "LUFS (I)",
    "LUFS (M)",
    "LUFS (S)",
    "LUFS Meter",
    "Latency",
    "Limit Max",
    "Limit Min",
    "Loading Settings...",
    "Loading {0} ({1}/{2})...",
    "M",
    "Math",
    "Math ({0})",
    "Math:",
    "Max",
    "Maximum gain applied by the inverse filter (Regularization).",
    "Measurement Bandwidth",
    "Measurement bandwidth is outside the valid range for this sample rate.",
    "Min",
    "Mode",
    "No GUI for {0}",
    "Normal",
    "Normalize Output (RMS to Input)",
    "Period:",
    "Pk: {0} {1}",
    "Play",
    "Playback Gain:",
    "R",
    "R: Vrms: 0.000 V  Vpp: 0.000 V",
    "R: Vrms: {0:.3f} V  Vpp: {1:.3f} V",
    "Reference Mode:",
    "Reference Trace",
    "Reset Stats",
    "S",
    "SR: -",
    "SR: {0}",
    "Select Settings to load.",
    "Select a module from the sidebar.",
    "Show Basic",
    "Show Detailed",
    "Show SPL",
    "Signal",
    "Speed:",
    "Sweep",
    "Sweep completed.\nMap normalized to 1kHz (Ref: {0:.2f} dB, {1:.2f} deg).\nThis map captures RELATIVE frequency response.\nUse 'Absolute Gain Calibration' to fix the absolute level.",
    "T1: {0:.2f}ms {1} | T2: {2:.2f}ms {3} | dT: {4:.2f}ms ({5:.1f}Hz) | {6}",
    "THD+N",
    "Tools",
    "Type:",
    "V1: {0:.3f}V",
    "V2: {0:.3f}V",
    "Vertical",
    "Welcome Image Not Found",
    "Width:",
    "dV: {0:.3f}V",
    "{0:.1f}",
    "{0:.1f} s",
    "{0:.2f} dB",
    "{0:.4f} %",
    "{0:d} m {1:.0f} s",
    "{0} dB",
    "{0} {1}",
]

# Missing keys in other language files (de, es, fr, ja, ko, pt, ru, zh)
MISSING_OTHER_KEYS = [
    "Invert X",
    "Invert Y",
    "Left/Right (L/R)",
    "Mapping:",
    "Mid/Side (M/S)",
]

def load_json(path):
    """Load JSON file preserving order"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    """Save JSON file with proper formatting"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write('\n')  # Add trailing newline

def update_en_json():
    """Add missing keys to en.json"""
    en_path = os.path.join(LANG_DIR, 'en.json')
    print(f"Updating {en_path}...")

    data = load_json(en_path)
    added = 0

    for key in MISSING_EN_KEYS:
        if key not in data:
            data[key] = key  # For English, key = value
            added += 1
            print(f"  Added: '{key}'")

    if added > 0:
        save_json(en_path, data)
        print(f"✓ Added {added} keys to en.json")
    else:
        print("✓ No keys needed to be added to en.json")

    return added

def update_other_lang_files():
    """Add missing keys to other language files"""
    lang_files = ['de.json', 'es.json', 'fr.json', 'ja.json', 'ko.json', 'pt.json', 'ru.json', 'zh.json']

    # Load en.json to get all keys
    en_path = os.path.join(LANG_DIR, 'en.json')
    en_data = load_json(en_path)

    total_added = 0

    for lang_file in lang_files:
        lang_path = os.path.join(LANG_DIR, lang_file)
        if not os.path.exists(lang_path):
            print(f"⚠ {lang_file} not found, skipping...")
            continue

        print(f"\nUpdating {lang_file}...")
        data = load_json(lang_path)
        added = 0

        # Add all missing keys from en.json
        for key in en_data.keys():
            if key not in data:
                # For non-English files, use the English value as placeholder
                data[key] = en_data[key]
                added += 1
                if key in MISSING_OTHER_KEYS:
                    print(f"  Added (required): '{key}'")

        if added > 0:
            save_json(lang_path, data)
            print(f"✓ Added {added} keys to {lang_file}")
            total_added += added
        else:
            print(f"✓ No keys needed to be added to {lang_file}")

    return total_added

def main():
    print("=== Translation File Update Script ===\n")

    # Update en.json first
    en_added = update_en_json()

    # Update other language files
    print("\n" + "="*50)
    other_added = update_other_lang_files()

    print("\n" + "="*50)
    print("\n✓ Update complete!")
    print(f"  - Added {en_added} keys to en.json")
    print(f"  - Added {other_added} keys total to other language files")
    print("\nNote: Non-English translations use English as placeholder.")
    print("Please review and translate them appropriately.")

if __name__ == "__main__":
    main()
