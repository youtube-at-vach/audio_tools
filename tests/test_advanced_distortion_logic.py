import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from src.core.analysis import AudioCalc


def test_mim():
    print("Testing MIM...")
    sr = 48000
    N = 32768
    freqs = np.fft.rfftfreq(N, 1/sr)

    # Generate Multitone (3 tones)
    tones = [100, 1000, 5000]
    sig = np.zeros(N)
    t = np.arange(N) / sr
    for f in tones:
        sig += 0.1 * np.sin(2*np.pi*f*t)

    # Add noise
    noise = np.random.normal(0, 0.0001, N)
    sig += noise

    window = np.blackman(N)
    fft_res = np.fft.rfft(sig * window)
    mag = np.abs(fft_res) * 2 / np.sum(window)

    res = AudioCalc.calculate_multitone_tdn(mag, freqs, tones)
    print(f"MIM TD+N: {res['tdn_db']:.2f} dB")

    if res['tdn_db'] > -40:
        print("FAIL: TD+N too high (expected low noise)")
    else:
        print("PASS")

def test_spdr():
    print("\nTesting SPDR...")
    sr = 48000
    N = 32768
    freqs = np.fft.rfftfreq(N, 1/sr)

    # Fundamental 1kHz
    t = np.arange(N) / sr
    sig = 1.0 * np.sin(2*np.pi*1000*t)

    # Spur at 2.5kHz, -60dB
    sig += 0.001 * np.sin(2*np.pi*2500*t)

    window = np.blackman(N)
    fft_res = np.fft.rfft(sig * window)
    mag = np.abs(fft_res) * 2 / np.sum(window)

    res = AudioCalc.calculate_spdr(mag, freqs, 1000.0)
    print(f"SPDR: {res['spdr_db']:.2f} dB")
    print(f"Max Spur: {res['max_spur_freq']:.0f} Hz")

    if abs(res['spdr_db'] - 60) < 1:
        print("PASS")
    else:
        print(f"FAIL: Expected ~60dB, got {res['spdr_db']:.2f}")

def test_pim():
    print("\nTesting PIM...")
    sr = 48000
    N = 32768
    freqs = np.fft.rfftfreq(N, 1/sr)

    f1 = 1800
    f2 = 2100
    t = np.arange(N) / sr

    # Carriers
    sig = 0.5 * np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t)

    # IM3 Lower: 2f1 - f2 = 3600 - 2100 = 1500
    # IM3 Upper: 2f2 - f1 = 4200 - 1800 = 2400
    # Add IM3 at -80dBc (relative to 0.5) => 0.00005
    sig += 0.00005 * np.sin(2*np.pi*1500*t)
    sig += 0.00005 * np.sin(2*np.pi*2400*t)

    window = np.blackman(N)
    fft_res = np.fft.rfft(sig * window)
    mag = np.abs(fft_res) * 2 / np.sum(window)

    res = AudioCalc.calculate_pim(mag, freqs, f1, f2)
    print(f"PIM: {res['pim_db']:.2f} dBc")

    if abs(res['pim_db'] - (-80)) < 2: # Windowing might affect slightly
        print("PASS")
    else:
        print(f"FAIL: Expected ~-80dB, got {res['pim_db']:.2f}")

if __name__ == "__main__":
    test_mim()
    test_spdr()
    test_pim()
