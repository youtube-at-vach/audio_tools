import sys
import os
import numpy as np
sys.path.append(os.getcwd())

from src.core.analysis import AudioCalc

def test_filters():
    print("Testing Filters...")
    sr = 48000
    N = 4800
    t = np.arange(N) / sr
    
    # Signal: 100Hz + 10kHz
    sig = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*10000*t)
    
    # LPF at 1kHz (Should remove 10kHz)
    lpf_sig = AudioCalc.lowpass_filter(sig, sr, 1000.0)
    
    # Check energy at 10kHz
    fft_orig = np.abs(np.fft.rfft(sig))
    fft_lpf = np.abs(np.fft.rfft(lpf_sig))
    freqs = np.fft.rfftfreq(N, 1/sr)
    
    idx_10k = np.argmin(np.abs(freqs - 10000))
    idx_100 = np.argmin(np.abs(freqs - 100))
    
    attenuation_10k = 20 * np.log10(fft_lpf[idx_10k] / fft_orig[idx_10k])
    print(f"LPF 1kHz Attenuation at 10kHz: {attenuation_10k:.2f} dB")
    
    if attenuation_10k < -20:
        print("LPF PASS")
    else:
        print("LPF FAIL")
        
    # HPF at 1kHz (Should remove 100Hz)
    hpf_sig = AudioCalc.highpass_filter(sig, sr, 1000.0)
    attenuation_100 = 20 * np.log10(hpf_sig[idx_100] / fft_orig[idx_100])
    print(f"HPF 1kHz Attenuation at 100Hz: {attenuation_100:.2f} dB")
    
    if attenuation_100 < -20:
        print("HPF PASS")
    else:
        print("HPF FAIL")

if __name__ == "__main__":
    test_filters()
