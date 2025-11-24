import numpy as np
from src.core.analysis import AudioCalc

def test_thdn_safety():
    print("Testing calculate_thdn_sine_fit safety...")
    sr = 48000
    sig = np.zeros(1024)
    
    # Test with NaN freq_guess
    res = AudioCalc.calculate_thdn_sine_fit(sig, sr, np.nan)
    print(f"Result for NaN: {res}")
    
    # Test with Inf freq_guess
    res = AudioCalc.calculate_thdn_sine_fit(sig, sr, np.inf)
    print(f"Result for Inf: {res}")
    
    print("Safety test passed if no crash.")

if __name__ == "__main__":
    test_thdn_safety()
