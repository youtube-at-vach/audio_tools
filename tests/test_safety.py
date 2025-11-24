import numpy as np
from src.core.analysis import AudioCalc

def test_optimize_frequency_safety():
    print("Testing optimize_frequency safety...")
    sr = 48000
    sig = np.zeros(1024)
    
    # Test with NaN
    res = AudioCalc.optimize_frequency(sig, sr, np.nan)
    print(f"Result for NaN: {res}")
    
    # Test with Inf
    res = AudioCalc.optimize_frequency(sig, sr, np.inf)
    print(f"Result for Inf: {res}")
    
    print("Safety test passed if no crash.")

if __name__ == "__main__":
    test_optimize_frequency_safety()
