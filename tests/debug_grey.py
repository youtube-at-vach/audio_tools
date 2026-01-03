import numpy as np


def analyze_grey_scaling():
    sample_rate = 48000
    duration = 1.0
    num_samples = int(sample_rate * duration)

    freqs = np.fft.rfftfreq(num_samples, d=1/sample_rate)

    f = freqs
    f**2
    c1 = 12194.217**2
    c2 = 20.6**2
    c3 = 107.7**2
    c4 = 737.9**2

    f_safe = f.copy()
    f_safe[0] = 1.0
    f2_safe = f_safe**2

    num = c1 * (f2_safe**2)
    denom = (f2_safe + c2) * np.sqrt((f2_safe + c3) * (f2_safe + c4)) * (f2_safe + c1)

    a_weight = num / denom
    scaling = 1.0 / (a_weight + 1e-12)
    scaling[0] = 0

    print(f"Max Scaling: {np.max(scaling):.2e}")
    print(f"Scaling at 1kHz: {scaling[np.argmin(np.abs(freqs - 1000))]:.2f}")
    print(f"Scaling at 20Hz: {scaling[np.argmin(np.abs(freqs - 20))]:.2f}")
    print(f"Scaling at 1Hz: {scaling[np.argmin(np.abs(freqs - 1))]:.2f}")

    # Normalize scaling to 1kHz
    ref_idx = np.argmin(np.abs(freqs - 1000))
    scaling_norm = scaling / scaling[ref_idx]

    print(f"Norm Scaling at 1Hz: {scaling_norm[np.argmin(np.abs(freqs - 1))]:.2f}")

if __name__ == "__main__":
    analyze_grey_scaling()
