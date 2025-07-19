
import numpy as np
import argparse

def generate_white_noise(duration, sample_rate, amplitude):
    """Generate white noise."""
    return np.random.uniform(-amplitude, amplitude, int(duration * sample_rate))

def generate_pink_noise(duration, sample_rate, amplitude):
    """Generate pink noise."""
    # This is a simplified implementation.
    # A more accurate implementation would use a filter.
    white_noise = generate_white_noise(duration, sample_rate, 1.0)
    fft = np.fft.rfft(white_noise)
    fft[1:] /= np.sqrt(np.arange(1, len(fft)))
    pink_noise = np.fft.irfft(fft)
    pink_noise /= np.max(np.abs(pink_noise))
    return pink_noise * amplitude

def generate_brown_noise(duration, sample_rate, amplitude):
    """Generate brown noise."""
    white_noise = generate_white_noise(duration, sample_rate, 1.0)
    brown_noise = np.cumsum(white_noise)
    brown_noise -= np.mean(brown_noise)
    brown_noise /= np.max(np.abs(brown_noise))
    return brown_noise * amplitude

def generate_gaussian_noise(duration, sample_rate, amplitude):
    """Generate Gaussian noise."""
    return np.random.normal(0, amplitude / 3, int(duration * sample_rate))

def generate_violet_noise(duration, sample_rate, amplitude):
    """Generate violet noise."""
    white_noise = generate_white_noise(duration, sample_rate, 1.0)
    violet_noise = np.diff(white_noise)
    violet_noise /= np.max(np.abs(violet_noise))
    return np.append(violet_noise, 0) * amplitude

def generate_periodic_noise(duration, sample_rate, amplitude, period_samples=1024):
    """Generate periodic noise."""
    num_samples = int(duration * sample_rate)
    period = np.random.uniform(-amplitude, amplitude, period_samples)
    repeats = int(np.ceil(num_samples / period_samples))
    periodic_noise = np.tile(period, repeats)
    return periodic_noise[:num_samples]

def generate_mls_noise(duration, sample_rate, amplitude, n_bits=16):
    """Generate Maximum Length Sequence (MLS) noise."""
    num_samples = int(duration * sample_rate)
    if n_bits > 24:
        raise ValueError("n_bits must be 24 or less")
    
    # Taps for common n_bits values
    taps = {
        10: [10, 7], 11: [11, 9], 12: [12, 11, 10, 4], 13: [13, 12, 11, 8],
        14: [14, 13, 12, 2], 15: [15, 14], 16: [16, 15, 13, 4], 17: [17, 14],
        18: [18, 11], 19: [19, 18, 17, 14], 20: [20, 17], 21: [21, 19],
        22: [22, 21], 23: [23, 18], 24: [24, 23, 22, 17]
    }
    if n_bits not in taps:
        raise ValueError(f"No defined taps for n_bits={n_bits}. Choose from {list(taps.keys())}")

    sequence_len = 2**n_bits - 1
    shift_register = np.ones(n_bits, dtype=int)
    mls_sequence = np.zeros(sequence_len, dtype=float)
    
    tap_indices = np.array(taps[n_bits]) - 1

    for i in range(sequence_len):
        feedback = np.bitwise_xor.reduce(shift_register[tap_indices])
        mls_sequence[i] = 1.0 if shift_register[-1] else -1.0
        shift_register = np.roll(shift_register, 1)
        shift_register[0] = feedback
        
    repeats = int(np.ceil(num_samples / sequence_len))
    mls_noise = np.tile(mls_sequence, repeats)
    return mls_noise[:num_samples] * amplitude


import sounddevice as sd
from scipy.io.wavfile import write as write_wav

def main():
    parser = argparse.ArgumentParser(description="Generate and play various types of noise.")
    parser.add_argument("noise_type", help="Type of noise to generate", choices=["white", "pink", "brown", "gaussian", "violet", "periodic", "mls"])
    parser.add_argument("-d", "--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("-r", "--rate", type=int, default=44100, help="Sample rate in Hz")
    parser.add_argument("-a", "--amplitude", type=float, default=0.5, help="Amplitude (0.0 to 1.0)")
    parser.add_argument("-o", "--output", help="Path to save the generated noise as a WAV file.")
    parser.add_argument("--list_devices", action="store_true", help="List available audio devices and exit.")
    parser.add_argument("--device", type=int, help="Output device ID. See --list_devices.")

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    noise_generators = {
        "white": generate_white_noise,
        "pink": generate_pink_noise,
        "brown": generate_brown_noise,
        "gaussian": generate_gaussian_noise,
        "violet": generate_violet_noise,
        "periodic": generate_periodic_noise,
        "mls": generate_mls_noise,
    }

    noise = noise_generators[args.noise_type](args.duration, args.rate, args.amplitude)

    if args.output:
        try:
            # Scale to 16-bit integer for WAV file
            scaled_noise = np.int16(noise / np.max(np.abs(noise)) * 32767)
            write_wav(args.output, args.rate, scaled_noise)
            print(f"Successfully saved {args.noise_type} noise to {args.output}")
        except Exception as e:
            print(f"Error saving WAV file: {e}")
    else:
        try:
            print(f"Playing {args.noise_type} noise for {args.duration} seconds...")
            sd.play(noise, args.rate, device=args.device)
            sd.wait()
            print("Playback finished.")
        except Exception as e:
            print(f"Error playing audio: {e}")

if __name__ == "__main__":
    main()
