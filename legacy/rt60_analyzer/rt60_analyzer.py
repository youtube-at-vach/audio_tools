
import argparse
import numpy as np
import sounddevice as sd
from scipy.signal import hilbert
from scipy.stats import linregress
import matplotlib.pyplot as plt

def record_impulse_response(duration=5, samplerate=48000, device=None):
    """Records audio from the specified device."""
    print("Recording for impulse response...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, blocking=True, device=device)
    sd.wait()
    print("Recording finished.")
    return recording.flatten()

def calculate_rt60(audio_data, samplerate):
    """Calculates the RT60 from the given audio data."""
    # Calculate the envelope of the signal
    analytic_signal = hilbert(audio_data)
    envelope = np.abs(analytic_signal)
    
    # Find the peak of the envelope, which is our starting point (t=0 for the decay)
    start_index = np.argmax(envelope)
    
    # Normalize with the peak value and convert to dB
    envelope_db = 20 * np.log10(envelope / envelope[start_index])
    
    # Get the time axis
    t = np.arange(len(envelope_db)) / samplerate
    
    # We analyze the decay from the peak onwards
    decay_envelope_db = envelope_db[start_index:]
    decay_t = t[start_index:]

    # Find the point where the level drops by 5 dB (start of T20/T30 range)
    try:
        t5_index = np.where(decay_envelope_db <= -5)[0][0]
    except IndexError:
        return None, None, None # Decay not long enough

    # Find the point where the level drops by 25 dB (end of T20 range)
    try:
        t25_index = np.where(decay_envelope_db <= -25)[0][0]
    except IndexError:
        return None, None, None # Decay not long enough for T20

    # Perform linear regression on the T20 part of the decay
    slope, intercept, _, _, _ = linregress(decay_t[t5_index:t25_index], decay_envelope_db[t5_index:t25_index])

    if slope >= 0:
        return None, None, None # Not a decay

    # Extrapolate to -60 dB to get RT60
    rt60 = -60 / slope

    return rt60, t, envelope_db

def main():
    parser = argparse.ArgumentParser(description='Measure RT60 from a recorded impulse response.')
    parser.add_argument('--duration', type=float, default=5.0, help='Recording duration in seconds.')
    parser.add_argument('--samplerate', type=int, default=48000, help='Sample rate in Hz.')
    parser.add_argument('--device', type=int, help='Input device ID.')
    parser.add_argument('--plot', action='store_true', help='Plot the decay curve.')

    args = parser.parse_args()

    impulse_response = record_impulse_response(args.duration, args.samplerate, args.device)
    rt60, t, envelope_db = calculate_rt60(impulse_response, args.samplerate)

    if rt60 is not None:
        print(f"RT60 (T20): {rt60:.2f} seconds")

        if args.plot and t is not None and envelope_db is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(t, envelope_db, label='Energy Decay Curve')
            plt.title('Reverberation Decay Curve')
            plt.xlabel('Time (s)')
            plt.ylabel('Level (dB)')
            plt.ylim(-80, 10)
            plt.grid(True)
            plt.legend()
            plt.show()
    else:
        print("Could not calculate RT60. The decay was not clear enough.")

if __name__ == '__main__':
    main()
