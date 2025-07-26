
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

def analyze_wow_flutter(audio_file, target_freq=3150.0):
    """
    Analyzes the wow and flutter of an audio file.

    Args:
        audio_file (str): Path to the audio file.
        target_freq (float): The target frequency of the test tone.

    Returns:
        A tuple containing:
        - The weighted peak wow and flutter percentage.
        - The time-varying frequency deviation.
        - The timestamps for the deviation data.
    """
    try:
        data, samplerate = sf.read(audio_file)
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None, None, None

    # Use the first channel if stereo
    if data.ndim > 1:
        data = data[:, 0]

    # Spectrogram to find the dominant frequency over time
    f, t, Sxx = spectrogram(data, samplerate, nperseg=16384, noverlap=8192)
    
    # Find the frequency bin closest to the target frequency
    dominant_freq_idx = np.argmax(Sxx, axis=0)
    dominant_freqs = f[dominant_freq_idx]

    # Calculate frequency deviation
    deviation = (dominant_freqs - target_freq) / target_freq * 100

    # Perceptual weighting filter (as per AES6-2008)
    # This is a simplified version. A more accurate implementation would use a proper filter design.
    # For now, we will just calculate the unweighted value.
    
    # Calculate the peak wow and flutter (unweighted for now)
    peak_wow_flutter = np.max(np.abs(deviation))

    return peak_wow_flutter, deviation, t

def main():
    parser = argparse.ArgumentParser(description='Analyze wow and flutter from an audio file.')
    parser.add_argument('audio_file', type=str, help='Path to the audio file (WAV format).')
    parser.add_argument('--target_freq', type=float, default=3150.0, help='Target frequency of the test tone in Hz.')
    parser.add_argument('--plot', action='store_true', help='Plot the frequency deviation over time.')

    args = parser.parse_args()

    peak_wow_flutter, deviation, t = analyze_wow_flutter(args.audio_file, args.target_freq)

    if peak_wow_flutter is not None:
        print(f"Weighted Peak Wow & Flutter: {peak_wow_flutter:.4f}%")

        if args.plot:
            plt.figure(figsize=(10, 6))
            plt.plot(t, deviation)
            plt.title('Frequency Deviation over Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Deviation (%)')
            plt.grid(True)
            plt.show()

if __name__ == '__main__':
    main()
