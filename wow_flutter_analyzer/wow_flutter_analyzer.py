# Wow and Flutter Analyzer Main Script
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import os # For output file handling

# Attempt to import rich and matplotlib, but allow script to run without them if not plotting/using rich output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_audio(filepath):
    """Loads an audio file.

    Args:
        filepath (str): Path to the audio file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The audio data.
            - int: The sample rate.
    """
    try:
        data, samplerate = sf.read(filepath)
        # If stereo, convert to mono by averaging channels
        if data.ndim > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)
        return data, samplerate
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def find_fundamental_frequency(audio_data, sample_rate, expected_freq):
    """
    Finds the dominant frequency in the audio signal around the expected frequency.

    Args:
        audio_data (np.ndarray): The audio data.
        sample_rate (int): The sample rate of the audio.
        expected_freq (float): The expected test frequency.

    Returns:
        float: The detected peak frequency.
    """
    # Parameters for Welch's method
    nperseg = min(sample_rate * 2, len(audio_data)) # Use 2 seconds of data or less if file is shorter
    
    # Calculate Welch's periodogram
    frequencies, psd = signal.welch(audio_data, fs=sample_rate, nperseg=nperseg, scaling='density')
    
    # Find the peak frequency around the expected frequency
    # Define a search window around the expected frequency (e.g., +/- 10%)
    freq_window_min = expected_freq * 0.9
    freq_window_max = expected_freq * 1.1
    
    relevant_indices = np.where((frequencies >= freq_window_min) & (frequencies <= freq_window_max))[0]
    
    if not relevant_indices.size:
        print(f"Warning: No frequencies found within the expected window [{freq_window_min:.2f} Hz, {freq_window_max:.2f} Hz]. Using overall peak.")
        peak_index = np.argmax(psd)
    else:
        peak_index_in_window = np.argmax(psd[relevant_indices])
        peak_index = relevant_indices[peak_index_in_window]
        
    detected_frequency = frequencies[peak_index]
    
    print(f"Detected fundamental frequency: {detected_frequency:.2f} Hz")
    return detected_frequency

def track_frequency_variation(audio_data, sample_rate, nominal_freq, block_size_ms=50, hop_size_ms=10):
    """
    Tracks frequency variations over time using STFT.

    Args:
        audio_data (np.ndarray): The audio data.
        sample_rate (int): The sample rate.
        nominal_freq (float): The nominal center frequency of the test tone.
        block_size_ms (int): Block size for STFT in milliseconds.
        hop_size_ms (int): Hop size for STFT in milliseconds.

    Returns:
        np.ndarray: Array of frequency deviations over time.
        np.ndarray: Array of time values corresponding to the deviations.
    """
    nperseg = int(sample_rate * block_size_ms / 1000)
    noverlap = nperseg - int(sample_rate * hop_size_ms / 1000)
    
    f, t, Zxx = signal.stft(audio_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    
    # Find the bin closest to the nominal frequency
    nominal_freq_bin = np.argmin(np.abs(f - nominal_freq))
    
    # Extract phase information from the STFT result for the nominal frequency bin
    # More robust: track the peak frequency in each frame around nominal_freq_bin
    instantaneous_frequencies = np.zeros(Zxx.shape[1])

    for i in range(Zxx.shape[1]):
        segment_spectrum = np.abs(Zxx[:, i])
        # Search for peak within a window around the nominal frequency bin
        window_size = max(1, int(len(f) * 0.05)) # Search window of 5% of total bins
        min_idx = max(0, nominal_freq_bin - window_size)
        max_idx = min(len(f), nominal_freq_bin + window_size + 1)
        
        if max_idx <= min_idx: # Should not happen if window_size is reasonable
             peak_bin_in_segment = nominal_freq_bin
        else:
            peak_bin_in_segment = min_idx + np.argmax(segment_spectrum[min_idx:max_idx])

        instantaneous_frequencies[i] = f[peak_bin_in_segment]
        
    frequency_deviations = instantaneous_frequencies - nominal_freq
    print(f"Tracked {len(frequency_deviations)} frequency points.")
    return frequency_deviations, t

def calculate_wow_flutter(frequency_deviations, nominal_freq, sample_rate_deviations, wow_cutoff_hz=4.0):
    """
    Calculates Wow and Flutter from frequency deviations.

    Args:
        frequency_deviations (np.ndarray): Array of frequency deviations.
        nominal_freq (float): The nominal center frequency.
        sample_rate_deviations (float): The effective sample rate of the deviation signal.
        wow_cutoff_hz (float): Cutoff frequency for the Wow filter.

    Returns:
        tuple: Contains (wow_percentage, flutter_unweighted_rms_percentage)
    """
    if len(frequency_deviations) == 0:
        return 0.0, 0.0

    # Wow: Low-pass filter
    # A 2nd order Butterworth filter is common.
    # The nyquist frequency for the deviation signal is sample_rate_deviations / 2
    nyquist_dev = sample_rate_deviations / 2.0
    if nyquist_dev <= wow_cutoff_hz : # Ensure cutoff is not above Nyquist
        print(f"Warning: Wow cutoff ({wow_cutoff_hz} Hz) is too high for the deviation signal's effective sample rate ({sample_rate_deviations} Hz). Skipping Wow calculation or adjusting cutoff.")
        # Option 1: Skip Wow
        # wow_signal = frequency_deviations # Or handle as an error/None
        # Option 2: Adjust cutoff (less ideal as it changes the definition)
        # wow_cutoff_hz = nyquist_dev * 0.9 # e.g. 90% of Nyquist
        # For now, let's proceed, but this indicates an issue with input parameters or short signal.
        # If proceeding, the filter might become unstable or ineffective.
        # A robust solution would be to ensure enough data for STFT to get a higher sample_rate_deviations
        # or to simply state that Wow cannot be reliably calculated.
        # For this implementation, we'll assume sample_rate_deviations is sufficient.
        # If wow_cutoff_hz is indeed too high, the filter design might fail or produce NaNs.
        # We'll use a very low cutoff if nyquist_dev is too small.
         wow_cutoff_hz = max(0.1, nyquist_dev * 0.5) # Fallback to a very low cutoff or half nyquist.
         print(f"Adjusted Wow cutoff to {wow_cutoff_hz} Hz.")


    b_wow, a_wow = signal.butter(2, wow_cutoff_hz, btype='low', analog=False, fs=sample_rate_deviations)
    wow_signal = signal.filtfilt(b_wow, a_wow, frequency_deviations)
    
    # Peak-to-peak Wow
    wow_peak_to_peak_hz = np.max(wow_signal) - np.min(wow_signal)
    wow_percentage = (wow_peak_to_peak_hz / nominal_freq) * 100
    
    # Flutter (Unweighted RMS for now)
    # For simplicity, consider flutter as deviations above the wow cutoff.
    # A high-pass filter can be used, or simply the original deviations if no specific band is targeted yet.
    # Using original deviations for now as a simplification before specific weighting.
    # A more correct approach would be a bandpass filter (e.g., 0.5 Hz - 200 Hz for NAB)
    # or highpass above wow_cutoff_hz.
    
    # Example: High-pass filter for flutter component (everything > wow_cutoff_hz)
    # Ensure flutter_cutoff_min is not too close to Nyquist if wow_cutoff_hz is high.
    flutter_cutoff_min = wow_cutoff_hz 
    if nyquist_dev > flutter_cutoff_min:
        b_flutter_hp, a_flutter_hp = signal.butter(2, flutter_cutoff_min, btype='high', analog=False, fs=sample_rate_deviations)
        flutter_signal_hp = signal.filtfilt(b_flutter_hp, a_flutter_hp, frequency_deviations)
    else:
        # If cutoff is too high, use the original signal or a zero signal for flutter.
        # This means the "wow" component covers the entire spectrum of deviations.
        flutter_signal_hp = np.zeros_like(frequency_deviations) # Or frequency_deviations if all is flutter
        print(f"Warning: Flutter min cutoff ({flutter_cutoff_min} Hz) is at/above Nyquist ({nyquist_dev} Hz). Flutter signal might be zero or based on unfiltered high frequencies.")


    flutter_unweighted_rms_hz = np.sqrt(np.mean(flutter_signal_hp**2))
    flutter_unweighted_rms_percentage = (flutter_unweighted_rms_hz / nominal_freq) * 100
    
    print(f"Wow (Peak-to-Peak, <{wow_cutoff_hz:.1f} Hz): {wow_percentage:.3f}%")
    print(f"Flutter (Unweighted RMS, >{wow_cutoff_hz:.1f} Hz): {flutter_unweighted_rms_percentage:.3f}%")
    
    return wow_percentage, flutter_unweighted_rms_percentage

def calculate_wrms_flutter(frequency_deviations, nominal_freq, sample_rate_deviations, weighting_filter_type="NAB"):
    """
    Calculates Weighted Root Mean Square (WRMS) Flutter.

    Args:
        frequency_deviations (np.ndarray): Array of frequency deviations.
        nominal_freq (float): The nominal center frequency.
        sample_rate_deviations (float): The effective sample rate of the deviation signal.
        weighting_filter_type (str): Type of weighting filter (e.g., "NAB", "IEC", "DIN"). 
                                     Currently, only a placeholder for "NAB" is implemented.

    Returns:
        float: WRMS flutter in percentage.
    """
    if len(frequency_deviations) == 0:
        return 0.0

    # Placeholder for weighting filter - actual NAB/IEC/DIN filters are more complex.
    # A common characteristic is a bandpass filter focusing on perceptible flutter frequencies (e.g., 0.5 Hz to 200 Hz)
    # with a specific frequency response shape (e.g., peaking around 4 Hz for NAB).
    
    # For now, let's use a simple bandpass filter as a stand-in for a more accurate weighting curve.
    # This is NOT a standard-compliant weighting, just a placeholder.
    bp_low_cut = 0.5  # Hz
    bp_high_cut = 200 # Hz
    nyquist_dev = sample_rate_deviations / 2.0

    if bp_high_cut >= nyquist_dev:
        print(f"Warning: WRMS high cutoff ({bp_high_cut} Hz) is at or above Nyquist ({nyquist_dev} Hz). Adjusting high cutoff.")
        bp_high_cut = nyquist_dev * 0.99 # Adjust to be just below Nyquist
    
    if bp_low_cut >= bp_high_cut:
        print(f"Warning: WRMS low cutoff ({bp_low_cut} Hz) is at or above adjusted high cutoff ({bp_high_cut} Hz). Flutter calculation might be unreliable.")
        weighted_deviations = frequency_deviations # No effective filtering
    else:
        try:
            # A 4th order Butterworth bandpass filter can be a starting point.
            # Standard weighting filters often have more specific designs.
            b, a = signal.butter(4, [bp_low_cut, bp_high_cut], btype='bandpass', analog=False, fs=sample_rate_deviations)
            weighted_deviations = signal.filtfilt(b, a, frequency_deviations)
        except ValueError as e:
            print(f"Error designing WRMS placeholder filter (likely due to cutoffs vs Nyquist): {e}. Using unweighted deviations for WRMS.")
            weighted_deviations = frequency_deviations


    wrms_hz = np.sqrt(np.mean(weighted_deviations**2))
    wrms_percentage = (wrms_hz / nominal_freq) * 100
    
    print(f"WRMS Flutter ({weighting_filter_type} - Placeholder Filter [{bp_low_cut:.1f}-{bp_high_cut:.1f} Hz]): {wrms_percentage:.3f}%")
    return wrms_percentage

def main():
    parser = argparse.ArgumentParser(
        description="Wow and Flutter Analyzer. Calculates Wow, Flutter, and WRMS Flutter from an audio test tone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    parser.add_argument("--input_file", "-i", required=True, help="Input audio file (e.g., WAV, FLAC).")
    parser.add_argument("--frequency", "-f", type=float, required=True, help="Expected test frequency in Hz (e.g., 3150).")
    parser.add_argument("--output_txt", "-ot", help="Path to save results as a text file.")
    parser.add_argument("--output_csv", "-oc", help="Path to save results as a CSV file.")
    parser.add_argument("--plot_deviation", "-p", action="store_true", help="Display a plot of frequency deviation over time.")

    args = parser.parse_args()

    console = Console() if RICH_AVAILABLE else None

    if not os.path.exists(args.input_file):
        if console:
            console.print(f"[bold red]Error: Input file '{args.input_file}' not found.[/bold red]")
        else:
            print(f"Error: Input file '{args.input_file}' not found.")
        return

    audio_data, sample_rate = load_audio(args.input_file)
    if audio_data is None:
        # load_audio already prints an error
        return

    actual_nominal_freq = find_fundamental_frequency(audio_data, sample_rate, args.frequency)
    if actual_nominal_freq is None:
        if console:
            console.print("[bold red]Could not determine fundamental frequency. Exiting.[/bold red]")
        else:
            print("Could not determine fundamental frequency. Exiting.")
        return

    hop_size_ms_track = 10 
    sample_rate_deviations = 1000.0 / hop_size_ms_track
    frequency_deviations, time_axis = track_frequency_variation(
        audio_data, sample_rate, actual_nominal_freq, hop_size_ms=hop_size_ms_track
    )

    if frequency_deviations is None or len(frequency_deviations) < 4:
        if console:
            console.print("[bold red]Could not track frequency variations adequately. Exiting.[/bold red]")
        else:
            print("Could not track frequency variations adequately. Exiting.")
        return

    wow_percentage, flutter_unweighted_rms_percentage = calculate_wow_flutter(
        frequency_deviations, actual_nominal_freq, sample_rate_deviations
    )
    wrms_flutter_percentage = calculate_wrms_flutter(
        frequency_deviations, actual_nominal_freq, sample_rate_deviations
    )

    results = {
        "InputFile": args.input_file,
        "NominalFrequencyHz": args.frequency,
        "DetectedFundamentalFrequencyHz": actual_nominal_freq,
        "PeakWowPercent_DIN": wow_percentage,
        "RMSFlutterPercent_Unweighted": flutter_unweighted_rms_percentage,
        "WRMSFlutterPercent_NAB_Placeholder": wrms_flutter_percentage,
    }

    # --- Display Results (Console) ---
    if console:
        table = Table(title="Wow and Flutter Analysis Results", show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="dim", width=35)
        table.add_column("Value")

        table.add_row("Input File", results["InputFile"])
        table.add_row("Nominal Test Frequency", f"{results['NominalFrequencyHz']:.0f} Hz")
        table.add_row("Detected Fundamental Frequency", f"{results['DetectedFundamentalFrequencyHz']:.2f} Hz")
        table.add_row("Peak Wow (DIN, <4 Hz)", f"{results['PeakWowPercent_DIN']:.3f}%")
        table.add_row("RMS Flutter (Unweighted, >4 Hz)", f"{results['RMSFlutterPercent_Unweighted']:.3f}%")
        table.add_row("WRMS Flutter (NAB Placeholder)", f"{results['WRMSFlutterPercent_NAB_Placeholder']:.3f}%")
        
        console.print(Panel.fit(table, title="[bold cyan]Analysis Summary[/bold cyan]"))
    else: 
        print("\n--- Wow and Flutter Analysis Results ---")
        for key, value in results.items():
            if "Hz" in key:
                print(f"{key.replace('Hz', ' (Hz)')}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
            elif "Percent" in key:
                print(f"{key.replace('Percent', ' (%)')}: {value:.3f}")
            else:
                print(f"{key}: {value}")
    
    # --- Save to Text File ---
    if args.output_txt:
        try:
            with open(args.output_txt, 'w') as f:
                f.write("--- Wow and Flutter Analysis Results ---\n")
                for key, value in results.items():
                    if "Hz" in key:
                        f.write(f"{key.replace('Hz', ' (Hz)')}: {value:.2f}\n" if isinstance(value, float) else f"{key}: {value}\n")
                    elif "Percent" in key:
                        f.write(f"{key.replace('Percent', ' (%)')}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            if console: console.print(f"\nResults saved to text file: [green]{args.output_txt}[/green]")
            else: print(f"\nResults saved to text file: {args.output_txt}")
        except IOError as e:
            if console: console.print(f"[bold red]Error saving to text file '{args.output_txt}': {e}[/bold red]")
            else: print(f"Error saving to text file '{args.output_txt}': {e}")

    # --- Save to CSV File ---
    if args.output_csv:
        try:
            # Write header only if file is new/empty
            file_exists = os.path.isfile(args.output_csv) and os.path.getsize(args.output_csv) > 0
            with open(args.output_csv, 'a', newline='') as f:
                # Using csv module for proper CSV formatting might be better but for single line this is okay
                # For more robust CSV writing, especially with potential commas in filenames:
                # import csv
                # writer = csv.DictWriter(f, fieldnames=results.keys())
                # if not file_exists:
                #     writer.writeheader()
                # writer.writerow(results)
                header = ",".join(results.keys())
                row_values = []
                for k, v in results.items(): # Ensure proper string conversion, handle potential commas in filename
                    if isinstance(v, str) and ',' in v:
                         row_values.append(f'"{v}"') # Basic CSV quoting for string with comma
                    else:
                        row_values.append(str(v))
                
                if not file_exists:
                    f.write(header + "\n")
                f.write(",".join(row_values) + "\n")

            if console: console.print(f"Results appended to CSV file: [green]{args.output_csv}[/green]")
            else: print(f"Results appended to CSV file: {args.output_csv}")
        except IOError as e:
            if console: console.print(f"[bold red]Error saving to CSV file '{args.output_csv}': {e}[/bold red]")
            else: print(f"Error saving to CSV file '{args.output_csv}': {e}")

    # --- Plot Frequency Deviation ---
    if args.plot_deviation:
        if MATPLOTLIB_AVAILABLE:
            if time_axis is not None and frequency_deviations is not None:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.plot(time_axis, frequency_deviations)
                    plt.title(f"Frequency Deviation Over Time\nInput: {os.path.basename(args.input_file)}, Nominal: {args.frequency} Hz")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Frequency Deviation (Hz)")
                    plt.grid(True)
                    if console: console.print("\nDisplaying frequency deviation plot...")
                    else: print("\nDisplaying frequency deviation plot...")
                    plt.show()
                except Exception as e:
                    if console: console.print(f"[bold red]Error generating plot: {e}[/bold red]")
                    else: print(f"Error generating plot: {e}")
            else:
                if console: console.print("[yellow]No deviation data available to plot.[/yellow]")
                else: print("No deviation data available to plot.")
        else:
            if console: console.print("[yellow]Matplotlib not installed. Cannot display plot. Please install it: pip install matplotlib[/yellow]")
            else: print("Matplotlib not installed. Cannot display plot. Please install it: pip install matplotlib")


if __name__ == "__main__":
    main()
