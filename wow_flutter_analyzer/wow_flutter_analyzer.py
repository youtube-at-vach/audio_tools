import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
import os

console = Console()

def load_audio_file(filepath: str):
    """
    Loads an audio file and returns the audio data and sample rate.

    Args:
        filepath: Path to the audio file.

    Returns:
        A tuple containing:
            - data (np.ndarray): Audio data as a NumPy array.
            - samplerate (int): Sample rate of the audio file.
    """
    try:
        data, samplerate = sf.read(filepath)
    except sf.LibsndfileError as e:
        console.print(f"[bold red]Error: Could not read audio file '{filepath}'.[/bold red]")
        console.print(e)
        return None, None
    except FileNotFoundError:
        console.print(f"[bold red]Error: Audio file '{filepath}' not found.[/bold red]")
        return None, None

    # Ensure audio is mono
    if data.ndim > 1:
        console.print("[yellow]Audio is stereo, converting to mono by taking the first channel.[/yellow]")
        data = data[:, 0]
    
    return data, samplerate

def main():
    """
    Main function to parse arguments and analyze wow and flutter.
    """
    parser = argparse.ArgumentParser(description="Analyze wow and flutter in an audio recording of a test tone.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file (e.g., recording of a 3.15kHz tone).")
    parser.add_argument("--ref_freq", type=float, default=3150.0, help="Reference frequency of the test tone in Hz (e.g., 3150 or 3000).")
    parser.add_argument("--weighting", type=str, default="din", choices=['unweighted', 'din'], help="Type of weighting filter to use ('unweighted' or 'din').")
    parser.add_argument("--min_wow_freq", type=float, default=0.5, help="Minimum frequency for wow component in Hz.")
    parser.add_argument("--max_wow_freq", type=float, default=6.0, help="Maximum frequency for wow component in Hz.")
    parser.add_argument("--min_flutter_freq", type=float, default=6.0, help="Minimum frequency for flutter component in Hz.")
    parser.add_argument("--max_flutter_freq", type=float, default=200.0, help="Maximum frequency for flutter component in Hz.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots and results. If None, plots are displayed and not saved.")
    
    args = parser.parse_args()

    console.print(f"Loading audio file: {args.input_file}")
    audio_data, sample_rate = load_audio_file(args.input_file)

    if audio_data is None:
        return

    console.print(f"  Sample rate: {sample_rate} Hz")
    console.print(f"  Number of samples: {len(audio_data)}")
    console.print(f"  Duration: {len(audio_data)/sample_rate:.2f} seconds")
    console.print(f"  Reference frequency: {args.ref_freq} Hz")
    console.print(f"  Weighting: {args.weighting}")
    console.print(f"  Wow frequency range: {args.min_wow_freq} Hz - {args.max_wow_freq} Hz")
    console.print(f"  Flutter frequency range: {args.min_flutter_freq} Hz - {args.max_flutter_freq} Hz")
    console.print(f"  Output directory: {args.output_dir if args.output_dir else 'Display plots directly'}")

    console.print("\nDemodulating frequency...")
    instantaneous_frequency = demodulate_frequency(audio_data, sample_rate, args.ref_freq)

    if instantaneous_frequency is None:
        console.print("[bold red]Frequency demodulation failed.[/bold red]")
        return

    console.print("  Demodulation successful.")
    console.print(f"  Mean instantaneous frequency: {np.mean(instantaneous_frequency):.2f} Hz")
    console.print(f"  Min instantaneous frequency: {np.min(instantaneous_frequency):.2f} Hz")
    console.print(f"  Max instantaneous frequency: {np.max(instantaneous_frequency):.2f} Hz")

    # --- Frequency Deviation Signal ---
    # Subtract the reference frequency (or mean) to get frequency deviation
    # Using the actual mean of the demodulated signal might be more robust to slight tuning errors
    # However, the standard often refers to deviation from the nominal reference frequency.
    # Let's stick to ref_freq as per common practice for now.
    frequency_deviation_signal = instantaneous_frequency - args.ref_freq
    console.print(f"\nCalculated frequency deviation signal (centered around 0 Hz):")
    console.print(f"  Mean: {np.mean(frequency_deviation_signal):.4f} Hz, Std: {np.std(frequency_deviation_signal):.4f} Hz")

    # --- Apply Weighting and Extract Components ---
    processed_deviation_signal: np.ndarray
    if args.weighting == 'din':
        console.print(f"\nApplying DIN weighting ({args.min_wow_freq} Hz - {args.max_flutter_freq} Hz)...")
        processed_deviation_signal = apply_din_weighting_filter(
            frequency_deviation_signal, 
            sample_rate,
            args.min_wow_freq, # min_freq for DIN approximation
            args.max_flutter_freq # max_freq for DIN approximation
        )
        console.print("  DIN weighting applied.")
    else: # 'unweighted'
        console.print("\nUsing unweighted frequency deviation.")
        processed_deviation_signal = frequency_deviation_signal

    console.print(f"  Processed deviation signal - Mean: {np.mean(processed_deviation_signal):.4f} Hz, Std: {np.std(processed_deviation_signal):.4f} Hz")

    # Extract Wow component
    console.print(f"\nExtracting Wow component ({args.min_wow_freq} Hz - {args.max_wow_freq} Hz)...")
    wow_signal = apply_bandpass_filter(
        processed_deviation_signal, 
        args.min_wow_freq, 
        args.max_wow_freq, 
        sample_rate
    )
    console.print(f"  Wow signal - Mean: {np.mean(wow_signal):.4f} Hz, Std: {np.std(wow_signal):.4f} Hz, Shape: {wow_signal.shape}")

    # Extract Flutter component
    console.print(f"\nExtracting Flutter component ({args.min_flutter_freq} Hz - {args.max_flutter_freq} Hz)...")
    flutter_signal = apply_bandpass_filter(
        processed_deviation_signal, 
        args.min_flutter_freq, 
        args.max_flutter_freq, 
        sample_rate
    )
    console.print(f"  Flutter signal - Mean: {np.mean(flutter_signal):.4f} Hz, Std: {np.std(flutter_signal):.4f} Hz, Shape: {flutter_signal.shape}")

    # Combined Wow and Flutter Signal
    combined_wow_flutter_signal: np.ndarray
    if args.weighting == 'din':
        # For DIN weighting, processed_deviation_signal is already filtered across the broad W&F range
        combined_wow_flutter_signal = processed_deviation_signal
        console.print(f"\nCombined Wow & Flutter (DIN weighted) signal uses the already processed deviation signal.")
    else: # unweighted
        console.print(f"\nCalculating combined Wow & Flutter (unweighted) signal ({args.min_wow_freq} Hz - {args.max_flutter_freq} Hz)...")
        combined_wow_flutter_signal = apply_bandpass_filter(
            frequency_deviation_signal, # Start from the original deviation for unweighted combined
            args.min_wow_freq,
            args.max_flutter_freq,
            sample_rate
        )
    console.print(f"  Combined W&F signal - Mean: {np.mean(combined_wow_flutter_signal):.4f} Hz, Std: {np.std(combined_wow_flutter_signal):.4f} Hz, Shape: {combined_wow_flutter_signal.shape}")

    # --- Calculate Metrics ---
    all_metrics = {}

    wow_metrics = calculate_metrics(wow_signal, args.ref_freq, "Wow")
    all_metrics.update(wow_metrics)

    flutter_metrics = calculate_metrics(flutter_signal, args.ref_freq, "Flutter")
    all_metrics.update(flutter_metrics)
    
    combined_wf_label = f"Combined W&F ({args.weighting})"
    combined_wf_metrics = calculate_metrics(combined_wow_flutter_signal, args.ref_freq, combined_wf_label)
    all_metrics.update(combined_wf_metrics)

    duration_sec = len(audio_data) / sample_rate
    drift_metrics = calculate_drift(instantaneous_frequency, args.ref_freq, sample_rate, duration_sec)
    all_metrics.update(drift_metrics)

    # --- Output Results Table ---
    console.print("\n[bold underline]Wow and Flutter Analysis Results:[/bold underline]")
    
    table = Table(title="Metrics Summary")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value (Hz)", style="magenta", justify="right")
    table.add_column("Value (%)", style="green", justify="right")

    # Wow metrics
    table.add_row("Wow Peak", f"{wow_metrics['Wow Peak (Hz)']:.4f}", f"{wow_metrics['Wow Peak (%)']:.4f}%")
    table.add_row("Wow RMS", f"{wow_metrics['Wow RMS (Hz)']:.4f}", f"{wow_metrics['Wow RMS (%)']:.4f}%")
    
    # Flutter metrics
    table.add_row("Flutter Peak", f"{flutter_metrics['Flutter Peak (Hz)']:.4f}", f"{flutter_metrics['Flutter Peak (%)']:.4f}%")
    table.add_row("Flutter RMS", f"{flutter_metrics['Flutter RMS (Hz)']:.4f}", f"{flutter_metrics['Flutter RMS (%)']:.4f}%")

    # Combined W&F metrics
    table.add_row(f"{combined_wf_label} Peak", f"{combined_wf_metrics[f'{combined_wf_label} Peak (Hz)']:.4f}", f"{combined_wf_metrics[f'{combined_wf_label} Peak (%)']:.4f}%")
    table.add_row(f"{combined_wf_label} RMS", f"{combined_wf_metrics[f'{combined_wf_label} RMS (Hz)']:.4f}", f"{combined_wf_metrics[f'{combined_wf_label} RMS (%)']:.4f}%")

    # Drift metrics
    table.add_section()
    table.add_row("Average Frequency", f"{drift_metrics['Average Frequency (Hz)']:.2f}", "-")
    table.add_row("Overall Drift vs Ref", f"{drift_metrics['Overall Drift (Hz)']:.2f}", f"{drift_metrics['Overall Drift (%)']:.4f}%")
    table.add_row("Initial Frequency (first 0.5s)", f"{drift_metrics['Initial Frequency (Hz)']:.2f}", "-")
    table.add_row("Final Frequency (last 0.5s)", f"{drift_metrics['Final Frequency (Hz)']:.2f}", "-")
    table.add_row("Drift over Measurement", f"{drift_metrics['Drift over Measurement (Hz)']:.2f}", f"{drift_metrics['Drift over Measurement (%)']:.4f}%")
    
    console.print(table)

    # --- Generate Plots ---
    if args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            console.print(f"\nPlots will be saved to: {args.output_dir}")
        except OSError as e:
            console.print(f"[bold red]Error creating output directory '{args.output_dir}': {e}[/bold red]")
            console.print("[yellow]Plots will be displayed directly instead of saving.[/yellow]")
            args.output_dir = None # Reset to display mode

    num_samples = len(audio_data)
    time_array = np.arange(num_samples) / sample_rate

    plot_frequency_deviation(
        time_array,
        frequency_deviation_signal, # Unweighted deviation
        processed_deviation_signal, # Potentially weighted deviation
        wow_signal,
        flutter_signal,
        args.weighting,
        args.output_dir,
        args.ref_freq
    )

    plot_deviation_spectrum(
        combined_wow_flutter_signal, # Signal to analyze (already appropriately weighted or unweighted)
        sample_rate, # This should be the effective sample rate of the deviation signal
        args.weighting,
        args.output_dir,
        "Combined Wow & Flutter",
        min_wow_freq=args.min_wow_freq,
        max_wow_freq=args.max_wow_freq,
        min_flutter_freq=args.min_flutter_freq,
        max_flutter_freq=args.max_flutter_freq
    )


def plot_frequency_deviation(time_array, deviation_signal, weighted_deviation_signal, wow_signal, flutter_signal, weighting_type, output_dir, ref_freq):
    """
    Plots frequency deviation, wow, and flutter components over time.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Determine which signal to plot for the main deviation plot
    main_deviation_signal = weighted_deviation_signal if weighting_type != 'unweighted' else deviation_signal
    
    # Top subplot: Frequency Deviation
    ax1 = axs[0]
    color1 = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency Deviation (Hz)', color=color1)
    ax1.plot(time_array, main_deviation_signal, color=color1, alpha=0.8, label=f'{weighting_type} Deviation')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title(f'Frequency Deviation over Time ({weighting_type})')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    # Secondary y-axis for percentage deviation
    ax1b = ax1.twinx()
    color2 = 'tab:red'
    ax1b.set_ylabel('Deviation (%)', color=color2)
    ax1b.plot(time_array, (main_deviation_signal / ref_freq) * 100, color=color2, linestyle='--', alpha=0.7, label=f'{weighting_type} Deviation (%)')
    ax1b.tick_params(axis='y', labelcolor=color2)
    
    # Add a combined legend for top plot if lines are distinct enough
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    # ax1b.legend(lines + lines2, labels + labels2, loc='upper right') # Can get crowded
    
    # Bottom subplot: Wow and Flutter Components
    ax2 = axs[1]
    ax2.plot(time_array, wow_signal, label='Wow Component', color='orange')
    ax2.plot(time_array, flutter_signal, label='Flutter Component', color='green', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency Deviation (Hz)')
    ax2.set_title(f'Wow and Flutter Components ({weighting_type})')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)

    # Secondary y-axis for percentage deviation on bottom plot
    ax2b = ax2.twinx()
    ax2b.set_ylabel('Deviation (%)', color=color2) # Re-use color2 for consistency
    # Calculate % for wow and flutter separately if needed, or use a general scale based on max deviation
    max_dev_wow_flutter = np.max(np.abs(np.concatenate((wow_signal, flutter_signal))))
    if max_dev_wow_flutter == 0: # Avoid division by zero if signals are flat zero
        ylim_percent_bottom = (-1, 1)
    else:
        ylim_percent_bottom = np.array([-max_dev_wow_flutter, max_dev_wow_flutter]) / ref_freq * 100
    ax2b.set_ylim(ylim_percent_bottom)
    ax2b.tick_params(axis='y', labelcolor=color2)


    fig.tight_layout() # Adjust layout to prevent overlapping titles/labels

    if output_dir:
        filepath = os.path.join(output_dir, "frequency_deviation_plot.png")
        try:
            plt.savefig(filepath)
            console.print(f"  Frequency deviation plot saved to: {filepath}")
        except Exception as e:
            console.print(f"[bold red]Error saving frequency deviation plot: {e}[/bold red]")
        plt.close(fig)
    else:
        plt.show()

def plot_deviation_spectrum(signal_to_analyze, sample_rate, weighting_type, output_dir, title_suffix, 
                            min_wow_freq, max_wow_freq, min_flutter_freq, max_flutter_freq):
    """
    Calculates and plots the Power Spectral Density (PSD) of the frequency deviation signal.
    """
    if signal_to_analyze is None or len(signal_to_analyze) == 0:
        console.print("[yellow]Warning: No signal data to plot spectrum. Skipping spectrum plot.[/yellow]")
        return

    nperseg = min(len(signal_to_analyze), 2048) # Use a segment length, e.g., 2048, or less if signal is shorter
    if nperseg == 0:
         console.print("[yellow]Warning: Signal too short for spectrum analysis. Skipping spectrum plot.[/yellow]")
         return

    frequencies, psd = signal.welch(signal_to_analyze, fs=sample_rate, window='hann', nperseg=nperseg, scaling='density')
    
    # Convert PSD to dB (optional, but common for spectra)
    # Adding a small epsilon to avoid log(0)
    psd_db = 10 * np.log10(psd + 1e-12) 

    plt.figure(figsize=(12, 7))
    plt.semilogx(frequencies, psd_db) # Plot frequency on a log scale

    plt.title(f'Spectrum of Frequency Deviation - {title_suffix} ({weighting_type})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)') # Or 'Amplitude' if not using dB
    plt.grid(True, which="both", ls="-", alpha=0.7)

    # Highlight Wow and Flutter regions
    # Ensure regions are within plot limits and min_freq < max_freq
    plot_min_freq = max(0.1, frequencies[1] if len(frequencies) > 1 else 0.1) # Smallest freq to plot, avoid 0 for log
    plot_max_freq = frequencies[-1] if len(frequencies) > 1 else sample_rate / 2

    # Wow region
    if min_wow_freq < max_wow_freq:
        plt.axvspan(max(min_wow_freq, plot_min_freq), min(max_wow_freq, plot_max_freq), 
                    color='skyblue', alpha=0.3, label=f'Wow ({min_wow_freq}-{max_wow_freq} Hz)')
    # Flutter region
    if min_flutter_freq < max_flutter_freq:
        plt.axvspan(max(min_flutter_freq, plot_min_freq), min(max_flutter_freq, plot_max_freq), 
                    color='lightgreen', alpha=0.3, label=f'Flutter ({min_flutter_freq}-{max_flutter_freq} Hz)')
    
    plt.xlim(plot_min_freq, plot_max_freq)
    if len(psd_db)>0:
        plt.ylim(np.percentile(psd_db,1)-10, np.percentile(psd_db,99)+10) # Adjust Y limits for better visualization
    
    plt.legend(loc='upper right')
    plt.tight_layout()

    if output_dir:
        filepath = os.path.join(output_dir, "deviation_spectrum_plot.png")
        try:
            plt.savefig(filepath)
            console.print(f"  Deviation spectrum plot saved to: {filepath}")
        except Exception as e:
            console.print(f"[bold red]Error saving deviation spectrum plot: {e}[/bold red]")
        plt.close() # Close the figure object
    else:
        plt.show()


def calculate_metrics(signal_hz: np.ndarray, ref_freq: float, label: str):
    """
    Calculates peak and RMS metrics for a given signal.

    Args:
        signal_hz: Input signal (frequency deviations in Hz).
        ref_freq: Reference frequency in Hz.
        label: Label for the metrics (e.g., "Wow", "Flutter").

    Returns:
        A dictionary containing peak and RMS metrics in Hz and %.
    """
    if signal_hz is None or len(signal_hz) == 0:
        console.print(f"[yellow]Warning: Empty or None signal provided for {label} metrics calculation. Returning zero values.[/yellow]")
        return {
            f"{label} Peak (Hz)": 0.0,
            f"{label} Peak (%)": 0.0,
            f"{label} RMS (Hz)": 0.0,
            f"{label} RMS (%)": 0.0,
        }

    peak_deviation_hz = np.max(np.abs(signal_hz))
    peak_deviation_percent = (peak_deviation_hz / ref_freq) * 100
    rms_deviation_hz = np.sqrt(np.mean(np.square(signal_hz)))
    rms_deviation_percent = (rms_deviation_hz / ref_freq) * 100

    return {
        f"{label} Peak (Hz)": peak_deviation_hz,
        f"{label} Peak (%)": peak_deviation_percent,
        f"{label} RMS (Hz)": rms_deviation_hz,
        f"{label} RMS (%)": rms_deviation_percent,
    }

def calculate_drift(instantaneous_frequency: np.ndarray, ref_freq: float, sample_rate: int, duration_sec: float):
    """
    Calculates various frequency drift metrics.

    Args:
        instantaneous_frequency: The raw demodulated frequency signal (Hz).
        ref_freq: Reference frequency (Hz).
        sample_rate: Sample rate of the audio (Hz).
        duration_sec: Total duration of the audio signal (seconds).

    Returns:
        A dictionary containing drift metrics.
    """
    if instantaneous_frequency is None or len(instantaneous_frequency) == 0:
        console.print("[yellow]Warning: Empty or None instantaneous_frequency for drift calculation. Returning zero values.[/yellow]")
        return {
            "Average Frequency (Hz)": 0.0,
            "Overall Drift (Hz)": 0.0,
            "Overall Drift (%)": 0.0,
            "Initial Frequency (Hz)": 0.0,
            "Final Frequency (Hz)": 0.0,
            "Drift over Measurement (Hz)": 0.0,
            "Drift over Measurement (%)": 0.0,
        }

    average_frequency_hz = np.mean(instantaneous_frequency)
    overall_drift_hz = average_frequency_hz - ref_freq
    overall_drift_percent = (overall_drift_hz / ref_freq) * 100

    # Calculate initial and final frequencies (average over ~0.5s)
    half_sec_samples = int(0.5 * sample_rate)
    if len(instantaneous_frequency) < half_sec_samples: # Handles case where audio is shorter than 0.5s
        console.print("[yellow]Warning: Audio duration is less than 0.5s, using full signal for initial/final frequency.[/yellow]")
        initial_frequency_hz = average_frequency_hz
        final_frequency_hz = average_frequency_hz
    elif len(instantaneous_frequency) < 2 * half_sec_samples: # Handles case where audio is between 0.5s and 1s
        console.print("[yellow]Warning: Audio duration is less than 1s (but >= 0.5s), initial/final frequency periods might overlap or be based on partial data if duration < 0.5s for one segment.[/yellow]")
        initial_frequency_hz = np.mean(instantaneous_frequency[:half_sec_samples])
        final_frequency_hz = np.mean(instantaneous_frequency[-half_sec_samples:]) # This will take from -0.5s to end
    else: # Audio is >= 1s
        initial_frequency_hz = np.mean(instantaneous_frequency[:half_sec_samples])
        final_frequency_hz = np.mean(instantaneous_frequency[-half_sec_samples:])

    drift_over_measurement_hz = final_frequency_hz - initial_frequency_hz
    drift_over_measurement_percent = (drift_over_measurement_hz / ref_freq) * 100
    
    return {
        "Average Frequency (Hz)": average_frequency_hz,
        "Overall Drift (Hz)": overall_drift_hz,
        "Overall Drift (%)": overall_drift_percent,
        "Initial Frequency (Hz)": initial_frequency_hz,
        "Final Frequency (Hz)": final_frequency_hz,
        "Drift over Measurement (Hz)": drift_over_measurement_hz,
        "Drift over Measurement (%)": drift_over_measurement_percent,
    }


def apply_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, sample_rate: int, order: int = 4):
    """
    Applies a Butterworth bandpass filter to the data.

    Args:
        data: Input signal (NumPy array).
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        sample_rate: Sample rate of the data in Hz.
        order: Order of the Butterworth filter.

    Returns:
        Filtered signal (NumPy array). Returns original data if filtering fails
        or if cutoff frequencies are invalid.
    """
    nyquist = 0.5 * sample_rate
    
    if lowcut <= 0:
        console.print(f"[yellow]Warning: lowcut frequency ({lowcut:.2f} Hz) is zero or negative. Adjusting to a small positive value (0.01 Hz) for filter design.[/yellow]")
        lowcut = 0.01 # Must be > 0 for butterworth

    if highcut >= nyquist:
        console.print(f"[yellow]Warning: highcut frequency ({highcut:.2f} Hz) is at or above Nyquist frequency ({nyquist:.2f} Hz). Clamping to {nyquist * 0.99:.2f} Hz.[/yellow]")
        highcut = nyquist * 0.99 # Ensure highcut is below Nyquist
    
    if lowcut >= highcut:
        console.print(f"[bold red]Error: Lowcut frequency ({lowcut:.2f} Hz) is greater than or equal to highcut frequency ({highcut:.2f} Hz). Cannot apply filter.[/bold red]")
        return data # Return original data

    try:
        b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=sample_rate, output='ba')
        filtered_data = signal.filtfilt(b, a, data)
        console.print(f"  Applied bandpass filter: {lowcut:.2f} Hz - {highcut:.2f} Hz, Order: {order}")
        return filtered_data
    except ValueError as e:
        console.print(f"[bold red]Error designing or applying bandpass filter ({lowcut}-{highcut} Hz): {e}[/bold red]")
        console.print("[yellow]Returning unfiltered data for this stage.[/yellow]")
        return data

def apply_din_weighting_filter(frequency_signal: np.ndarray, sample_rate: int, min_freq: float, max_freq: float):
    """
    Applies a DIN-style weighting filter to the frequency signal.
    For simplicity, this is currently implemented as a Butterworth bandpass filter
    covering the overall range of wow and flutter as specified by min_freq and max_freq.

    Args:
        frequency_signal: The demodulated frequency signal (deviation).
        sample_rate: The sample rate of the frequency_signal.
        min_freq: The minimum frequency for the overall bandpass (e.g., args.min_wow_freq).
        max_freq: The maximum frequency for the overall bandpass (e.g., args.max_flutter_freq).

    Returns:
        The weighted frequency signal.
    """
    # Using a 2nd order filter as often cited for simplified DIN weighting.
    # The task description mentions "A 2-pole Butterworth bandpass filter from 0.5 Hz to 200 Hz is a common simplification"
    # So, order = 2.
    return apply_bandpass_filter(frequency_signal, min_freq, max_freq, sample_rate, order=2)


def demodulate_frequency(audio_data: np.ndarray, sample_rate: int, ref_freq: float):
    """
    Demodulates the frequency from the audio signal.

    Args:
        audio_data: NumPy array of the input mono audio signal.
        sample_rate: Sample rate of the audio data in Hz.
        ref_freq: The expected reference frequency (e.g., 3150 Hz).

    Returns:
        A NumPy array containing the instantaneous frequency values, 
        or None if demodulation fails.
    """
    # 1. Initial Bandpass Filter (Optional but Recommended)
    # Design a bandpass filter to isolate the test tone.
    # Bandwidth is +/- 5% of ref_freq.
    lowcut = ref_freq * 0.90  # Lower cutoff frequency
    highcut = ref_freq * 1.10 # Upper cutoff frequency
    
    # Check if Nyquist frequency is high enough for the filter
    nyquist = 0.5 * sample_rate
    if highcut >= nyquist:
        console.print(f"[bold yellow]Warning: Highcut frequency ({highcut:.2f} Hz) is close to or above Nyquist frequency ({nyquist:.2f} Hz).[/bold yellow]")
        console.print(f"[bold yellow]Consider using a higher sample rate or a lower reference frequency if demodulation is poor.[/bold yellow]")
        # If highcut is too high, we might skip filtering or adjust it.
        # For now, we'll proceed but this could be a point of failure or poor performance.
        # If highcut is way too high, the filter design might fail.
        if highcut > nyquist * 1.5: # Arbitrary threshold to definitely skip
             console.print(f"[bold red]Error: Filter highcut {highcut:.2f} Hz is too far above Nyquist {nyquist:.2f} Hz. Skipping filtering.[/bold red]")
             filtered_audio = audio_data # Skip filtering
        else: # Proceed with caution
            highcut = nyquist * 0.99 # Clamp highcut to just below Nyquist to be safe
            if lowcut >= highcut:
                lowcut = highcut * 0.9 # Ensure lowcut is also below new highcut
    
    try:
        # Butterworth filter order (e.g., 4th order)
        order = 4
        # Get filter coefficients
        b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=sample_rate)
        # Apply the filter (zero-phase)
        filtered_audio = signal.filtfilt(b, a, audio_data)
        console.print(f"  Applied bandpass filter: {lowcut:.2f} Hz - {highcut:.2f} Hz")
    except ValueError as e:
        console.print(f"[bold red]Error designing or applying bandpass filter: {e}[/bold red]")
        console.print("[yellow]Proceeding with unfiltered audio for demodulation.[/yellow]")
        filtered_audio = audio_data


    # 2. Hilbert Transform
    # Compute the analytic signal.
    analytic_signal = signal.hilbert(filtered_audio)
    console.print("  Computed analytic signal using Hilbert transform.")

    # 3. Instantaneous Phase
    # Calculate the instantaneous phase from the analytic signal.
    # np.unwrap is crucial to correct phase jumps.
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    console.print("  Calculated instantaneous phase.")

    # 4. Instantaneous Frequency
    # The instantaneous frequency is the derivative of the phase.
    # Approximated by (np.diff(instantaneous_phase) / (2 * np.pi)) * sample_rate.
    # np.diff reduces the array length by one. Pad to match original length.
    instantaneous_freq_diff = (np.diff(instantaneous_phase) / (2 * np.pi)) * sample_rate
    instantaneous_frequency = np.pad(instantaneous_freq_diff, (1, 0), 'edge')
    console.print("  Calculated instantaneous frequency.")
    
    return instantaneous_frequency

if __name__ == "__main__":
    main()
