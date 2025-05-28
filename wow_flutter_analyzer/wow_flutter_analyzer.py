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
    ax1, ax2 = axs[0], axs[1] # Unpack axes

    # Determine which signal to plot for the main deviation plot
    main_dev_sig_to_plot = [] # Default to empty
    main_dev_sig_for_calc = weighted_deviation_signal if weighting_type != 'unweighted' else deviation_signal

    if main_dev_sig_for_calc is None or main_dev_sig_for_calc.size == 0 or np.all(np.isnan(main_dev_sig_for_calc)):
        console.print("[yellow]Warning: Main deviation signal for plotting is empty or all NaN. Top plot will be empty.[/yellow]")
    else:
        main_dev_sig_to_plot = main_dev_sig_for_calc
        # Ensure it's a numpy array for calculations
        if not isinstance(main_dev_sig_to_plot, np.ndarray): main_dev_sig_to_plot = np.array(main_dev_sig_to_plot)


    wow_sig_to_plot = []
    if wow_signal is None or wow_signal.size == 0 or np.all(np.isnan(wow_signal)):
        console.print("[yellow]Warning: Wow signal for plotting is empty or all NaN. Wow component will be empty in bottom plot.[/yellow]")
    else:
        wow_sig_to_plot = wow_signal
        if not isinstance(wow_sig_to_plot, np.ndarray): wow_sig_to_plot = np.array(wow_sig_to_plot)

    flutter_sig_to_plot = []
    if flutter_signal is None or flutter_signal.size == 0 or np.all(np.isnan(flutter_signal)):
        console.print("[yellow]Warning: Flutter signal for plotting is empty or all NaN. Flutter component will be empty in bottom plot.[/yellow]")
    else:
        flutter_sig_to_plot = flutter_signal
        if not isinstance(flutter_sig_to_plot, np.ndarray): flutter_sig_to_plot = np.array(flutter_sig_to_plot)
    
    # Top subplot: Frequency Deviation
    # ax1 = axs[0] # Already defined
    color1 = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency Deviation (Hz)', color=color1)
    if time_array.size > 0 and isinstance(main_dev_sig_to_plot, np.ndarray) and main_dev_sig_to_plot.size > 0:
        ax1.plot(time_array, main_dev_sig_to_plot, color=color1, alpha=0.8, label=f'{weighting_type} Deviation')
    elif time_array.size > 0:
        console.print("[yellow]Warning: Skipping plot for main deviation signal as no valid data was available, though time array exists.[/yellow]")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title(f'Frequency Deviation over Time ({weighting_type})')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    # Y-limits for ax1 (Hz)
    if len(main_dev_sig_to_plot) > 0: # Check if there's data to calculate limits from
        min_hz_main = np.nanmin(main_dev_sig_to_plot)
        max_hz_main = np.nanmax(main_dev_sig_to_plot)
        if np.isfinite(min_hz_main) and np.isfinite(max_hz_main):
            if min_hz_main < max_hz_main:
                ax1.set_ylim(min_hz_main - 0.1 * abs(min_hz_main), max_hz_main + 0.1 * abs(max_hz_main))
            elif min_hz_main == max_hz_main: # If flat line
                 ax1.set_ylim(min_hz_main - 1, max_hz_main + 1) # Default +-1 Hz range
            # else: let autoscale handle if min_hz_main > max_hz_main (should not happen with nanmin/nanmax)
        else:
            console.print("[yellow]Warning: Could not determine valid Y-axis limits (Hz) for main deviation plot. Autoscale will be used.[/yellow]")

    # Secondary y-axis for percentage deviation (ax1b)
    ax1b = ax1.twinx()
    color2 = 'tab:red'
    ax1b.set_ylabel('Deviation (%)', color=color2)
    
    main_dev_sig_percent_to_plot = []
    if len(main_dev_sig_to_plot) > 0 and abs(ref_freq) >= 1e-9:
        main_dev_sig_percent_to_plot = (main_dev_sig_to_plot / ref_freq) * 100
    elif abs(ref_freq) < 1e-9:
        console.print("[yellow]Warning: Reference frequency is near zero. Percentage deviation plot will be empty.[/yellow]")
        main_dev_sig_percent_to_plot = [] # Ensure it's an empty list if not plottable

    if time_array.size > 0 and isinstance(main_dev_sig_percent_to_plot, np.ndarray) and main_dev_sig_percent_to_plot.size > 0:
        ax1b.plot(time_array, main_dev_sig_percent_to_plot, color=color2, linestyle='--', alpha=0.7, label=f'{weighting_type} Deviation (%)')
    elif time_array.size > 0:
        # This warning might be redundant if the one above (ref_freq near zero) already fired
        if not (abs(ref_freq) < 1e-9 and len(main_dev_sig_to_plot) > 0) : # Avoid double warning
             console.print("[yellow]Warning: Skipping plot for main deviation signal (percentage) as no valid data was available, though time array exists.[/yellow]")   
    ax1b.tick_params(axis='y', labelcolor=color2)

    if isinstance(main_dev_sig_percent_to_plot, np.ndarray) and main_dev_sig_percent_to_plot.size > 0: # Check if it's a plottable array
        min_pct_main = np.nanmin(main_dev_sig_percent_to_plot)
        max_pct_main = np.nanmax(main_dev_sig_percent_to_plot)
        if np.isfinite(min_pct_main) and np.isfinite(max_pct_main):
            if min_pct_main < max_pct_main:
                 ax1b.set_ylim(min_pct_main - 0.1 * abs(min_pct_main), max_pct_main + 0.1 * abs(max_pct_main))
            elif min_pct_main == max_pct_main:
                 ax1b.set_ylim(min_pct_main - 0.1, max_pct_main + 0.1) # Default +-0.1% range
            # else: let autoscale
        else:
            console.print("[yellow]Warning: Could not determine valid Y-axis limits (%) for main deviation plot. Autoscale will be used.[/yellow]")

    # Bottom subplot: Wow and Flutter Components
    # ax2 = axs[1] # Already defined
    if time_array.size > 0 and isinstance(wow_sig_to_plot, np.ndarray) and wow_sig_to_plot.size > 0:
        ax2.plot(time_array, wow_sig_to_plot, label='Wow Component', color='orange')
    elif time_array.size > 0:
        console.print("[yellow]Info: Wow component plot skipped as it contains no data.[/yellow]")
        
    if time_array.size > 0 and isinstance(flutter_sig_to_plot, np.ndarray) and flutter_sig_to_plot.size > 0:
        ax2.plot(time_array, flutter_sig_to_plot, label='Flutter Component', color='green', alpha=0.7)
    elif time_array.size > 0:
        console.print("[yellow]Info: Flutter component plot skipped as it contains no data.[/yellow]")
        
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency Deviation (Hz)')
    ax2.set_title(f'Wow and Flutter Components ({weighting_type})')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)

    # Y-limits for ax2 (Hz)
    combined_wf_for_ylim = []
    if len(wow_sig_to_plot) > 0: combined_wf_for_ylim.extend(wow_sig_to_plot)
    if len(flutter_sig_to_plot) > 0: combined_wf_for_ylim.extend(flutter_sig_to_plot)
    
    if len(combined_wf_for_ylim) > 0:
        min_hz_wf = np.nanmin(combined_wf_for_ylim)
        max_hz_wf = np.nanmax(combined_wf_for_ylim)
        if np.isfinite(min_hz_wf) and np.isfinite(max_hz_wf):
            if min_hz_wf < max_hz_wf:
                ax2.set_ylim(min_hz_wf - 0.1 * abs(min_hz_wf), max_hz_wf + 0.1 * abs(max_hz_wf))
            elif min_hz_wf == max_hz_wf:
                ax2.set_ylim(min_hz_wf - 1, max_hz_wf + 1)
        else:
            console.print("[yellow]Warning: Could not determine valid Y-axis limits (Hz) for wow/flutter plot. Autoscale will be used.[/yellow]")


    # Secondary y-axis for percentage deviation on bottom plot (ax2b)
    ax2b = ax2.twinx()
    ax2b.set_ylabel('Deviation (%)', color=color2) # Re-use color2

    if len(combined_wf_for_ylim) > 0 and abs(ref_freq) >= 1e-9:
        wf_sig_percent_to_plot = (np.array(combined_wf_for_ylim) / ref_freq) * 100
        min_pct_wf = np.nanmin(wf_sig_percent_to_plot)
        max_pct_wf = np.nanmax(wf_sig_percent_to_plot)
        if np.isfinite(min_pct_wf) and np.isfinite(max_pct_wf):
            if min_pct_wf < max_pct_wf:
                ax2b.set_ylim(min_pct_wf - 0.1 * abs(min_pct_wf), max_pct_wf + 0.1 * abs(max_pct_wf))
            elif min_pct_wf == max_pct_wf:
                ax2b.set_ylim(min_pct_wf - 0.1, max_pct_wf + 0.1)
            # else autoscale
        else:
            console.print("[yellow]Warning: Could not determine valid Y-axis limits (%) for wow/flutter plot. Autoscale will be used.[/yellow]")
    elif abs(ref_freq) < 1e-9 and len(combined_wf_for_ylim) > 0:
         console.print("[yellow]Warning: Ref freq near zero, cannot plot % deviation for wow/flutter.[/yellow]")
    # If combined_wf_for_ylim is empty, this axis will be empty too.
    elif len(combined_wf_for_ylim) == 0 and time_array.size > 0: # if no data for y axis but time axis exists
        pass # Let it autoscale or be empty, no specific warning here as component specific ones are above

    ax2b.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()

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
    if signal_to_analyze is None or signal_to_analyze.size == 0 or np.all(np.isnan(signal_to_analyze)):
        console.print("[yellow]Warning: No valid signal data (None, empty, or all NaN) to plot spectrum. Skipping spectrum plot.[/yellow]")
        return

    # Ensure signal_to_analyze is a numpy array for np.isnan checks if it passed initial checks
    if not isinstance(signal_to_analyze, np.ndarray):
        signal_to_analyze = np.array(signal_to_analyze)

    # Remove NaNs before Welch, as it might not handle them well or might return all NaNs.
    valid_signal = signal_to_analyze[~np.isnan(signal_to_analyze)]
    if valid_signal.size == 0:
        console.print("[yellow]Warning: Signal contains only NaNs after cleaning. Skipping spectrum plot.[/yellow]")
        return

    nperseg = min(valid_signal.size, 2048)
    if nperseg == 0: # Should be caught by valid_signal.size == 0, but as a safeguard.
         console.print("[yellow]Warning: Signal too short for spectrum analysis after NaN removal. Skipping spectrum plot.[/yellow]")
         return

    try:
        frequencies, psd = signal.welch(valid_signal, fs=sample_rate, window='hann', nperseg=nperseg, scaling='density')
    except ValueError as e:
        console.print(f"[bold red]Error during Welch calculation for spectrum: {e}. Skipping spectrum plot.[/bold red]")
        return
    
    if frequencies.size == 0 or psd.size == 0 or np.all(np.isnan(psd)):
        console.print("[yellow]Warning: PSD calculation resulted in empty or all NaN data. Skipping spectrum plot.[/yellow]")
        return

    # Convert PSD to dB, adding a small epsilon to avoid log(0) or log(negative)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-12)) # Ensure psd is not negative before log
    
    # Check if psd_db itself became all NaN (e.g., if psd was all zero or negative)
    if np.all(np.isnan(psd_db)):
        console.print("[yellow]Warning: PSD_dB is all NaN. Skipping spectrum plot.[/yellow]")
        return

    plt.figure(figsize=(12, 7))
    plt.semilogx(frequencies, psd_db)

    plt.title(f'Spectrum of Frequency Deviation - {title_suffix} ({weighting_type})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.grid(True, which="both", ls="-", alpha=0.7)

    plot_min_freq = 0.1 # Default plot start frequency
    if frequencies.size > 1 and frequencies[1] > 0: # Frequencies from Welch should be positive
        plot_min_freq = max(0.1, frequencies[1]) # Smallest freq to plot, avoid 0 for log
    
    plot_max_freq = sample_rate / 2 # Default plot end frequency
    if frequencies.size > 0:
        plot_max_freq = frequencies[-1]

    # Highlight Wow and Flutter regions
    # Ensure region frequencies are finite numbers
    if np.isfinite(min_wow_freq) and np.isfinite(max_wow_freq) and min_wow_freq < max_wow_freq:
        plt.axvspan(max(min_wow_freq, plot_min_freq), min(max_wow_freq, plot_max_freq), 
                    color='skyblue', alpha=0.3, label=f'Wow ({min_wow_freq:.1f}-{max_wow_freq:.1f} Hz)')
    
    if np.isfinite(min_flutter_freq) and np.isfinite(max_flutter_freq) and min_flutter_freq < max_flutter_freq:
        plt.axvspan(max(min_flutter_freq, plot_min_freq), min(max_flutter_freq, plot_max_freq), 
                    color='lightgreen', alpha=0.3, label=f'Flutter ({min_flutter_freq:.1f}-{max_flutter_freq:.1f} Hz)')
    
    if plot_min_freq < plot_max_freq :
        plt.xlim(plot_min_freq, plot_max_freq)
    else: # Fallback if frequencies are somehow problematic
        plt.xlim(0.1, sample_rate / 2)

    # Y-axis limits for spectrum
    valid_psd_db = psd_db[np.isfinite(psd_db)] # Use only finite values for percentile calculation
    if valid_psd_db.size > 0:
        y_min_plot = np.percentile(valid_psd_db, 1) - 10
        y_max_plot = np.percentile(valid_psd_db, 99) + 10
        if np.isfinite(y_min_plot) and np.isfinite(y_max_plot) and y_min_plot < y_max_plot:
            plt.ylim(y_min_plot, y_max_plot)
        else:
            console.print("[yellow]Warning: Could not determine valid Y-axis limits for spectrum. Autoscale will be used.[/yellow]")
    else:
        console.print("[yellow]Warning: No finite PSD dB values to determine Y-axis limits for spectrum. Autoscale will be used.[/yellow]")

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
    # 1. Input Validation
    if signal_hz is None or signal_hz.size == 0 or np.all(np.isnan(signal_hz)):
        console.print(f"[yellow]Warning: Invalid input signal (None, empty, or all NaN) for '{label}'. Metrics will be zero.[/yellow]")
        return {
            f"{label} Peak (Hz)": 0.0,
            f"{label} Peak (%)": 0.0,
            f"{label} RMS (Hz)": 0.0,
            f"{label} RMS (%)": 0.0,
        }

    # 2. Safe Calculations
    # Use np.nanmax and np.nanmean to handle potential NaNs if not all elements are NaN
    # np.abs can introduce NaNs if the input contains them.
    abs_signal_hz = np.abs(signal_hz)
    peak_deviation_hz = np.nanmax(abs_signal_hz)
    
    # For RMS, square first, then nanmean, then sqrt. Squaring NaNs results in NaNs.
    squared_signal_hz = np.square(signal_hz)
    mean_square_hz = np.nanmean(squared_signal_hz)
    rms_deviation_hz = np.sqrt(mean_square_hz) if not np.isnan(mean_square_hz) else 0.0

    # Check for NaN/Inf results from calculations (e.g., if input was all Inf, or other edge cases)
    if not np.isfinite(peak_deviation_hz):
        console.print(f"[yellow]Warning: Peak deviation for '{label}' is not finite ({peak_deviation_hz}). Setting to 0.0.[/yellow]")
        peak_deviation_hz = 0.0
    
    if not np.isfinite(rms_deviation_hz):
        console.print(f"[yellow]Warning: RMS deviation for '{label}' is not finite ({rms_deviation_hz}). Setting to 0.0.[/yellow]")
        rms_deviation_hz = 0.0

    # Percentage calculations
    peak_deviation_percent = 0.0
    rms_deviation_percent = 0.0

    if abs(ref_freq) < 1e-9: # Check for near-zero ref_freq
        console.print(f"[yellow]Warning: Reference frequency for '{label}' is near zero. Percentage metrics will be zero.[/yellow]")
    else:
        peak_deviation_percent = (peak_deviation_hz / ref_freq) * 100
        rms_deviation_percent = (rms_deviation_hz / ref_freq) * 100
        
        if not np.isfinite(peak_deviation_percent):
            console.print(f"[yellow]Warning: Peak deviation (%) for '{label}' is not finite. Setting to 0.0.[/yellow]")
            peak_deviation_percent = 0.0
        if not np.isfinite(rms_deviation_percent):
            console.print(f"[yellow]Warning: RMS deviation (%) for '{label}' is not finite. Setting to 0.0.[/yellow]")
            rms_deviation_percent = 0.0
            
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
    # 1. Input Validation
    if instantaneous_frequency is None or instantaneous_frequency.size == 0 or np.all(np.isnan(instantaneous_frequency)):
        console.print("[yellow]Warning: Invalid input (None, empty, or all NaN) for drift calculation. Drift metrics will be zero.[/yellow]")
        return {
            "Average Frequency (Hz)": 0.0,
            "Overall Drift (Hz)": 0.0,
            "Overall Drift (%)": 0.0,
            "Initial Frequency (Hz)": 0.0,
            "Final Frequency (Hz)": 0.0,
            "Drift over Measurement (Hz)": 0.0,
            "Drift over Measurement (%)": 0.0,
        }

    # 2. Safe Calculations
    average_frequency_hz = np.nanmean(instantaneous_frequency)
    if not np.isfinite(average_frequency_hz):
        console.print(f"[yellow]Warning: Average frequency is not finite ({average_frequency_hz}). Setting to 0.0 for drift calculations.[/yellow]")
        average_frequency_hz = 0.0
        # If average is non-finite, instantaneous_frequency likely contains non-finite values everywhere,
        # so other means will also be non-finite.
    
    overall_drift_hz = average_frequency_hz - ref_freq
    if not np.isfinite(overall_drift_hz):
        overall_drift_hz = 0.0 # Should only happen if average_frequency_hz was Inf/NaN and not caught, or ref_freq is opposite Inf

    overall_drift_percent = 0.0
    drift_over_measurement_percent = 0.0

    if abs(ref_freq) < 1e-9:
        console.print("[yellow]Warning: Reference frequency for drift is near zero. Percentage drift metrics will be zero.[/yellow]")
    else:
        overall_drift_percent = (overall_drift_hz / ref_freq) * 100
        if not np.isfinite(overall_drift_percent): overall_drift_percent = 0.0


    # Calculate initial and final frequencies (average over ~0.5s)
    initial_frequency_hz = 0.0
    final_frequency_hz = 0.0
    
    half_sec_samples = int(0.5 * sample_rate)
    
    if half_sec_samples <= 0: # sample_rate might be very low or zero from bad input
        console.print("[yellow]Warning: half_sec_samples is zero or negative (invalid sample_rate?). Initial/Final frequency set to average.[/yellow]")
        initial_frequency_hz = average_frequency_hz
        final_frequency_hz = average_frequency_hz
    elif instantaneous_frequency.size < half_sec_samples:
        console.print("[yellow]Warning: Audio duration is less than 0.5s. Using full signal average for initial/final frequency.[/yellow]")
        initial_frequency_hz = average_frequency_hz
        final_frequency_hz = average_frequency_hz
    elif instantaneous_frequency.size < 2 * half_sec_samples:
        console.print("[yellow]Warning: Audio duration is less than 1s. Initial/final frequency periods might overlap.[/yellow]")
        # Still proceed with calculation for initial and final based on available data
        initial_slice = instantaneous_frequency[:half_sec_samples]
        final_slice = instantaneous_frequency[-half_sec_samples:]
        if initial_slice.size > 0:
            initial_frequency_hz = np.nanmean(initial_slice)
            if not np.isfinite(initial_frequency_hz): initial_frequency_hz = 0.0
        if final_slice.size > 0:
            final_frequency_hz = np.nanmean(final_slice)
            if not np.isfinite(final_frequency_hz): final_frequency_hz = 0.0
    else: # Audio is >= 1s
        initial_slice = instantaneous_frequency[:half_sec_samples]
        final_slice = instantaneous_frequency[-half_sec_samples:]
        if initial_slice.size > 0 : # Should always be true here given outer checks
            initial_frequency_hz = np.nanmean(initial_slice)
            if not np.isfinite(initial_frequency_hz): initial_frequency_hz = 0.0
        if final_slice.size > 0: # Should always be true here
            final_frequency_hz = np.nanmean(final_slice)
            if not np.isfinite(final_frequency_hz): final_frequency_hz = 0.0

    drift_over_measurement_hz = final_frequency_hz - initial_frequency_hz
    if not np.isfinite(drift_over_measurement_hz): drift_over_measurement_hz = 0.0

    if abs(ref_freq) >= 1e-9: # Avoid division by zero for the percentage
        drift_over_measurement_percent = (drift_over_measurement_hz / ref_freq) * 100
        if not np.isfinite(drift_over_measurement_percent): drift_over_measurement_percent = 0.0
    
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
        or an empty np.array([]) if demodulation fails or input is invalid.
    """
    # 0. Input Validation
    MIN_CYCLES_FOR_DEMOD = 5  # Minimum number of cycles of ref_freq needed
    MIN_SAMPLES_FILTER_PAD = 15 # Arbitrary padding for filter, filtfilt needs len(x) > 3 * order
    
    if audio_data is None or audio_data.size == 0:
        console.print("[yellow]Warning (demodulate_frequency): Input audio_data is None or empty. Returning empty array.[/yellow]")
        return np.array([])

    min_length_cycles = int(MIN_CYCLES_FOR_DEMOD * sample_rate / ref_freq) if ref_freq > 0 else MIN_SAMPLES_FILTER_PAD * 2
    min_length_filter = (4 * 3) + MIN_SAMPLES_FILTER_PAD # Assuming 4th order Butterworth for initial filter
    
    required_length = max(min_length_cycles, min_length_filter, 30) # Min 30 samples as a hard floor

    if audio_data.size < required_length:
        console.print(f"[yellow]Warning (demodulate_frequency): audio_data is too short ({audio_data.size} samples) for reliable processing (need ~{required_length}). Returning empty array.[/yellow]")
        return np.array([])
    
    if np.all(np.isnan(audio_data)):
        console.print("[yellow]Warning (demodulate_frequency): Input audio_data is all NaN. Returning empty array.[/yellow]")
        return np.array([])
    
    # If audio_data is all zero, it will likely result in zero instantaneous frequency, which is acceptable.

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
        filtered_audio_candidate = signal.filtfilt(b, a, audio_data)
        
        if filtered_audio_candidate.size == 0 or np.all(np.isnan(filtered_audio_candidate)) or np.all(filtered_audio_candidate == 0):
            console.print(f"[yellow]Warning (demodulate_frequency): Initial bandpass filter resulted in empty, all-NaN, or all-zero signal. Using unfiltered audio if possible, or failing.[/yellow]")
            # Check original audio again, if it was already problematic, then fail.
            if np.all(audio_data == 0) or np.all(np.isnan(audio_data)):
                 console.print(f"[bold red]Error (demodulate_frequency): Original audio also problematic. Cannot proceed.[/bold red]")
                 return np.array([])
            filtered_audio = audio_data # Fallback to original if it was not all zero/NaN
        else:
            filtered_audio = filtered_audio_candidate
            console.print(f"  Applied bandpass filter: {lowcut:.2f} Hz - {highcut:.2f} Hz")

    except ValueError as e:
        console.print(f"[bold red]Error designing or applying bandpass filter: {e}[/bold red]")
        # Check if error is due to data being too short for filtfilt
        if "len(x) must be >= 3 * order" in str(e) or filtered_audio.size < (3*order):
            console.print("[bold red]Error (demodulate_frequency): Data too short for initial bandpass filter. Cannot proceed.[/bold red]")
            return np.array([])
        console.print("[yellow]Proceeding with unfiltered audio for demodulation (if possible).[/yellow]")
        filtered_audio = audio_data # Fallback to original if not a length error.
        # If original audio itself is problematic (e.g. all zeros/NaNs), this will be handled below.

    if filtered_audio.size == 0 or np.all(np.isnan(filtered_audio)): # Double check after potential fallback
        console.print("[bold red]Error (demodulate_frequency): Filtered audio (or fallback) is empty or all NaN. Cannot proceed.[/bold red]")
        return np.array([])

    # If filtered_audio is all zeros, Hilbert will be all zeros, phase all zeros, diff all zeros. This is fine.

    # 2. Hilbert Transform
    analytic_signal = signal.hilbert(filtered_audio)
    if analytic_signal.size == 0 or np.all(np.isnan(analytic_signal)): # Should not happen if filtered_audio is okay
        console.print("[bold red]Error (demodulate_frequency): Analytic signal is empty or all NaN. Cannot proceed.[/bold red]")
        return np.array([])
    console.print("  Computed analytic signal using Hilbert transform.")
    
    # 3. Instantaneous Phase
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # np.angle and np.unwrap should handle all-zero analytic_signal correctly (phase becomes all zero).
    # If analytic_signal has NaNs, phase will have NaNs.

    # 4. Instantaneous Frequency
    if instantaneous_phase.size < 2: # np.diff needs at least 2 elements
        console.print("[yellow]Warning (demodulate_frequency): Phase array too short for frequency calculation. Returning empty array.[/yellow]")
        return np.array([])
        
    instantaneous_freq_diff = (np.diff(instantaneous_phase) / (2 * np.pi)) * sample_rate
    
    # Pad to match original length. 'edge' padding will replicate the single value if instantaneous_freq_diff has 1 element.
    # If instantaneous_freq_diff is empty (because instantaneous_phase had < 2 elements), this should ideally not be reached.
    # However, np.pad on empty array returns empty array.
    instantaneous_frequency = np.pad(instantaneous_freq_diff, (1, 0), 'edge') if instantaneous_freq_diff.size > 0 else np.array([])

    if instantaneous_frequency.size == 0:
         console.print("[yellow]Warning (demodulate_frequency): Resulting instantaneous frequency array is empty. Check input signal validity and length.[/yellow]")
         return np.array([]) # Explicitly return empty if it became empty

    # If instantaneous_frequency contains NaNs (propagated from analytic_signal), they are preserved.
    # This is acceptable as downstream functions (metrics, plots) are designed to handle NaNs.
    console.print("  Calculated instantaneous frequency.")
    
    return instantaneous_frequency

if __name__ == "__main__":
    main()
