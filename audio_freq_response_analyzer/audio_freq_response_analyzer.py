import numpy as np
import scipy.signal
import sys
import sounddevice as sd
import argparse
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
import csv
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt

# Initialize Rich Console for better terminal output
console = Console()
error_console = Console(stderr=True, style="bold red")


def generate_log_frequencies(start_freq, end_freq, points_per_octave):
    """
    Generates a list of frequencies logarithmically spaced by a certain number of points per octave.
    """
    if not (isinstance(start_freq, (int, float)) and start_freq > 0):
        raise ValueError("start_freq must be a positive number.")
    if not (isinstance(end_freq, (int, float)) and end_freq > 0):
        raise ValueError("end_freq must be a positive number.")
    if not (isinstance(points_per_octave, int) and points_per_octave > 0):
        raise ValueError("points_per_octave must be a positive integer.")
    if end_freq < start_freq:
        raise ValueError("end_freq must be greater than or equal to start_freq.")

    if np.isclose(start_freq, end_freq):
        return np.array([float(start_freq)])

    freqs = [float(start_freq)]
    current_freq = float(start_freq)
    ratio = 2**(1.0 / points_per_octave)

    while True:
        next_freq = current_freq * ratio
        if next_freq > end_freq * (1 + 1e-6): 
            break
        freqs.append(next_freq)
        current_freq = next_freq
    
    if not any(np.isclose(f, end_freq) for f in freqs):
        if freqs[-1] < end_freq * (1 - 1e-6) : 
            freqs.append(float(end_freq))

    return np.array(sorted(list(set(freqs))))


def generate_sine_segment(frequency, amplitude_dbfs, duration, sample_rate):
    """
    Generates a single sine wave segment.
    """
    if not (isinstance(frequency, (int, float)) and frequency > 0):
        raise ValueError("frequency must be a positive number.")
    if not isinstance(amplitude_dbfs, (int, float)):
        raise ValueError("amplitude_dbfs must be a number.")
    if not (isinstance(duration, (int, float)) and duration > 0):
        raise ValueError("duration must be a positive number.")
    if not (isinstance(sample_rate, int) and sample_rate > 0):
        raise ValueError("sample_rate must be a positive integer.")

    amp_linear = 10**(amplitude_dbfs / 20.0)
    if amp_linear > 1.0:
        error_console.print(f"Warning: Requested amplitude {amp_linear:.2f} (from {amplitude_dbfs:.1f} dBFS) exceeds 1.0. Capping at 1.0.")
        amp_linear = 1.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amp_linear * np.sin(2 * np.pi * frequency * t)
    return signal.astype(np.float32)


def analyze_frequency_segment(recorded_segment, target_freq, sample_rate, window_name):
    """
    Analyzes a segment of recorded audio to find the amplitude and phase of a target frequency.
    """
    if not isinstance(recorded_segment, np.ndarray) or recorded_segment.ndim != 1:
        raise ValueError("recorded_segment must be a 1D NumPy array.")
    if not isinstance(target_freq, (int, float)) or target_freq <= 0:
        raise ValueError("target_freq must be a positive number.")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if not isinstance(window_name, str):
        raise ValueError("window_name must be a string.")
        
    if len(recorded_segment) == 0:
        return (-np.inf, 0.0, 0.0) 

    try:
        window_samples = scipy.signal.get_window(window_name, len(recorded_segment))
    except ValueError:
        error_console.print(f"Warning: Invalid window name '{window_name}'. Using 'hann' as fallback.")
        window_name = 'hann'
        window_samples = scipy.signal.get_window(window_name, len(recorded_segment))

    windowed_segment = recorded_segment * window_samples
    
    fft_output = np.fft.rfft(windowed_segment)
    fft_freqs = np.fft.rfftfreq(len(recorded_segment), d=1.0/sample_rate)
    
    if np.sum(window_samples) == 0:
        scaled_fft_mag = np.abs(fft_output) 
    else:
        scaled_fft_mag = np.abs(fft_output) * (2.0 / np.sum(window_samples))

    idx = np.argmin(np.abs(fft_freqs - target_freq))
    actual_detected_freq = fft_freqs[idx]
    
    bin_resolution = float(sample_rate) / len(recorded_segment)
    if abs(actual_detected_freq - target_freq) > bin_resolution * 1.5:
        return (-np.inf, 0.0, actual_detected_freq)

    peak_magnitude_linear = scaled_fft_mag[idx]
    amplitude_dbfs = 20 * np.log10(peak_magnitude_linear) if peak_magnitude_linear > 1e-10 else -np.inf
    
    complex_fft_value_at_peak = fft_output[idx]
    phase_rad = np.angle(complex_fft_value_at_peak)
    phase_degrees = np.degrees(phase_rad)
    
    return (amplitude_dbfs, phase_degrees, actual_detected_freq)


def select_device():
    """Allows the user to select an audio device. Exits if no devices or error."""
    try:
        devices = sd.query_devices()
    except sd.PortAudioError as e:
        error_console.print(f"Error querying audio devices: {e}")
        sys.exit(1)
        
    if not devices:
        error_console.print("No audio devices found.")
        sys.exit(1)

    table = Table(title="Available Audio Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Max Input Ch", style="green")
    table.add_column("Max Output Ch", style="yellow")
    table.add_column("Def. Sample Rate", style="blue")

    for i, device in enumerate(devices):
        table.add_row(
            str(i),
            device['name'],
            str(device['max_input_channels']),
            str(device['max_output_channels']),
            str(device['default_samplerate'])
        )
    console.print(table)
    
    while True:
        try:
            device_id_str = Prompt.ask("Select device ID")
            device_id = int(device_id_str)
            if 0 <= device_id < len(devices):
                console.print(f"Selected device: ID {device_id} - {devices[device_id]['name']}")
                return device_id
            else:
                error_console.print(f"Invalid ID. Please choose between 0 and {len(devices) - 1}.")
        except ValueError:
            error_console.print("Invalid input. Please enter a number.")
        except Exception as e:
            error_console.print(f"An unexpected error occurred during device selection: {e}")
            sys.exit(1)


def play_record_tone_segment(tone_segment, sample_rate, device_idx, 
                             output_channel_idx_0based, input_channel_idx_0based, 
                             record_duration):
    played_signal_mono = tone_segment.astype(np.float32)
    actual_output_mapping = [output_channel_idx_0based + 1]
    actual_input_mapping = [input_channel_idx_0based + 1]

    try:
        recorded_data = sd.playrec(
            data=played_signal_mono, 
            samplerate=sample_rate, 
            channels=1, 
            input_mapping=actual_input_mapping, 
            output_mapping=actual_output_mapping, 
            device=device_idx, 
            blocking=True
        )
        sd.wait() 
        return recorded_data.flatten()
    except sd.PortAudioError as e:
        error_console.print(f"PortAudioError during play/record: {e}")
        return None
    except Exception as e:
        error_console.print(f"Unexpected error during play/record: {e}")
        return None


def save_results_to_csv(results_data, filename, console_instance):
    """Saves the frequency response results to a CSV file."""
    if not results_data:
        console_instance.print("[yellow]No results to save.[/yellow]")
        return

    header = ['Target Frequency (Hz)', 'Actual Frequency (Hz)', 'Amplitude (dBFS)', 'Phase (degrees)']
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for row in results_data:
                writer.writerow([
                    row.get('freq_target', ''),
                    row.get('freq_actual', ''),
                    row.get('amp_dbfs', ''),
                    # Use 'phase_unwrapped_deg' if available, otherwise 'phase_raw_deg'
                    row.get('phase_unwrapped_deg', row.get('phase_raw_deg', '')) 
                ])
        console_instance.print(f"[green]Results saved to {filename}[/green]")
    except IOError as e:
        error_console.print(f"Error saving results to CSV '{filename}': {e}")
    except Exception as e:
        error_console.print(f"An unexpected error occurred while saving CSV '{filename}': {e}")


def plot_frequency_response(results_data, amp_plot_filename, phase_plot_filename, display_plots, console_instance):
    """Plots frequency response (amplitude and phase) and saves to files if specified."""
    if not results_data:
        console_instance.print("[yellow]No data to plot.[/yellow]")
        return

    # Filter data for plotting: only include points with valid amplitude
    plot_data = [r for r in results_data if r.get('amp_dbfs', -np.inf) > -np.inf]

    if not plot_data:
        console_instance.print("[yellow]No valid data points to plot (all amplitudes are -inf).[/yellow]")
        return

    frequencies_hz = [r['freq_actual'] for r in plot_data] # Use actual detected frequency for plotting
    amplitudes_dbfs = [r['amp_dbfs'] for r in plot_data]
    
    # Check if there's valid phase data to plot
    phases_unwrapped_deg = [r.get('phase_unwrapped_deg') for r in plot_data]
    has_valid_phase_data = any(p is not None for p in phases_unwrapped_deg)
    # If phase_unwrapped_deg is None for some, filter them out for phase plot
    plot_phase_frequencies_hz = [frequencies_hz[i] for i, p in enumerate(phases_unwrapped_deg) if p is not None]
    plot_phases_unwrapped_deg = [p for p in phases_unwrapped_deg if p is not None]


    fig_amp, fig_phase = None, None # Initialize to None

    # Amplitude Plot
    try:
        fig_amp, ax_amp = plt.subplots()
        ax_amp.plot(frequencies_hz, amplitudes_dbfs, marker='.')
        ax_amp.set_xscale('log')
        ax_amp.set_title('Frequency Response - Amplitude')
        ax_amp.set_xlabel('Frequency (Hz)')
        ax_amp.set_ylabel('Amplitude (dBFS)')
        ax_amp.grid(True, which='both', linestyle='--')
        if amp_plot_filename:
            fig_amp.savefig(amp_plot_filename)
            console_instance.print(f"[green]Amplitude plot saved to {amp_plot_filename}[/green]")
    except Exception as e:
        error_console.print(f"Error creating/saving amplitude plot: {e}")


    # Phase Plot
    if has_valid_phase_data and plot_phase_frequencies_hz:
        try:
            fig_phase, ax_phase = plt.subplots()
            ax_phase.plot(plot_phase_frequencies_hz, plot_phases_unwrapped_deg, marker='.')
            ax_phase.set_xscale('log')
            ax_phase.set_title('Frequency Response - Phase')
            ax_phase.set_xlabel('Frequency (Hz)')
            ax_phase.set_ylabel('Phase (degrees)')
            ax_phase.grid(True, which='both', linestyle='--')
            if phase_plot_filename:
                fig_phase.savefig(phase_plot_filename)
                console_instance.print(f"[green]Phase plot saved to {phase_plot_filename}[/green]")
        except Exception as e:
            error_console.print(f"Error creating/saving phase plot: {e}")
    elif not has_valid_phase_data:
         console_instance.print("[yellow]No valid phase data to plot.[/yellow]")


    if display_plots:
        try:
            plt.show()
        except Exception as e:
            # This can happen in headless environments if 'Agg' wasn't forced or if plt.show() still fails
            error_console.print(f"Could not display plots (ensure you are in a GUI environment or plots are saved to file): {e}")

    if fig_amp:
        plt.close(fig_amp)
    if fig_phase:
        plt.close(fig_phase)


def main():
    parser = argparse.ArgumentParser(description="Measure audio frequency response.")
    # Existing args
    parser.add_argument("--start_freq", type=float, default=20.0, help="Start frequency (Hz)")
    parser.add_argument("--end_freq", type=float, default=20000.0, help="End frequency (Hz)")
    parser.add_argument("--points_per_octave", type=int, default=12, help="Number of points per octave")
    parser.add_argument("--amplitude", type=float, default=-20.0, help="Amplitude of test tone (dBFS)")
    parser.add_argument("--duration_per_step", type=float, default=0.2, help="Duration of each tone segment (seconds)")
    parser.add_argument("--device", type=int, help="Audio device ID. Prompts if not provided.")
    parser.add_argument("--output_channel", "-oc", type=str, choices=['L', 'R'], default='R', help="Output channel ('L' or 'R')")
    parser.add_argument("--input_channel", "-ic", type=str, choices=['L', 'R'], default='L', help="Input channel ('L' or 'R')")
    parser.add_argument("--sample_rate", type=int, default=48000, help="Sampling rate (Hz)")
    parser.add_argument("--window", type=str, default='hann', help="FFT window type for analysis")
    
    # New args for CSV and plotting
    parser.add_argument("--output_csv", type=str, default=None, help="Filename to save results as CSV.")
    parser.add_argument("--output_plot_amp", type=str, default=None, help="Filename to save amplitude plot (e.g., amp_response.png).")
    parser.add_argument("--output_plot_phase", type=str, default=None, help="Filename to save phase plot (e.g., phase_response.png).")
    parser.add_argument("--no_plot_display", action='store_true', help="Suppress displaying plots interactively.")
    
    args = parser.parse_args()

    # Device Selection and Validation (condensed for brevity, assumed correct from previous step)
    if args.device is None: 
        selected_device_idx = select_device() # select_device() handles its own errors
        # We can assume device_info will be valid if select_device() returns
        try:
            device_info = sd.query_devices(selected_device_idx)
        except sd.PortAudioError as e:
            error_console.print(f"Error querying selected device {selected_device_idx}: {e}")
            sys.exit(1)
        except Exception as e: # Should not happen if select_device is robust
            error_console.print(f"Unexpected error after device selection: {e}")
            sys.exit(1)
    else: 
        selected_device_idx = args.device
        try:
            devices = sd.query_devices()
            if not (0 <= selected_device_idx < len(devices)):
                error_console.print(f"Error: Device ID {selected_device_idx} is invalid. Available IDs are 0 to {len(devices)-1}.")
                sys.exit(1)
            device_info = sd.query_devices(selected_device_idx)
        except sd.PortAudioError as e:
            error_console.print(f"Error querying audio devices or device ID {selected_device_idx}: {e}")
            sys.exit(1)
        except Exception as e:
            error_console.print(f"An unexpected error occurred during device validation: {e}")
            sys.exit(1)

    console.print(f"Using device ID {selected_device_idx}: {device_info['name']}")
    output_ch_numeric = 0 if args.output_channel == 'L' else 1
    input_ch_numeric = 0 if args.input_channel == 'L' else 1

    if device_info['max_output_channels'] == 0:
        error_console.print(f"Error: Selected device '{device_info['name']}' has no output channels.")
        sys.exit(1)
    if output_ch_numeric >= device_info['max_output_channels']:
        error_console.print(f"Error: Output channel {args.output_channel} (idx {output_ch_numeric}) not available on device '{device_info['name']}'. Max output channels: {device_info['max_output_channels']-1}.")
        sys.exit(1)
    
    if device_info['max_input_channels'] == 0:
        error_console.print(f"Error: Selected device '{device_info['name']}' has no input channels.")
        sys.exit(1)
    if input_ch_numeric >= device_info['max_input_channels']:
        error_console.print(f"Error: Input channel {args.input_channel} (idx {input_ch_numeric}) not available on device '{device_info['name']}'. Max input channels: {device_info['max_input_channels']-1}.")
        sys.exit(1)

    freq_list = generate_log_frequencies(args.start_freq, args.end_freq, args.points_per_octave)
    console.print(f"Generated {len(freq_list)} frequency steps from {args.start_freq} Hz to {args.end_freq} Hz.")
    
    results_data = []
    for freq in freq_list:
        console.print(f"Measuring at {freq:.2f} Hz...")
        tone = generate_sine_segment(freq, args.amplitude, args.duration_per_step, args.sample_rate)
        
        recorded = play_record_tone_segment(
            tone, args.sample_rate, selected_device_idx, 
            output_ch_numeric, input_ch_numeric, 
            args.duration_per_step
        )
        
        if recorded is not None and len(recorded) > 0:
            amp_dbfs, phase_deg, actual_f = analyze_frequency_segment(recorded, freq, args.sample_rate, args.window)
            results_data.append({'freq_target': freq, 'freq_actual': actual_f, 'amp_dbfs': amp_dbfs, 'phase_raw_deg': phase_deg})
            console.print(f"  Actual: {actual_f:.2f} Hz, Amp: {amp_dbfs:.2f} dBFS, Phase: {phase_deg:.2f} deg")
        else:
            console.print(f"[yellow]Warning: No data recorded or error for {freq:.2f} Hz.[/yellow]")
            results_data.append({'freq_target': freq, 'freq_actual': 0.0, 'amp_dbfs': -np.inf, 'phase_raw_deg': 0.0, 'phase_unwrapped_deg': 0.0}) # Add unwrapped phase default


    # Phase Unwrapping
    valid_phases_raw_indices = [i for i, r in enumerate(results_data) if r.get('amp_dbfs', -np.inf) > -np.inf and 'phase_raw_deg' in r]
    
    if len(valid_phases_raw_indices) > 1:
        phases_to_unwrap_deg = [results_data[i]['phase_raw_deg'] for i in valid_phases_raw_indices]
        unwrapped_phases_deg_values = np.degrees(np.unwrap(np.deg2rad(phases_to_unwrap_deg)))
        
        unwrapped_idx = 0
        for i, r_entry in enumerate(results_data):
            if i in valid_phases_raw_indices:
                r_entry['phase_unwrapped_deg'] = unwrapped_phases_deg_values[unwrapped_idx]
                unwrapped_idx += 1
            else: 
                r_entry['phase_unwrapped_deg'] = r_entry.get('phase_raw_deg', None) # Use None for missing unwrapped phases
    elif len(valid_phases_raw_indices) == 1: 
        idx_single = valid_phases_raw_indices[0]
        results_data[idx_single]['phase_unwrapped_deg'] = results_data[idx_single]['phase_raw_deg']
        console.print("[yellow]Only one valid data point for phase; no unwrapping performed.[/yellow]")
    else:
        for r_entry in results_data: # Ensure all entries have the key even if no valid phase
             r_entry['phase_unwrapped_deg'] = r_entry.get('phase_raw_deg', None)
        console.print("[yellow]Not enough valid data points to unwrap phase.[/yellow]")


    # Print first 5 rows of results_data (already present)
    console.print("\n--- Frequency Response Measurement (First 5 points) ---")
    table = Table(title="Frequency Response Data (Sample)")
    table.add_column("Target Freq (Hz)", justify="right", style="cyan")
    table.add_column("Actual Freq (Hz)", justify="right", style="magenta")
    table.add_column("Amplitude (dBFS)", justify="right", style="green")
    table.add_column("Phase Raw (deg)", justify="right", style="yellow")
    table.add_column("Phase Unwrapped (deg)", justify="right", style="blue")

    for i, result in enumerate(results_data[:5]):
        table.add_row(
            f"{result['freq_target']:.2f}",
            f"{result.get('freq_actual', 0.0):.2f}",
            f"{result.get('amp_dbfs', -np.inf):.2f}",
            f"{result.get('phase_raw_deg', 0.0):.2f}",
            f"{result.get('phase_unwrapped_deg', result.get('phase_raw_deg', 'N/A')):.2f}"
        )
    console.print(table)

    # Save to CSV
    if args.output_csv:
        save_results_to_csv(results_data, args.output_csv, console)

    # Plotting
    plotting_needed = args.output_plot_amp or args.output_plot_phase or not args.no_plot_display
    if plotting_needed:
        plot_frequency_response(
            results_data, 
            args.output_plot_amp, 
            args.output_plot_phase, 
            not args.no_plot_display, 
            console
        )


if __name__ == '__main__':
    main()
