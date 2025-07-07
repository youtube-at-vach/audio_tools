import argparse
import sys
import numpy as np
import sounddevice as sd
import scipy.signal
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
import math
import csv
import matplotlib.pyplot as plt

# Initialize consoles
console = Console()
error_console = Console(stderr=True, style="bold red")

# --- Helper Functions ---

def dbfs_to_linear(dbfs):
    """Converts dBFS to linear amplitude."""
    return 10**(dbfs / 20)

def linear_to_dbfs(linear_amp):
    """Converts linear amplitude to dBFS. Handles -np.inf for 0 or negative input."""
    if linear_amp <= 1e-12: # Treat very small numbers as effectively zero / noise floor
        return -np.inf
    return 20 * math.log10(linear_amp)

def _find_peak_amplitude_in_band(fft_magnitudes, fft_frequencies, target_freq, search_half_width_hz=20.0):
    """
    Finds the peak amplitude and actual frequency of a target frequency within a specified band in an FFT spectrum.
    """
    min_freq = target_freq - search_half_width_hz
    max_freq = target_freq + search_half_width_hz
    
    valid_indices = np.where((fft_frequencies >= min_freq) & (fft_frequencies <= max_freq))[0]

    if not valid_indices.size:
        # This might happen if the signal is extremely weak or absent in the band.
        return target_freq, 0.0  # Return nominal target_freq and 0 amplitude

    band_magnitudes = fft_magnitudes[valid_indices]
    peak_index_in_band = np.argmax(band_magnitudes)
    peak_abs_index = valid_indices[peak_index_in_band]
    
    return fft_frequencies[peak_abs_index], fft_magnitudes[peak_abs_index]


def channel_spec_to_index(spec, device_channels, channel_type="input"):
    """
    Converts channel specifier ('L', 'R', or numeric string) to a 0-based integer index.
    Validates against the number of available channels on the device.
    """
    try:
        if isinstance(spec, int): 
            idx = spec
        elif spec.upper() == 'L':
            idx = 0
        elif spec.upper() == 'R':
            if device_channels < 2:
                error_console.print(f"Error: Channel 'R' selected for {channel_type}, but device only has {device_channels} channel(s).")
                return None
            idx = 1
        else: 
            parsed_idx = int(spec)
            idx = parsed_idx 
            if not (0 <= idx < device_channels):
                 error_console.print(f"Error: Numeric {channel_type} channel index {idx} (from spec '{spec}') is out of range for device with {device_channels} channels (0 to {device_channels-1}).")
                 return None
        
        if not (0 <= idx < device_channels): 
            error_console.print(f"Error: {channel_type.capitalize()} channel index {idx} (derived from spec '{spec}') is out of range for device with {device_channels} channels.")
            return None
        return idx
    except ValueError:
        error_console.print(f"Error: Invalid {channel_type} channel specifier '{spec}'. Use 'L', 'R', or a numeric 0-based index (e.g., '0', '1').")
        return None


def select_device():
    """Allows the user to select an audio device for both input and output. Exits if no devices found or error."""
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
    table.add_column("Default SR", style="blue")

    suitable_devices = []
    for i, device in enumerate(devices):
        table.add_row(
            str(i),
            device['name'],
            str(device['max_input_channels']),
            str(device['max_output_channels']),
            str(device['default_samplerate'])
        )
        if device['max_input_channels'] > 0 and device['max_output_channels'] > 0:
            suitable_devices.append(i)
            
    console.print(table)
    
    if not suitable_devices:
        error_console.print("No devices found with both input and output capabilities suitable for crosstalk test.")
        sys.exit(1)

    while True:
        try:
            device_id_str = Prompt.ask("Select device ID for playback and recording")
            device_id = int(device_id_str)
            if 0 <= device_id < len(devices):
                if devices[device_id]['max_output_channels'] == 0:
                    error_console.print(f"Device ID {device_id} ({devices[device_id]['name']}) has no output channels. Please select another.")
                    continue
                if devices[device_id]['max_input_channels'] == 0:
                    error_console.print(f"Device ID {device_id} ({devices[device_id]['name']}) has no input channels. Please select another.")
                    continue
                console.print(f"Selected device: ID {device_id} - {devices[device_id]['name']}")
                return device_id
            else:
                error_console.print(f"Invalid ID. Please choose from the list (0 to {len(devices) - 1}).")
        except ValueError:
            error_console.print("Invalid input. Please enter a number.")
        except Exception as e:
            error_console.print(f"An unexpected error occurred during device selection: {e}")
            sys.exit(1)


def generate_sine_wave(frequency, amplitude_dbfs, duration, sample_rate):
    """Generates a sine wave NumPy array."""
    amplitude_linear = dbfs_to_linear(amplitude_dbfs)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude_linear * np.sin(2 * np.pi * frequency * t)
    return np.clip(wave, -1.0, 1.0)


def play_and_record_multi_ch(signal_to_play, sample_rate, play_device_idx, 
                             output_channel_device_idx, input_channel_device_indices, 
                             duration_seconds, device_info_obj):
    """
    Plays a mono signal on a specific output channel and records from multiple input channels.
    """
    device_max_output_ch = device_info_obj['max_output_channels']

    output_buffer = np.zeros((len(signal_to_play), device_max_output_ch), dtype=signal_to_play.dtype)
    if not (0 <= output_channel_device_idx < device_max_output_ch): 
        error_console.print(f"Internal Error: Output channel index {output_channel_device_idx} invalid for device with {device_max_output_ch} output channels.")
        return None
    output_buffer[:, output_channel_device_idx] = signal_to_play

    playrec_input_mapping = [idx + 1 for idx in input_channel_device_indices] 
    
    try:
        recorded_audio = sd.playrec(output_buffer, 
                                    samplerate=sample_rate, 
                                    channels=output_buffer.shape[1], 
                                    input_mapping=playrec_input_mapping,
                                    device=play_device_idx, 
                                    blocking=True)
        return recorded_audio
    except sd.PortAudioError as e:
        error_console.print(f"Audio I/O error during playrec for device {play_device_idx} (SR: {sample_rate}, OutCh: {output_channel_device_idx}, InMap: {playrec_input_mapping}): {e}")
        return None
    except Exception as e:
        error_console.print(f"Unexpected error during playrec for device {play_device_idx}: {e}")
        return None


def analyze_recorded_channels(recorded_data_multi_ch, sample_rate, target_frequency, window_name):
    """
    Analyzes each channel in the recorded data for the amplitude of the target frequency.
    Returns a list of dicts: {'nominal_freq', 'actual_freq', 'amplitude_linear'} for each channel.
    """
    if recorded_data_multi_ch is None or recorded_data_multi_ch.ndim == 0 :
        return [] 
    
    if recorded_data_multi_ch.ndim == 1: 
        recorded_data_multi_ch = recorded_data_multi_ch[:, np.newaxis]

    num_channels_recorded = recorded_data_multi_ch.shape[1]
    N = recorded_data_multi_ch.shape[0]
    results = []

    if N == 0:
        return [{'nominal_freq': target_frequency, 'actual_freq': target_frequency, 'amplitude_linear': 0.0}] * num_channels_recorded if num_channels_recorded > 0 else []

    try:
        window_samples = scipy.signal.get_window(window_name, N)
    except (ValueError, TypeError) as e:
        error_console.print(f"Invalid FFT window '{window_name}': {e}. Using 'hann'.")
        window_name = 'hann'
        window_samples = scipy.signal.get_window(window_name, N)
    
    fft_frequencies = np.fft.rfftfreq(N, d=1/sample_rate)

    for i in range(num_channels_recorded):
        channel_data = recorded_data_multi_ch[:, i]
        windowed_signal = channel_data * window_samples
        fft_result = np.fft.rfft(windowed_signal)
        
        scaled_fft_magnitude = np.abs(fft_result) * (2 / np.sum(window_samples))
        
        actual_freq, linear_amp = _find_peak_amplitude_in_band(scaled_fft_magnitude, fft_frequencies, target_frequency)
        results.append({'nominal_freq': target_frequency, 'actual_freq': actual_freq, 'amplitude_linear': linear_amp})
    
    return results


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Audio Crosstalk Analyzer: Measures crosstalk between audio channels by playing a tone on one output channel and measuring the signal level on specified input channels.")

    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument("--frequency", type=float, default=1000.0, help="Frequency for single mode test (Hz). Default: 1000.0. Must be positive.")
    mode_group.add_argument("--sweep", action='store_true', help="Enable sweep mode. Overrides --frequency.")
    mode_group.add_argument("--start_freq", type=float, default=20.0, help="Start frequency for sweep mode (Hz). Default: 20.0. Must be positive.")
    mode_group.add_argument("--end_freq", type=float, default=20000.0, help="End frequency for sweep mode (Hz). Default: 20000.0. Must be positive.")
    mode_group.add_argument("--points_per_octave", "-ppo", type=int, default=3, help="Number of test points per octave in sweep mode. Default: 3. Must be positive.")

    signal_group = parser.add_argument_group('Signal Parameters')
    signal_group.add_argument("--amplitude", type=float, default=-12.0, help="Amplitude of the test signal (dBFS). Default: -12.0. Must be <= 0.")
    
    device_group = parser.add_argument_group('Audio Device and Channel Parameters')
    device_group.add_argument("--device", type=int, help="Audio device ID for playback and recording. Prompts if not provided.")
    device_group.add_argument("--sample_rate", type=int, default=48000, help="Sampling rate (Hz). Default: 48000")
    device_group.add_argument("--output_channel", "-oc", type=str, default='L', help="Output channel for test signal (e.g., 'L', 'R', or numeric 0-based index '0', '1', ...). Default: 'L'")
    device_group.add_argument("--input_channels", "-ic", type=str, nargs='+', required=True, help="List of input channels to record (e.g., 'L' 'R' or '0' '1' ...). First channel is the reference for crosstalk calculation (i.e., the channel receiving the direct signal or loopback of output_channel). All specified input channels must be unique.")

    analysis_group = parser.add_argument_group('Analysis Parameters')
    analysis_group.add_argument("--window", type=str, default='hann', help="FFT window type (e.g., hann, blackmanharris). Default: 'hann'")
    analysis_group.add_argument("--duration_per_step", type=float, default=0.5, help="Duration of tone and recording for each frequency step (seconds). Default: 0.5. Must be positive.")

    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("--output_csv", type=str, default=None, help="Path to save results in CSV format (e.g., results.csv).")
    output_group.add_argument("--output_plot", type=str, default=None, help="Path to save crosstalk plot as an image (e.g., plot.png). Plot is generated for sweep mode only.")
    output_group.add_argument("--no_plot_display", action='store_true', help="Suppress interactive display of the plot. Plot will still be saved if --output_plot is specified.")


    args = parser.parse_args()

    if args.amplitude > 0:
        error_console.print("Error: Signal amplitude (--amplitude) must be 0 dBFS or less.")
        sys.exit(1)
    if args.duration_per_step <= 0:
        error_console.print("Error: Duration per step (--duration_per_step) must be positive.")
        sys.exit(1)


    # --- Device Selection and Validation ---
    if args.device is None:
        selected_device_idx = select_device()
    else:
        selected_device_idx = args.device
        try:
            devices = sd.query_devices() 
            if not (0 <= selected_device_idx < len(devices)):
                error_console.print(f"Error: Device ID {selected_device_idx} is invalid. Max ID is {len(devices)-1}.")
                sys.exit(1)
            device_info_check = sd.query_devices(selected_device_idx) 
            if device_info_check['max_output_channels'] == 0:
                error_console.print(f"Error: Device ID {selected_device_idx} ({device_info_check['name']}) has no output channels.")
                sys.exit(1)
            if device_info_check['max_input_channels'] == 0:
                error_console.print(f"Error: Device ID {selected_device_idx} ({device_info_check['name']}) has no input channels.")
                sys.exit(1)
            console.print(f"Using specified device ID: {selected_device_idx} - {device_info_check['name']}")
        except sd.PortAudioError as e:
            error_console.print(f"Error querying audio devices: {e}")
            sys.exit(1)
        except Exception as e: 
            error_console.print(f"An unexpected error occurred validating device ID {args.device}: {e}")
            sys.exit(1)
            
    try:
        device_info = sd.query_devices(selected_device_idx)
    except sd.PortAudioError as e:
        error_console.print(f"Error querying capabilities for device {selected_device_idx}: {e}")
        sys.exit(1)

    # --- Channel Conversion and Validation ---
    output_channel_idx = channel_spec_to_index(args.output_channel, device_info['max_output_channels'], "output")
    if output_channel_idx is None:
        sys.exit(1)

    if len(args.input_channels) < 2:
        error_console.print("Error: At least two input channels must be specified (one reference, one or more for crosstalk measurement against).")
        sys.exit(1)

    input_channel_indices = []
    for spec in args.input_channels:
        idx = channel_spec_to_index(spec, device_info['max_input_channels'], "input")
        if idx is None:
            sys.exit(1)
        if idx in input_channel_indices: 
            error_console.print(f"Error: Input channel '{spec}' (resolved to index {idx}) is effectively a duplicate of a previously specified input channel. All resolved input channel indices must be unique.")
            sys.exit(1)
        input_channel_indices.append(idx)
    
    for i, undriven_idx_check in enumerate(input_channel_indices[1:]): 
        if output_channel_idx == undriven_idx_check:
            console.print(f"[yellow]Warning: Output channel ('{args.output_channel}' -> index {output_channel_idx}) is the same as undriven input channel '{args.input_channels[i+1]}' (index {undriven_idx_check}). This setup might not measure crosstalk correctly unless intended for a specific test configuration.[/yellow]")


    # --- Frequency List Generation ---
    frequencies_to_test = []
    if args.sweep:
        if args.start_freq <= 0 or args.end_freq <= 0:
            error_console.print("Error: Sweep frequencies (--start_freq, --end_freq) must be positive.")
            sys.exit(1)
        if args.start_freq >= args.end_freq:
            error_console.print("Error: Start frequency must be less than end frequency for sweep mode.")
            sys.exit(1)
        if args.points_per_octave <= 0:
            error_console.print("Error: Points per octave (--points_per_octave) must be positive.")
            sys.exit(1)
        
        current_freq = args.start_freq
        while current_freq <= args.end_freq * (1 + 1e-9): 
            frequencies_to_test.append(current_freq)
            octave_multiplier = 2**(1/args.points_per_octave)
            next_freq = current_freq * octave_multiplier
            
            if next_freq <= current_freq : 
                if len(frequencies_to_test) == 1 or frequencies_to_test[-1] < args.end_freq : 
                     if args.end_freq not in frequencies_to_test and args.end_freq > frequencies_to_test[-1]:
                          frequencies_to_test.append(args.end_freq) 
                break 
            if len(frequencies_to_test) >= 500 : 
                error_console.print("Warning: More than 500 frequency points generated for sweep, stopping. Adjust PPO or range.")
                if args.end_freq not in frequencies_to_test and args.end_freq > frequencies_to_test[-1]: 
                     if len(frequencies_to_test) < 501:
                         frequencies_to_test.append(args.end_freq) 
                break
            current_freq = next_freq
        
        if not frequencies_to_test: 
             error_console.print(f"Error: No frequencies generated for sweep from {args.start_freq} to {args.end_freq} with PPO {args.points_per_octave}.")
             sys.exit(1)
    else: 
        if args.frequency <=0:
            error_console.print("Error: Single test frequency (--frequency) must be positive.")
            sys.exit(1)
        frequencies_to_test.append(args.frequency)

    console.print("\n[cyan]Device Configuration:[/cyan]")
    console.print(f"  Playback/Record Device: {device_info['name']} (ID: {selected_device_idx}, Device SR: {device_info['default_samplerate']} Hz)")
    console.print(f"  Script Using Sample Rate: {args.sample_rate} Hz")
    console.print(f"  Test Signal Output Channel: '{args.output_channel}' (Device Index: {output_channel_idx})")
    console.print(f"  Input Channels (Recorded): {args.input_channels} (Device Indices: {input_channel_indices})")
    console.print(f"    Reference Input Channel (for driven signal level): '{args.input_channels[0]}' (Device Index: {input_channel_indices[0]})")
    
    console.print("\n[cyan]Test Parameters:[/cyan]")
    console.print(f"  Mode: {'Sweep' if args.sweep else 'Single Frequency'}")
    console.print("  Frequencies: " + ", ".join([f"{f:.2f}" for f in frequencies_to_test]) + " Hz")
    console.print(f"  Signal Amplitude: {args.amplitude} dBFS")
    console.print(f"  Duration per Step: {args.duration_per_step} s")
    console.print(f"  FFT Window: {args.window}")

    all_results_data = [] 

    console.print("\n[green]Starting crosstalk analysis...[/green]")
    for freq_hz in frequencies_to_test:
        console.print(f"  Testing frequency: {freq_hz:.2f} Hz...")
        
        signal = generate_sine_wave(freq_hz, args.amplitude, args.duration_per_step, args.sample_rate)
        
        recorded_data = play_and_record_multi_ch(signal, args.sample_rate, 
                                                 selected_device_idx, 
                                                 output_channel_idx, input_channel_indices,
                                                 args.duration_per_step, device_info) 
        
        freq_result_row = {'freq_hz': freq_hz, 'ref_amp_dbfs': np.nan} 
        for i in range(1, len(input_channel_indices)): 
            freq_result_row[f'undriven_ch_{i}_spec'] = args.input_channels[i] 
            freq_result_row[f'undriven_ch_{i}_idx'] = input_channel_indices[i]
            freq_result_row[f'undriven_ch_{i}_amp_dbfs'] = np.nan
            freq_result_row[f'crosstalk_ch_{i}_db'] = np.nan

        if recorded_data is None:
            error_console.print(f"    Failed to play/record at {freq_hz:.2f} Hz. Results for this frequency will be NaN.")
            all_results_data.append(freq_result_row) 
            continue

        analysis_results_per_input_ch = analyze_recorded_channels(recorded_data, args.sample_rate, freq_hz, args.window)

        if not analysis_results_per_input_ch or len(analysis_results_per_input_ch) != len(input_channel_indices):
            error_console.print(f"    Analysis failed or returned unexpected number of results at {freq_hz:.2f} Hz ({len(analysis_results_per_input_ch) if analysis_results_per_input_ch else 'None'} results for {len(input_channel_indices)} inputs). Results for this frequency will be NaN.")
            all_results_data.append(freq_result_row) 
            continue

        amp_driven_input_linear = analysis_results_per_input_ch[0]['amplitude_linear']
        ref_amp_dbfs = linear_to_dbfs(amp_driven_input_linear)
        freq_result_row['ref_amp_dbfs'] = ref_amp_dbfs
        
        console.print(f"    Reference Channel ('{args.input_channels[0]}'): {ref_amp_dbfs:.2f} dBFS (Actual freq: {analysis_results_per_input_ch[0]['actual_freq']:.2f} Hz)")

        for i in range(1, len(input_channel_indices)): 
            undriven_ch_spec = args.input_channels[i]
            amp_undriven_input_linear = analysis_results_per_input_ch[i]['amplitude_linear']
            undriven_amp_dbfs = linear_to_dbfs(amp_undriven_input_linear)
            
            crosstalk_db = np.nan 
            if amp_driven_input_linear > 1e-12 : 
                if amp_undriven_input_linear > 1e-12:
                    crosstalk_db = 20 * math.log10(amp_undriven_input_linear / amp_driven_input_linear)
                else: 
                    crosstalk_db = -np.inf 
            elif amp_undriven_input_linear > 1e-12: 
                 crosstalk_db = np.inf 
            
            console.print(f"    Undriven Channel ('{undriven_ch_spec}'): {undriven_amp_dbfs:.2f} dBFS (Actual freq: {analysis_results_per_input_ch[i]['actual_freq']:.2f} Hz). Crosstalk relative to '{args.input_channels[0]}': {crosstalk_db:.2f} dB")
            
            freq_result_row[f'undriven_ch_{i}_amp_dbfs'] = undriven_amp_dbfs
            freq_result_row[f'crosstalk_ch_{i}_db'] = crosstalk_db
        
        all_results_data.append(freq_result_row)

    # --- Display Results ---
    if not all_results_data:
        console.print("\n[yellow]No results to display (e.g., all frequencies failed or no frequencies tested).[/yellow]")
        sys.exit(0)
        
    console.print("\n[bold underline cyan]Crosstalk Analysis Results[/bold underline cyan]")
    results_table = Table(title=f"Crosstalk Measurement Summary (Output Ch: '{args.output_channel}')")
    results_table.add_column("Freq (Hz)", style="magenta", justify="right", min_width=10)
    results_table.add_column(f"Ref Ch ('{args.input_channels[0]}') Lvl (dBFS)", style="green", justify="right", min_width=22)

    for i in range(1, len(input_channel_indices)):
        undriven_ch_spec_header = args.input_channels[i] 
        results_table.add_column(f"Ch '{undriven_ch_spec_header}' Lvl (dBFS)", style="yellow", justify="right", min_width=20)
        results_table.add_column(f"Ch '{undriven_ch_spec_header}' Crosstalk (dB)", style="red", justify="right", min_width=22)

    for res_row in all_results_data:
        row_data_strings = [f"{res_row['freq_hz']:.2f}", f"{res_row['ref_amp_dbfs']:.2f}"]
        for i in range(1, len(input_channel_indices)):
            row_data_strings.append(f"{res_row.get(f'undriven_ch_{i}_amp_dbfs', np.nan):.2f}")
            row_data_strings.append(f"{res_row.get(f'crosstalk_ch_{i}_db', np.nan):.2f}")
        results_table.add_row(*row_data_strings)
    
    console.print(results_table)
    console.print(f"\n[green]Crosstalk analysis complete. Reference channel for crosstalk calculation is '{args.input_channels[0]}'. Negative crosstalk values indicate isolation (signal on undriven channel is lower than on reference channel).[/green]")

    # --- CSV Output ---
    if args.output_csv:
        try:
            with open(args.output_csv, 'w', newline='') as csvfile:
                # Define header based on the number of input channels
                header = ['Frequency (Hz)', f'Ref Ch ({args.input_channels[0]}) Lvl (dBFS)']
                for i in range(1, len(input_channel_indices)):
                    undriven_ch_spec_header = args.input_channels[i]
                    header.extend([
                        f"Ch '{undriven_ch_spec_header}' Lvl (dBFS)",
                        f"Ch '{undriven_ch_spec_header}' Crosstalk (dB)"
                    ])
                
                writer = csv.writer(csvfile)
                writer.writerow(header)
                
                for res_row in all_results_data:
                    row_data = [f"{res_row['freq_hz']:.2f}", f"{res_row['ref_amp_dbfs']:.2f}"]
                    for i in range(1, len(input_channel_indices)):
                        row_data.append(f"{res_row.get(f'undriven_ch_{i}_amp_dbfs', np.nan):.2f}")
                        row_data.append(f"{res_row.get(f'crosstalk_ch_{i}_db', np.nan):.2f}")
                    writer.writerow(row_data)
            console.print(f"\n[green]Results saved to CSV: {args.output_csv}[/green]")
        except IOError as e:
            error_console.print(f"\nError writing CSV file {args.output_csv}: {e}")
        except Exception as e:
            error_console.print(f"\nAn unexpected error occurred while writing CSV: {e}")

    # --- Plotting ---
    # Plot only if in sweep mode and there are results, and either output_plot is specified or interactive display is not suppressed.
    can_plot = args.sweep and all_results_data and (args.output_plot or not args.no_plot_display)
    
    if can_plot:
        console.print("\n[green]Generating plot...[/green]")
        try:
            frequencies = [r['freq_hz'] for r in all_results_data]
            
            plt.figure(figsize=(12, 7))
            
            for i in range(1, len(input_channel_indices)): # For each undriven channel
                crosstalk_values = [r.get(f'crosstalk_ch_{i}_db', np.nan) for r in all_results_data]
                # Filter out NaN for plotting if necessary, or let matplotlib handle them
                # For simplicity, we plot them; matplotlib usually breaks lines at NaNs.
                
                # Check if there's any valid data to plot for this channel
                if not all(np.isnan(crosstalk_values)):
                    plt.plot(frequencies, crosstalk_values, marker='o', linestyle='-', label=f"Crosstalk to Ch '{args.input_channels[i]}'")
                else:
                    console.print(f"[yellow]Skipping plot for Ch '{args.input_channels[i]}' as all crosstalk values are NaN.[/yellow]")

            plt.xscale('log')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Crosstalk (dB)")
            plt.title(f"Audio Crosstalk Analysis (Output Ch: '{args.output_channel}')")
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.5) # Grid for log scale
            
            # Add a horizontal line at -60dB as a common reference, if appropriate
            # plt.axhline(-60, color='grey', linestyle='--', linewidth=0.8, label='-60 dB Reference')
            
            # Improve Y-axis limits if possible, e.g., based on data range
            all_xtalk_values_flat = []
            for i in range(1, len(input_channel_indices)):
                all_xtalk_values_flat.extend([r.get(f'crosstalk_ch_{i}_db', np.nan) for r in all_results_data])
            
            valid_xtalk_values = [v for v in all_xtalk_values_flat if not np.isnan(v) and not np.isinf(v)]
            if valid_xtalk_values:
                min_val = min(valid_xtalk_values)
                max_val = max(valid_xtalk_values)
                y_margin = 10
                # Ensure min_val is not positive infinity if all are -inf
                if min_val == np.inf and max_val == np.inf: # All inf
                     pass # Keep default limits
                elif min_val == -np.inf and max_val == -np.inf: # all -inf
                     plt.ylim(-120, -30) # Example range for very low crosstalk
                else:
                    y_min_limit = min(-30, math.floor(min_val / 10) * 10 - y_margin) 
                    y_max_limit = max(0, math.ceil(max_val / 10) * 10 + y_margin)
                    if y_max_limit > y_min_limit +5 : #Ensure sensible range
                         plt.ylim(y_min_limit, y_max_limit)


            if args.output_plot:
                plt.savefig(args.output_plot)
                console.print(f"Plot saved to: {args.output_plot}")
            
            if not args.no_plot_display:
                console.print("Displaying plot window...")
                plt.show()
            
        except ImportError:
            error_console.print("\nMatplotlib is not installed. Cannot generate plot. Please install it: pip install matplotlib")
        except Exception as e:
            error_console.print(f"\nAn error occurred during plotting: {e}")
    elif args.output_plot and not args.sweep:
        console.print("\n[yellow]Plotting is enabled via --output_plot but only supported for sweep mode. No plot generated.[/yellow]")
    elif args.output_plot and not all_results_data:
         console.print("\n[yellow]Plotting is enabled via --output_plot but there are no results to plot. No plot generated.[/yellow]")


if __name__ == '__main__':
    main()
