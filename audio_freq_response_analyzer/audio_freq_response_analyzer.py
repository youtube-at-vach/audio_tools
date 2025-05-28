import numpy as np
# import scipy.signal # Replaced by signal_processing_utils
import sys
# import sounddevice as sd # Replaced by common_audio_lib
import argparse
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
# import csv # Replaced by common_audio_lib
# import matplotlib # No longer directly used for backend selection
# matplotlib.use('Agg') # Backend selection is handled by common_audio_lib.output_formatting_utils if needed
# import matplotlib.pyplot as plt # Replaced by common_audio_lib

# Common Audio Library Imports
from common_audio_lib.audio_device_manager import select_audio_device, get_device_info
from common_audio_lib.signal_processing_utils import (
    dbfs_to_linear, # For amplitude conversion if needed, though generate_sine_wave takes dBFS
    linear_to_dbfs,
    perform_fft,
    find_peak_magnitude
)
from common_audio_lib.audio_io_utils import (
    generate_log_spaced_frequencies,
    generate_sine_wave,
    play_and_record,
    resolve_channel_specifier
)
from common_audio_lib.output_formatting_utils import save_results_to_csv, generate_plot


# Initialize Rich Console for better terminal output
console = Console()
error_console = Console(stderr=True, style="bold red")


# Helper functions generate_log_frequencies, generate_sine_segment, select_device,
# play_record_tone_segment, save_results_to_csv, plot_frequency_response
# are replaced by common_audio_lib functions.

def analyze_frequency_segment_common(
    recorded_segment: np.ndarray, 
    target_freq: float, 
    sample_rate: int, 
    window_name: str,
    search_half_width_hz: float = 20.0 # For find_peak_magnitude
    ):
    """
    Analyzes a segment of recorded audio using common library functions.
    Returns amplitude (dBFS), phase (degrees), and actual detected frequency.
    """
    if not isinstance(recorded_segment, np.ndarray) or recorded_segment.ndim != 1:
        # error_console.print("Error: recorded_segment must be a 1D NumPy array.")
        # Consider raising ValueError or returning sentinel values
        return (-np.inf, 0.0, target_freq) 
        
    if len(recorded_segment) == 0:
        return (-np.inf, 0.0, target_freq) 

    try:
        # perform_fft is assumed to return (fft_freqs, scaled_magnitudes, complex_fft_results)
        fft_freqs, scaled_magnitudes, complex_fft_results = perform_fft(
            recorded_segment, sample_rate, window_name
        )
    except ValueError as e: # e.g. invalid window from common_perform_fft
        error_console.print(f"Warning: FFT analysis failed for target {target_freq} Hz (window: '{window_name}'): {e}. Using 'hann'.")
        fft_freqs, scaled_magnitudes, complex_fft_results = perform_fft(
            recorded_segment, sample_rate, 'hann'
        )
    
    if len(fft_freqs) == 0: # FFT failed
        return (-np.inf, 0.0, target_freq)

    actual_detected_freq, peak_magnitude_linear = find_peak_magnitude(
        scaled_magnitudes, fft_freqs, target_freq, search_half_width_hz
    )

    # Check if the detected peak is reasonably close to the target frequency
    # This check was in the original, it's a good heuristic.
    bin_resolution = float(sample_rate) / len(recorded_segment) 
    if abs(actual_detected_freq - target_freq) > bin_resolution * 1.5: # Heuristic threshold
        # Peak found is too far from target, likely noise or distortion peak.
        # Return low amplitude for this target frequency.
        return (-np.inf, 0.0, actual_detected_freq) # Or target_freq? Original returned actual_detected_freq.

    amplitude_dbfs = linear_to_dbfs(peak_magnitude_linear, min_dbfs=-np.inf) # Use common lib, allow -np.inf

    # Find index of actual_detected_freq to get phase from complex_fft_results
    # This assumes actual_detected_freq is one of the values in fft_freqs.
    # find_peak_magnitude should ideally return the index, or make it easy to get.
    # For now, search for it:
    # A small tolerance might be needed if actual_detected_freq has float precision issues.
    idx = np.argmin(np.abs(fft_freqs - actual_detected_freq))
    
    # Verify that the found index indeed corresponds to the peak magnitude returned by find_peak_magnitude
    # This is a sanity check.
    if not np.isclose(scaled_magnitudes[idx], peak_magnitude_linear):
        # This could happen if find_peak_magnitude does interpolation or if there are multiple identical peaks.
        # If find_peak_magnitude guarantees the returned actual_detected_freq is exactly from fft_freqs, this check is simpler.
        # For robustness, one might re-evaluate the peak at `idx` if there's a mismatch concern.
        # However, find_peak_magnitude is expected to return the frequency at the max magnitude in the band.
        pass # Assuming idx is correct for the peak reported.

    complex_fft_value_at_peak = complex_fft_results[idx]
    phase_rad = np.angle(complex_fft_value_at_peak)
    phase_degrees = np.degrees(phase_rad)
    
    return (amplitude_dbfs, phase_degrees, actual_detected_freq)


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

    # --- Device Selection and Validation ---
    if args.device is None: 
        selected_device_id = select_audio_device(console, require_input=True, require_output=True)
        if selected_device_id is None:
            error_console.print("No device selected. Exiting.")
            sys.exit(1)
    else: 
        selected_device_id = args.device
        try:
            # Validate device ID using get_device_info
            temp_device_info = get_device_info(selected_device_id) # Will raise error if invalid
            if temp_device_info['max_output_channels'] == 0 or temp_device_info['max_input_channels'] == 0:
                error_console.print(f"Error: Device ID {selected_device_id} ({temp_device_info['name']}) must have both input and output channels.")
                sys.exit(1)
            console.print(f"Using specified device ID: {selected_device_id} - {temp_device_info['name']}")
        except ValueError as e: # Handles invalid device ID format or device not found
             error_console.print(f"Error validating device ID {args.device}: {e}")
             sys.exit(1)
        except Exception as e:
            error_console.print(f"An unexpected error occurred during device validation: {e}")
            sys.exit(1)

    current_device_info = get_device_info(selected_device_id)
    if not current_device_info: # Should be caught above, but as safeguard
        error_console.print(f"Fatal: Could not get info for selected device {selected_device_id}.")
        sys.exit(1)

    console.print(f"Using device ID {selected_device_id}: {current_device_info['name']}")

    # --- Channel Selection and Validation using common_audio_lib ---
    output_channel_idx_0based = resolve_channel_specifier(
        args.output_channel, current_device_info['max_output_channels'], "output", error_console
    )
    if output_channel_idx_0based is None: sys.exit(1)

    input_channel_idx_0based = resolve_channel_specifier(
        args.input_channel, current_device_info['max_input_channels'], "input", error_console
    )
    if input_channel_idx_0based is None: sys.exit(1)

    # --- Frequency List Generation using common_audio_lib ---
    try:
        freq_list = generate_log_spaced_frequencies(args.start_freq, args.end_freq, args.points_per_octave)
    except ValueError as e:
        error_console.print(f"Error generating frequencies: {e}")
        sys.exit(1)
    console.print(f"Generated {len(freq_list)} frequency steps from {args.start_freq} Hz to {args.end_freq} Hz.")
    
    results_data = []
    for freq_target in freq_list: # Renamed freq to freq_target for clarity
        console.print(f"Measuring at {freq_target:.2f} Hz...")
        
        # Signal Generation using common_audio_lib
        # generate_sine_wave takes amplitude_dbfs, duration, sample_rate, phase (degrees, default 0)
        tone_segment = generate_sine_wave(
            frequency=freq_target, 
            amplitude_dbfs=args.amplitude, 
            duration=args.duration_per_step, 
            sample_rate=args.sample_rate
            # phase=0.0 # Default phase
        )
        
        # Playback and Recording using common_audio_lib
        # play_and_record takes: device_id, signal_to_play_mono, sample_rate, 
        # output_channel_device_idx_0based, input_channel_device_indices_0based_list, ...
        # For single input channel, pass it as a list: [input_channel_idx_0based]
        recorded_segment_multi_ch = play_and_record(
            device_id=selected_device_id,
            signal_to_play_mono=tone_segment,
            sample_rate=args.sample_rate,
            output_channel_device_idx_0based=output_channel_idx_0based,
            input_channel_device_indices_0based_list=[input_channel_idx_0based], # Expects a list
            record_duration_secs=args.duration_per_step, # Match playback duration
            error_console=error_console
        )
        
        if recorded_segment_multi_ch is not None and recorded_segment_multi_ch.shape[0] > 0:
            # play_and_record returns [samples, channels], we need mono for analysis here.
            recorded_segment_mono = recorded_segment_multi_ch[:, 0] # Take the first (and only) recorded channel

            amp_dbfs, phase_deg, actual_f = analyze_frequency_segment_common(
                recorded_segment_mono, freq_target, args.sample_rate, args.window
            )
            results_data.append({'freq_target': freq_target, 'freq_actual': actual_f, 'amp_dbfs': amp_dbfs, 'phase_raw_deg': phase_deg})
            console.print(f"  Actual: {actual_f:.2f} Hz, Amp: {amp_dbfs:.2f} dBFS, Phase: {phase_deg:.2f} deg")
        else:
            console.print(f"[yellow]Warning: No data recorded or error for {freq_target:.2f} Hz.[/yellow]")
            # Add placeholder for this frequency to maintain data structure consistency for unwrapping/plotting
            results_data.append({'freq_target': freq_target, 'freq_actual': freq_target, 'amp_dbfs': -np.inf, 'phase_raw_deg': 0.0, 'phase_unwrapped_deg': 0.0})


    # Phase Unwrapping (logic remains similar, operates on results_data)
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


    # Print first 5 rows of results_data (logic remains similar)
    console.print("\n--- Frequency Response Measurement (First 5 points) ---")
    table = Table(title="Frequency Response Data (Sample)")
    table.add_column("Target Freq (Hz)", justify="right", style="cyan")
    table.add_column("Actual Freq (Hz)", justify="right", style="magenta")
    table.add_column("Amplitude (dBFS)", justify="right", style="green")
    table.add_column("Phase Raw (deg)", justify="right", style="yellow")
    table.add_column("Phase Unwrapped (deg)", justify="right", style="blue")

    for i, result_row in enumerate(results_data[:5]): # Renamed result to result_row
        table.add_row(
            f"{result_row['freq_target']:.2f}",
            f"{result_row.get('freq_actual', 0.0):.2f}",
            f"{result_row.get('amp_dbfs', -np.inf):.2f}",
            f"{result_row.get('phase_raw_deg', 0.0):.2f}",
            # Handle case where unwrapped might be None if raw was None
            f"{result_row.get('phase_unwrapped_deg', result_row.get('phase_raw_deg', 'N/A')) if result_row.get('phase_unwrapped_deg') is not None else 'N/A'}"
        )
    console.print(table)

    # Save to CSV using common_audio_lib
    if args.output_csv:
        # output_formatting_utils.save_results_to_csv expects list of dicts and fieldnames
        csv_fieldnames = ['freq_target', 'freq_actual', 'amp_dbfs', 'phase_raw_deg', 'phase_unwrapped_deg']
        # Ensure all dicts in results_data have these keys, even if values are None or NaN
        # The current loop populates these keys.
        save_results_to_csv(args.output_csv, results_data, csv_fieldnames, console=console)

    # Plotting using common_audio_lib
    plotting_needed = args.output_plot_amp or args.output_plot_phase or not args.no_plot_display
    if plotting_needed:
        # Filter data for plotting: only include points with valid amplitude
        plot_data_valid_amp = [r for r in results_data if r.get('amp_dbfs', -np.inf) > -float('inf')]

        if not plot_data_valid_amp:
            console.print("[yellow]No valid data points to plot (all amplitudes are -inf).[/yellow]")
        else:
            plot_frequencies_hz = [r['freq_actual'] for r in plot_data_valid_amp]
            
            if args.output_plot_amp or not args.no_plot_display:
                amplitudes_dbfs_plot = [r['amp_dbfs'] for r in plot_data_valid_amp]
                generate_plot(
                    x_data=plot_frequencies_hz,
                    y_data_list=[amplitudes_dbfs_plot],
                    legend_labels_list=['Amplitude'],
                    title='Frequency Response - Amplitude',
                    x_label='Frequency (Hz)',
                    y_label='Amplitude (dBFS)',
                    output_filename=args.output_plot_amp,
                    show_plot=(not args.no_plot_display and not args.output_plot_phase), # Show if no phase plot to follow or if only amp plot
                    log_x_scale=True,
                    console=console
                )

            # Phase Plot
            phases_unwrapped_plot = [r.get('phase_unwrapped_deg') for r in plot_data_valid_amp]
            # Filter out None values for phase plotting, and corresponding frequencies
            phase_plot_points = [(freq, phase) for freq, phase in zip(plot_frequencies_hz, phases_unwrapped_plot) if phase is not None]
            
            if phase_plot_points and (args.output_plot_phase or not args.no_plot_display):
                phase_plot_freqs = [p[0] for p in phase_plot_points]
                phase_plot_values = [p[1] for p in phase_plot_points]
                generate_plot(
                    x_data=phase_plot_freqs,
                    y_data_list=[phase_plot_values],
                    legend_labels_list=['Phase (Unwrapped)'],
                    title='Frequency Response - Phase',
                    x_label='Frequency (Hz)',
                    y_label='Phase (degrees)',
                    output_filename=args.output_plot_phase,
                    show_plot=(not args.no_plot_display),
                    log_x_scale=True,
                    console=console
                )
            elif not phase_plot_points and (args.output_plot_phase or not args.no_plot_display) :
                 console.print("[yellow]No valid phase data to plot.[/yellow]")


if __name__ == '__main__':
    main()
