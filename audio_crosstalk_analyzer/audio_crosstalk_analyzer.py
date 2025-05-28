import argparse
import sys
import numpy as np
# import sounddevice as sd # Replaced by audio_device_manager and audio_io_utils
# import scipy.signal # Replaced by signal_processing_utils
from rich.console import Console
from rich.table import Table # Kept for direct use if any, though common lib might handle some tables
from rich.prompt import Prompt # Kept for direct use
import math
# import csv # Replaced by output_formatting_utils
# import matplotlib.pyplot as plt # Replaced by output_formatting_utils

# Common Audio Library Imports
from common_audio_lib.audio_device_manager import list_available_devices, select_audio_device, get_device_info
from common_audio_lib.signal_processing_utils import (
    dbfs_to_linear, 
    linear_to_dbfs,
    perform_fft,
    find_peak_magnitude
)
from common_audio_lib.audio_io_utils import (
    resolve_channel_specifier,
    generate_sine_wave,
    play_and_record # Replaces play_and_record_multi_ch
)
from common_audio_lib.output_formatting_utils import save_results_to_csv, generate_plot


# Initialize consoles
console = Console()
error_console = Console(stderr=True, style="bold red")


# --- Helper Functions (Removed or Replaced) ---
# dbfs_to_linear: Replaced by common_audio_lib.signal_processing_utils.dbfs_to_linear
# linear_to_dbfs: Replaced by common_audio_lib.signal_processing_utils.linear_to_dbfs
# _find_peak_amplitude_in_band: Replaced by common_audio_lib.signal_processing_utils.find_peak_magnitude
# channel_spec_to_index: Replaced by common_audio_lib.audio_io_utils.resolve_channel_specifier
# select_device: Replaced by common_audio_lib.audio_device_manager.select_audio_device
# generate_sine_wave: Replaced by common_audio_lib.audio_io_utils.generate_sine_wave
# play_and_record_multi_ch: Replaced by common_audio_lib.audio_io_utils.play_and_record


def analyze_recorded_channels_common_lib(
    recorded_data_multi_ch: np.ndarray, 
    sample_rate: float, 
    target_frequency: float, 
    window_name: str,
    search_half_width_hz: float = 20.0 # Added for find_peak_magnitude
    ):
    """
    Analyzes each channel in the recorded data using common library functions.
    Returns a list of dicts: {'nominal_freq', 'actual_freq', 'amplitude_linear'} for each channel.
    """
    if recorded_data_multi_ch is None or recorded_data_multi_ch.ndim == 0:
        return [] 
    
    # Ensure recorded_data_multi_ch is 2D [samples, channels]
    if recorded_data_multi_ch.ndim == 1: 
        recorded_data_multi_ch = recorded_data_multi_ch[:, np.newaxis]

    num_channels_recorded = recorded_data_multi_ch.shape[1]
    N_samples = recorded_data_multi_ch.shape[0]
    results = []

    if N_samples == 0:
        # Return a list of default results for each expected channel
        return [{'nominal_freq': target_frequency, 'actual_freq': target_frequency, 'amplitude_linear': 0.0}] * num_channels_recorded if num_channels_recorded > 0 else []

    for i in range(num_channels_recorded):
        channel_data = recorded_data_multi_ch[:, i]
        
        # Perform FFT using common library function
        try:
            fft_freqs, fft_mags_scaled = perform_fft(channel_data, sample_rate, window_name=window_name)
        except ValueError as e: # e.g. invalid window name from perform_fft
             error_console.print(f"FFT error for channel {i} (nominal freq {target_frequency} Hz): {e}. Using 'hann'.")
             fft_freqs, fft_mags_scaled = perform_fft(channel_data, sample_rate, window_name='hann')


        if len(fft_freqs) == 0: # FFT failed or empty data
            actual_freq, linear_amp = target_frequency, 0.0
        else:
            # Find peak magnitude using common library function
            actual_freq, linear_amp = find_peak_magnitude(
                fft_mags_scaled, 
                fft_freqs, 
                target_frequency, 
                search_half_width_hz=search_half_width_hz
            )
        
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
        # Use common_audio_lib to select device
        selected_device_id = select_audio_device(console, require_input=True, require_output=True)
        if selected_device_id is None:
            error_console.print("No device selected. Exiting.")
            sys.exit(1)
    else:
        selected_device_id = args.device
        # Validate the user-provided device ID
        try:
            # list_available_devices(console) # Optionally list them first
            device_info_val = get_device_info(selected_device_id) # Use common lib
            if not device_info_val: # Should raise error in get_device_info if invalid
                error_console.print(f"Error: Could not retrieve info for device ID {selected_device_id}.")
                sys.exit(1)
            if device_info_val['max_output_channels'] == 0:
                error_console.print(f"Error: Device ID {selected_device_id} ({device_info_val['name']}) has no output channels.")
                sys.exit(1)
            if device_info_val['max_input_channels'] == 0:
                error_console.print(f"Error: Device ID {selected_device_id} ({device_info_val['name']}) has no input channels.")
                sys.exit(1)
            console.print(f"Using specified device ID: {selected_device_id} - {device_info_val['name']}")
        except ValueError as e: # Handles invalid device ID format or device not found from get_device_info
             error_console.print(f"Error validating device ID {args.device}: {e}")
             sys.exit(1)
        except Exception as e: 
            error_console.print(f"An unexpected error occurred validating device ID {args.device}: {e}")
            sys.exit(1)
            
    # Get full device info for the selected device
    current_device_info = get_device_info(selected_device_id) # Renamed
    if not current_device_info: # Should not happen if previous checks passed
        error_console.print(f"Critical Error: Failed to get device info for selected device {selected_device_id} after validation.")
        sys.exit(1)

    # --- Channel Conversion and Validation ---
    # Use common_audio_lib.audio_io_utils.resolve_channel_specifier
    output_channel_idx_0based = resolve_channel_specifier( # Renamed
        args.output_channel, 
        current_device_info['max_output_channels'], 
        "output", 
        error_console=error_console
    )
    if output_channel_idx_0based is None:
        sys.exit(1)

    if len(args.input_channels) < 2:
        error_console.print("Error: At least two input channels must be specified (one reference, one or more for crosstalk measurement against).")
        sys.exit(1)

    input_channel_indices_0based = [] # Renamed
    for spec in args.input_channels:
        idx = resolve_channel_specifier(
            spec, 
            current_device_info['max_input_channels'], 
            "input", 
            error_console=error_console
        )
        if idx is None:
            sys.exit(1)
        if idx in input_channel_indices_0based: 
            error_console.print(f"Error: Input channel '{spec}' (resolved to index {idx}) is effectively a duplicate of a previously specified input channel. All resolved input channel indices must be unique.")
            sys.exit(1)
        input_channel_indices_0based.append(idx)
    
    # Check if output channel is also one of the undriven input channels
    # input_channel_indices_0based[0] is the reference channel.
    # input_channel_indices_0based[1:] are the undriven channels.
    for i, undriven_idx_check in enumerate(input_channel_indices_0based[1:]): 
        if output_channel_idx_0based == undriven_idx_check:
            # Original args.input_channels[i+1] corresponds to this undriven_idx_check
            console.print(f"[yellow]Warning: Output channel ('{args.output_channel}' -> index {output_channel_idx_0based}) is the same as undriven input channel '{args.input_channels[i+1]}' (index {undriven_idx_check}). This setup might not measure crosstalk correctly unless intended for a specific test configuration.[/yellow]")


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
        
        # Use common_audio_lib.audio_io_utils.generate_log_spaced_frequencies for sweep
        try:
            frequencies_to_test = generate_log_spaced_frequencies(
                args.start_freq, args.end_freq, args.points_per_octave
            )
            if len(frequencies_to_test) == 0 : # Should not happen if start_freq <= end_freq
                 error_console.print(f"Error: No frequencies generated for sweep from {args.start_freq} to {args.end_freq} with PPO {args.points_per_octave}.")
                 sys.exit(1)
            if len(frequencies_to_test) > 500: # Arbitrary limit to prevent excessive runtimes
                error_console.print(f"Warning: Sweep generated {len(frequencies_to_test)} frequencies, which may take a long time. Consider adjusting range or PPO.")

        except ValueError as e:
            error_console.print(f"Error generating sweep frequencies: {e}")
            sys.exit(1)
    else: 
        if args.frequency <=0:
            error_console.print("Error: Single test frequency (--frequency) must be positive.")
            sys.exit(1)
        frequencies_to_test = np.array([args.frequency])

    console.print(f"\n[cyan]Device Configuration:[/cyan]")
    console.print(f"  Playback/Record Device: {current_device_info['name']} (ID: {selected_device_id}, Device SR: {current_device_info['default_samplerate']} Hz)")
    console.print(f"  Script Using Sample Rate: {args.sample_rate} Hz")
    console.print(f"  Test Signal Output Channel: '{args.output_channel}' (Device Index: {output_channel_idx_0based})")
    console.print(f"  Input Channels (Recorded): {args.input_channels} (Device Indices: {input_channel_indices_0based})")
    console.print(f"    Reference Input Channel (for driven signal level): '{args.input_channels[0]}' (Device Index: {input_channel_indices_0based[0]})")
    
    console.print(f"\n[cyan]Test Parameters:[/cyan]")
    console.print(f"  Mode: {'Sweep' if args.sweep else 'Single Frequency'}")
    console.print(f"  Frequencies: " + ", ".join([f"{f:.2f}" for f in frequencies_to_test]) + " Hz")
    console.print(f"  Signal Amplitude: {args.amplitude} dBFS")
    console.print(f"  Duration per Step: {args.duration_per_step} s")
    console.print(f"  FFT Window: {args.window}")

    all_results_data = [] 

    console.print(f"\n[green]Starting crosstalk analysis...[/green]")
    for freq_hz in frequencies_to_test:
        console.print(f"  Testing frequency: {freq_hz:.2f} Hz...")
        
        # Use common_audio_lib.audio_io_utils.generate_sine_wave
        # It takes amplitude_dbfs and phase in degrees (default 0)
        signal_to_play_mono = generate_sine_wave(
            frequency=freq_hz, 
            amplitude_dbfs=args.amplitude, 
            duration=args.duration_per_step, 
            sample_rate=args.sample_rate,
            # phase=0.0 # Default phase is 0.0 degrees
        )
        
        # Use common_audio_lib.audio_io_utils.play_and_record
        # It takes device_id (int|None), signal_to_play_mono, sample_rate, 
        # output_channel_device_idx_0based, input_channel_device_indices_0based_list,
        # record_duration_secs (optional), error_console (optional)
        recorded_data = play_and_record(
            device_id=selected_device_id,
            signal_to_play_mono=signal_to_play_mono,
            sample_rate=args.sample_rate,
            output_channel_device_idx_0based=output_channel_idx_0based,
            input_channel_device_indices_0based_list=input_channel_indices_0based,
            record_duration_secs=args.duration_per_step, # Match playback duration
            error_console=error_console
        )
        
        # Prepare a dictionary to store results for this frequency
        freq_result_row = {'freq_hz': freq_hz, 'ref_amp_dbfs': np.nan} 
        for i in range(1, len(input_channel_indices_0based)): # For undriven channels
            # Store original specifier and resolved index for clarity in results/CSV
            freq_result_row[f'undriven_ch_{i}_spec'] = args.input_channels[i] 
            freq_result_row[f'undriven_ch_{i}_idx'] = input_channel_indices_0based[i]
            freq_result_row[f'undriven_ch_{i}_amp_dbfs'] = np.nan
            freq_result_row[f'crosstalk_ch_{i}_db'] = np.nan

        if recorded_data is None: # play_and_record returns None on failure
            error_console.print(f"    Failed to play/record at {freq_hz:.2f} Hz. Results for this frequency will be NaN.")
            all_results_data.append(freq_result_row) 
            continue

        # Analyze recorded data using the new common-lib based function
        analysis_results_per_input_ch = analyze_recorded_channels_common_lib(
            recorded_data, 
            args.sample_rate, 
            freq_hz, 
            args.window
            # search_half_width_hz could be made configurable if needed
        )

        if not analysis_results_per_input_ch or len(analysis_results_per_input_ch) != len(input_channel_indices_0based):
            error_console.print(f"    Analysis failed or returned unexpected number of results at {freq_hz:.2f} Hz. Expected {len(input_channel_indices_0based)} results, got {len(analysis_results_per_input_ch) if analysis_results_per_input_ch else 'None'}. Results for this frequency will be NaN.")
            all_results_data.append(freq_result_row) 
            continue

        # Process results: first input channel is reference
        amp_driven_input_linear = analysis_results_per_input_ch[0]['amplitude_linear']
        # Use common_audio_lib.signal_processing_utils.linear_to_dbfs
        ref_amp_dbfs = linear_to_dbfs(amp_driven_input_linear, min_dbfs=-120.0) # Use a defined min_dbfs
        freq_result_row['ref_amp_dbfs'] = ref_amp_dbfs
        
        console.print(f"    Reference Channel ('{args.input_channels[0]}'): {ref_amp_dbfs:.2f} dBFS (Actual freq: {analysis_results_per_input_ch[0]['actual_freq']:.2f} Hz)")

        for i in range(1, len(input_channel_indices_0based)): # Iterate through undriven channels
            undriven_ch_spec = args.input_channels[i] # Original specifier for messages
            amp_undriven_input_linear = analysis_results_per_input_ch[i]['amplitude_linear']
            undriven_amp_dbfs = linear_to_dbfs(amp_undriven_input_linear, min_dbfs=-120.0)
            
            crosstalk_db = np.nan 
            # Crosstalk calculation logic remains similar
            if amp_driven_input_linear > 1e-12 : # Avoid division by zero or very small numbers
                if amp_undriven_input_linear > 1e-12:
                    crosstalk_db = 20 * math.log10(amp_undriven_input_linear / amp_driven_input_linear)
                else: # Undriven is effectively silent
                    crosstalk_db = -np.inf 
            elif amp_undriven_input_linear > 1e-12: # Driven is silent, but undriven has signal (unusual)
                 crosstalk_db = np.inf 
            # else: both are silent, crosstalk is NaN or undefined.
            
            console.print(f"    Undriven Channel ('{undriven_ch_spec}'): {undriven_amp_dbfs:.2f} dBFS (Actual freq: {analysis_results_per_input_ch[i]['actual_freq']:.2f} Hz). Crosstalk relative to '{args.input_channels[0]}': {crosstalk_db:.2f} dB")
            
            freq_result_row[f'undriven_ch_{i}_amp_dbfs'] = undriven_amp_dbfs
            freq_result_row[f'crosstalk_ch_{i}_db'] = crosstalk_db
        
        all_results_data.append(freq_result_row)

    # --- Display Results ---
    if not all_results_data:
        console.print("\n[yellow]No results to display (e.g., all frequencies failed or no frequencies tested).[/yellow]")
        sys.exit(0)
        
    console.print(f"\n[bold underline cyan]Crosstalk Analysis Results[/bold underline cyan]")
    results_table = Table(title=f"Crosstalk Measurement Summary (Output Ch: '{args.output_channel}')")
    results_table.add_column("Freq (Hz)", style="magenta", justify="right", min_width=10)
    results_table.add_column(f"Ref Ch ('{args.input_channels[0]}') Lvl (dBFS)", style="green", justify="right", min_width=22)

    for i in range(1, len(input_channel_indices_0based)): # Iterate through undriven channels
        undriven_ch_spec_header = args.input_channels[i] 
        results_table.add_column(f"Ch '{undriven_ch_spec_header}' Lvl (dBFS)", style="yellow", justify="right", min_width=20)
        results_table.add_column(f"Ch '{undriven_ch_spec_header}' Crosstalk (dB)", style="red", justify="right", min_width=22)

    for res_row in all_results_data:
        row_data_strings = [f"{res_row['freq_hz']:.2f}", f"{res_row['ref_amp_dbfs']:.2f}"]
        for i in range(1, len(input_channel_indices_0based)): # Iterate through undriven channels
            row_data_strings.append(f"{res_row.get(f'undriven_ch_{i}_amp_dbfs', np.nan):.2f}")
            row_data_strings.append(f"{res_row.get(f'crosstalk_ch_{i}_db', np.nan):.2f}")
        results_table.add_row(*row_data_strings)
    
    console.print(results_table)
    console.print(f"\n[green]Crosstalk analysis complete. Reference channel for crosstalk calculation is '{args.input_channels[0]}'. Negative crosstalk values indicate isolation (signal on undriven channel is lower than on reference channel).[/green]")

    # --- CSV Output ---
    if args.output_csv:
        # Prepare data for save_results_to_csv
        # The function expects a list of dictionaries, where each dictionary is a row.
        # The fieldnames should match the keys in these dictionaries.
        
        csv_fieldnames = ['Frequency_Hz', f'Ref_Ch_{args.input_channels[0]}_Lvl_dBFS']
        for i in range(1, len(input_channel_indices_0based)):
            undriven_ch_spec_header_safe = args.input_channels[i].replace("'", "") # Sanitize for key
            csv_fieldnames.extend([
                f'Ch_{undriven_ch_spec_header_safe}_Lvl_dBFS',
                f'Ch_{undriven_ch_spec_header_safe}_Crosstalk_dB'
            ])
        
        csv_data_rows = []
        for res_row in all_results_data:
            row_dict = {
                'Frequency_Hz': res_row['freq_hz'],
                f'Ref_Ch_{args.input_channels[0]}_Lvl_dBFS': res_row['ref_amp_dbfs']
            }
            for i in range(1, len(input_channel_indices_0based)):
                undriven_ch_spec_header_safe = args.input_channels[i].replace("'", "")
                row_dict[f'Ch_{undriven_ch_spec_header_safe}_Lvl_dBFS'] = res_row.get(f'undriven_ch_{i}_amp_dbfs', np.nan)
                row_dict[f'Ch_{undriven_ch_spec_header_safe}_Crosstalk_dB'] = res_row.get(f'crosstalk_ch_{i}_db', np.nan)
            csv_data_rows.append(row_dict)
            
        save_results_to_csv(args.output_csv, csv_data_rows, csv_fieldnames, console=console)
        # save_results_to_csv prints its own success/error message

    # --- Plotting ---
    can_plot = args.sweep and all_results_data and (args.output_plot or not args.no_plot_display)
    
    if can_plot:
        plot_frequencies = [r['freq_hz'] for r in all_results_data]
        plot_y_data_list = []
        plot_legend_labels = []
        
        for i in range(1, len(input_channel_indices_0based)): # For each undriven channel
            crosstalk_values = [r.get(f'crosstalk_ch_{i}_db', np.nan) for r in all_results_data]
            if not all(np.isnan(crosstalk_values)): # Only add if there's some valid data
                plot_y_data_list.append(crosstalk_values)
                plot_legend_labels.append(f"Crosstalk to Ch '{args.input_channels[i]}'")
            else:
                 console.print(f"[yellow]Skipping plot for Ch '{args.input_channels[i]}' as all crosstalk values are NaN.[/yellow]")

        if plot_y_data_list: # If there's anything to plot
            # Determine Y-axis limits for better visualization (optional, generate_plot might handle this)
            # For now, pass data as is. generate_plot has its own logic for scales.
            
            generate_plot(
                x_data=plot_frequencies,
                y_data_list=plot_y_data_list,
                legend_labels_list=plot_legend_labels,
                title=f"Audio Crosstalk Analysis (Output Ch: '{args.output_channel}')",
                x_label="Frequency (Hz)",
                y_label="Crosstalk (dB)",
                output_filename=args.output_plot,
                show_plot=(not args.no_plot_display),
                log_x_scale=True, # Crosstalk plots are typically log-x
                log_y_scale=False, # Y-axis (dB) is usually linear
                console=console
            )
        else:
            console.print("[yellow]No valid data to plot after filtering NaN-only series.[/yellow]")
            
    elif args.output_plot and not args.sweep:
        console.print(f"\n[yellow]Plotting is enabled via --output_plot but only supported for sweep mode. No plot generated.[/yellow]")
    elif args.output_plot and not all_results_data:
         console.print(f"\n[yellow]Plotting is enabled via --output_plot but there are no results to plot. No plot generated.[/yellow]")


if __name__ == '__main__':
    main()
