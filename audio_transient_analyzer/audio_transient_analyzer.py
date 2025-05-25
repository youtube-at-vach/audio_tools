import argparse
import numpy as np
import sounddevice as sd
from rich.console import Console # Removed ErrorConsole
from rich.table import Table
from rich.prompt import Prompt
import sys
import math
import numpy as np # Ensure numpy is imported, though it was already present
from scipy.signal import windows # For tukey window, and hanning if not using np.hanning
import csv # For CSV output
from typing import Optional, Union # Added for 3.8+ compatibility

def dbfs_to_linear(dbfs: float) -> float:
    """Converts a dBFS value to a linear amplitude value (0 dBFS = 1.0 linear)."""
    return 10**(dbfs / 20.0)

def generate_impulse(amplitude_dbfs: float, sample_rate: int) -> np.ndarray:
    """Generates a simple impulse signal."""
    amplitude_linear = dbfs_to_linear(amplitude_dbfs)
    # Create a short signal, e.g., 10 samples, with the impulse at the first sample.
    # impulse_signal = np.zeros(10)
    # A slightly longer impulse might be better for some sound cards, 1ms.
    num_samples = max(10, int(0.001 * sample_rate)) # Ensure at least 10 samples or 1ms worth
    impulse_signal = np.zeros(num_samples)
    impulse_signal[0] = amplitude_linear 
    # For a 3-sample pulse:
    # if num_samples >=3:
    #   impulse_signal[0:3] = amplitude_linear
    # else: # if num_samples is < 3, just make all samples the peak
    #   impulse_signal[:] = amplitude_linear
    return np.clip(impulse_signal, -1.0, 1.0)


def generate_tone_burst(frequency: float, amplitude_dbfs: float, cycles: int, envelope_type: str, sample_rate: int) -> np.ndarray:
    """Generates a tone burst signal."""
    amplitude_linear = dbfs_to_linear(amplitude_dbfs)
    num_samples_burst = int(cycles * sample_rate / frequency)
    if num_samples_burst == 0:
        # This can happen if cycles is too low or freq too high for the sample rate
        # Return a single sample impulse as a fallback or handle error appropriately
        return generate_impulse(amplitude_dbfs, sample_rate)


    t = np.linspace(0, cycles / frequency, num_samples_burst, endpoint=False)
    wave = amplitude_linear * np.sin(2 * np.pi * frequency * t)

    if envelope_type == 'hann':
        if num_samples_burst > 0: # Hanning window requires non-zero length
            envelope = np.hanning(num_samples_burst)
            wave *= envelope
    elif envelope_type == 'tukey': # Assuming scipy.signal.windows is available
        if num_samples_burst > 0: # Tukey window requires non-zero length
            envelope = windows.tukey(num_samples_burst, alpha=0.5) # alpha=0.5 is a common choice
            wave *= envelope
    # Rectangular is default (no change or multiply by ones)
    
    return np.clip(wave, -1.0, 1.0)

def play_and_record(
    signal_to_play: np.ndarray, 
    sample_rate: int, 
    play_device_idx: int, 
    output_channel_device_idx: int, 
    input_channel_device_idx: int, 
    record_duration_samples: int, 
    device_info_obj,
    console: Console,
    error_console: Console # Changed ErrorConsole to Console
) -> Optional[np.ndarray]: # Changed from "np.ndarray | None" to "Optional[np.ndarray]"
    """Plays a signal on a specific output channel and records from a specific input channel."""
    try:
        device_max_output_ch = device_info_obj['max_output_channels']
        
        # Create an output buffer for all device channels, padded to record_duration_samples
        output_buffer_padded = np.zeros((record_duration_samples, device_max_output_ch), dtype=signal_to_play.dtype)
        
        # Ensure signal_to_play is not longer than record_duration_samples
        play_len = min(len(signal_to_play), record_duration_samples)
        
        # Place the signal_to_play into the correct output channel
        output_buffer_padded[:play_len, output_channel_device_idx] = signal_to_play[:play_len]

        console.print(f"Playing signal on device {play_device_idx}, output channel index {output_channel_device_idx}, "
                      f"recording from input channel index {input_channel_device_idx}.")
        console.print(f"Output buffer shape: {output_buffer_padded.shape}, Max output channels: {device_max_output_ch}")
        console.print(f"Input mapping: {[input_channel_device_idx + 1]}")


        recorded_audio = sd.playrec(
            output_buffer_padded,
            samplerate=sample_rate,
            channels=output_buffer_padded.shape[1], # Number of output channels for the device
            input_mapping=[input_channel_device_idx + 1],  # 1-based index for input channel
            device=play_device_idx,
            blocking=True
        )
        
        # sd.playrec returns a 2D array (samples, channels). We want the first channel.
        if recorded_audio.ndim > 1 and recorded_audio.shape[1] > 0:
             return recorded_audio[:, 0]
        elif recorded_audio.ndim == 1: # Should not happen with input_mapping as list, but good to check
            return recorded_audio
        else: # Should not happen if recording was successful
            error_console.print(f"Unexpected shape for recorded audio: {recorded_audio.shape}")
            return None

    except sd.PortAudioError as e:
        error_console.print(f"[bold red]PortAudioError during playback/recording: {e}[/bold red]")
        error_console.print("This could be due to incorrect device ID, channel selection, sample rate, or other device-specific issues.")
        error_console.print(f"Device Info: {device_info_obj}")
        error_console.print(f"Selected SR: {sample_rate}, Output Ch Idx: {output_channel_device_idx}, Input Ch Idx: {input_channel_device_idx}")
        return None
    except Exception as e:
        error_console.print(f"[bold red]An unexpected error occurred in play_and_record: {e}[/bold red]")
        return None

# Analysis functions
def find_signal_start(audio_data: np.ndarray, sample_rate: int, threshold_factor: float = 0.05, pre_trigger_samples: int = 10) -> int:
    """
    Finds the first point that robustly exceeds a threshold and returns an index 
    slightly before it to capture the beginning of the rise.
    """
    if audio_data is None or len(audio_data) == 0:
        return 0
    
    max_abs_val = np.max(np.abs(audio_data))
    if max_abs_val == 0: # Handle silent audio
        return 0
        
    normalized_data = np.abs(audio_data / max_abs_val)
    
    threshold = threshold_factor # Threshold is applied on normalized data
    
    start_indices = np.where(normalized_data >= threshold)[0]
    
    if len(start_indices) == 0:
        return 0 # Signal never exceeds threshold
        
    first_exceed_point = start_indices[0]
    
    return max(0, first_exceed_point - pre_trigger_samples)

def calculate_rise_time(audio_data: np.ndarray, sample_rate: int, start_index: int) -> float:
    """Calculates 10% to 90% rise time of the signal segment starting from start_index."""
    if audio_data is None or len(audio_data) == 0 or start_index < 0 or start_index >= len(audio_data):
        return 0.0

    segment = audio_data[start_index:]
    if len(segment) == 0:
        return 0.0

    # Find the peak in the absolute value of the segment
    peak_value = np.max(np.abs(segment))
    if peak_value == 0: # Segment is all zeros
        return 0.0

    val_10 = 0.1 * peak_value
    val_90 = 0.9 * peak_value

    # Find first index where segment's absolute value crosses 10% of peak
    indices_above_10 = np.where(np.abs(segment) >= val_10)[0]
    if len(indices_above_10) == 0:
        return 0.0 # Signal doesn't reach 10% of its peak

    idx_10 = indices_above_10[0]

    # Find first index at or after idx_10 where segment's absolute value crosses 90% of peak
    # Search in the sub-segment starting from idx_10
    segment_from_idx_10 = segment[idx_10:]
    indices_above_90_in_sub_segment = np.where(np.abs(segment_from_idx_10) >= val_90)[0]
    
    if len(indices_above_90_in_sub_segment) == 0:
        return 0.0 # Signal doesn't reach 90% after reaching 10%

    idx_90_in_sub_segment = indices_above_90_in_sub_segment[0]
    idx_90 = idx_90_in_sub_segment + idx_10 # Map back to segment's indexing

    if idx_90 < idx_10: # Should ideally not happen with the logic above
        return 0.0 

    rise_time_samples = idx_90 - idx_10
    rise_time_seconds = rise_time_samples / sample_rate
    
    return rise_time_seconds

def calculate_overshoot(audio_data: np.ndarray, sample_rate: int, start_index: int) -> float:
    if audio_data is None or len(audio_data) == 0 or start_index < 0 or start_index >= len(audio_data):
        return 0.0

    segment = audio_data[start_index:]
    min_segment_len_ms = 20 # 20ms
    min_segment_samples = int(min_segment_len_ms / 1000 * sample_rate)

    if len(segment) < min_segment_samples:
        return 0.0

    abs_segment = np.abs(segment)
    peak_idx_in_segment = np.argmax(abs_segment)
    abs_peak_value = abs_segment[peak_idx_in_segment]

    if abs_peak_value == 0: # No signal
        return 0.0

    # Estimate steady-state value
    # Start looking for steady state 5ms after the peak
    post_peak_start_offset_ms = 5
    post_peak_start_offset_samples = int(post_peak_start_offset_ms / 1000 * sample_rate)
    
    steady_state_search_start_idx = peak_idx_in_segment + post_peak_start_offset_samples
    
    post_peak_segment = segment[steady_state_search_start_idx:]

    # Require at least 10ms of signal for steady state estimation
    min_post_peak_len_ms = 10
    min_post_peak_samples = int(min_post_peak_len_ms / 1000 * sample_rate)

    if len(post_peak_segment) < min_post_peak_samples:
        return 0.0 # Not enough data after peak for a reliable steady-state value

    # Heuristic: mean of abs values in the last half of post-peak segment
    steady_state_value = np.mean(np.abs(post_peak_segment[len(post_peak_segment)//2:]))

    if steady_state_value < 0.01 * abs_peak_value: # Likely decaying to zero
        return 0.0 

    if steady_state_value == 0: # Avoid division by zero
        return 0.0

    overshoot = ((abs_peak_value - steady_state_value) / steady_state_value) * 100
    return overshoot if overshoot > 0 else 0.0 # Overshoot should be positive

def calculate_settling_time(audio_data: np.ndarray, sample_rate: int, start_index: int, settle_percentage: float = 0.05) -> float:
    if audio_data is None or len(audio_data) == 0 or start_index < 0 or start_index >= len(audio_data):
        return 0.0

    segment = audio_data[start_index:] # This segment is not abs()
    if len(segment) == 0:
        return 0.0

    # Estimate final/steady-state value
    stable_part_min_len_ms = 10
    min_stable_samples = int(stable_part_min_len_ms / 1000 * sample_rate)
    
    last_quarter_start_idx = len(segment) * 3 // 4
    stable_part = segment[last_quarter_start_idx:]

    final_value_estimate_abs = 0.0
    if len(stable_part) >= min_stable_samples:
        final_value_estimate_abs = np.mean(np.abs(stable_part)) # Using abs for magnitude
    # else: final_value_estimate_abs remains 0.0, implies decay to zero if not enough data

    peak_abs_segment = np.max(np.abs(segment))
    if peak_abs_segment == 0: # No signal
        return 0.0

    tolerance = 0.0
    upper_bound = 0.0
    lower_bound = 0.0

    # Determine bounds based on whether signal is decaying to zero or settling to non-zero
    if final_value_estimate_abs < 0.1 * peak_abs_segment: # Likely decaying to zero
        tolerance = settle_percentage * peak_abs_segment
        upper_bound = tolerance
        lower_bound = -tolerance
    else: # Settling to a non-zero value (use the actual mean, not its abs, for center of band)
        # Re-calculate final_value for non-decaying case using the mean of the actual values (not abs)
        # to correctly center the band for bipolar signals.
        final_value_signed_estimate = np.mean(stable_part) if len(stable_part) >= min_stable_samples else 0.0
        tolerance = settle_percentage * final_value_estimate_abs # Tolerance magnitude based on abs mean
        upper_bound = final_value_signed_estimate + tolerance
        lower_bound = final_value_signed_estimate - tolerance
        
    # Start search for settling time from the peak of the segment
    peak_idx_segment = np.argmax(np.abs(segment))
    search_segment = segment[peak_idx_segment:] # Search from the peak onwards

    if len(search_segment) == 0:
        return 0.0

    settled_sample_relative_to_peak = -1

    for i in range(len(search_segment)):
        current_sample_value = search_segment[i]
        # Check if this sample is within bounds
        if lower_bound <= current_sample_value <= upper_bound:
            # If it is, assume settled from this point and verify the rest
            all_subsequent_settled = True
            for j in range(i, len(search_segment)):
                if not (lower_bound <= search_segment[j] <= upper_bound):
                    all_subsequent_settled = False
                    break 
            
            if all_subsequent_settled:
                settled_sample_relative_to_peak = i
                break 
        # If current_sample_value is not within bounds, continue search from next sample
        # (the all_subsequent_settled check will be skipped)

    if settled_sample_relative_to_peak != -1:
        settling_time_seconds = settled_sample_relative_to_peak / sample_rate
        return settling_time_seconds
    else:
        return 0.0 # Did not settle within the analyzed part of the segment

def select_device(console: Console, error_console: Console): # Changed ErrorConsole to Console
    """Lists available audio devices and allows the user to select one."""
    try:
        devices = sd.query_devices()
    except Exception as e:
        error_console.print(f"Error querying audio devices: {e}")
        sys.exit(1)

    input_devices = [dev for dev in devices if dev['max_input_channels'] > 0]
    output_devices = [dev for dev in devices if dev['max_output_channels'] > 0]

    if not input_devices or not output_devices:
        error_console.print("No suitable input/output devices found.")
        sys.exit(1)

    console.print("Available audio devices:")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=4)
    table.add_column("Name")
    table.add_column("API")
    table.add_column("In (Ch)", style="green")
    table.add_column("Out (Ch)", style="blue")
    table.add_column("Default SR (Hz)", style="yellow")

    for i, device in enumerate(devices):
        table.add_row(
            str(i),
            device['name'],
            device.get('hostapi_name', 'N/A'), # Changed to use .get()
            str(device['max_input_channels']),
            str(device['max_output_channels']),
            str(int(device['default_samplerate'])),
        )
    console.print(table)

    while True:
        try:
            device_id_str = Prompt.ask("Select device ID")
            device_id = int(device_id_str)
            if not (0 <= device_id < len(devices)):
                error_console.print(f"Invalid device ID. Please choose between 0 and {len(devices) - 1}.")
                continue
            selected_device = devices[device_id]
            if selected_device['max_input_channels'] == 0 or selected_device['max_output_channels'] == 0:
                error_console.print(f"Device ID {device_id} ('{selected_device['name']}') does not support both input and output.")
                continue
            return device_id, selected_device
        except ValueError:
            error_console.print("Invalid input. Please enter a numeric ID.")
        except Exception as e:
            error_console.print(f"An unexpected error occurred: {e}")
            sys.exit(1)

def channel_spec_to_index(channel_spec, device_channels, channel_type="output"):
    """Converts channel specifier ('L', 'R', or numeric string) to a 0-based integer index."""
    if isinstance(channel_spec, str):
        if channel_spec.upper() == 'L':
            return 0
        elif channel_spec.upper() == 'R':
            if device_channels < 2:
                raise ValueError(f"Device does not have a Right {channel_type} channel.")
            return 1
        elif channel_spec.isdigit():
            idx = int(channel_spec) - 1
            if not (0 <= idx < device_channels):
                raise ValueError(f"Invalid {channel_type} channel index '{channel_spec}'. Must be between 1 and {device_channels}.")
            return idx
        else:
            raise ValueError(f"Invalid {channel_type} channel specifier '{channel_spec}'. Use 'L', 'R', or a numeric index.")
    else: # Assuming it's already an int (e.g. from argparse default if not overridden)
        idx = channel_spec -1
        if not (0 <= idx < device_channels):
                raise ValueError(f"Invalid {channel_type} channel index '{channel_spec}'. Must be between 1 and {device_channels}.")
        return idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audio Transient Analyzer")
    parser.add_argument('--signal_type', choices=['impulse', 'tone_burst'], default='impulse', help='Type of transient signal to generate')
    parser.add_argument('--amplitude', type=float, default=-6.0, help='Amplitude of the test signal in dBFS. Must be <= 0')
    parser.add_argument('--device', type=int, help='Audio device ID. Prompts if not provided')
    parser.add_argument('--sample_rate', type=int, default=48000, help='Sampling rate in Hz')
    parser.add_argument('--output_channel', type=str, default='L', help="Output channel for test signal (e.g., 'L', 'R', or a 1-based numeric index like '1', '2', ...)")
    parser.add_argument('--input_channel', type=str, default='L', help="Input channel for recording (e.g., 'L', 'R', or a 1-based numeric index like '1', '2', ...)")
    parser.add_argument('--duration', type=float, default=0.1, help='Duration of the recording in seconds. Must be positive')
    parser.add_argument('--burst_freq', type=float, default=1000.0, help="Frequency of the tone burst in Hz (for signal_type 'tone_burst'). Must be positive")
    parser.add_argument('--burst_cycles', type=int, default=10, help="Number of cycles in the tone burst (for signal_type 'tone_burst'). Must be positive")
    # Updated choices to include tukey as per doc, assuming scipy will be handled.
    parser.add_argument('--burst_envelope', type=str, choices=['hann', 'rectangular', 'tukey'], default='hann', help="Envelope for the tone burst (for signal_type 'tone_burst')")
    parser.add_argument('--output_csv', type=str, help='Path to save results in CSV format')

    args = parser.parse_args()

    console = Console()
    error_console = Console(stderr=True, style="stderr") # Defined error_console

    if args.amplitude > 0:
        error_console.print("Error: Amplitude must be less than or equal to 0 dBFS.")
        sys.exit(1)
    if args.duration <= 0:
        error_console.print("Error: Duration must be positive.")
        sys.exit(1)
    if args.signal_type == 'tone_burst':
        if args.burst_freq <= 0:
            error_console.print("Error: Burst frequency must be positive for tone_burst signal type.")
            sys.exit(1)
        if args.burst_cycles <= 0:
            error_console.print("Error: Burst cycles must be positive for tone_burst signal type.")
            sys.exit(1)

    selected_device_id = args.device
    device_info = None

    if selected_device_id is None:
        selected_device_id, device_info = select_device(console, error_console)
    else:
        try:
            devices = sd.query_devices()
            if not (0 <= selected_device_id < len(devices)):
                error_console.print(f"Error: Device ID {selected_device_id} is invalid.")
                sys.exit(1)
            device_info = devices[selected_device_id]
            if device_info['max_input_channels'] == 0 or device_info['max_output_channels'] == 0:
                error_console.print(f"Error: Device '{device_info['name']}' (ID: {selected_device_id}) must support both input and output.")
                sys.exit(1)
        except Exception as e:
            error_console.print(f"Error accessing device ID {selected_device_id}: {e}")
            sys.exit(1)

    console.print(f"Using device: ID {selected_device_id} - {device_info['name']}")

    try:
        output_channel_idx = channel_spec_to_index(args.output_channel, device_info['max_output_channels'], "output")
        input_channel_idx = channel_spec_to_index(args.input_channel, device_info['max_input_channels'], "input")
    except ValueError as e:
        error_console.print(f"Error: {e}")
        sys.exit(1)

    console.print("\nSelected Parameters:")
    console.print(f"  Signal Type: {args.signal_type}")
    console.print(f"  Amplitude: {args.amplitude} dBFS")
    console.print(f"  Sample Rate: {args.sample_rate} Hz")
    console.print(f"  Output Channel: {args.output_channel} (Index: {output_channel_idx})")
    console.print(f"  Input Channel: {args.input_channel} (Index: {input_channel_idx})")
    console.print(f"  Duration: {args.duration} s")
    if args.signal_type == 'tone_burst':
        console.print(f"  Burst Frequency: {args.burst_freq} Hz")
        console.print(f"  Burst Cycles: {args.burst_cycles}")
        console.print(f"  Burst Envelope: {args.burst_envelope}")
    if args.output_csv:
        console.print(f"  Output CSV: {args.output_csv}")

    # Generate the test signal
    # The first block that was here has been removed.
    # This is the second, correct block, which was previously marked with:
    # test_signal = None # Renamed from signal_to_play to test_signal for clarity before play_and_record
    # That specific comment is now part of this combined comment.
    test_signal = None 
    if args.signal_type == 'impulse':
        test_signal = generate_impulse(args.amplitude, args.sample_rate)
        console.print(f"\nGenerated impulse signal of length {len(test_signal)} samples.")
    elif args.signal_type == 'tone_burst':
        test_signal = generate_tone_burst(
            frequency=args.burst_freq,
            amplitude_dbfs=args.amplitude,
            cycles=args.burst_cycles,
            envelope_type=args.burst_envelope,
            sample_rate=args.sample_rate
        )
        console.print(f"\nGenerated tone burst signal of length {len(test_signal)} samples.")
        if len(test_signal) == int(0.001 * args.sample_rate) and len(test_signal) <=10 : # check if it fell back to impulse
             console.print(f"[yellow]Warning: Tone burst generation might have defaulted to a short impulse due to parameters (freq/cycles/sr).[/yellow]")


    if test_signal is not None:
        console.print(f"Test signal peak amplitude: {np.max(np.abs(test_signal)):.4f} linear")

        record_duration_samples = int(args.duration * args.sample_rate)
        
        console.print(f"Preparing for playback and recording: Duration {args.duration}s ({record_duration_samples} samples)")

        recorded_audio = play_and_record(
            signal_to_play=test_signal,
            sample_rate=args.sample_rate,
            play_device_idx=selected_device_id, # This is the actual device ID
            output_channel_device_idx=output_channel_idx, # This is the 0-based index for the chosen channel on the device
            input_channel_device_idx=input_channel_idx,   # This is the 0-based index for the chosen channel on the device
            record_duration_samples=record_duration_samples,
            device_info_obj=device_info, # Full device info object
            console=console,
            error_console=error_console
        )

        if recorded_audio is not None:
            console.print(f"Playback and recording complete. Length of recorded audio: {len(recorded_audio)} samples.")
            
            # Analyze the recorded audio
            console.print("\nAnalyzing recorded audio...")
            
            # Use a pre-trigger of e.g. 50 samples, or make it configurable if needed
            # For now, using a fixed pre_trigger_samples for find_signal_start
            pre_trigger_samples_config = 50 # Could be an argparse parameter later
            actual_start_index = find_signal_start(
                recorded_audio, 
                sample_rate=args.sample_rate, 
                threshold_factor=0.05, # Using the previous threshold factor
                pre_trigger_samples=pre_trigger_samples_config
            )
            console.print(f"  Detected signal start (after pre-trigger adjustment): Index {actual_start_index}")

            rise_time = calculate_rise_time(recorded_audio, args.sample_rate, actual_start_index)
            console.print(f"  Calculated Rise Time (10%-90%): {rise_time:.6f} s")

            overshoot = calculate_overshoot(recorded_audio, args.sample_rate, actual_start_index)
            overshoot = calculate_overshoot(recorded_audio, args.sample_rate, actual_start_index)
            # console.print(f"  Calculated Overshoot: {overshoot:.2f} %") # Will be printed in table

            settling_time = calculate_settling_time(recorded_audio, args.sample_rate, actual_start_index, settle_percentage=0.05)
            # console.print(f"  Calculated Settling Time (to +/-5% from peak): {settling_time:.6f} s") # Will be printed in table
            
            peak_amplitude_recorded = 0.0
            if actual_start_index < len(recorded_audio):
                segment_for_peak = recorded_audio[actual_start_index:]
                if len(segment_for_peak) > 0:
                    peak_amplitude_recorded = np.max(np.abs(segment_for_peak))

            # --- Display results in a Rich Table ---
            results_table = Table(title="Transient Response Analysis Results")
            results_table.add_column("Parameter", style="cyan", no_wrap=True)
            results_table.add_column("Value", style="magenta")

            results_table.add_row("Signal Type", args.signal_type)
            if args.signal_type == 'tone_burst':
                results_table.add_row("Test Frequency (Hz)", f"{args.burst_freq:.1f}")
            results_table.add_row("Peak Amplitude (Recorded)", f"{peak_amplitude_recorded:.4f} linear")
            results_table.add_row("Rise Time (s)", f"{rise_time:.4f}")
            results_table.add_row("Overshoot (%)", f"{overshoot:.2f}")
            results_table.add_row("Settling Time (s)", f"{settling_time:.4f}")
            
            console.print("\n") # Add a newline before the table
            console.print(results_table)

            # --- Save results to CSV if requested ---
            if args.output_csv:
                csv_data = [
                    ['Parameter', 'Value'],
                    ['Signal Type', args.signal_type],
                ]
                if args.signal_type == 'tone_burst':
                    csv_data.append(['Test Frequency (Hz)', f"{args.burst_freq:.1f}"])
                csv_data.extend([
                    ['Peak Amplitude (Recorded)', f"{peak_amplitude_recorded:.4f}"],
                    ['Rise Time (s)', f"{rise_time:.4f}"],
                    ['Overshoot (%)', f"{overshoot:.2f}"],
                    ['Settling Time (s)', f"{settling_time:.4f}"],
                ])
                
                try:
                    with open(args.output_csv, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(csv_data)
                    console.print(f"\n[green]Results saved to {args.output_csv}[/green]")
                except IOError as e:
                    error_console.print(f"\n[bold red]Error saving CSV file {args.output_csv}: {e}[/bold red]")

        else:
            error_console.print("Failed to record audio. Skipping analysis.")
    else:
        error_console.print("Test signal generation failed. Skipping playback and recording.")

    # The old placeholder print is removed as we now have actual (though placeholder) analysis.
    # console.print("\nAnalysis (placeholder)...")
