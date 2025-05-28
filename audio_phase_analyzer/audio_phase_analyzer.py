import numpy as np
# import sounddevice as sd # Replaced by common_audio_lib
import argparse
from scipy import signal as scisignal # Kept for scisignal.correlate
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
# import matplotlib.pyplot as plt # Replaced by common_audio_lib

# Common Audio Library Imports
from common_audio_lib.audio_device_manager import (
    list_available_devices, 
    select_audio_device, 
    get_device_info
)
from common_audio_lib.audio_io_utils import (
    generate_sine_wave, # Replaces local generate_sine_wave
    play_and_record,    # Replaces play_and_record_stereo
    resolve_channel_specifier
)
from common_audio_lib.output_formatting_utils import generate_plot


# Initialize Rich Console
console = Console()

# Default values
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_DURATION = 2.0 
DEFAULT_FREQUENCY = 1000.0
DEFAULT_AMPLITUDE_DBFS = -6.0
# DEFAULT_OUTPUT_CHANNELS = "1,2" # Original default, will be re-interpreted for single output
# For single output, let's default to the first channel '1' (0-indexed '0')
DEFAULT_OUTPUT_CHANNEL_SINGLE = "1" # For the new single channel output
DEFAULT_INPUT_CHANNELS = "1,2" # Still need two for phase comparison

# Local generate_sine_wave: Removed, using common_audio_lib.audio_io_utils.generate_sine_wave
# play_and_record_stereo: Removed, using common_audio_lib.audio_io_utils.play_and_record
# list_audio_devices: Removed, using common_audio_lib.audio_device_manager.list_available_devices
# parse_channel_mapping: Removed, logic incorporated into main using resolve_channel_specifier
# plot_lissajous: Removed, using common_audio_lib.output_formatting_utils.generate_plot


def calculate_phase_difference(recorded_audio, sample_rate, frequency):
    """
    Calculates the phase difference between two channels of an audio signal
    using cross-correlation.

    Args:
        recorded_audio (numpy.ndarray): A 2-channel numpy array.
        sample_rate (int): The sample rate of the audio.
        frequency (float): The frequency of the test tone.

    Returns:
        float or None: The phase difference in degrees (-180 to +180),
                       or None if calculation fails.
    """
    if recorded_audio is None or recorded_audio.ndim < 2 or recorded_audio.shape[0] == 0:
        console.print("[bold red]Error:[/bold red] No valid recorded audio data to process.")
        return None
    if recorded_audio.shape[1] < 2:
        console.print("[bold red]Error:[/bold red] Need at least two channels to compare phase.")
        return None

    ch1 = recorded_audio[:, 0]
    ch2 = recorded_audio[:, 1]

    # Ensure signals are not flat (zero standard deviation) or too short
    if np.std(ch1) < 1e-9 or np.std(ch2) < 1e-9:
        console.print("[yellow]Warning:[/yellow] One or both channels appear to be silent or DC. Phase calculation may be unreliable.")
        return 0.0 # Return 0.0 for silent signals after warning
    
    min_length_for_correlation = 10 # Arbitrary small number, might need adjustment based on frequency
    if len(ch1) < min_length_for_correlation or len(ch2) < min_length_for_correlation:
        console.print(f"[yellow]Warning:[/yellow] Signals are too short for reliable correlation (length {len(ch1)}).")
        return None

    # Cross-correlation
    try:
        # Using scipy.signal.correlate
        correlation = scisignal.correlate(ch1, ch2, mode='full', method='fft') # method='fft' is generally faster
        # Lags for 'full' mode
        lags = scisignal.correlation_lags(len(ch1), len(ch2), mode='full')
    except Exception as e:
        console.print(f"[bold red]Error during cross-correlation:[/bold red] {e}")
        return None
    
    if correlation is None or lags is None or len(correlation) == 0:
        console.print("[bold red]Error:[/bold red] Cross-correlation resulted in empty output.")
        return None

    # Find the lag corresponding to the maximum correlation
    try:
        delay_samples = lags[np.argmax(correlation)]
    except ValueError as e: # Handles cases like empty correlation array if not caught above
        console.print(f"[bold red]Error finding peak correlation:[/bold red] {e}. Signals might be problematic (e.g., all zeros).")
        return None

    # Convert delay to phase
    # phase_rad = (delay_samples / sample_rate) * frequency * 2 * np.pi
    # phase_deg = np.degrees(phase_rad)
    # The above calculates phase of ch1 relative to ch2.
    # If ch1 leads ch2 (e.g. ch1 = sin(wt), ch2 = sin(wt-phi)), delay_samples is positive.
    # This means ch1 must be delayed to match ch2, so ch1 is ahead.
    # A positive delay_samples means ch2 lags ch1. The phase of ch2 relative to ch1 is positive.
    # So, we want the phase of ch2 relative to ch1.
    # If delay_samples is positive, ch2 is lagging, so (phase_ch2 - phase_ch1) should be positive.
    # The formula (delay_samples / sample_rate) * frequency * 2 * np.pi gives:
    # if ch1 leads ch2, delay is positive, phase is positive. This is phase_ch1 - phase_ch2.
    # We want phase_ch2 - phase_ch1. So we should negate the delay or the resulting phase.
    # Let's use -delay_samples to represent the shift of ch2 relative to ch1.
    
    phase_rad = (-delay_samples / sample_rate) * frequency * 2 * np.pi
    phase_deg = np.degrees(phase_rad)

    # Normalize phase to +/- 180 degrees
    phase_deg_normalized = (phase_deg + 180) % 360 - 180
    
    # Sanity check for very large delays that might indicate issues
    # One period in samples:
    samples_per_period = sample_rate / frequency
    if abs(delay_samples) > samples_per_period * 2: # Arbitrary threshold, e.g., more than 2 periods off
        console.print(f"[yellow]Warning:[/yellow] Large delay detected ({delay_samples} samples). Phase result might be ambiguous or incorrect.")

    return phase_deg_normalized


def plot_lissajous(recorded_audio, sample_rate, frequency, input_ch_labels, target_duration_ms=100):
    """
    Plots a Lissajous figure for the first `target_duration_ms` of the recorded audio.

    Args:
        recorded_audio (numpy.ndarray): The 2-channel numpy array.
        sample_rate (int): The sample rate.
        frequency (float): The test tone frequency (for title).
        input_ch_labels (list[str]): Labels for the input channels (e.g., ['1', '2']).
        target_duration_ms (int): Duration of the audio segment to plot in milliseconds.
    """
    if recorded_audio is None or recorded_audio.ndim < 2 or recorded_audio.shape[1] < 2 or recorded_audio.shape[0] == 0:
        console.print("[bold red]Error:[/bold red] Not enough data or channels to plot Lissajous figure.")
        return

    ch1 = recorded_audio[:, 0]
    ch2 = recorded_audio[:, 1]

    num_samples = int(sample_rate * (target_duration_ms / 1000.0))
    
    # Ensure we don't try to plot more samples than available
    num_samples = min(num_samples, len(ch1), len(ch2))
    if num_samples < 2: # Need at least 2 points to plot a line
        console.print("[yellow]Warning:[/yellow] Not enough samples for a meaningful Lissajous plot after slicing.")
        return

    ch1_plot = ch1[:num_samples]
    ch2_plot = ch2[:num_samples]

    # Use common_audio_lib.output_formatting_utils.generate_plot
    # generate_plot takes x_data, y_data_list (list of y-datasets), legend_labels_list, title, etc.
    # For Lissajous, x_data is ch1_plot, y_data_list is [ch2_plot].
    
    plot_title = f"Lissajous Figure (Ch {input_ch_labels[0]} vs Ch {input_ch_labels[1]} @ {frequency:.0f} Hz)"
    
    # generate_plot might not have an 'axis equal' option directly.
    # This is a limitation if the common lib doesn't support aspect ratio control.
    # For now, we'll plot it as is.
    generate_plot(
        x_data=ch1_plot,
        y_data_list=[ch2_plot], # Y data as a list of datasets
        legend_labels_list=[f"Ch {input_ch_labels[1]}"], # Legend for the Y data
        title=plot_title,
        x_label=f"Amplitude - Input Channel {input_ch_labels[0]}",
        y_label=f"Amplitude - Input Channel {input_ch_labels[1]}",
        output_filename=None, # No file saving by default, only display
        show_plot=True,
        log_x_scale=False, # Lissajous are linear scale
        log_y_scale=False,
        console=console
        # Note: Aspect ratio 'equal' is not directly supported by generate_plot's current signature.
        # The plot might look stretched if this is not handled internally by generate_plot for X-Y type plots.
    )
    console.print(f"Lissajous plot for the first {target_duration_ms}ms displayed (if GUI available).")


# list_audio_devices: Removed, replaced by common_audio_lib.audio_device_manager.list_available_devices
# parse_channel_mapping: Removed, logic incorporated into main using resolve_channel_specifier
# get_device_id_from_arg: Removed, device selection handled by common_audio_lib


def main():
    # console is already initialized globally
    parser = argparse.ArgumentParser(
        description="Measures stereo phase characteristics using a test tone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help
    )
    parser.add_argument(
        '--list_devices', 
        action='store_true', 
        help='List available audio devices and their details, then exit.'
    )
    parser.add_argument(
        '-f', '--frequency', 
        type=float, 
        default=DEFAULT_FREQUENCY, 
        help='Test frequency in Hz.'
    )
    parser.add_argument(
        '-d', '--duration', 
        type=float, 
        default=DEFAULT_DURATION, 
        help='Duration of the test signal in seconds.'
    )
    parser.add_argument(
        '-sr', '--samplerate', 
        type=int, 
        default=DEFAULT_SAMPLE_RATE, 
        help='Sample rate in Hz.'
    )
    parser.add_argument(
        '-a', '--amplitude', 
        type=float, 
        default=DEFAULT_AMPLITUDE_DBFS, 
        help='Amplitude of the test tone in dBFS (0 dBFS = 1.0 linear).'
    )
    parser.add_argument(
        '--input_device', 
        type=str, # Can be int ID or string name
        default=None, 
        help='Input device ID (integer) or name (string). Uses system default if not specified.'
    )
    parser.add_argument(
        '--output_device', 
        type=str, # Can be int ID or string name
        default=None, 
        help='Output device ID (integer) or name (string). Uses system default if not specified.'
    )
    parser.add_argument(
        '--output_channels', 
        type=str, 
        default=DEFAULT_OUTPUT_CHANNEL_SINGLE, # Changed default
        help='Output channel for the test signal (e.g., "1" or "L"). common_audio_lib plays mono on one channel.'
    )
    parser.add_argument(
        '--input_channels', 
        type=str, 
        default=DEFAULT_INPUT_CHANNELS,
        help='Comma-separated physical input channels to record from (e.g., "1,2"). These are used for phase comparison.'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help="Display a Lissajous figure (X-Y plot) of the recorded stereo channels using matplotlib."
    )

    args = parser.parse_args()

    if args.list_devices:
        list_available_devices(console) # Use common lib
        exit(0)

    # --- Device Selection and Validation ---
    # Output device (single device for both play and record in common_audio_lib.play_and_record)
    # If different input/output devices are absolutely needed, play_and_record would need modification,
    # or separate sd.InputStream/sd.OutputStream would be required (more complex).
    # For now, assume one device for both, selected by user.
    if args.output_device and args.input_device and args.output_device != args.input_device:
        console.print("[yellow]Warning:[/yellow] Different input and output devices specified. "
                      "The common `play_and_record` function uses a single device for both. "
                      f"Using output device '{args.output_device}' for both operations.")
        # Or, could prompt user again here or exit. For now, prioritize output_device.
    
    device_specifier_for_selection = args.output_device if args.output_device else args.input_device # Prioritize output, then input, then None
    
    if device_specifier_for_selection is None:
        selected_device_id = select_audio_device(console, require_input=True, require_output=True)
        if selected_device_id is None:
            error_console.print("No device selected. Exiting.")
            sys.exit(1)
    else:
        try:
            # Try to parse as int first
            selected_device_id = int(device_specifier_for_selection)
        except ValueError:
            # If not int, assume it's a name to be resolved by get_device_info
            selected_device_id = device_specifier_for_selection 
        
        # Validate the specified device
        try:
            temp_device_info = get_device_info(selected_device_id) # Will raise error if invalid or not found
            if temp_device_info['max_output_channels'] == 0 or temp_device_info['max_input_channels'] == 0:
                 error_console.print(f"Error: Device '{temp_device_info['name']}' (ID/Spec: {selected_device_id}) must have both input and output channels.")
                 sys.exit(1)
            selected_device_id = temp_device_info['index'] # Ensure we use the integer ID hereafter
            console.print(f"Using specified device: {temp_device_info['name']} (ID: {selected_device_id})")
        except ValueError as e:
             error_console.print(f"Error validating device '{selected_device_id}': {e}")
             list_available_devices(console)
             sys.exit(1)

    current_device_info = get_device_info(selected_device_id) # Get full info

    # --- Channel Parsing and Validation ---
    # Output channel: common play_and_record takes a single 0-based output channel index.
    # The script originally took a pair "1,2" for output_channels, implying stereo output of the mono tone.
    # We will adapt to use only the *first* channel specified in args.output_channels.
    
    output_channels_str_list = args.output_channels.split(',')
    if not output_channels_str_list:
        error_console.print("Error: Output channels string is empty.")
        sys.exit(1)
    
    # Use the first specified output channel
    # Convert 1-based user input (e.g. "1") or "L"/"R" to 0-based index
    output_channel_idx_0based = resolve_channel_specifier(
        output_channels_str_list[0].strip(), 
        current_device_info['max_output_channels'], 
        "output", 
        error_console
    )
    if output_channel_idx_0based is None: sys.exit(1)
    console.print(f"[yellow]Note:[/yellow] Test signal will be played on a single output channel: '{output_channels_str_list[0].strip()}' (Device Index: {output_channel_idx_0based}).")

    # Input channels: need two for phase comparison. common play_and_record takes a list of 0-based input indices.
    input_channels_str_list = args.input_channels.split(',')
    if len(input_channels_str_list) != 2:
        error_console.print(f"Error: Exactly two input channels must be specified for phase comparison. Got: '{args.input_channels}'")
        sys.exit(1)
        
    input_channel_indices_0based = []
    input_channel_labels_for_plot = [] # Store original specifiers for plot labels
    for ch_spec_str in input_channels_str_list:
        ch_spec_str_stripped = ch_spec_str.strip()
        idx = resolve_channel_specifier(
            ch_spec_str_stripped, 
            current_device_info['max_input_channels'], 
            "input", 
            error_console
        )
        if idx is None: sys.exit(1)
        if idx in input_channel_indices_0based: # Check for duplicates
            error_console.print(f"Error: Duplicate input channel specified ('{ch_spec_str_stripped}' resolved to index {idx}).")
            sys.exit(1)
        input_channel_indices_0based.append(idx)
        input_channel_labels_for_plot.append(ch_spec_str_stripped)


    # Parameter Summary Table
    param_table = Table(title="[bold dodger_blue1]Measurement Parameters[/bold dodger_blue1]", show_header=True, header_style="bold magenta")
    param_table.add_column("Parameter", style="dim", width=20)
    param_table.add_column("Value")
    param_table.add_row("Test Frequency", f"{args.frequency:.1f} Hz")
    param_table.add_row("Duration", f"{args.duration:.2f} s")
    param_table.add_row("Sample Rate", f"{args.samplerate} Hz")
    param_table.add_row("Amplitude", f"{args.amplitude:.1f} dBFS")
    param_table.add_row("Audio Device", f"{current_device_info['name']} (ID: {selected_device_id})")
    param_table.add_row("Output Channel (0-based)", f"{output_channel_idx_0based} (Orig spec: '{output_channels_str_list[0].strip()}')")
    param_table.add_row("Input Channels (0-based)", f"{input_channel_indices_0based[0]} (Ref: '{input_channel_labels_for_plot[0]}'), {input_channel_indices_0based[1]} (DUT: '{input_channel_labels_for_plot[1]}')")
    console.print(param_table)
    console.print("-" * 30)

    # Device settings check is implicitly handled by get_device_info and resolve_channel_specifier
    # and will be further checked by play_and_record.

    console.print(f"Generating {args.duration}s mono tone at {args.frequency}Hz, Sample Rate: {args.samplerate}Hz...")
    # Use common_audio_lib.audio_io_utils.generate_sine_wave
    test_signal_mono = generate_sine_wave(args.frequency, args.amplitude, args.duration, args.samplerate)
    console.print(f"Signal generated. Shape: {test_signal_mono.shape}")

    # Use common_audio_lib.audio_io_utils.play_and_record
    recorded_data_multi_ch = play_and_record(
        device_id=selected_device_id,
        signal_to_play_mono=test_signal_mono,
        sample_rate=args.samplerate,
        output_channel_device_idx_0based=output_channel_idx_0based, # Single output channel
        input_channel_device_indices_0based_list=input_channel_indices_0based, # List of two input channels
        record_duration_secs=args.duration,
        error_console=error_console
    )
    
    if recorded_data_multi_ch is None or recorded_data_multi_ch.ndim != 2 or recorded_data_multi_ch.shape[1] != 2:
         console.print("[bold red]Recording failed, produced no/incomplete data, or did not return 2 channels. Cannot calculate phase.[/bold red]")
    else:
         console.print(f"Recorded data shape: {recorded_data_multi_ch.shape}")
         console.print("[green]Recording successful.[/green] Calculating phase difference...")
         # calculate_phase_difference expects a 2-channel numpy array [samples, 2]
         phase = calculate_phase_difference(recorded_data_multi_ch, args.samplerate, args.frequency)
         
         if phase is not None:
             phase_text_val = f"{phase:.2f}°"
             style = "cyan"
             if abs(phase) >= 170: style = "bold red"
             elif abs(phase) >= 90: style = "yellow"
             elif abs(phase) < 10: style = "bold green"

             phase_text_display = Text()
             phase_text_display.append("Calculated Phase Difference: ", style="default")
             phase_text_display.append(phase_text_val, style=style)
             
             console.print(Panel(
                 phase_text_display, 
                 title="[bold #2070b2]Phase Analysis Result[/bold #2070b2]",
                 subtitle=f"Input Ch '{input_channel_labels_for_plot[1]}' relative to Ch '{input_channel_labels_for_plot[0]}' @ {args.frequency:.1f} Hz",
                 expand=False,
                 border_style="dim #2070b2"
             ))
             
             if args.plot:
                 if recorded_data_multi_ch.shape[0] > 0 : # Check if there are samples
                     console.print("Preparing Lissajous figure...")
                     # Pass original channel specifiers for labels
                     plot_lissajous(recorded_data_multi_ch, args.samplerate, args.frequency, 
                                    input_channel_labels_for_plot)
                 else:
                     console.print("[yellow]Skipping plot:[/yellow] No valid recorded data available for plotting.")
         else:
             console.print("[bold red]Phase calculation failed or produced no result.[/bold red]")
             if args.plot:
                 console.print("[yellow]Skipping plot:[/yellow] Phase calculation failed, no data to plot reliably.")
    
    
    # Optional: Save recorded audio for inspection
    # try:
    #     import soundfile as sf # type: ignore
    #     sf.write('recorded_audio.wav', recorded_data, sr)
    #     print("Saved recorded audio to recorded_audio.wav (if soundfile is installed)")
    # except ImportError:
    #     pass # print("soundfile module not found, cannot save WAV. pip install soundfile")
    # except Exception as e:
    #     print(f"Error saving .wav: {e}")

if __name__ == '__main__':
    main()
