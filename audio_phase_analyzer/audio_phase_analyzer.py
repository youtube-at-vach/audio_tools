import numpy as np
import sounddevice as sd
import argparse
from scipy import signal as scisignal
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
import matplotlib.pyplot as plt

# Initialize Rich Console
console = Console()

# Default values
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_DURATION = 2.0 # Changed default to 2.0 as per common use case
DEFAULT_FREQUENCY = 1000.0
DEFAULT_AMPLITUDE_DBFS = -6.0
DEFAULT_OUTPUT_CHANNELS = "1,2"
DEFAULT_INPUT_CHANNELS = "1,2"

def generate_sine_wave(frequency, duration, sample_rate, amplitude_dbfs=DEFAULT_AMPLITUDE_DBFS):
    """
    Generates a mono sine wave.

    Args:
        frequency (float): Frequency of the sine wave in Hz.
        duration (float): Duration of the sine wave in seconds.
        sample_rate (int): The sample rate in Hz.
        amplitude_dbfs (float): Amplitude in dBFS (decibels relative to full scale).
                               0 dBFS = 1.0 linear.

    Returns:
        tuple: (numpy.ndarray, int)
               A tuple containing the generated mono sine wave (numpy array)
               and the sample_rate.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    amplitude_linear = 10**(amplitude_dbfs / 20.0)
    wave = amplitude_linear * np.sin(frequency * t * 2 * np.pi)
    return wave, sample_rate

def play_and_record_stereo(signal, sample_rate, 
                           output_device=None, input_device=None, 
                           output_mapping_channels=None, input_mapping_channels=None):
    """
    Plays a mono signal on specified output channels and records from specified input channels.

    Args:
        signal (numpy.ndarray): The mono numpy array signal to play.
        sample_rate (int): The sample rate for playback and recording.
        output_device (int or str, optional): Output device ID or name.
        input_device (int or str, optional): Input device ID or name.
        output_mapping_channels (list[int], optional): Physical output channels. Defaults to [1, 2].
        input_mapping_channels (list[int], optional): Physical input channels. Defaults to [1, 2].

    Returns:
        numpy.ndarray: A 2-channel numpy array containing the recorded audio.
    """
    output_map = output_mapping_channels or [1, 2]
    input_map = input_mapping_channels or [1, 2]

    if len(output_map) != 2:
        raise ValueError("Output mapping must specify exactly two channels for stereo playback of the mono test tone.")
    if len(input_map) != 2:
        raise ValueError("Input mapping must specify exactly two channels for stereo recording.")

    # The mono signal will be duplicated to two output streams, mapped to output_map[0] and output_map[1]
    if signal.ndim == 1:
        output_signal_stereo = np.tile(signal.reshape(-1, 1), (1, 2))
    elif signal.ndim == 2 and signal.shape[1] == 1: # Mono signal in a 2D array
        output_signal_stereo = np.tile(signal, (1, 2))
    else:
        # This case should ideally not be hit if generate_sine_wave always returns mono
        raise ValueError("Input signal to play_and_record_stereo must be mono.")

    num_input_channels_to_record = len(input_map) # Should be 2

    console.print(f"Preparing to play mono signal on output channels {output_map} and record from input channels {input_map} "
                  f"for {output_signal_stereo.shape[0] / sample_rate:.2f} seconds...")
    
    try:
        recorded_audio = sd.playrec(output_signal_stereo, # Play the prepared stereo signal
                                   samplerate=sample_rate,
                                   channels=num_input_channels_to_record, # Number of channels to record
                                   input_mapping=input_map,
                                   output_mapping=output_map,
                                   device=(input_device, output_device),
                                   blocking=True)
        console.print("[green]Finished playing and recording.[/green]")
    except sd.PortAudioError as pae:
        console.print(f"[bold red]PortAudioError during playback and recording:[/bold red] {pae}")
        console.print("This often indicates an issue with device capabilities (e.g., sample rate, channel count) or device selection.")
        console.print("Please check your device settings and selected channels.")
        console.print("Available devices:")
        list_audio_devices() # list_audio_devices will use console.print
        num_frames = int(signal.shape[0])
        recorded_audio = np.zeros((num_frames, num_input_channels_to_record)) # Return empty data matching expected shape
    except Exception as e:
        console.print(f"[bold red]Error during playback and recording:[/bold red] {e}")
        console.print("Please ensure you have valid input and output devices selected.")
        console.print("Available devices:")
        list_audio_devices() # list_audio_devices will use console.print
        num_frames = int(signal.shape[0])
        recorded_audio = np.zeros((num_frames, num_input_channels_to_record))

    return recorded_audio


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

    plt.figure(figsize=(6, 6))
    plt.plot(ch1_plot, ch2_plot, color='dodgerblue') # Changed color for better visibility
    
    title = f"Lissajous Figure (Ch {input_ch_labels[0]} vs Ch {input_ch_labels[1]} @ {frequency:.0f} Hz)"
    plt.title(title)
    plt.xlabel(f"Amplitude - Input Channel {input_ch_labels[0]}")
    plt.ylabel(f"Amplitude - Input Channel {input_ch_labels[1]}")
    
    plt.axis('equal')  # Crucial for correct phase representation
    plt.grid(True, linestyle=':', alpha=0.7) # Added linestyle and alpha for subtlety
    
    # Add a small margin to prevent points from being exactly on the edge
    max_val = np.max(np.abs(np.concatenate((ch1_plot, ch2_plot)))) * 1.1 
    if max_val == 0: max_val = 1 # Avoid zero limits if signal is silent
    plt.xlim([-max_val, max_val])
    plt.ylim([-max_val, max_val])

    console.print(f"Displaying Lissajous plot for the first {target_duration_ms}ms of the recording...")
    try:
        plt.show()
    except Exception as e:
        console.print(f"[bold red]Error displaying plot:[/bold red] {e}")
        console.print("It's possible that your environment does not support GUI display for matplotlib (e.g., a headless server).")
        console.print("If you are running in such an environment, plotting may not work as expected.")


def list_audio_devices():
    """Prints a list of available audio devices using Rich Table."""
    devices = sd.query_devices()
    if not devices:
        console.print("[yellow]No audio devices found.[/yellow]")
        return

    table = Table(title="Available Audio Devices", expand=True)
    table.add_column("ID", style="dim", width=3)
    table.add_column("Name", style="cyan", min_width=20, ratio=1)
    table.add_column("API", style="green", min_width=10, ratio=0.5)
    table.add_column("Max In", justify="right", style="magenta")
    table.add_column("Max Out", justify="right", style="magenta")
    table.add_column("Def. SR (Hz)", justify="right", style="yellow")

    for i, device in enumerate(devices):
        default_sr_str = str(int(device.get('default_samplerate', 0)))
        table.add_row(
            str(i),
            device['name'],
            device['hostapi_name'],
            str(device['max_input_channels']),
            str(device['max_output_channels']),
            default_sr_str
        )
    console.print(table)

def parse_channel_mapping(channel_str, arg_name="channel mapping"):
    """
    Parses a comma-separated channel string (e.g., "1,2") into a list of integers.
    Sounddevice uses 1-based indexing.
    """
    if not channel_str: # Should be caught by argparse default if not provided
        raise ValueError(f"{arg_name} string is empty.")
    try:
        channels = [int(ch.strip()) for ch in channel_str.split(',')]
        if not channels: # e.g. if input was just ","
            raise ValueError("No channel numbers found.")
        if not all(c > 0 for c in channels):
            raise ValueError("Channel indices must be positive (1-based).")
        # For this application, we typically expect two channels for stereo.
        if len(channels) != 2:
            console.print(f"[yellow]Warning:[/yellow] {arg_name} '{channel_str}' resulted in {len(channels)} channel(s). "
                  "This application expects two channels for stereo phase comparison.")
            # Depending on strictness, could raise ValueError here.
            # For now, allow it but play_and_record_stereo might raise error if not 2 for output.
        return channels
    except ValueError as e:
        # Catch specific errors from int() conversion or our own raises
        console.print(f"[bold red]Error:[/bold red] Invalid {arg_name} string '{channel_str}'. Expected format like '1,2'. Details: {e}")
        exit(1) # Exit because this is a CLI argument parsing error
    except Exception as e: # Catch any other unexpected errors during parsing
        console.print(f"[bold red]Error:[/bold red] Unexpected problem parsing {arg_name} string '{channel_str}': {e}")
        exit(1)


def get_device_id_from_arg(device_arg, kind="input"):
    """
    Attempts to get a valid device ID from a CLI argument.
    The argument can be an integer ID or a string name.
    """
    if device_arg is None:
        return None # Use system default
    try:
        # Try to convert to int first. If device_arg is "0", "1", etc.
        return int(device_arg)
    except ValueError:
        # If not an int, assume it's a device name or part of a name.
        # Sounddevice handles this directly.
        return device_arg


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
        default=DEFAULT_OUTPUT_CHANNELS,
        help='Comma-separated physical output channels for the stereo signal (e.g., "1,2"). Maps the generated stereo pair to these device channels.'
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
        list_audio_devices()
        exit(0)

    # Parse channel mappings
    try:
        output_mapping = parse_channel_mapping(args.output_channels, arg_name="output_channels")
        input_mapping = parse_channel_mapping(args.input_channels, arg_name="input_channels")
        if len(output_mapping) != 2:
            console.print(f"[bold red]Error:[/bold red] --output_channels must specify exactly two channels. Got: {args.output_channels}")
            exit(1)
        if len(input_mapping) != 2:
            console.print(f"[bold red]Error:[/bold red] --input_channels must specify exactly two channels. Got: {args.input_channels}")
            exit(1)
    except ValueError as e: # Should be caught by parse_channel_mapping, but as a fallback
        console.print(f"[bold red]Exiting due to channel mapping error:[/bold red] {e}")
        exit(1)

    selected_input_device_arg = get_device_id_from_arg(args.input_device, "input")
    selected_output_device_arg = get_device_id_from_arg(args.output_device, "output")

    # Parameter Summary Table
    param_table = Table(title="[bold dodger_blue1]Measurement Parameters[/bold dodger_blue1]", show_header=True, header_style="bold magenta")
    param_table.add_column("Parameter", style="dim", width=20)
    param_table.add_column("Value")

    def get_device_name(device_arg_value, device_list):
        if device_arg_value is None:
            return "System Default"
        try:
            dev_id = int(device_arg_value)
            if 0 <= dev_id < len(device_list):
                return f"{device_list[dev_id]['name']} (ID: {dev_id})"
            return f"Unknown ID: {dev_id}"
        except ValueError: # It's a name string
            # Try to find the device by name to display more info if possible
            try:
                dev_info = sd.query_devices(device_arg_value)
                return f"{dev_info['name']} (Name: '{device_arg_value}')"
            except ValueError:
                return f"'{device_arg_value}' (Name not directly resolved, using as is)"
        except Exception:
             return str(device_arg_value) # Fallback

    all_devs_list = sd.query_devices() # Query once for names
    input_dev_name = get_device_name(selected_input_device_arg, all_devs_list)
    output_dev_name = get_device_name(selected_output_device_arg, all_devs_list)

    param_table.add_row("Test Frequency", f"{args.frequency:.1f} Hz")
    param_table.add_row("Duration", f"{args.duration:.2f} s")
    param_table.add_row("Sample Rate", f"{args.samplerate} Hz")
    param_table.add_row("Amplitude", f"{args.amplitude:.1f} dBFS")
    param_table.add_row("Output Device", output_dev_name)
    param_table.add_row("Input Device", input_dev_name)
    param_table.add_row("Output Channels (1-based)", str(output_mapping))
    param_table.add_row("Input Channels (1-based)", f"{input_mapping[0]} (Ref), {input_mapping[1]} (DUT)")
    
    console.print(param_table)
    console.print("-" * 30)

    # Verify devices if specified
    if selected_input_device_arg is not None:
        try:
            sd.check_input_settings(device=selected_input_device_arg, channels=max(input_mapping), samplerate=args.samplerate)
        except Exception as e:
            console.print(f"[bold red]Error with selected input device '{selected_input_device_arg}':[/bold red] {e}")
            list_audio_devices()
            exit(1)
    if selected_output_device_arg is not None:
        try:
            sd.check_output_settings(device=selected_output_device_arg, channels=max(output_mapping), samplerate=args.samplerate)
        except Exception as e:
            console.print(f"[bold red]Error with selected output device '{selected_output_device_arg}':[/bold red] {e}")
            list_audio_devices()
            exit(1)

    console.print(f"Generating {args.duration}s mono tone at {args.frequency}Hz, Sample Rate: {args.samplerate}Hz...")
    test_signal, sr = generate_sine_wave(args.frequency, args.duration, args.samplerate, args.amplitude)
    console.print(f"Signal generated. Shape: {test_signal.shape}")

    recorded_data = play_and_record_stereo(
        test_signal, 
        sr,
        output_device=selected_output_device_arg,
        input_device=selected_input_device_arg,
        output_mapping_channels=output_mapping,
        input_mapping_channels=input_mapping
    )
    
    console.print(f"Recorded data shape: {recorded_data.shape}")

    if recorded_data.size == 0 or recorded_data.shape[0] == 0 or recorded_data.shape[1] != len(input_mapping):
         console.print("[bold red]Recording seems to have failed, produced no/incomplete data, or channel count mismatch. Cannot calculate phase.[/bold red]")
    else:
         console.print("[green]Recording successful.[/green] Calculating phase difference...")
         phase = calculate_phase_difference(recorded_data, sr, args.frequency)
         
         if phase is not None:
             phase_text_val = f"{phase:.2f}°"
             style = "cyan"
             if abs(phase) >= 170: style = "bold red" # Very out of phase
             elif abs(phase) >= 90: style = "yellow" # Significantly out of phase
             elif abs(phase) < 10: style = "bold green" # In phase

             phase_text_display = Text()
             phase_text_display.append("Calculated Phase Difference: ", style="default")
             phase_text_display.append(phase_text_val, style=style)
             
             console.print(Panel(
                 phase_text_display, 
                 title="[bold #2070b2]Phase Analysis Result[/bold #2070b2]", # Using a hex color
                 subtitle=f"Input Ch {input_mapping[1]} relative to Ch {input_mapping[0]} @ {args.frequency:.1f} Hz",
                 expand=False,
                 border_style="dim #2070b2"
             ))
             
             # Call plotting function if requested
             if args.plot:
                 if recorded_data is not None and recorded_data.shape[0] > 0 and recorded_data.shape[1] == 2:
                     console.print("Preparing Lissajous figure...")
                     plot_lissajous(recorded_data, sr, args.frequency, 
                                    [str(input_mapping[0]), str(input_mapping[1])])
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
