#!/usr/bin/env python3
import argparse
import math
import numpy as np
# import sounddevice as sd # Replaced by common_audio_lib
from rich.console import Console
from rich.table import Table
# import scipy.signal # Not directly used, was likely for potential filters not implemented

# Common Audio Library Imports
from common_audio_lib.audio_device_manager import (
    list_available_devices as common_list_available_devices,
    select_audio_device,
    get_device_info
)
from common_audio_lib.audio_io_utils import (
    resolve_channel_specifier,
    generate_sine_wave as common_generate_sine_wave,
    play_and_record,
    record_audio # Newly added
)
# No specific signal processing utils like FFT are used here directly, only for generate_sine_wave's internals.

# Initialize Rich Console
console = Console()

# map_lr_to_channel_index: Removed, replaced by common_audio_lib.audio_io_utils.resolve_channel_specifier
# list_audio_devices: Removed, replaced by common_audio_lib.audio_device_manager.list_available_devices
# generate_sine_wave: Removed, replaced by common_audio_lib.audio_io_utils.generate_sine_wave


def calculate_rms(audio_segment):
    """
    Calculates the RMS (Root Mean Square) value of an audio segment.

    Args:
        audio_segment (np.ndarray): NumPy array of the audio segment.

    Returns:
        float: The RMS value.
    """
    if audio_segment is None or audio_segment.size == 0:
        console.print("[yellow]Warning: Audio segment is empty or None. RMS considered 0.[/yellow]")
        return 1e-12 # Return a very small number to avoid log(0)
    rms = np.sqrt(np.mean(audio_segment**2))
    if rms < 1e-12: # Threshold for effective zero
        console.print("[yellow]Warning: RMS value is close to zero.[/yellow]")
        return 1e-12
    return rms

def measure_snr(device_id: int, output_channel_idx: int, input_channel_idx: int,
                samplerate: int, signal_freq: float, signal_amp: float,
                signal_duration: float, noise_duration: float,
                signal_file_path=None):
    """
    Measures the Signal-to-Noise Ratio (SNR).
    Uses 0-based channel indices internally after resolving from user input.

    Args:
        device_id (int): ID of the audio device for both input and output.
        output_channel_idx_0based (int): 0-based output channel index.
        input_channel_idx_0based (int): 0-based input channel index.
        samplerate (int): Samplerate in Hz.
        signal_freq (float): Frequency of the test sine wave in Hz.
        signal_amp_dbfs (float): Amplitude of the test sine wave in dBFS.
        signal_duration (float): Duration of signal playback/recording in seconds.
        noise_duration (float): Duration of noise floor recording in seconds.
        signal_file_path (str, optional): Path to an audio file for the signal. Defaults to None.

    Returns:
        tuple: (snr_db, rms_signal_only, rms_noise) or (None, None, None) if error.
    """
    try:
        device_info = get_device_info(device_id) # Use common lib
        if not device_info: # Should be caught by get_device_info raising error
            console.print(f"[bold red]Error: Could not get info for device ID {device_id}.[/bold red]")
            return None, None, None
    except ValueError as e:
        console.print(f"[bold red]Error: Invalid device ID {device_id}. {e}[/bold red]")
        return None, None, None

    # Prepare Signal
    if signal_file_path:
        console.print("[yellow]Warning: Signal file path provided, but file loading is not yet implemented. Using generated sine wave.[/yellow]")
        # common_generate_sine_wave takes amplitude_dbfs
        signal_to_play = common_generate_sine_wave(signal_freq, signal_amp_dbfs, signal_duration, samplerate)
    else:
        signal_to_play = common_generate_sine_wave(signal_freq, signal_amp_dbfs, signal_duration, samplerate)

    if signal_to_play is None: # Should not happen if common_generate_sine_wave is robust
        console.print("[bold red]Error: Failed to generate test signal.[/bold red]")
        return None, None, None

    try:
        # Playback Signal and Record (Signal + Noise)
        console.print(f"\n[cyan]Preparing to play signal on '{device_info['name']}' (Output Channel Index {output_channel_idx_0based}) "
                      f"and record from '{device_info['name']}' (Input Channel Index {input_channel_idx_0based})...[/cyan]")
        console.print(f"[yellow]Ensure your audio loopback or measurement setup is ready.[/yellow]")
        
        console.print("Playing signal and recording (signal + noise)...")
        recorded_signal_plus_noise_multi_ch = play_and_record(
            device_id=device_id,
            signal_to_play_mono=signal_to_play,
            sample_rate=samplerate,
            output_channel_device_idx_0based=output_channel_idx_0based,
            input_channel_device_indices_0based_list=[input_channel_idx_0based], # List with one channel
            record_duration_secs=signal_duration,
            error_console=console
        )
        
        if recorded_signal_plus_noise_multi_ch is None:
            console.print("[bold red]Signal+Noise recording failed.[/bold red]")
            return None, None, None
        # Select the single recorded channel
        recorded_signal_plus_noise = recorded_signal_plus_noise_multi_ch[:, 0]
        console.print("[green]Signal playback and recording complete.[/green]")

        # Record Noise Floor using new common_audio_lib.record_audio
        console.print(f"\n[cyan]Preparing to record noise floor from '{device_info['name']}' (Input Channel Index {input_channel_idx_0based})...[/cyan]")
        console.print("[yellow]Ensure the environment is silent for noise measurement.[/yellow]")

        console.print(f"Recording noise for {noise_duration} seconds...")
        recorded_noise_multi_ch = record_audio(
            duration_secs=noise_duration,
            sample_rate=samplerate,
            device_id=device_id,
            input_channel_device_indices_0based_list=[input_channel_idx_0based], # List with one channel
            console_instance=console
        )
        
        if recorded_noise_multi_ch is None:
            console.print("[bold red]Noise recording failed.[/bold red]")
            return None, None, None
        # Select the single recorded channel
        recorded_noise = recorded_noise_multi_ch[:, 0]
        console.print("[green]Noise recording complete.[/green]")

        # Calculate RMS (local function calculate_rms)
        if recorded_signal_plus_noise.ndim > 1 and recorded_signal_plus_noise.shape[1] == 1:
            recorded_signal_plus_noise = recorded_signal_plus_noise.squeeze()
        if recorded_noise.ndim > 1 and recorded_noise.shape[1] == 1:
            recorded_noise = recorded_noise.squeeze()

        rms_signal_plus_noise = calculate_rms(recorded_signal_plus_noise)
        rms_noise = calculate_rms(recorded_noise)

        if rms_noise < 1e-12 : # Effectively zero noise
            console.print("[yellow]Warning: RMS of noise is extremely low. SNR might be very high or infinite.[/yellow]")
            return float('inf'), rms_signal_plus_noise, rms_noise
        
        power_signal_plus_noise = rms_signal_plus_noise**2
        power_noise = rms_noise**2

        if power_signal_plus_noise < power_noise:
            console.print("[bold red]Warning: Measured signal power is less than noise power. This indicates an issue with the measurement or high noise levels. SNR will be 0 dB or negative.[/bold red]")
            rms_signal_only_val = 1e-12 # effectively zero signal
        else:
            rms_signal_only_val = np.sqrt(power_signal_plus_noise - power_noise)
        
        if rms_signal_only_val < 1e-12: # effectively zero signal
             console.print("[yellow]Warning: Calculated signal RMS is very low or zero. SNR will be very low or negative.[/yellow]")
             snr_db = 20 * math.log10(1e-12 / rms_noise) # Avoid log(0)
        else:
             snr_db = 20 * math.log10(rms_signal_only_val / rms_noise)

        return snr_db, rms_signal_only_val, rms_noise

    except sd.PortAudioError as e:
        console.print(f"[bold red]PortAudio Error on device {device_id}: {e}[/bold red]")
        console.print("[bold yellow]This might be due to incorrect device ID, channel configurations, or system audio issues.[/bold yellow]")
        return None, None, None
    except ValueError as e:
        console.print(f"[bold red]ValueError: {e}[/bold red]")
        console.print("[bold yellow]This could be due to invalid channel numbers for the selected device, data shape issues, or other parameter problems.[/bold yellow]")
        return None, None, None
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="SNR (Signal-to-Noise Ratio) Analyzer Tool")
    parser.add_argument("--list_devices", action="store_true", help="List available audio devices and exit.")
    
    parser.add_argument("--device", type=str, required=False, # Changed to str for ID or name
                        help="ID or name of the audio device for both input and output.")
    
    parser.add_argument("--output_channel", type=str, default='L', # Default L for 0-based
                        help="Output channel (e.g., 'L', 'R', or 0-based index '0', '1'). Default: 'L'.")
    parser.add_argument("--input_channel", type=str, default='L', # Default L for 0-based
                        help="Input channel (e.g., 'L', 'R', or 0-based index '0', '1'). Default: 'L'.")
    
    parser.add_argument("--samplerate", type=int, default=48000, help="Samplerate in Hz (default: 48000).")
    parser.add_argument("--frequency", type=float, default=1000.0, help="Frequency of the test sine wave in Hz (default: 1000.0).")
    parser.add_argument("--amplitude_dbfs", type=float, default=-6.0, help="Amplitude of the test sine wave in dBFS (default: -6.0).")
    parser.add_argument("--signal_duration", type=float, default=5.0, help="Duration of signal playback/recording in seconds (default: 5.0).")
    parser.add_argument("--noise_duration", type=float, default=5.0, help="Duration of noise floor recording in seconds (default: 5.0).")
    # parser.add_argument("--signal_file", type=str, help="Path to an audio file to use as the signal (optional).")

    args = parser.parse_args()

    if args.list_devices:
        common_list_available_devices(console) # Use common lib
        return

    # If not listing devices, --device becomes required (or prompt).
    selected_device_id = None
    if args.device is None:
        console.print("No device specified. Please select a device:")
        selected_device_id = select_audio_device(console, require_input=True, require_output=True)
        if selected_device_id is None:
            console.print("[bold red]No device selected. Exiting.[/bold red]")
            return
    else:
        try: # Validate user-provided device
            selected_device_id_arg = int(args.device) if args.device.isdigit() else args.device
            device_info_val = get_device_info(selected_device_id_arg)
            if not device_info_val: # Should be caught by get_device_info
                raise ValueError("Device not found or info retrieval failed.")
            selected_device_id = device_info_val['index'] # Use integer ID
            console.print(f"Using specified device: {device_info_val['name']} (ID: {selected_device_id})")
        except ValueError as e:
            console.print(f"[bold red]Error with specified device '{args.device}': {e}[/bold red]")
            common_list_available_devices(console)
            return
            
    current_device_info = get_device_info(selected_device_id) # Get full info for selected device
    if not current_device_info:
         console.print(f"[bold red]Failed to get info for device ID {selected_device_id}. Exiting.[/bold red]")
         return

    # Resolve channel specifiers to 0-based indices
    output_channel_0based = resolve_channel_specifier(
        args.output_channel, current_device_info['max_output_channels'], "output", console
    )
    if output_channel_0based is None: return

    input_channel_0based = resolve_channel_specifier(
        args.input_channel, current_device_info['max_input_channels'], "input", console
    )
    if input_channel_0based is None: return

    # PortAudio check is implicitly handled by sounddevice/common lib calls.

    snr_db, rms_signal, rms_noise = measure_snr(
        device_id=selected_device_id,
        output_channel_idx_0based=output_channel_0based,
        input_channel_idx_0based=input_channel_0based,
        samplerate=args.samplerate,
        signal_freq=args.frequency,
        signal_amp_dbfs=args.amplitude_dbfs, # Pass dBFS amplitude
        signal_duration=args.signal_duration,
        noise_duration=args.noise_duration
        # signal_file_path=args.signal_file
    )

    if snr_db is not None:
        table = Table(title="SNR Measurement Results")
        # Using the column headers from the updated README.md
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        # table.add_column("Unit", style="green") # Removed as per new README format

        table.add_row("SNR", f"{snr_db:.2f} dB" if snr_db != float('inf') else "Infinite dB")
        # Example Vrms, actual unit depends on calibration. The README suggests this format.
        table.add_row("RMS Signal (Estimated)", f"{rms_signal:.6f} Vrms") 
        table.add_row("RMS Noise", f"{rms_noise:.6f} Vrms")

        console.print(table)

if __name__ == "__main__":
    main()
