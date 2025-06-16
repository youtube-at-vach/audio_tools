#!/usr/bin/env python3
import argparse
import math
import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.table import Table
import scipy.signal

# Initialize Rich Console
console = Console()

def map_lr_to_channel_index(lr_string: str) -> int:
    """Converts 'L' or 'R' string to 1-based channel index."""
    if lr_string.upper() == 'L':
        return 1
    elif lr_string.upper() == 'R':
        return 2
    else:
        # This case should ideally be prevented by argparse choices
        raise ValueError(f"Invalid channel string: {lr_string}. Expected 'L' or 'R'.")

def list_audio_devices():
    """Lists available audio input and output devices."""
    try:
        devices = sd.query_devices()
        table = Table(title="Available Audio Devices")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Max Input Channels", style="green")
        table.add_column("Max Output Channels", style="yellow")
        table.add_column("Default Samplerate", style="blue")

        for i, device in enumerate(devices):
            table.add_row(
                str(i),
                device["name"],
                str(device["max_input_channels"]),
                str(device["max_output_channels"]),
                str(device["default_samplerate"])
            )
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error listing audio devices: {e}[/bold red]")

def generate_sine_wave(frequency, duration, amplitude, samplerate):
    """
    Generates a mono sine wave NumPy array.

    Args:
        frequency (float): Frequency in Hz.
        duration (float): Duration in seconds.
        amplitude (float): Amplitude (0.0 to 1.0).
        samplerate (int): Samplerate in Hz.

    Returns:
        np.ndarray: The generated sine wave.
    """
    t = np.linspace(0, duration, int(samplerate * duration), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave.astype(np.float32)

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
        # Return a very small positive number (epsilon) instead of true zero
        # to prevent math.log10(0) errors in SNR calculations.
        return 1e-12
    rms = np.sqrt(np.mean(audio_segment**2))
    if rms < 1e-12: # Threshold for effective zero
        console.print("[yellow]Warning: RMS value is close to zero.[/yellow]")
        # Return epsilon for effectively zero RMS as well, to prevent log(0) issues.
        return 1e-12
    return rms

def measure_snr(device_id: int, output_channel_idx: int, input_channel_idx: int,
                samplerate: int, signal_freq: float, signal_amp: float,
                signal_duration: float, noise_duration: float):
    """
    Measures the Signal-to-Noise Ratio (SNR).

    Args:
        device_id (int): ID of the audio device for both input and output.
        output_channel_idx (int): Output channel index (1-based).
        input_channel_idx (int): Input channel index (1-based).
        samplerate (int): Samplerate in Hz.
        signal_freq (float): Frequency of the test sine wave in Hz.
        signal_amp (float): Amplitude of the test sine wave (0.0-1.0).
        signal_duration (float): Duration of signal playback/recording in seconds.
        noise_duration (float): Duration of noise floor recording in seconds.
        signal_file_path (str, optional): Path to an audio file for the signal. Defaults to None.

    Returns:
        tuple: (snr_db, rms_signal_only, rms_noise) or (None, None, None) if error.
    """
    try:
        # Validate device ID
        device_info = sd.query_devices(device_id)
    except (ValueError, sd.PortAudioError) as e:
        console.print(f"[bold red]Error: Invalid device ID {device_id}. {e}[/bold red]")
        return None, None, None

    # Prepare Signal
    signal = generate_sine_wave(signal_freq, signal_duration, signal_amp, samplerate)

    try:
        max_out_channels = device_info['max_output_channels']
        max_in_channels = device_info['max_input_channels']

        if not (0 < output_channel_idx <= max_out_channels):
            console.print(f"[bold red]Error: Output channel {output_channel_idx} is out of range for device {device_id} (1-{max_out_channels}).[/bold red]")
            return None, None, None
        if not (0 < input_channel_idx <= max_in_channels):
            console.print(f"[bold red]Error: Input channel {input_channel_idx} is out of range for device {device_id} (1-{max_in_channels}).[/bold red]")
            return None, None, None

        # Playback Signal and Record (Signal + Noise)
        console.print(f"\n[cyan]Preparing to play signal on '{device_info['name']}' (Output Channel {output_channel_idx}) and record from '{device_info['name']}' (Input Channel {input_channel_idx})...[/cyan]")
        console.print(f"[yellow]Ensure your audio loopback or measurement setup is ready.[/yellow]")
        
        # The 'signal' is a 1D NumPy array.
        # For sd.playrec, if output_mapping specifies a single channel,
        # the data should be a 2D array with shape (n_frames, 1).
        console.print("Playing signal and recording (signal + noise)...")
        recorded_signal_plus_noise = sd.playrec(
            signal.reshape(-1, 1), # Play the mono signal on the specified output_channel_idx
            samplerate=samplerate,
            device=device_id, 
            channels=1, # Record 1 channel specified by input_mapping
            input_mapping=[input_channel_idx], # 1-based
            output_mapping=[output_channel_idx], # 1-based, directs the single channel from data
            blocking=True
        )
        sd.wait() # Ensure playback and recording are finished
        console.print("[green]Signal playback and recording complete.[/green]")

        # Record Noise Floor
        console.print(f"\n[cyan]Preparing to record noise floor from '{device_info['name']}' (Input Channel {input_channel_idx})...[/cyan]")
        console.print("[yellow]Ensure the environment is silent for noise measurement.[/yellow]")

        console.print(f"Recording noise for {noise_duration} seconds...")
        recorded_noise = sd.rec(
            int(noise_duration * samplerate),
            samplerate=samplerate,
            device=device_id, 
            channels=1, # Record 1 channel specified by mapping
            mapping=[input_channel_idx], # 1-based
            blocking=True
        )
        sd.wait() # Ensure recording is finished
        console.print("[green]Noise recording complete.[/green]")

        # Calculate RMS
        if recorded_signal_plus_noise.ndim > 1 and recorded_signal_plus_noise.shape[1] == 1:
            recorded_signal_plus_noise = recorded_signal_plus_noise.squeeze()
        if recorded_noise.ndim > 1 and recorded_noise.shape[1] == 1:
            recorded_noise = recorded_noise.squeeze()

        # Check if the recorded signal+noise is unexpectedly low
        # This is a heuristic check. A very low RMS might indicate that the signal
        # generated by --amplitude was not properly recorded.
        # We compare against signal_amp / 100 as an arbitrary small fraction.
        # A more sophisticated check might consider expected loopback attenuation.
        # Check if signal_amp is not zero to avoid division by zero if it was set to 0.
        if signal_amp > 1e-9: # Avoid issues if signal_amp is virtually zero
            # Calculate RMS of the original generated signal for a rough reference
            # This doesn't account for system gain/attenuation, it's a sanity check.
            # The generated signal is 'signal', its RMS is signal_amp / sqrt(2) for a sine wave.
            expected_min_rms_recorded_signal = (signal_amp / np.sqrt(2)) / 100.0 # e.g. 40dB lower than expected direct signal RMS

            # Calculate RMS of recorded_signal_plus_noise before it's used further
            # This avoids calculating it twice if it's already calculated below
            # However, the current structure calculates it with calculate_rms(recorded_signal_plus_noise)
            # So, let's calculate it here for the check.
            temp_rms_signal_plus_noise = np.sqrt(np.mean(recorded_signal_plus_noise**2))

            if temp_rms_signal_plus_noise < expected_min_rms_recorded_signal and temp_rms_signal_plus_noise < 0.01 : # Added an absolute threshold too
                console.print(f"[bold yellow]Warning: The recorded 'signal + noise' level ({temp_rms_signal_plus_noise:.4e}) is very low "
                              f"compared to the generated signal amplitude ({signal_amp:.2f}).[/bold yellow]")
                console.print("[yellow]  Please check your audio device levels, connections, and loopback setup. "
                              "The signal might not be recorded correctly or is extremely attenuated.[/yellow]")

        rms_signal_plus_noise = calculate_rms(recorded_signal_plus_noise)
        rms_noise = calculate_rms(recorded_noise)

        if rms_noise < 1e-12 : # Effectively zero noise
            console.print("[yellow]Warning: RMS of noise is extremely low. SNR might be very high or infinite.[/yellow]")
            return float('inf'), rms_signal_plus_noise, rms_noise
        
        power_signal_plus_noise = rms_signal_plus_noise**2
        power_noise = rms_noise**2

        # Estimate signal power by subtracting noise power from (signal+noise) power.
        # This common method assumes that the signal and noise are uncorrelated,
        # and that the noise characteristics are consistent between the two measurements.
        if power_signal_plus_noise < power_noise:
            console.print("[bold red]Warning: Measured signal power is less than noise power. This indicates an issue with the measurement or high noise levels. SNR will be 0 dB or negative.[/bold red]")
            rms_signal_only_val = 1e-12 # effectively zero signal
        else:
            # Signal power = (Signal+Noise Power) - Noise Power
            # RMS_signal_only = sqrt(Signal Power)
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
    
    parser.add_argument("--device", type=int, required=False, # Will be made required if not list_devices
                        help="ID of the audio device for both input and output.")
    
    parser.add_argument("--output_channel", type=str, default='R', choices=['L', 'R'],
                        help="Output channel ('L' or 'R'). Default: 'R'.")
    parser.add_argument("--input_channel", type=str, default='L', choices=['L', 'R'],
                        help="Input channel ('L' or 'R'). Default: 'L'.")
    
    parser.add_argument("--samplerate", type=int, default=48000, help="Samplerate in Hz (default: 48000).")
    parser.add_argument("--frequency", type=float, default=1000.0, help="Frequency of the test sine wave in Hz (default: 1000.0).")
    parser.add_argument("--amplitude", type=float, default=0.8, help="Amplitude of the test sine wave (0.0-1.0, default: 0.8).")
    parser.add_argument("--signal_duration", type=float, default=5.0, help="Duration of signal playback/recording in seconds (default: 5.0).")
    parser.add_argument("--noise_duration", type=float, default=5.0, help="Duration of noise floor recording in seconds (default: 5.0).")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # If not listing devices, --device becomes required.
    if args.device is None:
        parser.error("argument --device is required when not listing devices.")
        
    try:
        output_channel_int = map_lr_to_channel_index(args.output_channel)
        input_channel_int = map_lr_to_channel_index(args.input_channel)
    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        parser.print_help()
        return

    # Check if system has libportaudio2, if not, instruct user.
    # This check is basic and might not cover all PortAudio installation issues.
    try:
        sd.check_hostapi()
    except Exception as e: 
        if "PortAudio library not found" in str(e) or "portaudio" in str(e).lower():
            console.print("[bold red]PortAudio library seems to be missing or not configured correctly.[/bold red]")
            console.print("Please ensure libportaudio2 is installed (e.g., `sudo apt-get install libportaudio2`) and detectable by sounddevice.")
            return

    snr_db, rms_signal, rms_noise = measure_snr(
        device_id=args.device,
        output_channel_idx=output_channel_int,
        input_channel_idx=input_channel_int,
        samplerate=args.samplerate,
        signal_freq=args.frequency,
        signal_amp=args.amplitude,
        signal_duration=args.signal_duration,
        noise_duration=args.noise_duration
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
