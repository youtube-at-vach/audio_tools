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
        return 1e-12 # Return a very small number to avoid log(0)
    rms = np.sqrt(np.mean(audio_segment**2))
    if rms < 1e-12: # Threshold for effective zero
        console.print("[yellow]Warning: RMS value is close to zero.[/yellow]")
        return 1e-12
    return rms

def measure_snr(output_device_id, input_device_id, output_channel, input_channel,
                samplerate, signal_freq, signal_amp, signal_duration, noise_duration,
                signal_file_path=None):
    """
    Measures the Signal-to-Noise Ratio (SNR).

    Args:
        output_device_id (int): ID of the output audio device.
        input_device_id (int): ID of the input audio device.
        output_channel (int): Output channel (1-based).
        input_channel (int): Input channel (1-based).
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
        # Validate device IDs
        sd.query_devices(output_device_id)
        sd.query_devices(input_device_id)
    except (ValueError, sd.PortAudioError) as e:
        console.print(f"[bold red]Error: Invalid device ID. {e}[/bold red]")
        return None, None, None

    # Prepare Signal
    if signal_file_path:
        # File loading to be implemented as a future enhancement
        console.print("[yellow]Warning: Signal file path provided, but file loading is not yet implemented. Using generated sine wave.[/yellow]")
        signal = generate_sine_wave(signal_freq, signal_duration, signal_amp, samplerate)
    else:
        signal = generate_sine_wave(signal_freq, signal_duration, signal_amp, samplerate)

    try:
        output_device_info = sd.query_devices(output_device_id)
        input_device_info = sd.query_devices(input_device_id)

        max_out_channels = output_device_info['max_output_channels']
        # max_in_channels = input_device_info['max_input_channels'] # Not directly used by playrec's channels with input_mapping

        if not (0 < output_channel <= max_out_channels):
            console.print(f"[bold red]Error: Output channel {output_channel} is out of range for device {output_device_id} (1-{max_out_channels}).[/bold red]")
            return None, None, None
        if not (0 < input_channel <= input_device_info['max_input_channels']):
            console.print(f"[bold red]Error: Input channel {input_channel} is out of range for device {input_device_id} (1-{input_device_info['max_input_channels']}).[/bold red]")
            return None, None, None

        output_channel_idx = output_channel - 1  # 0-based index
        input_channel_idx = input_channel - 1    # 0-based index

        # Playback Signal and Record (Signal + Noise)
        console.print(f"\n[cyan]Preparing to play signal on '{output_device_info['name']}' (Channel {output_channel}) and record from '{input_device_info['name']}' (Channel {input_channel})...[/cyan]")
        console.print(f"[yellow]Ensure your audio loopback or measurement setup is ready.[/yellow]")
        # User prompt before starting might be good here, e.g., input("Press Enter to start signal measurement...")

        output_buffer = np.zeros((len(signal), max_out_channels), dtype=signal.dtype)
        output_buffer[:, output_channel_idx] = signal
        
        console.print("Playing signal and recording (signal + noise)...")
        recorded_signal_plus_noise = sd.playrec(
            output_buffer,
            samplerate=samplerate,
            channels=1, # Number of channels to record, corresponds to len(input_mapping)
            input_mapping=[input_channel], # 1-based
            output_mapping=[output_channel], # 1-based
            blocking=True
        )
        sd.wait() # Ensure playback and recording are finished
        console.print("[green]Signal playback and recording complete.[/green]")

        # Record Noise Floor
        console.print(f"\n[cyan]Preparing to record noise floor from '{input_device_info['name']}' (Channel {input_channel})...[/cyan]")
        console.print("[yellow]Ensure the environment is silent for noise measurement.[/yellow]")
        # User prompt: input("Press Enter to start noise measurement...")

        console.print(f"Recording noise for {noise_duration} seconds...")
        recorded_noise = sd.rec(
            int(noise_duration * samplerate),
            samplerate=samplerate,
            channels=1, # Number of channels to record
            mapping=[input_channel], # 1-based
            blocking=True
        )
        sd.wait() # Ensure recording is finished
        console.print("[green]Noise recording complete.[/green]")

        # Calculate RMS
        # Squeeze to make it 1D if it's 2D with one channel
        if recorded_signal_plus_noise.ndim > 1 and recorded_signal_plus_noise.shape[1] == 1:
            recorded_signal_plus_noise = recorded_signal_plus_noise.squeeze()
        if recorded_noise.ndim > 1 and recorded_noise.shape[1] == 1:
            recorded_noise = recorded_noise.squeeze()

        rms_signal_plus_noise = calculate_rms(recorded_signal_plus_noise)
        rms_noise = calculate_rms(recorded_noise)

        if rms_noise < 1e-12 : # Effectively zero noise
            console.print("[yellow]Warning: RMS of noise is extremely low. SNR might be very high or infinite.[/yellow]")
            # Avoid division by zero; can return a very large number or handle as 'Infinity'
            return float('inf'), rms_signal_plus_noise, rms_noise

        # Calculate SNR
        # SNR = 20 * log10 (RMS_signal / RMS_noise)
        # RMS_signal^2 = RMS_signal+noise^2 - RMS_noise^2
        
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
        console.print(f"[bold red]PortAudio Error: {e}[/bold red]")
        console.print("[bold yellow]This might be due to incorrect device IDs, channel configurations, or system audio issues (like libportaudio2 missing).[/bold yellow]")
        return None, None, None
    except ValueError as e:
        console.print(f"[bold red]ValueError: {e}[/bold red]")
        console.print("[bold yellow]This could be due to invalid channel numbers for the selected device or other parameter issues.[/bold yellow]")
        return None, None, None
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="SNR (Signal-to-Noise Ratio) Analyzer Tool")
    parser.add_argument("--list_devices", action="store_true", help="List available audio devices and exit.")
    parser.add_argument("--output_device", type=int, help="ID of the output audio device.")
    parser.add_argument("--input_device", type=int, help="ID of the input audio device.")
    parser.add_argument("--output_channel", type=int, default=1, help="Output channel (1-based, default: 1).")
    parser.add_argument("--input_channel", type=int, default=1, help="Input channel (1-based, default: 1).")
    parser.add_argument("--samplerate", type=int, default=48000, help="Samplerate in Hz (default: 48000).")
    parser.add_argument("--frequency", type=float, default=1000.0, help="Frequency of the test sine wave in Hz (default: 1000.0).")
    parser.add_argument("--amplitude", type=float, default=0.8, help="Amplitude of the test sine wave (0.0-1.0, default: 0.8).")
    parser.add_argument("--signal_duration", type=float, default=5.0, help="Duration of signal playback/recording in seconds (default: 5.0).")
    parser.add_argument("--noise_duration", type=float, default=5.0, help="Duration of noise floor recording in seconds (default: 5.0).")
    # parser.add_argument("--signal_file", type=str, help="Path to an audio file to use as the signal (optional).") # Future enhancement

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.output_device is None or args.input_device is None:
        console.print("[bold red]Error: Both --output_device and --input_device must be specified if not listing devices.[/bold red]")
        parser.print_help()
        return
        
    # Check if system has libportaudio2, if not, install it
    try:
        sd.check_hostapi()
    except Exception as e: # Broad exception to catch if sounddevice itself is not working due to system deps
        if "PortAudio library not found" in str(e) or "portaudio" in str(e).lower():
            console.print("[bold yellow]PortAudio library seems to be missing. Attempting to install libportaudio2...[/bold yellow]")
            # This command might need sudo and user interaction, which is problematic here.
            # For a CLI tool, it's better to instruct the user.
            console.print("[bold red]Please install libportaudio2: `sudo apt-get install libportaudio2` (or equivalent for your system) and try again.[/bold red]")
            # In a controlled environment (like a Dockerfile or CI), you could run:
            # import subprocess
            # subprocess.run(["sudo", "apt-get", "update"], check=True)
            # subprocess.run(["sudo", "apt-get", "install", "-y", "libportaudio2"], check=True)
            # console.print("[green]Attempted to install libportaudio2. Please try running the script again.[/green]")
            return # Exit after instruction

    snr_db, rms_signal, rms_noise = measure_snr(
        output_device_id=args.output_device,
        input_device_id=args.input_device,
        output_channel=args.output_channel,
        input_channel=args.input_channel,
        samplerate=args.samplerate,
        signal_freq=args.frequency,
        signal_amp=args.amplitude,
        signal_duration=args.signal_duration,
        noise_duration=args.noise_duration
        # signal_file_path=args.signal_file # Future
    )

    if snr_db is not None:
        table = Table(title="SNR Measurement Results")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Unit", style="green")

        table.add_row("SNR", f"{snr_db:.2f}" if snr_db != float('inf') else "Infinite", "dB")
        table.add_row("RMS Signal (derived)", f"{rms_signal:.6f}", "Linear") # Assuming uncalibrated units
        table.add_row("RMS Noise", f"{rms_noise:.6f}", "Linear")

        console.print(table)

if __name__ == "__main__":
    main()
