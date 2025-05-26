import argparse
import argparse
import math
import time # Added import
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Initialize Rich Console
console = Console()


def list_audio_devices():
    """Lists available audio input and output devices using Rich."""
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
                str(device["default_samplerate"]),
            )
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error listing audio devices: {e}[/bold red]")


def generate_sine_wave(frequency, duration, amplitude, samplerate):
    """Generates a mono sine wave NumPy array."""
    t = np.linspace(0, duration, int(samplerate * duration), False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave.astype(np.float32)


def calculate_rms(audio_segment):
    """Calculates and returns the RMS (Root Mean Square) value of an audio segment."""
    if audio_segment is None or audio_segment.size == 0:
        console.print("[yellow]Warning: Audio segment is empty or None. RMS treated as 0.[/yellow]")
        return 0.0
    rms = np.sqrt(np.mean(audio_segment**2))
    if rms < 1e-9: # Effectively zero
        return 1e-9 # Return a very small number to avoid log(0) issues
    return rms


def measure_snr(
    output_device_id,
    input_device_id,
    output_channel,
    input_channel,
    samplerate,
    signal_freq,
    signal_amp,
    signal_duration,
    noise_duration,
    signal_file_path=None,
):
    """Measures the Signal-to-Noise Ratio (SNR)."""
    try:
        console.print(f"\n[cyan]Selected Output Device ID: {output_device_id}[/cyan]")
        console.print(f"[cyan]Selected Input Device ID: {input_device_id}[/cyan]")
        console.print(f"[cyan]Output Channel: {output_channel}, Input Channel: {input_channel}[/cyan]")
        console.print(f"[cyan]Samplerate: {samplerate} Hz[/cyan]")


        # Prepare Signal
        if signal_file_path:
            # File loading to be implemented in future if needed
            console.print("[yellow]Signal file path provided, but file loading is not yet implemented. Using generated sine wave.[/yellow]")
            signal = generate_sine_wave(signal_freq, signal_duration, signal_amp, samplerate)
        else:
            console.print(f"[cyan]Generating sine wave: {signal_freq} Hz, {signal_amp} amplitude, {signal_duration}s duration[/cyan]")
            signal = generate_sine_wave(signal_freq, signal_duration, signal_amp, samplerate)

        output_device_info = sd.query_devices(output_device_id)
        input_device_info = sd.query_devices(input_device_id)

        max_out_channels = output_device_info['max_output_channels']
        
        if output_channel > max_out_channels:
            console.print(f"[bold red]Error: Output channel {output_channel} exceeds max output channels ({max_out_channels}) for device {output_device_id}.[/bold red]")
            return None, None, None
        if input_channel > input_device_info['max_input_channels']:
            console.print(f"[bold red]Error: Input channel {input_channel} exceeds max input channels ({input_device_info['max_input_channels']}) for device {input_device_id}.[/bold red]")
            return None, None, None

        # Prepare the output signal for sd.playrec.
        # If output_mapping is used, 'data' must have len(output_mapping) columns.
        # Since 'signal' is mono (1D) and we map it to a single output channel,
        # it needs to be reshaped to a 2D array with one column.
        output_signal_for_playrec = signal.reshape(-1, 1)

        # Playback Signal and Record (Signal + Noise)
        console.print("\n[bold yellow]Prepare for signal playback and recording (Signal + Noise)...[/bold yellow]")
        console.print("Ensure your microphone is positioned to capture the output.")
        # Adding a small delay to allow user to prepare
        sd.sleep(2000) 

        console.print(f"Playing signal on output device {output_device_id} (channel {output_channel}) and recording from input device {input_device_id} (channel {input_channel})...")
        
        recorded_signal_plus_noise = sd.playrec(
            output_signal_for_playrec, # Use the (N,1) reshaped signal
            samplerate=samplerate,
            channels=1, # Number of input channels to record, len([input_channel]) = 1
            input_mapping=[input_channel], # 1-based physical input channel
            output_mapping=[output_channel], # 1-based physical output channel
            blocking=True # Wait for playback and recording to complete
        )
        sd.wait() # Ensure all audio has been processed
        sd.stop(ignore_errors=True)  # Stop any active streams before delay
        console.print("[green]Signal playback and recording complete.[/green]")

        time.sleep(0.2) # Delay for 200ms to allow device to be released

        # Record Noise Floor
        console.print(f"\n[bold yellow]Prepare for noise floor recording ({noise_duration}s). Ensure silence or typical noise conditions.[/bold yellow]")
        # Adding a small delay
        sd.sleep(2000)

        # Check input settings before attempting to record noise
        try:
            console.print(f"Verifying input settings for device {input_device_id} (channel {input_channel}) at {samplerate} Hz...")
            sd.check_input_settings(
                device=input_device_id,
                channels=1, # Check if the device supports recording at least 1 channel
                samplerate=samplerate
                # The 'mapping' argument is not valid for check_input_settings.
                # Channel-specific validation happens implicitly during sd.rec() if mapping is used there.
            )
            # console.print("[green]Input device settings appear valid for noise recording.[/green]") # Optional success message
        except sd.PortAudioError as e:
            # Note: The error message might be less specific now if channel mapping was the true cause of a previous error.
            # However, the primary check is for device + samplerate + basic channel count.
            console.print(f"[bold red]Error: Input device {input_device_id} does not support the required settings (samplerate: {samplerate}Hz, channels: 1) for noise recording.[/bold red]")
            console.print(f"[bold red]PortAudio Error details: {e}[/bold red]")
            console.print("Please check the device capabilities (e.g., using --list_devices) or try different settings.")
            raise # Re-raise to be caught by the main PortAudioError handler in measure_snr
        
        # Check input settings before attempting to record noise
        try:
            console.print(f"Verifying input settings for device {input_device_id} (channel {input_channel}) at {samplerate} Hz...")
            sd.check_input_settings(
                device=input_device_id,
                channels=1, # Check if the device supports recording at least 1 channel
                samplerate=samplerate
                # The 'mapping' argument is not valid for check_input_settings.
            )
        except sd.PortAudioError as e:
            console.print(f"[bold red]Error: Input device {input_device_id} does not support the required settings (samplerate: {samplerate}Hz, channels: 1) for noise recording.[/bold red]")
            console.print(f"[bold red]PortAudio Error details: {e}[/bold red]")
            console.print("Please check the device capabilities (e.g., using --list_devices) or try different settings.")
            raise 
        
        recorded_noise = None  # Initialize
        num_noise_frames = int(noise_duration * samplerate)

        try:
            console.print(f"Recording noise from input device {input_device_id} (channel {input_channel}) using 'with sd.InputStream(...)'...")
            
            with sd.InputStream(
                device=input_device_id,
                mapping=[input_channel],  # Use the 1-based channel from args
                channels=1,               # Number of channels defined by mapping
                samplerate=samplerate,
                dtype='float32'           # Explicitly set dtype
            ) as stream_noise:
                console.print(f"Recording {noise_duration}s of noise...")
                # stream.read() for sd.InputStream returns (data, overflow_status)
                temp_recorded_data, overflowed = stream_noise.read(num_noise_frames)
                if overflowed:
                    console.print("[yellow]Warning: Input overflow during noise recording.[/yellow]")
                recorded_noise = temp_recorded_data # Assign data for later RMS calc
                # No explicit stream_noise.start(), stop(), or close() needed with 'with' statement for blocking read.
            
            console.print("[green]Noise recording complete.[/green]")

        except sd.PortAudioError as e:
            # This is the existing specific handler for PortAudioErrors during noise recording
            console.print(f"[bold red]Error during noise recording with device ID {input_device_id} (channel {input_channel}):[/bold red]")
            console.print(f"[bold red]PortAudio Error: {e}[/bold red]")
            console.print("\n[bold yellow]Suggestions:[/bold yellow]")
            console.print("- Verify the input device ID (`--input_device`) using `--list_devices`.")
            console.print("- Ensure the input device is not currently in use by another application.")
            console.print("- Check your system's audio settings and confirm that other applications can record from this device.")
            console.print("- This could be a system-specific issue with ALSA/PortAudio or device permissions.")
            raise # Re-raise to be caught by the main PortAudioError handler in measure_snr
        
        except Exception as e: # Catch other potential errors during stream ops
            console.print(f"[bold red]An unexpected error occurred during noise recording stream operations: {e}[/bold red]")
            recorded_noise = None # Ensure recorded_noise is None before re-raising
            raise # Re-raise to be caught by the main generic Exception handler in measure_snr

        # Calculate RMS
        rms_signal_plus_noise = calculate_rms(recorded_signal_plus_noise)
        rms_noise = calculate_rms(recorded_noise)

        console.print(f"\n[blue]RMS of (Signal + Noise): {rms_signal_plus_noise:.6f}[/blue]")
        console.print(f"[blue]RMS of Noise: {rms_noise:.6f}[/blue]")

        # Calculate SNR
        # SNR = 20 * log10 (RMS_signal / RMS_noise)
        # RMS_signal^2 = RMS_signal+noise^2 - RMS_noise^2
        
        rms_signal_plus_noise_sq = rms_signal_plus_noise**2
        rms_noise_sq = rms_noise**2

        if rms_signal_plus_noise_sq < rms_noise_sq:
            console.print("[bold red]Warning: Measured RMS of (Signal+Noise) is less than RMS of Noise. This indicates an issue with the measurement (e.g., signal too weak, noise too high, or measurement error). SNR will be reported as 0 dB or negative.[/bold red]")
            rms_signal_only = 0.0 # Or handle as per specific requirement, e.g. very small number
        else:
            rms_signal_only = np.sqrt(rms_signal_plus_noise_sq - rms_noise_sq)

        console.print(f"[blue]Calculated RMS of Signal (estimated): {rms_signal_only:.6f}[/blue]")

        if rms_noise <= 1e-9: # Effectively zero noise (handles the 1e-9 floor from calculate_rms)
            console.print("[green]Noise level is extremely low. SNR is considered very high (Infinity).[/green]")
            snr_db = float('inf')
        elif rms_signal_only < 1e-9 : # Effectively zero signal after noise subtraction
             console.print("[yellow]Signal level after noise subtraction is effectively zero. SNR might be very low or negative.[/yellow]")
             snr_db = 20 * math.log10(1e-9 / rms_noise) # Avoid log10(0)
        else:
            snr_db = 20 * math.log10(rms_signal_only / rms_noise)
            
        return snr_db, rms_signal_only, rms_noise

    except sd.PortAudioError as e:
        console.print(f"[bold red]PortAudio Error: {e}[/bold red]")
        console.print("[bold yellow]This might be due to incorrect device IDs, channel configurations, or system audio issues (like libportaudio2 not being installed). Try running with --list_devices to check IDs.[/bold yellow]")
        return None, None, None
    except ValueError as e:
        console.print(f"[bold red]ValueError: {e}[/bold red]")
        console.print("[bold yellow]This could be due to invalid device IDs, channel numbers, or other parameters. Please check your inputs.[/bold yellow]")
        return None, None, None
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
        return None, None, None


def main():
    """Main function to parse arguments and run the SNR measurement."""
    parser = argparse.ArgumentParser(description="SNR (Signal-to-Noise Ratio) Analyzer Tool")
    parser.add_argument(
        "--list_devices", action="store_true", help="List available audio devices and exit."
    )
    parser.add_argument(
        "--output_device", type=int, help="ID of the output audio device."
    )
    parser.add_argument(
        "--input_device", type=int, help="ID of the input audio device."
    )
    parser.add_argument(
        "--output_channel", type=int, default=1, help="Output channel (1-based, default: 1)."
    )
    parser.add_argument(
        "--input_channel", type=int, default=1, help="Input channel (1-based, default: 1)."
    )
    parser.add_argument(
        "--samplerate", type=int, default=48000, help="Samplerate in Hz (default: 48000)."
    )
    parser.add_argument(
        "--frequency", type=float, default=1000.0, help="Frequency of the test sine wave in Hz (default: 1000.0)."
    )
    parser.add_argument(
        "--amplitude", type=float, default=0.8, help="Amplitude of the test sine wave (0.0-1.0, default: 0.8)."
    )
    parser.add_argument(
        "--signal_duration", type=float, default=5.0, help="Duration of signal playback/recording in seconds (default: 5.0)."
    )
    parser.add_argument(
        "--noise_duration", type=float, default=5.0, help="Duration of noise floor recording in seconds (default: 5.0)."
    )
    
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.output_device is None or args.input_device is None:
        console.print("[bold red]Error: --output_device and --input_device must be specified unless --list_devices is used.[/bold red]")
        parser.print_help()
        return
        
    # Validate device IDs before querying
    try:
        devices = sd.query_devices()
        if not (0 <= args.output_device < len(devices)):
            console.print(f"[bold red]Error: Invalid output_device ID: {args.output_device}. Use --list_devices to see valid IDs.[/bold red]")
            return
        if not (0 <= args.input_device < len(devices)):
            console.print(f"[bold red]Error: Invalid input_device ID: {args.input_device}. Use --list_devices to see valid IDs.[/bold red]")
            return
            
        # Check if selected devices have the required capabilities
        output_device_info = sd.query_devices(args.output_device)
        if output_device_info['max_output_channels'] == 0:
            console.print(f"[bold red]Error: Selected output device '{output_device_info['name']}' (ID: {args.output_device}) has no output channels.[/bold red]")
            return
        if args.output_channel > output_device_info['max_output_channels']:
             console.print(f"[bold red]Error: Specified output channel {args.output_channel} for device '{output_device_info['name']}' exceeds its maximum of {output_device_info['max_output_channels']} output channels.[/bold red]")
             return


        input_device_info = sd.query_devices(args.input_device)
        if input_device_info['max_input_channels'] == 0:
            console.print(f"[bold red]Error: Selected input device '{input_device_info['name']}' (ID: {args.input_device}) has no input channels.[/bold red]")
            return
        if args.input_channel > input_device_info['max_input_channels']:
            console.print(f"[bold red]Error: Specified input channel {args.input_channel} for device '{input_device_info['name']}' exceeds its maximum of {input_device_info['max_input_channels']} input channels.[/bold red]")
            return

    except Exception as e:
        console.print(f"[bold red]Error querying device capabilities: {e}. Ensure correct device IDs are provided or system audio is configured.[/bold red]")
        return


    snr_db, rms_signal, rms_noise = measure_snr(
        output_device_id=args.output_device,
        input_device_id=args.input_device,
        output_channel=args.output_channel,
        input_channel=args.input_channel,
        samplerate=args.samplerate,
        signal_freq=args.frequency,
        signal_amp=args.amplitude,
        signal_duration=args.signal_duration,
        noise_duration=args.noise_duration,
    )

    if snr_db is not None and rms_signal is not None and rms_noise is not None:
        results_table = Table(title="SNR Measurement Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="magenta")
        results_table.add_column("Unit", style="green")

        results_table.add_row("RMS of Signal (estimated)", f"{rms_signal:.6f}", "Linear")
        results_table.add_row("RMS of Noise", f"{rms_noise:.6f}", "Linear")
        if math.isinf(snr_db):
             results_table.add_row("Signal-to-Noise Ratio (SNR)", "Infinity", "dB")
        else:
             results_table.add_row("Signal-to-Noise Ratio (SNR)", f"{snr_db:.2f}", "dB")
        
        console.print("\n")
        console.print(results_table)

if __name__ == "__main__":
    main()
