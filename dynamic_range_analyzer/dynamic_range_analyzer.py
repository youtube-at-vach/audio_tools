#!/usr/bin/env python3
import argparse
import numpy as np
import sounddevice as sd
import scipy.signal as sig
from scipy.signal import get_window
from rich.console import Console
from rich.table import Table

# Initialize Rich Console
console = Console()

# --- AES17-2015 Standard Constants ---
# Test signal frequency (recommended: 997 Hz)
TEST_FREQUENCY = 997.0
# Test signal level (recommended: -60.0 dBFS)
TEST_LEVEL_DBFS = -60.0
# Corresponding linear amplitude
TEST_AMPLITUDE = 10**(TEST_LEVEL_DBFS / 20.0)

def list_audio_devices():
    """Lists available audio input and output devices."""
    try:
        devices = sd.query_devices()
        table = Table(title="Available Audio Devices")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Max Input Ch", style="green")
        table.add_column("Max Output Ch", style="yellow")
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

def measure_unweighted_dynamic_range(device_id, output_channel, input_channel, samplerate, duration):
    """
    Measures Unweighted Dynamic Range.
    """
    try:
        device_info = sd.query_devices(device_id)
    except (ValueError, sd.PortAudioError) as e:
        console.print(f"[bold red]Error: Invalid device ID {device_id}. {e}[/bold red]")
        return None, None

    # --- Validate Channels ---
    if not (0 < output_channel <= device_info['max_output_channels']):
        console.print(f"[bold red]Error: Output channel {output_channel} is out of range for device {device_id} (1-{device_info['max_output_channels']}).[/bold red]")
        return None, None
    if not (0 < input_channel <= device_info['max_input_channels']):
        console.print(f"[bold red]Error: Input channel {input_channel} is out of range for device {device_id} (1-{device_info['max_input_channels']}).[/bold red]")
        return None, None

    # --- Generate Test Signal ---
    num_samples = int(samplerate * duration)
    t = np.linspace(0, duration, num_samples, False)
    test_signal = TEST_AMPLITUDE * np.sin(2 * np.pi * TEST_FREQUENCY * t)
    test_signal = test_signal.astype(np.float32)

    # --- Play and Record ---
    console.print(f"\n[cyan]Playing -60 dBFS test tone on '{device_info['name']}' (Output Ch: {output_channel}) and recording (Input Ch: {input_channel})...[/cyan]")
    console.print("[yellow]Ensure your audio loopback or measurement setup is ready.[/yellow]")
    
    recorded_data = sd.playrec(
        test_signal.reshape(-1, 1),
        samplerate=samplerate,
        device=device_id,
        channels=1,
        input_mapping=[input_channel],
        output_mapping=[output_channel],
        blocking=True
    )
    sd.wait()
    console.print("[green]Recording complete.[/green]")

    recorded_data = recorded_data.squeeze()

    # --- Analysis ---
    console.print("[cyan]Analyzing recorded data...[/cyan]")

    # 1. Remove the test tone using FFT filtering.
    n_samples = len(recorded_data)

    # Apply a window function to the recorded data to reduce spectral leakage
    window = get_window('hann', n_samples)
    windowed_data = recorded_data * window

    # Compute the FFT
    fft_data = np.fft.rfft(windowed_data)
    fft_freq = np.fft.rfftfreq(n_samples, 1 / samplerate)

    # Find the bin corresponding to the test frequency
    target_bin = np.argmin(np.abs(fft_freq - TEST_FREQUENCY))

    # Zero out the target bin and a few bins around it to ensure full removal
    # A wider notch helps to remove the peak's energy more effectively.
    notch_width_bins = 3 # Bins on each side of the target
    fft_data[max(0, target_bin - notch_width_bins) : target_bin + notch_width_bins + 1] = 0

    # Reconstruct the signal without the test tone
    data_notched = np.fft.irfft(fft_data, n=n_samples)

    # 2. Calculate RMS of the remaining noise (unweighted)
    rms_noise = np.sqrt(np.mean(data_notched**2))
    if rms_noise < 1e-12:
        return float('inf'), 1e-12 # Avoid log(0)

    # 3. Calculate Dynamic Range
    dynamic_range_db = -20 * np.log10(rms_noise)

    return dynamic_range_db, rms_noise

def main():
    parser = argparse.ArgumentParser(description="Unweighted Dynamic Range (DR) Analyzer.")
    parser.add_argument("--list_devices", action="store_true", help="List available audio devices and exit.")
    parser.add_argument("--device", type=int, help="ID of the audio device for input and output.")
    parser.add_argument("--output_channel", type=int, default=1, help="1-based index of the output channel (default: 1).")
    parser.add_argument("--input_channel", type=int, default=1, help="1-based index of the input channel (default: 1).")
    parser.add_argument("--samplerate", type=int, default=48000, help="Samplerate in Hz (default: 48000).")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds (default: 5.0).")

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.device is None:
        parser.error("argument --device is required when not listing devices.")

    dr_db, rms_noise = measure_unweighted_dynamic_range(
        device_id=args.device,
        output_channel=args.output_channel,
        input_channel=args.input_channel,
        samplerate=args.samplerate,
        duration=args.duration
    )

    if dr_db is not None:
        table = Table(title="Unweighted Dynamic Range Measurement Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Dynamic Range (Unweighted)", f"{dr_db:.2f} dB")
        table.add_row("Noise Level (Unweighted)", f"{20*np.log10(rms_noise):.2f} dBFS")
        table.add_row("RMS Noise (Unweighted)", f"{rms_noise:.6e}")
        
        console.print(table)

if __name__ == "__main__":
    main()