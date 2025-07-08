
import argparse
import sounddevice as sd
from rich.console import Console
from rich.table import Table

def get_devices():
    """Lists available audio devices."""
    devices = sd.query_devices()
    console = Console()
    table = Table(title="Available Audio Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Max Input Channels", style="green")
    table.add_column("Max Output Channels", style="yellow")

    for i, device in enumerate(devices):
        table.add_row(str(i), device['name'], str(device['max_input_channels']), str(device['max_output_channels']))
    
    console.print(table)

import numpy as np

def find_loopback(device_id, sample_rate=48000, test_freq=440, duration=0.1, threshold=0.1):
    """
    Finds loopback paths on a specified audio device.
    """
    try:
        device_info = sd.query_devices(device_id)
    except ValueError:
        print(f"Error: Device ID {device_id} not found.")
        return

    max_out = device_info['max_output_channels']
    max_in = device_info['max_input_channels']

    if max_out == 0 or max_in == 0:
        print(f"Device {device_id} ('{device_info['name']}') does not support both input and output.")
        return

    console = Console()
    console.print(f"Testing device: [bold magenta]{device_info['name']}[/bold magenta]")
    
    found_paths = []

    t = np.linspace(0, duration, int(sample_rate * duration), False, dtype=np.float32)
    test_signal = np.sin(2 * np.pi * test_freq * t)

    for out_ch in range(1, max_out + 1):
        output_signal = np.zeros((len(test_signal), max_out), dtype=np.float32)
        output_signal[:, out_ch - 1] = test_signal
        
        recorded_signal = sd.playrec(output_signal, samplerate=sample_rate, channels=max_in, device=device_id)
        sd.wait()

        for in_ch in range(1, max_in + 1):
            input_fft = np.fft.rfft(recorded_signal[:, in_ch - 1])
            freqs = np.fft.rfftfreq(len(recorded_signal), 1/sample_rate)
            
            # Find the magnitude at the test frequency
            target_bin = np.argmin(np.abs(freqs - test_freq))
            magnitude = np.abs(input_fft[target_bin])

            if magnitude > threshold:
                found_paths.append((out_ch, in_ch))

    if found_paths:
        table = Table(title="Found Loopback Paths")
        table.add_column("Output Channel", style="cyan")
        table.add_column("Input Channel", style="magenta")
        for path in found_paths:
            table.add_row(str(path[0]), str(path[1]))
        console.print(table)
    else:
        console.print("[yellow]No loopback paths found.[/yellow]")


def main():
    """Main function to run the loopback finder."""
    parser = argparse.ArgumentParser(description="Finds active loopback paths on an audio device.")
    parser.add_argument("-d", "--device", type=int, help="The ID of the audio device to test.")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit.")
    
    args = parser.parse_args()

    if args.list_devices:
        get_devices()
        return

    if args.device is None:
        get_devices()
        print("\nPlease specify a device ID with the -d or --device option.")
        return
        
    find_loopback(args.device)

if __name__ == "__main__":
    main()
