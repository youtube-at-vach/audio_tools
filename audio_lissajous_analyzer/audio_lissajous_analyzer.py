# audio_lissajous_analyzer.py

import argparse
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rich.console import Console
from rich.table import Table

# --- Configuration ---
DEFAULT_INPUT_DEVICE_ID = None
DEFAULT_INPUT_CHANNELS = [1, 2]  # Default to Left and Right channels
DEFAULT_SAMPLERATE = 48000
DEFAULT_BLOCK_DURATION_MS = 50  # How much audio to grab at a time
DEFAULT_PLOT_UPDATE_INTERVAL_MS = 20 # How often to update the plot

console = Console()

def list_audio_devices():
    """Lists available audio devices using rich."""
    try:
        devices = sd.query_devices()
    except Exception as e:
        console.print(f"[bold red]Error querying audio devices: {e}[/bold red]")
        console.print("Please ensure PortAudio is installed and configured correctly.")
        console.print("On Debian/Ubuntu, try: [cyan]sudo apt-get install libportaudio2[/cyan]")
        return

    table = Table(title="Available Audio Devices")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="magenta")
    table.add_column("Max In", justify="right", style="green")
    table.add_column("Max Out", justify="right", style="yellow")
    table.add_column("Default SR", justify="right", style="blue")

    for i, device in enumerate(devices):
        table.add_row(
            str(i),
            device['name'],
            str(device['max_input_channels']),
            str(device['max_output_channels']),
            str(int(device['default_samplerate'])),
        )
    console.print(table)

def get_device_info(device_id):
    """Gets information for a specific audio device."""
    try:
        return sd.query_devices(device_id)
    except ValueError:
        console.print(f"[bold red]Error: Device ID {device_id} not found.[/bold red]")
        return None

def main():
    """Main function to run the Lissajous analyzer."""
    parser = argparse.ArgumentParser(description="Real-time audio Lissajous figure visualizer.")
    parser.add_argument(
        "-l", "--list-devices", action="store_true",
        help="List available audio devices and exit."
    )
    parser.add_argument(
        "-d", "--device", type=int, default=DEFAULT_INPUT_DEVICE_ID,
        help="Input device ID. Defaults to the system's default input device."
    )
    parser.add_argument(
        "-c", "--channels", type=int, nargs=2, default=DEFAULT_INPUT_CHANNELS,
        metavar=("L", "R"),
        help=f"1-based input channel indices for X and Y axes. Default: {DEFAULT_INPUT_CHANNELS[0]} {DEFAULT_INPUT_CHANNELS[1]}"
    )
    parser.add_argument(
        "-r", "--samplerate", type=int, default=DEFAULT_SAMPLERATE,
        help=f"Sample rate in Hz. Default: {DEFAULT_SAMPLERATE}"
    )
    parser.add_argument(
        "-b", "--blocksize", type=int,
        help="Block size (number of frames). Default is calculated from --block-duration."
    )
    parser.add_argument(
        "--block-duration", type=int, default=DEFAULT_BLOCK_DURATION_MS,
        help=f"Duration of each audio block in milliseconds. Default: {DEFAULT_BLOCK_DURATION_MS}ms"
    )
    parser.add_argument(
        "--update-interval", type=int, default=DEFAULT_PLOT_UPDATE_INTERVAL_MS,
        help=f"Plot update interval in milliseconds. Default: {DEFAULT_PLOT_UPDATE_INTERVAL_MS}ms"
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    

    device_id = args.device
    if device_id is None:
        try:
            device_id = sd.default.device[0] # Get default input device ID
        except Exception as e:
            console.print(f"[bold red]Error: Could not determine default input device. {e}[/bold red]")
            return

    device_info = get_device_info(device_id)
    if device_info is None:
        # get_device_info already printed an error
        return

    # --- Validate Channels ---
    ch_x, ch_y = args.channels
    max_in_ch = device_info['max_input_channels']
    if not (1 <= ch_x <= max_in_ch and 1 <= ch_y <= max_in_ch):
        console.print("[bold red]Error: Invalid channel index.[/bold red]")
        console.print(f"Device {device_id} ('{device_info['name']}') has {max_in_ch} input channels.")
        console.print(f"Please provide channel indices between 1 and {max_in_ch}.")
        return
    if ch_x == ch_y:
        console.print("[bold yellow]Warning: Both axes are mapped to the same input channel.[/bold yellow]")

    samplerate = args.samplerate
    blocksize = args.blocksize or int(samplerate * args.block_duration / 1000)

    num_channels_to_open = max(ch_x, ch_y)

    console.print("Starting Lissajous Analyzer...")
    console.print(f"  Device: [cyan]{device_info['name']} (ID: {device_id})[/cyan]")
    console.print(f"  Channels (X, Y): [cyan]{ch_x}, {ch_y}[/cyan]")
    console.print(f"  Sample Rate: [cyan]{samplerate} Hz[/cyan]")
    console.print(f"  Block Size: [cyan]{blocksize} frames[/cyan]")


    # --- Matplotlib Setup ---
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_title(f"Lissajous Figure (Ch {ch_x} vs Ch {ch_y})")
    ax.set_xlabel(f"Channel {ch_x} Amplitude")
    ax.set_ylabel(f"Channel {ch_y} Amplitude")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True)
    ax.set_aspect('equal', 'box') # Crucial for correct phase visualization
    plt.tight_layout()

    # This queue will hold the latest block of audio data
    # We use a fixed-size numpy array as a circular buffer for simplicity
    q_size = blocksize * 2 # A bit of buffer
    audio_buffer = np.zeros((q_size, num_channels_to_open))
    latest_data_ptr = [0] # Use a list to make it mutable inside the callback

    def audio_callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            console.print(f"[yellow]Status: {status}[/yellow]")
        
        # Extract the desired channels based on 0-based indexing
        # ch_x and ch_y are 1-based, so subtract 1
        extracted_data = indata[:, [ch_x - 1, ch_y - 1]]

        start = latest_data_ptr[0]
        end = start + len(extracted_data)
        audio_buffer[start:end, :] = extracted_data
        latest_data_ptr[0] = end if end < q_size else 0


    def update_plot(frame):
        """This is called by matplotlib for each plot update."""
        # For simplicity, we just plot the whole buffer.
        # A more sophisticated approach might use a deque or handle buffer wrapping.
        line.set_data(audio_buffer[:, 0], audio_buffer[:, 1])
        return line,

    try:
        stream = sd.InputStream(
            device=device_id,
            channels=num_channels_to_open,
            samplerate=samplerate,
            blocksize=blocksize,
            callback=audio_callback
        )
        with stream:
            _ani = FuncAnimation(
                fig,
                update_plot,
                interval=args.update_interval,
                blit=True,
                cache_frame_data=False # Important for performance
            )
            plt.show()

    except Exception as e:
        console.print(f"\n[bold red]An error occurred: {e}[/bold red]")
        # This can be triggered by invalid sample rates, etc.
        parser.print_help()


if __name__ == "__main__":
    main()
