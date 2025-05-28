import sounddevice as sd
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

def list_available_devices(console: Console):
    """
    Queries and displays available audio devices in a table.
    """
    try:
        devices = sd.query_devices()
        if not devices:
            console.print("No audio devices found.")
            return

        table = Table(title="Available Audio Devices")
        table.add_column("ID", style="dim", width=5)
        table.add_column("Name", style="cyan", min_width=20)
        table.add_column("Max Input Ch", style="magenta", justify="right")
        table.add_column("Max Output Ch", style="green", justify="right")
        table.add_column("Default SR (Hz)", style="yellow", justify="right")

        for i, device in enumerate(devices):
            table.add_row(
                str(i),
                device['name'],
                str(device['max_input_channels']),
                str(device['max_output_channels']),
                str(int(device['default_samplerate'])),
            )
        console.print(table)
    except sd.PortAudioError as e:
        console.print(f"[bold red]Error querying audio devices: {e}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")


def select_audio_device(console: Console, prompt_message="Select device ID", require_input=True, require_output=True):
    """
    Prompts the user to select an audio device and validates the selection.
    """
    list_available_devices(console)
    devices = sd.query_devices()

    if not devices:
        console.print("[bold red]No devices available to select.[/bold red]")
        return None

    while True:
        try:
            device_id_str = Prompt.ask(prompt_message)
            device_id = int(device_id_str)
            
            if not (0 <= device_id < len(devices)):
                console.print(f"[bold red]Invalid device ID. Please select an ID from the table.[/bold red]")
                continue

            device_info = get_device_info(device_id)
            if device_info is None: # Should not happen if previous check is correct, but good for safety
                console.print(f"[bold red]Could not retrieve information for device ID {device_id}.[/bold red]")
                continue

            if require_input and device_info['max_input_channels'] == 0:
                console.print(f"[bold red]Device ID {device_id} ({device_info['name']}) does not have input channels. Please select another.[/bold red]")
                continue
            
            if require_output and device_info['max_output_channels'] == 0:
                console.print(f"[bold red]Device ID {device_id} ({device_info['name']}) does not have output channels. Please select another.[/bold red]")
                continue
            
            console.print(f"Selected device: [bold green]{device_info['name']} (ID: {device_id})[/bold green]")
            return device_id

        except ValueError:
            console.print("[bold red]Invalid input. Please enter a number (device ID).[/bold red]")
        except sd.PortAudioError as e: # Should be caught by get_device_info, but as a fallback
            console.print(f"[bold red]Error accessing device information: {e}[/bold red]")
            return None
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred during selection: {e}[/bold red]")
            return None


def get_device_info(device_id):
    """
    Retrieves information for a specific audio device.
    """
    try:
        # Ensure device_id is an integer if it's a string representation of one
        parsed_device_id = int(device_id)
        device_info = sd.query_devices(parsed_device_id)
        return device_info
    except ValueError:
        # If device_id is not a number, try it as a device name (string)
        # This part of sounddevice.query_devices() is a bit tricky as it might still raise ValueError for names
        # For simplicity, we'll assume device_id is primarily an integer ID as per typical use with sounddevice.
        # If string names were a primary use case, this would need more robust handling.
        try:
            device_info = sd.query_devices(device_id)
            return device_info
        except ValueError as e_val:
            # Using Console here is not ideal as this is a utility function.
            # Consider raising the error or returning None and letting the caller handle console output.
            # For now, let's print to stderr for debugging, though this is bad practice in a library.
            # print(f"Error: Invalid device ID or name '{device_id}'. {e_val}", file=sys.stderr)
            raise ValueError(f"Invalid device ID or name '{device_id}'.") from e_val
    except sd.PortAudioError as e:
        # print(f"PortAudioError: Could not query device '{device_id}'. {e}", file=sys.stderr)
        raise sd.PortAudioError(f"Could not query device '{device_id}'.") from e
    except Exception as e: # Catch any other unexpected errors
        # print(f"An unexpected error occurred while querying device '{device_id}': {e}", file=sys.stderr)
        raise Exception(f"An unexpected error occurred while querying device '{device_id}'.") from e
