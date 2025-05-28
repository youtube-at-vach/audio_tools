import numpy as np
import sounddevice as sd
from rich.console import Console # For error_console and example usage

# Assuming signal_processing_utils.py is in the same package (common_audio_lib)
try:
    from .signal_processing_utils import dbfs_to_linear, linear_to_dbfs
except ImportError:
    # Fallback for direct execution or if common_audio_lib is not installed as a package
    # This is a simplified version for the utility; real use expects package structure.
    print("Warning: Could not import .signal_processing_utils. Defining dbfs_to_linear locally for audio_io_utils.")
    def dbfs_to_linear(dbfs: float) -> float:
        if dbfs is None: return 0.0
        return 10**(dbfs / 20.0)
    def linear_to_dbfs(linear_amp: float, min_dbfs: float = -120.0) -> float:
        if linear_amp <= 0: return min_dbfs
        return 20 * np.log10(linear_amp)


def resolve_channel_specifier(spec_str: str, num_device_channels: int, channel_type_name: str = "channel", error_console: Console = None) -> int | None:
    """
    Converts a channel specifier string (e.g., 'L', 'R', '0', '1') to a 0-based integer index.
    Validates the index against num_device_channels.
    Returns the 0-based channel index or None if invalid.
    """
    effective_console = error_console if error_console else Console(stderr=True)
    spec_str_lower = spec_str.lower()
    
    if spec_str_lower == 'l':
        channel_idx = 0
    elif spec_str_lower == 'r':
        if num_device_channels >= 2:
            channel_idx = 1
        else:
            effective_console.print(f"[bold red]Error: Cannot select 'R' {channel_type_name}, device only has {num_device_channels} {channel_type_name}(s).[/bold red]")
            return None
    else:
        try:
            channel_idx = int(spec_str)
        except ValueError:
            effective_console.print(f"[bold red]Error: Invalid {channel_type_name} specifier '{spec_str}'. Must be 'L', 'R', or an integer.[/bold red]")
            return None

    if not (0 <= channel_idx < num_device_channels):
        effective_console.print(f"[bold red]Error: {channel_type_name.capitalize()} index {channel_idx} out of range (0-{num_device_channels-1}).[/bold red]")
        return None
        
    return channel_idx

def generate_sine_wave(frequency: float, amplitude_dbfs: float, duration: float, sample_rate: float, phase: float = 0.0) -> np.ndarray:
    """
    Generates a NumPy array representing a sine wave.
    Clips the signal to [-1.0, 1.0].
    """
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    if duration < 0:
        raise ValueError("Duration cannot be negative.")
    if frequency <= 0 : # Or allow negative for phase inversion, but typically positive.
        # Could print warning or raise error. For now, allow it.
        pass

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    amplitude_linear = dbfs_to_linear(amplitude_dbfs)
    
    wave = amplitude_linear * np.sin(2 * np.pi * frequency * t + np.deg2rad(phase))
    
    # Clip the signal to the valid range [-1.0, 1.0]
    return np.clip(wave, -1.0, 1.0)

def generate_log_spaced_frequencies(start_freq: float, end_freq: float, points_per_octave: int) -> np.ndarray:
    """
    Generates a list of frequencies logarithmically spaced.
    Returns a NumPy array of frequencies.
    """
    if start_freq <= 0 or end_freq <= 0:
        raise ValueError("Start and end frequencies must be positive.")
    if start_freq >= end_freq:
        # Return start_freq as a single point, or raise error.
        # Original code might have returned a single point or empty based on num_octaves.
        return np.array([start_freq]) 
    if points_per_octave <= 0:
        raise ValueError("Points per octave must be positive.")

    num_octaves = np.log2(end_freq / start_freq)
    total_points = int(np.ceil(num_octaves * points_per_octave)) + 1 # +1 to include the start_freq
    
    if total_points <= 1 and start_freq != end_freq : # if only 1 point, just return start_freq or avg
        # This condition ensures that if total_points suggests only start_freq, but end_freq is different,
        # we still provide a sequence. Let's ensure at least two points if start != end.
        # total_points = max(2, total_points) # This might not be what user wants if points_per_octave is very low.
        # A common approach for logspace:
        log_start = np.log2(start_freq)
        log_end = np.log2(end_freq)
        log_freqs = np.linspace(log_start, log_end, total_points)
        frequencies = 2**log_freqs
        # Ensure end_freq is included if it's very close due to linspace behavior
        if frequencies[-1] < end_freq and not np.isclose(frequencies[-1], end_freq):
             frequencies = np.append(frequencies, end_freq)
        # Ensure uniqueness and sort
        frequencies = np.unique(frequencies)
        # Filter out frequencies beyond end_freq if linspace overshot due to few points
        frequencies = frequencies[frequencies <= end_freq * 1.0001] # allow for small float inaccuracies
        return frequencies

    # If start_freq == end_freq, total_points calculation results in 1.
    if total_points <=1 :
        return np.array([start_freq])

    # Use np.logspace if points_per_octave directly translates to number of points over the whole range
    # The original logic implies points_per_octave is a density measure.
    # Let's use the octave-based generation method.
    
    frequencies = []
    current_freq = float(start_freq)
    frequencies.append(current_freq)
    
    if start_freq == end_freq: # Should have been caught by total_points <=1
        return np.array(frequencies)

    # Factor for each step within an octave
    step_factor = 2**(1.0 / points_per_octave)
    
    while current_freq < end_freq:
        next_freq = current_freq * step_factor
        if next_freq > end_freq * (1 + 1e-9): # Check if we are overshooting, with tolerance
            # If the very next step significantly overshoots, decide whether to add end_freq
            # This ensures that we don't add a point very slightly over end_freq,
            # but also that we don't miss end_freq if it's just beyond the last step.
            break 
        current_freq = next_freq
        frequencies.append(current_freq)
    
    # Ensure the last frequency is end_freq, without creating a duplicate if already close.
    if not np.isclose(frequencies[-1], end_freq):
        if frequencies[-1] < end_freq : # Only add if we haven't passed it
            frequencies.append(float(end_freq))
        elif len(frequencies) > 1 and frequencies[-1] > end_freq: # if overshot, replace last
            frequencies[-1] = float(end_freq)


    return np.unique(np.array(frequencies))


def play_and_record(
    device_id: int | None, 
    signal_to_play_mono: np.ndarray, 
    sample_rate: float, 
    output_channel_device_idx_0based: int, 
    input_channel_device_indices_0based_list: list[int], 
    record_duration_secs: float | None = None, 
    error_console: Console = None
) -> np.ndarray | None:
    """
    Plays a mono signal on a specific output channel and records from specified input channels.
    Returns the recorded audio as a NumPy array or None on failure.
    """
    effective_console = error_console if error_console else Console(stderr=True, style="bold red")
    
    if device_id is None:
        try:
            # Try to use default devices if no ID is given
            # Note: This might not always pick the desired device in complex setups.
            # Explicit device_id is usually better.
            sd.check_output_settings(samplerate=sample_rate, channels=output_channel_device_idx_0based + 1)
            sd.check_input_settings(samplerate=sample_rate, channels=max(input_channel_device_indices_0based_list) + 1 if input_channel_device_indices_0based_list else 1)
        except Exception as e: # Catches PortAudioError from check_settings or ValueError
            effective_console.print(f"Error with default audio device settings: {e}")
            return None
    else:
        try:
            dev_info = sd.query_devices(device_id)
            # Check output channel
            if output_channel_device_idx_0based >= dev_info['max_output_channels']:
                effective_console.print(f"Output channel {output_channel_device_idx_0based} is out of range for device {device_id} (max: {dev_info['max_output_channels']-1}).")
                return None
            # Check input channels
            for ch_idx in input_channel_device_indices_0based_list:
                if ch_idx >= dev_info['max_input_channels']:
                    effective_console.print(f"Input channel {ch_idx} is out of range for device {device_id} (max: {dev_info['max_input_channels']-1}).")
                    return None
        except (sd.PortAudioError, ValueError) as e:
            effective_console.print(f"Error querying device ID {device_id}: {e}")
            return None
        except Exception as e: # Catch any other error like device_id not being an int
            effective_console.print(f"Unexpected error with device ID {device_id}: {e}")
            return None


    num_playback_samples = len(signal_to_play_mono)
    playback_duration_secs = num_playback_samples / sample_rate
    
    actual_record_duration_secs = record_duration_secs if record_duration_secs is not None else playback_duration_secs
    if actual_record_duration_secs <= 0:
        effective_console.print("Recording duration must be positive.")
        return None

    num_record_samples = int(actual_record_duration_secs * sample_rate)

    # --- Prepare output buffer ---
    # Get max output channels for the selected device to build the buffer correctly
    try:
        if device_id is None: # Use default device
            dev_info_out = sd.query_devices(sd.default.device[1]) # Default output device
        else:
            dev_info_out = sd.query_devices(device_id)
        device_max_output_channels = dev_info_out['max_output_channels']
    except Exception as e:
        effective_console.print(f"Could not query output device info: {e}")
        return None

    if output_channel_device_idx_0based >= device_max_output_channels:
        effective_console.print(f"Output channel index {output_channel_device_idx_0based} exceeds device capabilities ({device_max_output_channels} channels).")
        return None

    output_buffer = np.zeros((num_playback_samples, device_max_output_channels), dtype=signal_to_play_mono.dtype)
    output_buffer[:, output_channel_device_idx_0based] = signal_to_play_mono

    # --- Prepare input mapping ---
    # sounddevice uses 1-based indexing for channel mapping
    input_mapping_1based = [ch + 1 for ch in input_channel_device_indices_0based_list]
    if not input_mapping_1based: # No input channels requested
        effective_console.print(f"No input channels specified for recording.")
        return None # Or modify to only play if that's a valid use case (not for this func name)

    try:
        recorded_audio = sd.playrec(
            output_buffer,
            samplerate=sample_rate,
            # For playrec, 'channels' refers to the number of *input* channels to record
            channels=len(input_mapping_1based), 
            input_mapping=input_mapping_1based, # Specifies which device channels to use for recording
            device=device_id, # Specify the device for both play and record
            blocking=True,
            # dtype of recorded audio can be specified if needed, defaults to float32
        )
        # sd.playrec already waits for completion if blocking=True.
        # If record_duration is longer than playback, playrec pads output_buffer with zeros.
        # If record_duration is shorter, playback is truncated.
        # We need to ensure recorded_audio has `num_record_samples`.
        # sd.playrec returns an array with `samplerate * duration` frames if duration is passed,
        # or `len(data)` frames if duration is None.
        # However, sd.playrec does not have a 'duration' argument directly.
        # It records for the duration of the 'data' (output_buffer) argument.
        # If we need to record longer, we need to pad output_buffer.
        # If we need to record shorter, we can truncate recorded_audio.

        if num_record_samples > num_playback_samples: # Need to record longer than playback
            # Pad the output_buffer to extend playback (effectively playing silence)
            padding_samples = num_record_samples - num_playback_samples
            padding_zeros = np.zeros((padding_samples, device_max_output_channels), dtype=output_buffer.dtype)
            extended_output_buffer = np.vstack((output_buffer, padding_zeros))
            
            recorded_audio = sd.playrec(
                extended_output_buffer,
                samplerate=sample_rate,
                channels=len(input_mapping_1based),
                input_mapping=input_mapping_1based,
                device=device_id,
                blocking=True
            )
        elif num_record_samples < num_playback_samples: # Need to record shorter than playback
            # Play the full signal but only care about the first part of the recording
            # This is tricky because playrec records for the duration of output_buffer.
            # A workaround is to use a callback, or to simply truncate the result.
            # For simplicity with blocking playrec:
            recorded_audio = sd.playrec(
                output_buffer, # Play full signal
                samplerate=sample_rate,
                channels=len(input_mapping_1based),
                input_mapping=input_mapping_1based,
                device=device_id,
                blocking=True
            )
            recorded_audio = recorded_audio[:num_record_samples, :]
        # else: num_record_samples == num_playback_samples, initial recorded_audio is fine.

        if recorded_audio.shape[0] < num_record_samples:
            # This might happen if sounddevice stops early for some reason
            effective_console.print(f"Warning: Recorded audio is shorter ({recorded_audio.shape[0]} samples) than requested ({num_record_samples} samples).")
            # Pad with zeros to meet requested length
            padding = np.zeros((num_record_samples - recorded_audio.shape[0], recorded_audio.shape[1]), dtype=recorded_audio.dtype)
            recorded_audio = np.vstack((recorded_audio, padding))
        
        return recorded_audio

    except sd.PortAudioError as e:
        effective_console.print(f"PortAudioError during play/record: {e}")
        return None
    except ValueError as e: # e.g. invalid channel mapping
        effective_console.print(f"ValueError during play/record setup: {e}")
        return None
    except Exception as e:
        effective_console.print(f"An unexpected error occurred during play/record: {e}")
        return None


def record_audio(duration_secs: float, sample_rate: int, device_id: int | None, 
                 input_channel_device_indices_0based_list: list[int], 
                 console_instance: Console = None) -> np.ndarray | None:
    '''Records audio from specified input channels of a device.

    Args:
        duration_secs: Duration of the recording in seconds.
        sample_rate: The sample rate in Hz.
        device_id: The ID of the audio device to use. Can be None for default.
        input_channel_device_indices_0based_list: List of 0-based input channel indices to record from.
        console_instance: Optional Rich Console for printing messages.

    Returns:
        A NumPy array containing the recorded audio (samples, channels), or None on error.
    '''
    effective_console = console_instance if console_instance else Console(stderr=True, style="bold red")
    try:
        if not input_channel_device_indices_0based_list:
            effective_console.print("[bold red]Error: No input channels specified for recording.[/bold red]")
            return None

        # sd.rec maps channels based on the number of columns in 'mapping' if it's a list of lists,
        # or if 'channels' arg is given and 'mapping' is a simple list, it maps them sequentially.
        # For sd.rec, mapping should be a list of input channel numbers (1-based).
        input_mapping_1based = [idx + 1 for idx in input_channel_device_indices_0based_list]
        
        num_channels_to_record = len(input_mapping_1based)
        num_frames = int(duration_secs * sample_rate)

        if num_frames <= 0:
            effective_console.print("[bold red]Error: Recording duration must be positive.[/bold red]")
            return None

        # Validate device and channels before recording
        if device_id is None:
            sd.check_input_settings(samplerate=sample_rate, channels=max(input_channel_device_indices_0based_list) + 1 if input_channel_device_indices_0based_list else 1)
        else:
            dev_info = sd.query_devices(device_id)
            for ch_idx in input_channel_device_indices_0based_list:
                if ch_idx >= dev_info['max_input_channels']:
                    effective_console.print(f"[bold red]Input channel {ch_idx} is out of range for device {device_id} (max: {dev_info['max_input_channels']-1}).[/bold red]")
                    return None
        
        effective_console.print(f"Recording {duration_secs}s from device {device_id if device_id is not None else 'default'}, channels {input_mapping_1based} at {sample_rate}Hz.")

        recorded_data = sd.rec(
            frames=num_frames,
            samplerate=sample_rate,
            device=device_id,
            mapping=input_mapping_1based, # Use 1-based mapping for sd.rec
            channels=num_channels_to_record, # Explicitly state number of channels from mapping
            blocking=True,
            dtype='float32' # Consistent data type
        )
        sd.wait() # Ensure recording is finished
        
        effective_console.print("[green]Recording complete.[/green]")
        return recorded_data
    except sd.PortAudioError as e:
        effective_console.print(f"[bold red]PortAudioError during recording: {e}[/bold red]")
        return None
    except ValueError as e: # e.g. from device validation
        effective_console.print(f"[bold red]ValueError during recording setup: {e}[/bold red]")
        return None
    except Exception as e:
        effective_console.print(f"[bold red]An unexpected error occurred during recording: {e}[/bold red]")
        return None


if __name__ == '__main__':
    console = Console()
    console.rule("[bold cyan]Testing common_audio_lib.audio_io_utils[/bold cyan]")

    # --- Test resolve_channel_specifier ---
    console.print("\n[bold]1. Testing resolve_channel_specifier[/bold]")
    console.print(f"Spec 'L', 2 channels: {resolve_channel_specifier('L', 2, 'output', console)}")
    console.print(f"Spec 'R', 2 channels: {resolve_channel_specifier('R', 2, 'output', console)}")
    console.print(f"Spec '0', 2 channels: {resolve_channel_specifier('0', 2, 'output', console)}")
    console.print(f"Spec '1', 2 channels: {resolve_channel_specifier('1', 2, 'output', console)}")
    console.print(f"Spec 'R', 1 channel: {resolve_channel_specifier('R', 1, 'output', console)}") # Expected error
    console.print(f"Spec '2', 2 channels: {resolve_channel_specifier('2', 2, 'output', console)}") # Expected error
    console.print(f"Spec 'X', 2 channels: {resolve_channel_specifier('X', 2, 'output', console)}") # Expected error

    # --- Test generate_sine_wave ---
    console.print("\n[bold]2. Testing generate_sine_wave[/bold]")
    sample_rate_test = 48000
    sine_wave = generate_sine_wave(frequency=1000, amplitude_dbfs=-6, duration=0.1, sample_rate=sample_rate_test)
    console.print(f"Generated 1kHz sine wave, -6 dBFS, 0.1s. Shape: {sine_wave.shape}, Max amp: {np.max(sine_wave):.4f} (Expected linear: {dbfs_to_linear(-6):.4f})")
    clipped_sine = generate_sine_wave(frequency=1000, amplitude_dbfs=6, duration=0.1, sample_rate=sample_rate_test) # +6 dBFS should clip
    console.print(f"Generated 1kHz sine wave, +6 dBFS (clipped). Max amp: {np.max(clipped_sine):.4f} (Expected: <= 1.0)")


    # --- Test generate_log_spaced_frequencies ---
    console.print("\n[bold]3. Testing generate_log_spaced_frequencies[/bold]")
    freqs = generate_log_spaced_frequencies(start_freq=20, end_freq=20000, points_per_octave=3)
    console.print(f"Log spaced frequencies (20-20k Hz, 3 pts/octave): {len(freqs)} points. First few: {freqs[:5]}, Last few: {freqs[-5:]}")
    freqs_single = generate_log_spaced_frequencies(1000, 1000, 1)
    console.print(f"Log spaced for 1000-1000 Hz: {freqs_single}")
    freqs_short = generate_log_spaced_frequencies(20, 200, 1) # Test few points
    console.print(f"Log spaced for 20-200 Hz (1 pt/octave): {freqs_short}")

    # --- Test play_and_record ---
    console.print("\n[bold]4. Testing play_and_record[/bold]")
    # This test is interactive and device-dependent. It might not run in all CI environments.
    try:
        # Query devices once for all tests if needed
        all_devices = sd.query_devices()
        default_input_device_id = sd.default.device[0]
        default_output_device_id = sd.default.device[1]
        
        default_input_info = get_device_info(default_input_device_id)
        default_output_info = get_device_info(default_output_device_id)

        if default_input_info and default_output_info and \
           default_output_info['max_output_channels'] > 0 and \
           default_input_info['max_input_channels'] > 0:
            
            console.print(f"Default Input Device: ID {default_input_device_id}, Name: {default_input_info['name']}")
            console.print(f"Default Output Device: ID {default_output_device_id}, Name: {default_output_info['name']}")

            output_ch_idx_test_playrec = 0 
            input_ch_indices_test_playrec = [0]

            console.print(f"Attempting play_and_record on output ch {output_ch_idx_test_playrec} and input ch {input_ch_indices_test_playrec} of default device(s).")
            
            test_signal_duration_playrec = 0.5 # seconds
            test_tone_playrec = generate_sine_wave(440, -24, test_signal_duration_playrec, sample_rate_test) # Lower amplitude
            
            console.print(f"Playing 440 Hz tone at -24 dBFS for {test_signal_duration_playrec}s and recording...")
            
            recorded_audio_playrec = play_and_record(
                device_id=None, 
                signal_to_play_mono=test_tone_playrec,
                sample_rate=sample_rate_test,
                output_channel_device_idx_0based=output_ch_idx_test_playrec,
                input_channel_device_indices_0based_list=input_ch_indices_test_playrec,
                record_duration_secs=test_signal_duration_playrec + 0.05, 
                error_console=console
            )

            if recorded_audio_playrec is not None:
                console.print(f"play_and_record: Successfully recorded audio. Shape: {recorded_audio_playrec.shape}")
                max_recorded_amp_playrec = np.max(np.abs(recorded_audio_playrec))
                console.print(f"play_and_record: Max absolute amplitude in recorded audio: {max_recorded_amp_playrec:.4f}")
                if max_recorded_amp_playrec < 0.0001: # Adjusted threshold
                    console.print("[yellow]play_and_record: Recorded audio seems very silent. Check microphone and levels.[/yellow]")
            else:
                console.print("[bold red]play_and_record: Failed to play and record audio.[/bold red]")
        else:
            console.print("[bold yellow]Warning: Default audio devices not suitable for play_and_record test. Skipping.[/bold yellow]")

    except Exception as e:
        console.print(f"[bold red]Could not run play_and_record test due to an error: {e}[/bold red]")

    # --- Test record_audio ---
    console.print("\n[bold]5. Testing record_audio[/bold]")
    try:
        if default_input_info and default_input_info['max_input_channels'] > 0:
            input_ch_indices_test_record = [0] # Record from first channel
            test_record_duration = 0.5 # seconds
            
            console.print(f"Attempting record_audio from input ch {input_ch_indices_test_record} of default input device for {test_record_duration}s.")

            recorded_audio_record = record_audio(
                duration_secs=test_record_duration,
                sample_rate=sample_rate_test,
                device_id=default_input_device_id, # Use default input device explicitly
                input_channel_device_indices_0based_list=input_ch_indices_test_record,
                console_instance=console
            )
            if recorded_audio_record is not None:
                console.print(f"record_audio: Successfully recorded audio. Shape: {recorded_audio_record.shape}")
                max_recorded_amp_record = np.max(np.abs(recorded_audio_record))
                console.print(f"record_audio: Max absolute amplitude: {max_recorded_amp_record:.4f}")
            else:
                console.print("[bold red]record_audio: Failed to record audio.[/bold red]")
        else:
            console.print("[bold yellow]Warning: Default input device not suitable for record_audio test. Skipping.[/bold yellow]")
            
    except Exception as e:
        console.print(f"[bold red]Could not run record_audio test due to an error: {e}[/bold red]")

    console.rule("[bold cyan]End of tests[/bold cyan]")
