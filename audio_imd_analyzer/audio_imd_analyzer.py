import argparse
import sys
import numpy as np
import scipy.signal
import sounddevice as sd
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

# Global variable to keep track of playback position in play_and_record callback
current_frame_playback = 0

# Initialize consoles
console = Console()
error_console = Console(stderr=True, style="bold red")

def dbfs_to_linear(dbfs):
    """Converts dBFS to linear amplitude."""
    return 10**(dbfs / 20)

def generate_dual_tone(freq1, amp1_dbfs, freq2, ratio_f1_f2, duration, sample_rate):
    """
    Generates a dual-tone signal.
    """
    amp1_linear = dbfs_to_linear(amp1_dbfs)
    if ratio_f1_f2 == 0:
        if amp1_linear > 0:
            # Error already printed to stderr by previous versions, raising ValueError is key
            raise ValueError("Ratio f1/f2 cannot be zero if amp1_linear is non-zero.")
        else:
            amp2_linear = 0 
    else:
        amp2_linear = amp1_linear / ratio_f1_f2

    if (amp1_linear + amp2_linear) > 1.0:
        raise ValueError(
            f"Combined linear amplitude ({amp1_linear + amp2_linear:.4f}) exceeds 1.0. "
            f"amp1_linear: {amp1_linear:.4f}, amp2_linear: {amp2_linear:.4f}"
        )

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    s1 = amp1_linear * np.sin(2 * np.pi * freq1 * t)
    s2 = amp2_linear * np.sin(2 * np.pi * freq2 * t)
    s_combined = s1 + s2
    s_combined = np.clip(s_combined, -1.0, 1.0) 
    return s_combined

def select_device():
    """Allows the user to select an audio device. Exits if no devices found or error."""
    try:
        devices = sd.query_devices()
    except sd.PortAudioError as e:
        error_console.print(f"Error querying audio devices: {e}")
        sys.exit(1)
        
    if not devices:
        error_console.print("No audio devices found.")
        sys.exit(1)

    table = Table(title="Available Audio Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Max Input Channels", style="green")
    table.add_column("Max Output Channels", style="yellow")
    table.add_column("Default Sample Rate", style="blue")

    for i, device in enumerate(devices):
        table.add_row(
            str(i),
            device['name'],
            str(device['max_input_channels']),
            str(device['max_output_channels']),
            str(device['default_samplerate'])
        )
    console.print(table)
    
    while True:
        try:
            device_id_str = Prompt.ask("Select device ID")
            device_id = int(device_id_str)
            if 0 <= device_id < len(devices):
                console.print(f"Selected device: ID {device_id} - {devices[device_id]['name']}")
                return device_id
            else:
                error_console.print(f"Invalid ID. Please choose between 0 and {len(devices) - 1}.")
        except ValueError:
            error_console.print("Invalid input. Please enter a number.")
        except Exception as e: # Catch other potential Prompt issues
            error_console.print(f"An unexpected error occurred during device selection: {e}")
            sys.exit(1)


def play_and_record(signal_to_play, device_max_output_channels, device_max_input_channels, sample_rate, 
                    play_device_idx, record_device_idx,
                    output_channel_numeric_idx, record_channel_numeric_idx, duration_seconds):
    """
    Plays a signal and records simultaneously. Returns recorded audio or None on error.
    """
    global current_frame_playback
    current_frame_playback = 0 
    recorded_chunks = []
    total_frames_to_play = len(signal_to_play)

    def callback(indata, outdata, frames, time, status):
        global current_frame_playback # Python 3 specific
        if status:
            # Use sys.stderr directly for callback context as passing error_console is complex
            sys.stderr.write(f"Stream callback status: {status}\n")
        
        outdata.fill(0)
        remaining_frames_in_signal = total_frames_to_play - current_frame_playback
        frames_to_output_this_callback = min(frames, remaining_frames_in_signal)

        if frames_to_output_this_callback > 0:
            signal_chunk = signal_to_play[current_frame_playback : current_frame_playback + frames_to_output_this_callback]
            
            if len(signal_chunk) < frames: # Pad if signal chunk is shorter than frames
                temp_chunk = np.zeros(frames)
                temp_chunk[:len(signal_chunk)] = signal_chunk
                signal_chunk_to_assign = temp_chunk
            else:
                signal_chunk_to_assign = signal_chunk

            # Assign to the correct output channel
            if output_channel_numeric_idx < outdata.shape[1]:
                 outdata[:, output_channel_numeric_idx] = signal_chunk_to_assign
            else: # Fallback, though validated in main
                outdata[:, 0] = signal_chunk_to_assign 
            current_frame_playback += frames_to_output_this_callback
        
        # Record from the correct input channel
        if record_channel_numeric_idx < indata.shape[1]:
            recorded_chunks.append(indata[:, record_channel_numeric_idx].copy())
        else: # Fallback, though validated in main
            recorded_chunks.append(np.zeros(frames))

    try:
        with sd.Stream(device=(play_device_idx, record_device_idx), 
                       samplerate=sample_rate,
                       channels=(device_max_output_channels, device_max_input_channels), 
                       callback=callback):
            sd.sleep(int(duration_seconds * 1000))
            
    except sd.PortAudioError as e:
        error_console.print(f"Audio stream error: {e}")
        return None # Indicate error
    except Exception as e:
        error_console.print(f"An unexpected error occurred during play/record: {e}")
        return None # Indicate error
    finally:
        current_frame_playback = 0 # Ensure reset

    if not recorded_chunks:
        error_console.print("No audio data was recorded.")
        return None
    return np.concatenate(recorded_chunks)

def _find_peak_amplitude_in_band(fft_magnitudes, fft_frequencies, target_freq, search_half_width_hz=20.0):
    min_freq = target_freq - search_half_width_hz
    max_freq = target_freq + search_half_width_hz
    band_indices = np.where((fft_frequencies >= min_freq) & (fft_frequencies <= max_freq))[0]
    if not band_indices.size:
        return 0.0, 0.0
    peak_index_in_band = np.argmax(fft_magnitudes[band_indices])
    peak_abs_index = band_indices[peak_index_in_band]
    return fft_frequencies[peak_abs_index], fft_magnitudes[peak_abs_index]

def analyze_imd_smpte(recorded_audio, sample_rate, f1, f2, window_name='blackmanharris', num_sideband_pairs=3):
    N = len(recorded_audio)
    if N == 0:
        # This case should ideally be caught by play_and_record returning None
        error_console.print("Cannot analyze empty recorded audio.")
        return None 

    try:
        window_samples = scipy.signal.get_window(window_name, N)
    except (ValueError, TypeError) as e: # TypeError if window_name is not a string
        error_console.print(f"Invalid FFT window '{window_name}': {e}. Using 'blackmanharris'.")
        window_name = 'blackmanharris'
        window_samples = scipy.signal.get_window(window_name, N)

    windowed_signal = recorded_audio * window_samples
    fft_result = np.fft.rfft(windowed_signal)
    scaled_fft_magnitude = np.abs(fft_result) * (2 / np.sum(window_samples))
    fft_frequencies = np.fft.rfftfreq(N, d=1/sample_rate)

    amp_f2_freq_actual, amp_f2_linear = _find_peak_amplitude_in_band(scaled_fft_magnitude, fft_frequencies, f2, search_half_width_hz=max(50.0, f1*0.1)) # Reduced search width a bit

    if amp_f2_linear < 1e-9:
        console.print(f"[yellow]Warning: Reference tone f2 ({f2:.1f} Hz) amplitude ({amp_f2_linear:.2e}) is too low for meaningful IMD analysis.[/yellow]")
        return {
            'imd_percentage': 0.0, 'imd_db': -np.inf, 'amp_f2_dbfs': -np.inf, 
            'amp_f2_linear': amp_f2_linear, 'amp_f2_freq_actual': amp_f2_freq_actual,
            'imd_products_details': []
        }

    imd_products_details = []
    sideband_amplitudes_linear = []
    for n in range(1, num_sideband_pairs + 1):
        for sign, type_str in [(1, '+'), (-1, '-')]:
            sb_freq_nominal = f2 + sign * n * f1
            if sb_freq_nominal <= 0:
                continue
            actual_freq, amp_sb_linear = _find_peak_amplitude_in_band(scaled_fft_magnitude, fft_frequencies, sb_freq_nominal)
            if amp_sb_linear > 1e-12:
                dbr_f2 = 20 * np.log10(amp_sb_linear / amp_f2_linear) if amp_f2_linear > 0 else -np.inf
                imd_products_details.append({
                    'order_n': n, 'type': type_str, 'freq_hz_nominal': sb_freq_nominal,
                    'freq_hz_actual': actual_freq, 'amp_linear': amp_sb_linear, 'amp_dbr_f2': dbr_f2
                })
                sideband_amplitudes_linear.append(amp_sb_linear)
    
    rms_sum_sidebands = np.sqrt(np.sum(np.array(sideband_amplitudes_linear)**2)) if sideband_amplitudes_linear else 0.0
    amp_f2_dbfs = 20 * np.log10(amp_f2_linear) if amp_f2_linear > 1e-9 else -np.inf
    
    if amp_f2_linear < 1e-9:
        imd_percentage, imd_db = 0.0, -np.inf
    else:
        imd_percentage = (rms_sum_sidebands / amp_f2_linear) * 100
        imd_db = 20 * np.log10(rms_sum_sidebands / amp_f2_linear) if rms_sum_sidebands > 1e-9 else -np.inf

    return {
        'imd_percentage': imd_percentage, 'imd_db': imd_db, 'amp_f2_dbfs': amp_f2_dbfs,
        'amp_f2_linear': amp_f2_linear, 'amp_f2_freq_actual': amp_f2_freq_actual,
        'imd_products_details': imd_products_details
    }

def analyze_imd_ccif(recorded_audio, sample_rate, f1, f2, window_name='blackmanharris'):
    """
    Analyzes recorded audio for CCIF (twin-tone) Intermodulation Distortion.
    f1 and f2 are the nominal frequencies of the two input tones.
    """
    N = len(recorded_audio)
    if N == 0:
        error_console.print("Cannot analyze empty recorded audio for CCIF.")
        return None

    try:
        window_samples = scipy.signal.get_window(window_name, N)
    except (ValueError, TypeError) as e:
        error_console.print(f"Invalid FFT window '{window_name}': {e}. Using 'blackmanharris'.")
        window_name = 'blackmanharris'
        window_samples = scipy.signal.get_window(window_name, N)

    windowed_signal = recorded_audio * window_samples
    fft_result = np.fft.rfft(windowed_signal)
    # Scale FFT magnitude: (2 / sum_of_window_samples) for single-sided spectrum
    # This scaling gives peak amplitude of sine waves, not RMS.
    scaled_fft_magnitude = np.abs(fft_result) * (2 / np.sum(window_samples))
    fft_frequencies = np.fft.rfftfreq(N, d=1/sample_rate)

    # Determine a reasonable search width for f1 and f2, e.g., 1/4 of their difference or min 50Hz
    f_diff = abs(f2 - f1)
    search_half_width_f1_f2 = max(50.0, f_diff / 4.0) 

    amp_f1_freq_actual, amp_f1_linear = _find_peak_amplitude_in_band(scaled_fft_magnitude, fft_frequencies, f1, search_half_width_hz=search_half_width_f1_f2)
    amp_f2_freq_actual, amp_f2_linear = _find_peak_amplitude_in_band(scaled_fft_magnitude, fft_frequencies, f2, search_half_width_hz=search_half_width_f1_f2)

    if amp_f1_linear < 1e-9 or amp_f2_linear < 1e-9:
        console.print(f"[yellow]Warning: One or both reference tones f1 ({f1:.1f} Hz, amp: {amp_f1_linear:.2e}) "
                      f"or f2 ({f2:.1f} Hz, amp: {amp_f2_linear:.2e}) are too low for meaningful CCIF IMD analysis.[/yellow]")
        return {
            'imd_percentage': 0.0, 'imd_db': -np.inf,
            'amp_f1_dbfs': 20 * np.log10(amp_f1_linear) if amp_f1_linear > 1e-9 else -np.inf,
            'amp_f1_linear': amp_f1_linear, 'amp_f1_freq_actual': amp_f1_freq_actual,
            'amp_f2_dbfs': 20 * np.log10(amp_f2_linear) if amp_f2_linear > 1e-9 else -np.inf,
            'amp_f2_linear': amp_f2_linear, 'amp_f2_freq_actual': amp_f2_freq_actual,
            'imd_products_details': []
        }

    sum_amp_f1_f2 = amp_f1_linear + amp_f2_linear
    if sum_amp_f1_f2 < 1e-9: # Should be caught by above, but as a safeguard
        error_console.print("Sum of reference tone amplitudes is too low for CCIF analysis.")
        # Return structure consistent with low signal warning
        return {
            'imd_percentage': 0.0, 'imd_db': -np.inf,
            'amp_f1_dbfs': 20 * np.log10(amp_f1_linear) if amp_f1_linear > 1e-9 else -np.inf,
            'amp_f1_linear': amp_f1_linear, 'amp_f1_freq_actual': amp_f1_freq_actual,
            'amp_f2_dbfs': 20 * np.log10(amp_f2_linear) if amp_f2_linear > 1e-9 else -np.inf,
            'amp_f2_linear': amp_f2_linear, 'amp_f2_freq_actual': amp_f2_freq_actual,
            'imd_products_details': []
        }


    imd_products_details = []
    distortion_products_linear_amps_sq = [] # For RMS sum

    # Difference tone (d2)
    freq_d2_nominal = abs(f2 - f1)
    # Search width for d2 can be tighter, e.g. 20Hz, as it's usually well separated
    actual_d2_freq, amp_d2_linear = _find_peak_amplitude_in_band(scaled_fft_magnitude, fft_frequencies, freq_d2_nominal, search_half_width_hz=20.0)
    if amp_d2_linear > 1e-12: # Threshold for considering a product significant
        dbr_sum = 20 * np.log10(amp_d2_linear / sum_amp_f1_f2)
        imd_products_details.append({
            'type': "d2 (f2-f1)", 'freq_hz_nominal': freq_d2_nominal,
            'freq_hz_actual': actual_d2_freq, 'amp_linear': amp_d2_linear, 'amp_dbr_f_sum': dbr_sum
        })
        distortion_products_linear_amps_sq.append(amp_d2_linear**2)

    # Third-order product 1 (d3_lower): 2*f1 - f2
    freq_d3_lower_nominal = 2 * f1 - f2
    if freq_d3_lower_nominal > 0:
        actual_d3_lower_freq, amp_d3_lower_linear = _find_peak_amplitude_in_band(scaled_fft_magnitude, fft_frequencies, freq_d3_lower_nominal, search_half_width_hz=20.0)
        if amp_d3_lower_linear > 1e-12:
            dbr_sum = 20 * np.log10(amp_d3_lower_linear / sum_amp_f1_f2)
            imd_products_details.append({
                'type': "d3 (2f1-f2)", 'freq_hz_nominal': freq_d3_lower_nominal,
                'freq_hz_actual': actual_d3_lower_freq, 'amp_linear': amp_d3_lower_linear, 'amp_dbr_f_sum': dbr_sum
            })
            distortion_products_linear_amps_sq.append(amp_d3_lower_linear**2)

    # Third-order product 2 (d3_upper): 2*f2 - f1
    freq_d3_upper_nominal = 2 * f2 - f1
    if freq_d3_upper_nominal > 0: # Should always be true if f1, f2 > 0
        actual_d3_upper_freq, amp_d3_upper_linear = _find_peak_amplitude_in_band(scaled_fft_magnitude, fft_frequencies, freq_d3_upper_nominal, search_half_width_hz=20.0)
        if amp_d3_upper_linear > 1e-12:
            dbr_sum = 20 * np.log10(amp_d3_upper_linear / sum_amp_f1_f2)
            imd_products_details.append({
                'type': "d3 (2f2-f1)", 'freq_hz_nominal': freq_d3_upper_nominal,
                'freq_hz_actual': actual_d3_upper_freq, 'amp_linear': amp_d3_upper_linear, 'amp_dbr_f_sum': dbr_sum
            })
            distortion_products_linear_amps_sq.append(amp_d3_upper_linear**2)
    
    rms_sum_distortion_products = np.sqrt(np.sum(distortion_products_linear_amps_sq)) if distortion_products_linear_amps_sq else 0.0
    
    imd_percentage = (rms_sum_distortion_products / sum_amp_f1_f2) * 100
    imd_db = 20 * np.log10(rms_sum_distortion_products / sum_amp_f1_f2) if rms_sum_distortion_products > 1e-9 else -np.inf # Avoid log(0)

    amp_f1_dbfs = 20 * np.log10(amp_f1_linear) if amp_f1_linear > 1e-9 else -np.inf
    amp_f2_dbfs = 20 * np.log10(amp_f2_linear) if amp_f2_linear > 1e-9 else -np.inf

    return {
        'imd_percentage': imd_percentage, 'imd_db': imd_db,
        'amp_f1_dbfs': amp_f1_dbfs, 'amp_f1_linear': amp_f1_linear, 'amp_f1_freq_actual': amp_f1_freq_actual,
        'amp_f2_dbfs': amp_f2_dbfs, 'amp_f2_linear': amp_f2_linear, 'amp_f2_freq_actual': amp_f2_freq_actual,
        'imd_products_details': imd_products_details
    }

def save_results_to_csv(imd_results, standard, output_csv_path):
    """Saves IMD analysis results to a CSV file."""
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write summary information
            writer.writerow([f'IMD Standard:', standard.upper()])
            if standard == 'smpte':
                writer.writerow(['IMD Percentage:', f'{imd_results['imd_percentage']:.4f} %'])
                writer.writerow(['IMD (dB):', f'{imd_results['imd_db']:.2f} dB'])
                writer.writerow(['Reference f2 (Hz):', f'{imd_results['amp_f2_freq_actual']:.1f}'])
                writer.writerow(['Reference f2 (dBFS):', f'{imd_results['amp_f2_dbfs']:.2f}'])
                writer.writerow([]) # Blank row for separation
                writer.writerow(['Order (n)', 'Type', 'Nom. Freq (Hz)', 'Act. Freq (Hz)', 'Amplitude (Lin)', 'Level (dBr f2)'])
                for p in imd_results['imd_products_details']:
                    writer.writerow([p['order_n'], p['type'], f'{p['freq_hz_nominal']:.1f}', 
                                     f'{p['freq_hz_actual']:.1f}', f'{p['amp_linear']:.2e}', f'{p['amp_dbr_f2']:.2f}'])
            elif standard == 'ccif':
                writer.writerow(['IMD Percentage:', f'{imd_results['imd_percentage']:.4f} %'])
                writer.writerow(['IMD (dB):', f'{imd_results['imd_db']:.2f} dB'])
                writer.writerow(['Reference f1 (Hz):', f'{imd_results['amp_f1_freq_actual']:.1f}'])
                writer.writerow(['Reference f1 (dBFS):', f'{imd_results['amp_f1_dbfs']:.2f}'])
                writer.writerow(['Reference f2 (Hz):', f'{imd_results['amp_f2_freq_actual']:.1f}'])
                writer.writerow(['Reference f2 (dBFS):', f'{imd_results['amp_f2_dbfs']:.2f}'])
                writer.writerow([]) # Blank row for separation
                writer.writerow(['Product Type', 'Nom. Freq (Hz)', 'Act. Freq (Hz)', 'Amplitude (Lin)', 'Level (dBr f1+f2)'])
                for p in imd_results['imd_products_details']:
                    writer.writerow([p['type'], f'{p['freq_hz_nominal']:.1f}', 
                                     f'{p['freq_hz_actual']:.1f}', f'{p['amp_linear']:.2e}', f'{p['amp_dbr_f_sum']:.2f}'])
        console.print(f"[green]Results saved to CSV: {output_csv_path}[/green]")
    except IOError as e:
        error_console.print(f"Error writing CSV file {output_csv_path}: {e}")
    except Exception as e:
        error_console.print(f"An unexpected error occurred while writing CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate, play, record, and analyze a dual-tone signal for IMD testing.")
    # Group arguments for clarity
    gen_group = parser.add_argument_group('Signal Generation Parameters')
    gen_group.add_argument("--f1", type=float, default=60.0, help="Frequency of the first tone (Hz)")
    gen_group.add_argument("--f2", type=float, default=7000.0, help="Frequency of the second tone (Hz)")
    gen_group.add_argument("--amplitude", type=float, default=-12.0, help="Amplitude of the first tone (dBFS)")
    gen_group.add_argument("--ratio", type=float, default=4.0, help="Linear amplitude ratio of f1/f2 (e.g., 4 for 4:1)")
    gen_group.add_argument("--duration", type=float, default=1.0, help="Signal duration (seconds)")
    gen_group.add_argument("--sample_rate", type=int, default=48000, help="Sampling rate (Hz)")
    
    audio_group = parser.add_argument_group('Audio Device Parameters')
    audio_group.add_argument("--device", type=int, help="Audio device ID. Prompts if not provided.")
    audio_group.add_argument("--output_channel", "-oc", type=str, choices=['L', 'R'], default='R', help="Output channel ('L' or 'R')")
    audio_group.add_argument("--input_channel", "-ic", type=str, choices=['L', 'R'], default='L', help="Input channel ('L' or 'R')")

    analysis_group = parser.add_argument_group('Analysis Parameters')
    analysis_group.add_argument("--window", type=str, default='blackmanharris', help="FFT window type")
    analysis_group.add_argument("--num_sidebands", type=int, default=3, help="Number of sideband pairs for SMPTE IMD")
    analysis_group.add_argument("--standard", "-std", type=str, default='smpte', choices=['smpte', 'ccif'], help="IMD standard to use (smpte or ccif)")

    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("--output-csv", type=str, default=None, help="Path to save IMD product details as a CSV file.")
    args = parser.parse_args()

    # --- Adjust defaults for CCIF ---
    if args.standard == 'ccif':
        # Check if f1, f2, and ratio were provided by the user (not default)
        # Default values are specified in add_argument
        user_provided_f1 = args.f1 != parser.get_default("f1")
        user_provided_f2 = args.f2 != parser.get_default("f2")
        user_provided_ratio = args.ratio != parser.get_default("ratio")

        if not user_provided_f1 and not user_provided_f2:
            args.f1 = 19000.0
            args.f2 = 20000.0
            console.print(f"[yellow]CCIF standard selected. Defaulting f1={args.f1}Hz, f2={args.f2}Hz.[/yellow]")
        elif user_provided_f1 != user_provided_f2: # if user only provided one, it's ambiguous for CCIF
            error_console.print("For CCIF, please provide both --f1 and --f2, or neither to use defaults.")
            sys.exit(1)
        
        if not user_provided_ratio:
            args.ratio = 1.0
            console.print(f"[yellow]CCIF standard selected. Defaulting --ratio={args.ratio}.[/yellow]")
        elif args.ratio != 1.0:
            console.print(f"[yellow]Warning: CCIF standard typically uses a 1:1 ratio. User specified --ratio={args.ratio}.[/yellow]")

    # --- Device Selection and Validation ---
    if args.device is None:
        selected_device_idx = select_device() # select_device handles its own errors and exits
    else:
        selected_device_idx = args.device
        try:
            devices = sd.query_devices()
            if not (0 <= selected_device_idx < len(devices)):
                error_console.print(f"Device ID {selected_device_idx} is invalid. Max ID is {len(devices)-1}.")
                sys.exit(1)
            console.print(f"Using specified device ID: {selected_device_idx} - {devices[selected_device_idx]['name']}")
        except sd.PortAudioError as e:
            error_console.print(f"Error querying audio devices: {e}")
            sys.exit(1)
        except Exception as e: # Catch any other error during device validation
            error_console.print(f"An unexpected error occurred during device validation: {e}")
            sys.exit(1)
            
    try:
        device_info = sd.query_devices(selected_device_idx)
        device_max_output_channels = device_info['max_output_channels']
        device_max_input_channels = device_info['max_input_channels']
    except sd.PortAudioError as e:
        error_console.print(f"Error querying capabilities for device {selected_device_idx}: {e}")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"Unexpected error querying device {selected_device_idx} capabilities: {e}")
        sys.exit(1)

    output_channel_numeric_idx = 0 if args.output_channel == 'L' else 1
    input_channel_numeric_idx = 0 if args.input_channel == 'L' else 1

    if device_max_output_channels == 0:
        error_console.print(f"Selected device '{device_info['name']}' has no output channels.")
        sys.exit(1)
    if output_channel_numeric_idx >= device_max_output_channels:
        error_console.print(f"Output channel {args.output_channel} (idx {output_channel_numeric_idx}) exceeds device max output channels ({device_max_output_channels}).")
        sys.exit(1)
    
    if device_max_input_channels == 0:
        error_console.print(f"Selected device '{device_info['name']}' has no input channels.")
        sys.exit(1)
    if input_channel_numeric_idx >= device_max_input_channels:
        error_console.print(f"Input channel {args.input_channel} (idx {input_channel_numeric_idx}) exceeds device max input channels ({device_max_input_channels}).")
        sys.exit(1)

    # --- Signal Generation ---
    console.print("\n[green]Generating dual-tone signal...[/green]")
    console.print(f"  Parameters: f1={args.f1}Hz, f2={args.f2}Hz, Amp(f1)={args.amplitude}dBFS, Ratio={args.ratio}, SR={args.sample_rate}Hz, Dur={args.duration}s")
    try:
        dual_tone_signal = generate_dual_tone(args.f1, args.amplitude, args.f2, args.ratio, args.duration, args.sample_rate)
        console.print(f"  Signal generated. Max/Min: {np.max(dual_tone_signal):.4f} / {np.min(dual_tone_signal):.4f}")
    except ValueError as e:
        error_console.print(f"Error generating signal: {e}")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"An unexpected error occurred during signal generation: {e}")
        sys.exit(1)

    # --- Playback and Recording ---
    console.print("\n[green]Starting audio playback and recording...[/green]")
    console.print(f"  Device: {device_info['name']} (ID {selected_device_idx})")
    console.print(f"  Output Ch: {args.output_channel} ({output_channel_numeric_idx}), Input Ch: {args.input_channel} ({input_channel_numeric_idx})")
    
    recorded_audio = play_and_record(
        dual_tone_signal, device_max_output_channels, device_max_input_channels, args.sample_rate,
        selected_device_idx, selected_device_idx, 
        output_channel_numeric_idx, input_channel_numeric_idx, args.duration
    )

    if recorded_audio is None or recorded_audio.size == 0: # play_and_record returns None on error
        error_console.print("Audio playback/recording failed or yielded no data. Exiting.")
        sys.exit(1)
    console.print(f"  Audio recorded. Shape: {recorded_audio.shape}, Max/Min: {np.max(recorded_audio):.4f} / {np.min(recorded_audio):.4f}")

    # --- IMD Analysis ---
    console.print(f"\n[green]Analyzing recorded audio for IMD ({args.standard.upper()})...[/green]")
    
    imd_results = None
    if args.standard == 'smpte':
        console.print(f"  Analysis Parameters (SMPTE): f1={args.f1}Hz, f2={args.f2}Hz, Window={args.window}, Sidebands={args.num_sidebands}")
        try:
            imd_results = analyze_imd_smpte(
                recorded_audio, args.sample_rate, args.f1, args.f2, 
                window_name=args.window, num_sideband_pairs=args.num_sidebands
            )
        except Exception as e:
            error_console.print(f"An unexpected error occurred during SMPTE IMD analysis: {e}")
            sys.exit(1)

        if imd_results:
            f2_nominal = args.f2
            f2_actual = imd_results['amp_f2_freq_actual']
            console.print(f"  Reference f2 (Nominal: {f2_nominal:.1f}Hz, Actual: {f2_actual:.1f}Hz): {imd_results['amp_f2_dbfs']:.2f} dBFS ({imd_results['amp_f2_linear']:.4f} linear)")
            console.print(f"  IMD (SMPTE, {args.num_sidebands} pairs): [bold cyan]{imd_results['imd_percentage']:.4f} %[/bold cyan] / [bold cyan]{imd_results['imd_db']:.2f} dB[/bold cyan]")

            if imd_results['imd_products_details']:
                table = Table(title="SMPTE IMD Product Details")
                table.add_column("Order (n)", style="cyan")
                table.add_column("Type", style="cyan")
                table.add_column("Nom. Freq (Hz)", style="magenta", justify="right")
                table.add_column("Act. Freq (Hz)", style="magenta", justify="right")
                table.add_column("Amplitude (Lin)", style="green", justify="right")
                table.add_column("Level (dBr f2)", style="yellow", justify="right")

                for p in imd_results['imd_products_details']:
                    table.add_row(str(p['order_n']), p['type'], f"{p['freq_hz_nominal']:.1f}", 
                                  f"{p['freq_hz_actual']:.1f}", f"{p['amp_linear']:.2e}", f"{p['amp_dbr_f2']:.2f}")
                console.print(table)
            else:
                console.print("  No significant SMPTE IMD products found above threshold.")
        else: # Should not happen if analyze_imd_smpte always returns a dict or raises error
            error_console.print("SMPTE IMD analysis returned no results.")
            sys.exit(1)

    elif args.standard == 'ccif':
        console.print(f"  Analysis Parameters (CCIF): f1={args.f1}Hz, f2={args.f2}Hz, Window={args.window}")
        try:
            imd_results = analyze_imd_ccif(
                recorded_audio, args.sample_rate, args.f1, args.f2, 
                window_name=args.window
            )
        except Exception as e:
            error_console.print(f"An unexpected error occurred during CCIF IMD analysis: {e}")
            sys.exit(1)

        if imd_results:
            f1_nom, f1_act = args.f1, imd_results['amp_f1_freq_actual']
            f2_nom, f2_act = args.f2, imd_results['amp_f2_freq_actual']
            console.print(f"  Reference f1 (Nom: {f1_nom:.1f}Hz, Act: {f1_act:.1f}Hz): {imd_results['amp_f1_dbfs']:.2f} dBFS ({imd_results['amp_f1_linear']:.4f} lin)")
            console.print(f"  Reference f2 (Nom: {f2_nom:.1f}Hz, Act: {f2_act:.1f}Hz): {imd_results['amp_f2_dbfs']:.2f} dBFS ({imd_results['amp_f2_linear']:.4f} lin)")
            console.print(f"  IMD (CCIF): [bold cyan]{imd_results['imd_percentage']:.4f} %[/bold cyan] / [bold cyan]{imd_results['imd_db']:.2f} dB[/bold cyan]")

            if imd_results['imd_products_details']:
                table = Table(title="CCIF IMD Product Details")
                table.add_column("Product Type", style="cyan", width=12)
                table.add_column("Nom. Freq (Hz)", style="magenta", justify="right")
                table.add_column("Act. Freq (Hz)", style="magenta", justify="right")
                table.add_column("Amplitude (Lin)", style="green", justify="right")
                table.add_column("Level (dBr f1+f2)", style="yellow", justify="right")

                for p in imd_results['imd_products_details']:
                    table.add_row(p['type'], f"{p['freq_hz_nominal']:.1f}", 
                                  f"{p['freq_hz_actual']:.1f}", f"{p['amp_linear']:.2e}", f"{p['amp_dbr_f_sum']:.2f}")
                console.print(table)
            else:
                console.print("  No significant CCIF IMD products found above threshold.")
        else: # Should not happen if analyze_imd_ccif always returns a dict or raises error
            error_console.print("CCIF IMD analysis returned no results.")
            sys.exit(1)
    
    else: # Should be caught by argparse choices
        error_console.print(f"Unknown IMD standard: {args.standard}")
        sys.exit(1)

    if imd_results: # General success message if any analysis was done
        console.print(f"
[green]IMD analysis ({args.standard.upper()}) complete.[/green]")
        if args.output_csv:
            save_results_to_csv(imd_results, args.standard, args.output_csv)
    else: # Fallback, though specific errors should have exited earlier
        error_console.print("IMD analysis could not be performed or returned no meaningful results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
