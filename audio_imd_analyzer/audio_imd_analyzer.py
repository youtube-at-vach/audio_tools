import argparse
import sys
import numpy as np
# import scipy.signal # Keep for get_window if common_perform_fft doesn't take window name directly
# import sounddevice as sd # Replaced by common_audio_lib
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

# Common Audio Library Imports
from common_audio_lib.audio_device_manager import select_audio_device, get_device_info
from common_audio_lib.signal_processing_utils import (
    dbfs_to_linear, 
    linear_to_dbfs, # May not be directly needed if analysis returns dBFS
    perform_fft,
    find_peak_magnitude
)
from common_audio_lib.audio_io_utils import (
    resolve_channel_specifier,
    play_and_record # Replaces local play_and_record
)
# generate_dual_tone remains local as it's specific to IMD tests.


# Global variable current_frame_playback is removed.

# Initialize consoles
console = Console()
error_console = Console(stderr=True, style="bold red")

# dbfs_to_linear is replaced by common_audio_lib.signal_processing_utils.dbfs_to_linear

def generate_dual_tone(freq1, amp1_dbfs, freq2, ratio_f1_f2, duration, sample_rate):
    """
    Generates a dual-tone signal. Uses common dbfs_to_linear.
    """
    # Use common_audio_lib.signal_processing_utils.dbfs_to_linear
    amp1_linear = dbfs_to_linear(amp1_dbfs) 
    if ratio_f1_f2 == 0:
        if amp1_linear > 1e-9: # Check against a small threshold
            raise ValueError("Ratio f1/f2 cannot be zero if amp1_linear is non-zero.")
        else:
            amp2_linear = 0.0
    else:
        amp2_linear = amp1_linear / ratio_f1_f2

    # Check combined peak, assuming worst-case phase alignment (direct sum of amplitudes)
    if (amp1_linear + amp2_linear) > 1.0:
        # This is a peak check. If it's an RMS check, it would be sqrt(amp1^2 + amp2^2)
        # For safety, clipping is applied anyway, but good to warn/error if peak sum is > 1.0
        error_console.print(
            f"[yellow]Warning: Combined peak linear amplitude ({amp1_linear + amp2_linear:.4f}) exceeds 1.0. "
            f"amp1_linear: {amp1_linear:.4f}, amp2_linear: {amp2_linear:.4f}. Signal will be clipped.[/yellow]"
        )
        # Depending on strictness, could raise ValueError here.
        # raise ValueError(f"Combined peak linear amplitude ({amp1_linear + amp2_linear:.4f}) exceeds 1.0.")


    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    s1 = amp1_linear * np.sin(2 * np.pi * freq1 * t)
    s2 = amp2_linear * np.sin(2 * np.pi * freq2 * t)
    s_combined = s1 + s2
    s_combined = np.clip(s_combined, -1.0, 1.0) # Ensure output is within [-1, 1]
    return s_combined

# select_device is replaced by common_audio_lib.audio_device_manager.select_audio_device
# play_and_record is replaced by common_audio_lib.audio_io_utils.play_and_record
# _find_peak_amplitude_in_band is replaced by common_audio_lib.signal_processing_utils.find_peak_magnitude


def analyze_imd_smpte(recorded_audio, sample_rate, f1, f2, window_name='blackmanharris', num_sideband_pairs=3):
    N = len(recorded_audio)
    if N == 0:
        error_console.print("Cannot analyze empty recorded audio.")
        return None 

    # Use common_audio_lib.signal_processing_utils.perform_fft
    # Assuming perform_fft returns: fft_frequencies, scaled_fft_magnitude
    # If it also returns complex_fft_results, it's not used in this IMD analysis.
    try:
        fft_frequencies, scaled_fft_magnitude, _ = perform_fft(recorded_audio, sample_rate, window_name)
    except ValueError as e: # From perform_fft if window_name is invalid
        error_console.print(f"FFT analysis failed (window: '{window_name}'): {e}. Using 'blackmanharris'.")
        fft_frequencies, scaled_fft_magnitude, _ = perform_fft(recorded_audio, sample_rate, 'blackmanharris')

    if len(fft_frequencies) == 0: # perform_fft failed
        error_console.print("FFT analysis returned no data.")
        return None

    # Use common_audio_lib.signal_processing_utils.find_peak_magnitude
    amp_f2_freq_actual, amp_f2_linear = find_peak_magnitude(
        scaled_fft_magnitude, fft_frequencies, f2, search_half_width_hz=max(50.0, f1 * 0.1)
    )

    if amp_f2_linear < 1e-9: # Threshold for meaningful analysis
        console.print(f"[yellow]Warning: Reference tone f2 ({f2:.1f} Hz) amplitude ({amp_f2_linear:.2e}) is too low for meaningful IMD analysis.[/yellow]")
        return {
            'imd_percentage': 0.0, 'imd_db': -np.inf, 'amp_f2_dbfs': -np.inf, 
            'amp_f2_linear': amp_f2_linear, 'amp_f2_freq_actual': amp_f2_freq_actual,
            'imd_products_details': []
        }

    imd_products_details = []
    sideband_amplitudes_linear = []
    for n_order in range(1, num_sideband_pairs + 1): # Renamed n to n_order
        for sign, type_str in [(1, '+'), (-1, '-')]:
            sb_freq_nominal = f2 + sign * n_order * f1
            if sb_freq_nominal <= 0: continue
            # Use common find_peak_magnitude for sidebands
            actual_freq, amp_sb_linear = find_peak_magnitude(
                scaled_fft_magnitude, fft_frequencies, sb_freq_nominal, search_half_width_hz=20.0 # Default search width for sidebands
            )
            if amp_sb_linear > 1e-12: # Threshold for considering a sideband
                dbr_f2 = 20 * np.log10(amp_sb_linear / amp_f2_linear) if amp_f2_linear > 1e-9 else -np.inf
                imd_products_details.append({
                    'order_n': n_order, 'type': type_str, 'freq_hz_nominal': sb_freq_nominal,
                    'freq_hz_actual': actual_freq, 'amp_linear': amp_sb_linear, 'amp_dbr_f2': dbr_f2
                })
                sideband_amplitudes_linear.append(amp_sb_linear)
    
    rms_sum_sidebands = np.sqrt(np.sum(np.array(sideband_amplitudes_linear)**2)) if sideband_amplitudes_linear else 0.0
    # Use common linear_to_dbfs
    amp_f2_dbfs = linear_to_dbfs(amp_f2_linear, min_dbfs=-np.inf) if amp_f2_linear > 1e-9 else -np.inf
    
    if amp_f2_linear < 1e-9: # If f2 is too low
        calculated_imd_percentage, calculated_imd_db = 0.0, -np.inf # Renamed local vars
    else:
        calculated_imd_percentage = (rms_sum_sidebands / amp_f2_linear) * 100
        calculated_imd_db = 20 * np.log10(rms_sum_sidebands / amp_f2_linear) if rms_sum_sidebands > 1e-9 and amp_f2_linear > 1e-9 else -np.inf

    return {
        'imd_percentage': calculated_imd_percentage, 'imd_db': calculated_imd_db, 'amp_f2_dbfs': amp_f2_dbfs,
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

    # Use common_audio_lib.signal_processing_utils.perform_fft
    try:
        fft_frequencies, scaled_fft_magnitude, _ = perform_fft(recorded_audio, sample_rate, window_name)
    except ValueError as e: # From perform_fft if window_name is invalid
        error_console.print(f"FFT analysis failed (window: '{window_name}'): {e}. Using 'blackmanharris'.")
        fft_frequencies, scaled_fft_magnitude, _ = perform_fft(recorded_audio, sample_rate, 'blackmanharris')

    if len(fft_frequencies) == 0: # perform_fft failed
        error_console.print("FFT analysis returned no data for CCIF.")
        return None
        
    f_diff = abs(f2 - f1)
    search_half_width_f1_f2 = max(50.0, f_diff / 4.0)

    # Use common_audio_lib.signal_processing_utils.find_peak_magnitude for f1 and f2
    amp_f1_freq_actual, amp_f1_linear = find_peak_magnitude(
        scaled_fft_magnitude, fft_frequencies, f1, search_half_width_hz=search_half_width_f1_f2
    )
    amp_f2_freq_actual, amp_f2_linear = find_peak_magnitude(
        scaled_fft_magnitude, fft_frequencies, f2, search_half_width_hz=search_half_width_f1_f2
    )

    if amp_f1_linear < 1e-9 or amp_f2_linear < 1e-9:
        console.print(f"[yellow]Warning: One or both reference tones f1 ({f1:.1f} Hz, amp: {amp_f1_linear:.2e}) "
                      f"or f2 ({f2:.1f} Hz, amp: {amp_f2_linear:.2e}) are too low for meaningful CCIF IMD analysis.[/yellow]")
        # Use common linear_to_dbfs
        amp_f1_dbfs_val = linear_to_dbfs(amp_f1_linear, min_dbfs=-np.inf) if amp_f1_linear > 1e-9 else -np.inf
        amp_f2_dbfs_val = linear_to_dbfs(amp_f2_linear, min_dbfs=-np.inf) if amp_f2_linear > 1e-9 else -np.inf
        return {
            'imd_percentage': 0.0, 'imd_db': -np.inf,
            'amp_f1_dbfs': amp_f1_dbfs_val,
            'amp_f1_linear': amp_f1_linear, 'amp_f1_freq_actual': amp_f1_freq_actual,
            'amp_f2_dbfs': amp_f2_dbfs_val,
            'amp_f2_linear': amp_f2_linear, 'amp_f2_freq_actual': amp_f2_freq_actual,
            'imd_products_details': []
        }

    sum_amp_f1_f2 = amp_f1_linear + amp_f2_linear
    if sum_amp_f1_f2 < 1e-9: 
        error_console.print("Sum of reference tone amplitudes is too low for CCIF analysis.")
        amp_f1_dbfs_val = linear_to_dbfs(amp_f1_linear, min_dbfs=-np.inf) if amp_f1_linear > 1e-9 else -np.inf
        amp_f2_dbfs_val = linear_to_dbfs(amp_f2_linear, min_dbfs=-np.inf) if amp_f2_linear > 1e-9 else -np.inf
        return {
            'imd_percentage': 0.0, 'imd_db': -np.inf,
            'amp_f1_dbfs': amp_f1_dbfs_val,
            'amp_f1_linear': amp_f1_linear, 'amp_f1_freq_actual': amp_f1_freq_actual,
            'amp_f2_dbfs': amp_f2_dbfs_val,
            'amp_f2_linear': amp_f2_linear, 'amp_f2_freq_actual': amp_f2_freq_actual,
            'imd_products_details': []
        }


    imd_products_details = []
    distortion_products_linear_amps_sq = [] # For RMS sum

    # Difference tone (d2)
    freq_d2_nominal = abs(f2 - f1)
    actual_d2_freq, amp_d2_linear = find_peak_magnitude(
        scaled_fft_magnitude, fft_frequencies, freq_d2_nominal, search_half_width_hz=20.0
    )
    if amp_d2_linear > 1e-12:
        dbr_sum = 20 * np.log10(amp_d2_linear / sum_amp_f1_f2) if sum_amp_f1_f2 > 1e-9 else -np.inf
        imd_products_details.append({
            'type': "d2 (f2-f1)", 'freq_hz_nominal': freq_d2_nominal,
            'freq_hz_actual': actual_d2_freq, 'amp_linear': amp_d2_linear, 'amp_dbr_f_sum': dbr_sum
        })
        distortion_products_linear_amps_sq.append(amp_d2_linear**2)

    # Third-order product 1 (d3_lower): 2*f1 - f2
    freq_d3_lower_nominal = 2 * f1 - f2
    if freq_d3_lower_nominal > 0:
        actual_d3_lower_freq, amp_d3_lower_linear = find_peak_magnitude(
            scaled_fft_magnitude, fft_frequencies, freq_d3_lower_nominal, search_half_width_hz=20.0
        )
        if amp_d3_lower_linear > 1e-12:
            dbr_sum = 20 * np.log10(amp_d3_lower_linear / sum_amp_f1_f2) if sum_amp_f1_f2 > 1e-9 else -np.inf
            imd_products_details.append({
                'type': "d3 (2f1-f2)", 'freq_hz_nominal': freq_d3_lower_nominal,
                'freq_hz_actual': actual_d3_lower_freq, 'amp_linear': amp_d3_lower_linear, 'amp_dbr_f_sum': dbr_sum
            })
            distortion_products_linear_amps_sq.append(amp_d3_lower_linear**2)

    # Third-order product 2 (d3_upper): 2*f2 - f1
    freq_d3_upper_nominal = 2 * f2 - f1
    if freq_d3_upper_nominal > 0:
        actual_d3_upper_freq, amp_d3_upper_linear = find_peak_magnitude(
            scaled_fft_magnitude, fft_frequencies, freq_d3_upper_nominal, search_half_width_hz=20.0
        )
        if amp_d3_upper_linear > 1e-12:
            dbr_sum = 20 * np.log10(amp_d3_upper_linear / sum_amp_f1_f2) if sum_amp_f1_f2 > 1e-9 else -np.inf
            imd_products_details.append({
                'type': "d3 (2f2-f1)", 'freq_hz_nominal': freq_d3_upper_nominal,
                'freq_hz_actual': actual_d3_upper_freq, 'amp_linear': amp_d3_upper_linear, 'amp_dbr_f_sum': dbr_sum
            })
            distortion_products_linear_amps_sq.append(amp_d3_upper_linear**2)
    
    rms_sum_distortion_products = np.sqrt(np.sum(distortion_products_linear_amps_sq)) if distortion_products_linear_amps_sq else 0.0
    
    calculated_imd_percentage = (rms_sum_distortion_products / sum_amp_f1_f2) * 100 if sum_amp_f1_f2 > 1e-9 else 0.0
    calculated_imd_db = 20 * np.log10(rms_sum_distortion_products / sum_amp_f1_f2) if rms_sum_distortion_products > 1e-9 and sum_amp_f1_f2 > 1e-9 else -np.inf

    # Use common linear_to_dbfs
    amp_f1_dbfs_val = linear_to_dbfs(amp_f1_linear, min_dbfs=-np.inf) if amp_f1_linear > 1e-9 else -np.inf
    amp_f2_dbfs_val = linear_to_dbfs(amp_f2_linear, min_dbfs=-np.inf) if amp_f2_linear > 1e-9 else -np.inf

    return {
        'imd_percentage': calculated_imd_percentage, 'imd_db': calculated_imd_db,
        'amp_f1_dbfs': amp_f1_dbfs_val, 'amp_f1_linear': amp_f1_linear, 'amp_f1_freq_actual': amp_f1_freq_actual,
        'amp_f2_dbfs': amp_f2_dbfs_val, 'amp_f2_linear': amp_f2_linear, 'amp_f2_freq_actual': amp_f2_freq_actual,
        'imd_products_details': imd_products_details
    }

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
        selected_device_id = select_audio_device(console, require_input=True, require_output=True)
        if selected_device_id is None:
            error_console.print("No device selected. Exiting.")
            sys.exit(1)
    else:
        selected_device_id = args.device
        try:
            # Validate device ID
            temp_device_info = get_device_info(selected_device_id) # Will raise error if invalid
            if temp_device_info['max_output_channels'] == 0 or temp_device_info['max_input_channels'] == 0:
                 error_console.print(f"Error: Device ID {selected_device_id} ({temp_device_info['name']}) must have both input and output channels.")
                 sys.exit(1)
            console.print(f"Using specified device ID: {selected_device_id} - {temp_device_info['name']}")
        except ValueError as e: # Handles invalid device ID format or device not found
             error_console.print(f"Error validating device ID {args.device}: {e}")
             sys.exit(1)
        except Exception as e:
            error_console.print(f"An unexpected error occurred during device validation: {e}")
            sys.exit(1)
            
    current_device_info = get_device_info(selected_device_id) # Renamed
    if not current_device_info: # Safeguard
        error_console.print(f"Fatal: Could not get info for selected device {selected_device_id}.")
        sys.exit(1)

    # Use common_audio_lib.audio_io_utils.resolve_channel_specifier
    output_channel_idx_0based = resolve_channel_specifier( # Renamed
        args.output_channel, 
        current_device_info['max_output_channels'], 
        "output", 
        error_console=error_console
    )
    if output_channel_idx_0based is None: sys.exit(1)

    input_channel_idx_0based = resolve_channel_specifier( # Renamed
        args.input_channel, 
        current_device_info['max_input_channels'], 
        "input", 
        error_console=error_console
    )
    if input_channel_idx_0based is None: sys.exit(1)


    # --- Signal Generation ---
    console.print(f"\n[green]Generating dual-tone signal...[/green]")
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
    console.print(f"\n[green]Starting audio playback and recording...[/green]")
    console.print(f"  Device: {current_device_info['name']} (ID {selected_device_id})") # Use current_device_info
    console.print(f"  Output Ch: {args.output_channel} ({output_channel_idx_0based}), Input Ch: {args.input_channel} ({input_channel_idx_0based})") # Use resolved indices
    
    # Use common_audio_lib.audio_io_utils.play_and_record
    # It expects input_channel_device_indices_0based_list to be a list.
    recorded_audio_multi_ch = play_and_record(
        device_id=selected_device_id,
        signal_to_play_mono=dual_tone_signal,
        sample_rate=args.sample_rate,
        output_channel_device_idx_0based=output_channel_idx_0based,
        input_channel_device_indices_0based_list=[input_channel_idx_0based], # Pass as list
        record_duration_secs=args.duration, # Match playback duration
        error_console=error_console
    )

    if recorded_audio_multi_ch is None or recorded_audio_multi_ch.size == 0:
        error_console.print("Audio playback/recording failed or yielded no data using common_audio_lib. Exiting.")
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
        console.print(f"\n[green]IMD analysis ({args.standard.upper()}) complete.[/green]")
    else: # Fallback, though specific errors should have exited earlier
        error_console.print("IMD analysis could not be performed or returned no meaningful results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
