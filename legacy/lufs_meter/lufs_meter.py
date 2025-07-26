import soundfile as sf
import numpy as np
from scipy import signal
import argparse
import csv
from typing import Tuple, List, Dict, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def load_audio_file(filepath: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Loads an audio file, resamples to 48kHz, and converts to mono.

    Args:
        filepath: Path to the audio file.

    Returns:
        A tuple of (audio_data, sample_rate) or (None, None) if an error occurs.
    """
    try:
        audio_data, original_sample_rate = sf.read(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None, None
    except Exception as e:
        print(f"Error: Could not read audio file '{filepath}'. Unsupported format or other error: {e}")
        return None, None

    # Resample to 48kHz if necessary
    if original_sample_rate != 48000:
        num_samples = int(len(audio_data) * 48000 / original_sample_rate)
        audio_data = signal.resample(audio_data, num_samples)
        sample_rate = 48000
        print(f"Info: Resampled audio from {original_sample_rate} Hz to {sample_rate} Hz.")
    else:
        sample_rate = original_sample_rate

    # Convert to mono
    if audio_data.ndim > 1:
        if audio_data.shape[1] == 1: # Already mono but in 2D array
            audio_data = audio_data.flatten()
        elif audio_data.shape[1] == 2: # Stereo
            audio_data = np.mean(audio_data, axis=1)
            print("Info: Converted stereo audio to mono.")
        else: # More than 2 channels
            print(f"Warning: Audio has {audio_data.shape[1]} channels. Using the average of the first two channels.")
            audio_data = np.mean(audio_data[:, :2], axis=1)
    
    return audio_data, sample_rate


def apply_k_weighting(audio_data: np.ndarray, sample_rate: int):
    """
    Applies K-weighting filter to mono audio data.
    Assumes input audio_data is already at 48kHz.

    Args:
        audio_data: Mono audio data as a NumPy array.
        sample_rate: Sample rate of the audio data (should be 48000 Hz).

    Returns:
        K-weighted audio data as a NumPy array.
    """
    if sample_rate != 48000:
        # This should ideally not happen if load_audio_file is used correctly
        print(f"Warning: K-weighting is designed for 48kHz, but input is {sample_rate}Hz. Results may be inaccurate.")

    # K-weighting filter coefficients (ITU-R BS.1770-4)
    # First stage: High-shelf filter
    b0_shelf = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a0_shelf = [1.0, -1.69065929318241, 0.73248077421585]
    
    # Second stage: High-pass filter
    b1_hp = [1.0, -2.0, 1.0]
    a1_hp = [1.0, -1.99004745483398, 0.99007225036621]

    # Apply filters sequentially
    # Stage 1
    stage1_output = signal.lfilter(b0_shelf, a0_shelf, audio_data)
    # Stage 2
    k_weighted_audio = signal.lfilter(b1_hp, a1_hp, stage1_output)
    
    return k_weighted_audio


def calculate_mean_square(audio_block: np.ndarray) -> float:
    """
    Calculates the mean of the square of the samples in the block.
    (1/N) * sum(x_i^2)
    """
    if audio_block.size == 0:
        return 0.0
    return np.mean(np.square(audio_block))


def mean_square_to_lufs(mean_square_value: float) -> float:
    """
    Converts a mean square value to LUFS.
    Formula: -0.691 + 10 * log10(mean_square_value)
    """
    if mean_square_value <= 1e-10:  # Threshold to avoid log10 of zero or very small numbers
        return -np.inf # Or a very small LUFS value like -99.0 or -70.0 as per some standards for silence
    return -0.691 + 10 * np.log10(mean_square_value)


def calculate_momentary_loudness(k_weighted_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Calculates momentary loudness using a 400ms window with 75% overlap.
    Updates occur every 100ms.
    """
    if sample_rate == 0: # Should not happen with proper loading
        return np.array([])
        
    window_samples = int(0.400 * sample_rate)
    hop_samples = int(0.100 * sample_rate) # Update every 100ms for 75% overlap on 400ms window

    if k_weighted_audio.size < window_samples:
        # Handle cases where audio is shorter than window
        ms = calculate_mean_square(k_weighted_audio)
        lufs = mean_square_to_lufs(ms)
        return np.array([lufs]) if lufs > -np.inf else np.array([])

    momentary_lufs_values = []
    for i in range(0, k_weighted_audio.size - window_samples + 1, hop_samples):
        window = k_weighted_audio[i : i + window_samples]
        mean_square = calculate_mean_square(window)
        lufs = mean_square_to_lufs(mean_square)
        momentary_lufs_values.append(lufs)
    
    return np.array(momentary_lufs_values)


def calculate_short_term_loudness(k_weighted_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Calculates short-term loudness using a 3s window with 75% overlap.
    Updates occur every 0.75s (for 75% overlap). EBU R128 suggests updates every 1s. Let's use 1s for now.
    """
    if sample_rate == 0: # Should not happen
        return np.array([])

    window_samples = int(3.0 * sample_rate)
    hop_samples = int(1.0 * sample_rate) # Update every 1 second

    if k_weighted_audio.size < window_samples:
        ms = calculate_mean_square(k_weighted_audio)
        lufs = mean_square_to_lufs(ms)
        return np.array([lufs]) if lufs > -np.inf else np.array([])

    short_term_lufs_values = []
    for i in range(0, k_weighted_audio.size - window_samples + 1, hop_samples):
        window = k_weighted_audio[i : i + window_samples]
        mean_square = calculate_mean_square(window)
        lufs = mean_square_to_lufs(mean_square)
        short_term_lufs_values.append(lufs)
        
    return np.array(short_term_lufs_values)


def calculate_integrated_loudness(k_weighted_audio: np.ndarray, sample_rate: int) -> Tuple[float, np.ndarray]:
    """
    Calculates integrated loudness with gating (ITU-R BS.1770-4).
    """
    if sample_rate == 0 or k_weighted_audio.size == 0:
        return -np.inf, np.array([])

    block_samples = int(0.400 * sample_rate) # 400ms non-overlapping blocks
    
    if k_weighted_audio.size < block_samples:
        # Not enough audio for even one block
        return -np.inf, np.array([])

    num_blocks = k_weighted_audio.size // block_samples
    
    block_mean_squares = []
    block_lufs_values = [] # For returning as momentary LUFS for these blocks

    for i in range(num_blocks):
        start = i * block_samples
        end = start + block_samples
        audio_block = k_weighted_audio[start:end]
        ms = calculate_mean_square(audio_block)
        block_mean_squares.append(ms)
        block_lufs_values.append(mean_square_to_lufs(ms))

    block_mean_squares = np.array(block_mean_squares)
    momentary_lufs_for_gating = np.array(block_lufs_values)

    # Absolute Gating
    abs_gate_threshold_lufs = -70.0
    # Identify blocks above absolute gate
    # We need indices to filter both mean_square_values and original blocks if needed,
    # but for BS.1770, we only need the mean square values of blocks that pass.
    passed_abs_gate_indices = [i for i, lufs in enumerate(momentary_lufs_for_gating) if lufs > abs_gate_threshold_lufs]

    if not passed_abs_gate_indices:
        return -np.inf, momentary_lufs_for_gating # No blocks passed absolute gate

    ms_passed_abs_gate = block_mean_squares[passed_abs_gate_indices]
    
    # Relative Gating
    # Calculate average of mean square values of blocks that passed absolute gate
    avg_ms_passed_abs_gate = np.mean(ms_passed_abs_gate)
    if avg_ms_passed_abs_gate <= 1e-10: # Avoid log error if all are silent
         return -np.inf, momentary_lufs_for_gating

    relative_gate_ref_lufs = mean_square_to_lufs(avg_ms_passed_abs_gate)
    relative_gate_threshold_lufs = relative_gate_ref_lufs - 10.0

    # Filter blocks from those that passed absolute gating
    # We use their LUFS values (derived from their mean squares) for this comparison
    lufs_of_ms_passed_abs_gate = momentary_lufs_for_gating[passed_abs_gate_indices]

    passed_relative_gate_indices_of_abs_gated = [i for i, lufs in enumerate(lufs_of_ms_passed_abs_gate) if lufs > relative_gate_threshold_lufs]
    
    if not passed_relative_gate_indices_of_abs_gated:
        return -np.inf, momentary_lufs_for_gating # No blocks passed relative gate

    # Get the mean square values of blocks that passed *both* gates
    ms_passed_both_gates = ms_passed_abs_gate[passed_relative_gate_indices_of_abs_gated]

    if ms_passed_both_gates.size == 0: # Should be caught by previous check, but for safety
        return -np.inf, momentary_lufs_for_gating

    # Final Integrated Loudness Calculation
    final_avg_mean_square = np.mean(ms_passed_both_gates)
    integrated_lufs = mean_square_to_lufs(final_avg_mean_square)
    
    return integrated_lufs, momentary_lufs_for_gating


def calculate_loudness_range(short_term_lufs_values: np.ndarray, integrated_lufs: float) -> float:
    """
    Calculates Loudness Range (LRA) according to EBU R128.
    """
    if short_term_lufs_values.size < 3: # Not enough data for percentiles
        return 0.0 

    # 1. Filter short_term_lufs_values
    # Absolute threshold
    filtered_lufs = short_term_lufs_values[short_term_lufs_values >= -70.0]
    
    # Relative threshold (only if integrated_lufs is defined)
    if integrated_lufs > -np.inf: # Check if integrated_lufs is a valid number
        filtered_lufs = filtered_lufs[filtered_lufs >= (integrated_lufs - 20.0)]
    else: # If integrated_lufs is undefined, LRA might also be considered undefined or based purely on absolute
        # For now, if integrated LUFS is -inf, we'll proceed with only abs-gated values
        # or return 0.0 if too few values remain.
        pass


    # 2. If fewer than 3 values remain
    if filtered_lufs.size < 3:
        return 0.0 # Or np.nan if preferred for "undefined"

    # 3. Sort the remaining values
    sorted_lufs = np.sort(filtered_lufs)

    # 4. Calculate percentile boundaries
    # 5. Determine LUFS values at these percentiles
    val_10th = np.percentile(sorted_lufs, 10, interpolation='lower')
    val_95th = np.percentile(sorted_lufs, 95, interpolation='lower')

    # 6. LRA is the difference
    lra = val_95th - val_10th
    return lra


def calculate_true_peak(audio_data: np.ndarray, sample_rate: int) -> float:
    """
    Calculates True Peak (dBTP) according to ITU-R BS.1770-4.
    Assumes audio_data is mono and at 48kHz.
    """
    if audio_data.size == 0:
        return -np.inf

    # 1. Upsampling
    upsampling_factor = 4 # Target >= 192kHz, for 48kHz input, 4x is 192kHz
    if sample_rate < 48000: # Adjust if input SR is lower, though load_audio_file should ensure 48k
        # This case should ideally be handled or warned, as 4x might not be enough
        # For simplicity, we stick to 4x, assuming load_audio_file enforced 48kHz.
        pass
    
    # Using resample_poly for potentially better quality with integer factors
    try:
        upsampled_audio = signal.resample_poly(audio_data, upsampling_factor, 1)
    except ValueError as e:
        # This can happen if audio_data is too short, e.g. for polyphase filtering.
        # Fallback to a simpler method or just use original if too short.
        # For now, if too short, just use the original audio's peak.
        print(f"Warning: resample_poly failed ('{e}'), using original audio for peak. True Peak may be less accurate.")
        upsampled_audio = audio_data


    # 2. Peak Detection
    peak_abs_value = np.max(np.abs(upsampled_audio))

    # 3. dBTP Conversion
    if peak_abs_value == 0:
        return -np.inf
    
    dbtp = 20 * np.log10(peak_abs_value)
    return dbtp


def format_value(value: float, unit: str, target: Optional[float] = None) -> Text:
    """Helper function to format LUFS or dBTP values for printing with Rich."""
    text = Text()
    if value == -np.inf:
        text.append("-INF ", style="dim")
    elif value == np.nan or value is None:
        text.append("Undefined ", style="dim")
    else:
        text.append(f"{value:.1f} ", style="bold")
        if unit == "LUFS" and target is not None:
            diff = value - target
            if diff > 0:
                text.append(f"({diff:+.1f} vs target)", style="red")
            else:
                text.append(f"({diff:+.1f} vs target)", style="green")
    text.append(unit, style="dim")
    return text


def save_results_to_csv(filepath: str, results: Dict[str, Any], headers: List[str]):
    """Saves the loudness measurement results to a CSV file."""
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            # Prepare a row with formatted values, handling None for Target Loudness
            row_to_write: Dict[str, str] = {} # Ensure row_to_write keys are strings
            for header in headers:
                val = results.get(header)
                if isinstance(val, float) and val == -np.inf:
                    row_to_write[header] = "-INF"
                elif val is None or (isinstance(val, float) and np.isnan(val)):
                    row_to_write[header] = "N/A" # Or "Undefined"
                elif isinstance(val, float):
                     row_to_write[header] = f"{val:.1f}"
                else: # Should not happen if results dict is structured correctly, but good to handle
                    row_to_write[header] = str(val) if val is not None else "N/A"
            writer.writerow(row_to_write)
        print(f"Info: Results saved to '{filepath}'")
    except IOError as e:
        print(f"Error: Could not write to CSV file '{filepath}'. Reason: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="LUFS Meter: Analyze audio files for loudness according to ITU-R BS.1770-4 / EBU R128.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument(
        "filepath", 
        type=str, 
        help="Path to the input audio file."
    )
    parser.add_argument(
        "-o", "--output_file", 
        type=str, 
        help="Optional path to a CSV file to save the main loudness results."
    )
    parser.add_argument(
        "-t", "--target_loudness", 
        type=float, 
        default=None, # E.g. -23.0 for EBU R128 general broadcasting
        help="Optional target integrated loudness in LUFS. For informational comparison."
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Print full arrays of momentary and short-term LUFS values."
    )
    args = parser.parse_args()

    console = Console()

    resampled_mono_audio, sample_rate = load_audio_file(args.filepath)

    if resampled_mono_audio is None or sample_rate is None:
        console.print(f"[bold red]Error: Could not load or process audio file: {args.filepath}[/bold red]")
        return

    if args.verbose:
        console.print(f"Info: Audio loaded: {len(resampled_mono_audio)} samples at {sample_rate} Hz.")
        
    true_peak_dbtp = calculate_true_peak(resampled_mono_audio, sample_rate)
    
    k_weighted_audio = apply_k_weighting(resampled_mono_audio, sample_rate)
    if args.verbose:
        console.print(f"Info: K-weighting applied. Output data shape: {k_weighted_audio.shape}")

    momentary_lufs_values = calculate_momentary_loudness(k_weighted_audio, sample_rate)
    max_momentary_lufs = -np.inf
    if momentary_lufs_values.size > 0:
        max_momentary_lufs = np.max(momentary_lufs_values)
    
    short_term_lufs_values = calculate_short_term_loudness(k_weighted_audio, sample_rate)
    max_short_term_lufs = -np.inf
    if short_term_lufs_values.size > 0:
        max_short_term_lufs = np.max(short_term_lufs_values)
            
    integrated_lufs, momentary_lufs_for_gating = calculate_integrated_loudness(k_weighted_audio, sample_rate)
    
    loudness_range_lu = calculate_loudness_range(short_term_lufs_values, integrated_lufs)

    # --- Rich Console Output ---
    table = Table(title=f"Loudness Analysis: {args.filepath}", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=25)
    table.add_column("Value", min_width=20)

    table.add_row("Integrated Loudness", format_value(integrated_lufs, "LUFS", args.target_loudness))
    table.add_row("Loudness Range (LRA)", format_value(loudness_range_lu, "LU"))
    table.add_row("Max Momentary Loudness", format_value(max_momentary_lufs, "LUFS"))
    table.add_row("Max Short-Term Loudness", format_value(max_short_term_lufs, "LUFS"))
    table.add_row("True Peak", format_value(true_peak_dbtp, "dBTP"))

    if args.target_loudness is not None:
        table.add_row("Target Loudness", format_value(args.target_loudness, "LUFS"))
        if integrated_lufs > -np.inf: # Check if integrated_lufs is a valid number
            diff = integrated_lufs - args.target_loudness
            color = "red" if diff > 0.5 else "green" if diff < -0.5 else "default" # Adding a bit of tolerance
            status_message = f"Measured Integrated Loudness is [bold {color}]{abs(diff):.1f} LUFS {'above' if diff > 0 else 'below' if diff < 0 else 'at'}[/bold {color}] target."
            if abs(diff) <= 0.5:
                status_message = "Measured Integrated Loudness is [bold green]at target[/bold green]."
            console.print(Panel(status_message, title="Target Comparison", expand=False))


    console.print(table)

    if args.verbose:
        console.print("\n[bold]Momentary LUFS values (400ms windows, 100ms hop):[/bold]")
        console.print(np.round(momentary_lufs_values, 1).tolist()) # More readable list
        console.print("\n[bold]Short-Term LUFS values (3s windows, 1s hop):[/bold]")
        console.print(np.round(short_term_lufs_values, 1).tolist())
        console.print("\n[bold]Momentary LUFS for Gating (400ms non-overlapping blocks):[/bold]")
        console.print(np.round(momentary_lufs_for_gating,1).tolist())


    # --- CSV Output ---
    if args.output_file:
        results_data = {
            "Filepath": args.filepath, # Added for context in CSV
            "Integrated LUFS": integrated_lufs,
            "Loudness Range LU": loudness_range_lu,
            "Max Momentary LUFS": max_momentary_lufs,
            "Max Short-Term LUFS": max_short_term_lufs,
            "True Peak dBTP": true_peak_dbtp,
            "Target Loudness LUFS": args.target_loudness if args.target_loudness is not None else np.nan
        }
        # Ensure headers match the keys in results_data, including Filepath and Target
        csv_headers = ["Filepath", "Integrated LUFS", "Loudness Range LU", "Max Momentary LUFS", "Max Short-Term LUFS", "True Peak dBTP", "Target Loudness LUFS"]
        save_results_to_csv(args.output_file, results_data, csv_headers)


if __name__ == "__main__":
    main()
