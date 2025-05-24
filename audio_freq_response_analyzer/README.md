# Audio Frequency Response Analyzer

## Overview

This script, `audio_freq_response_analyzer.py`, is a command-line tool for measuring the frequency response (amplitude and phase) of audio devices or systems. It works by generating a series of sine wave tones at logarithmically spaced frequencies, playing them through a selected output audio device channel, recording via a selected input audio device channel (typically in a loopback configuration or through a Device Under Test - DUT), and then analyzing each recorded segment to determine the amplitude and phase at the target frequency.

The tool provides options for specifying the frequency range, points per octave, signal amplitude, duration of each test tone, audio devices, channels, and FFT analysis parameters. Results can be displayed in a table, saved to a CSV file, and plotted as graphs.

## Dependencies

The script requires Python 3.8+ and the following Python libraries:
- **NumPy**: For numerical operations, array manipulation, and FFT.
- **SoundDevice**: For audio playback and recording.
- **SciPy**: For signal processing functions, particularly FFT windowing (`scipy.signal.get_window`) and phase unwrapping (`numpy.unwrap` is used, which is part of NumPy but often associated with SciPy's signal processing context).
- **Rich**: For enhanced terminal output (tables, styled text, prompts).
- **Matplotlib**: For generating plots of the frequency response.

These dependencies can generally be installed using pip:
```bash
pip install numpy sounddevice scipy rich matplotlib
```

## Usage

The script is run from the command line, typically from the repository root:
```bash
python audio_freq_response_analyzer/audio_freq_response_analyzer.py [OPTIONS]
```

### Main Options

| Option                | Alias | Default Value      | Description                                                                                                |
|-----------------------|-------|--------------------|------------------------------------------------------------------------------------------------------------|
| `--start_freq`        |       | `20.0`             | Start frequency for the sweep in Hz.                                                                       |
| `--end_freq`          |       | `20000.0`          | End frequency for the sweep in Hz.                                                                         |
| `--points_per_octave` |       | `12`               | Number of frequency points to measure per octave.                                                          |
| `--amplitude`         |       | `-20.0`            | Amplitude of the generated test tones in dBFS.                                                             |
| `--duration_per_step` |       | `0.2`              | Duration of each individual sine wave tone segment in seconds.                                               |
| `--device`            |       | Prompts user       | Integer ID of the audio device to use for both playback and recording. Prompts if not provided.            |
| `--output_channel`    | `-oc` | `R`                | Output channel for playback: 'L' (left) or 'R' (right).                                                    |
| `--input_channel`     | `-ic` | `L`                | Input channel for recording: 'L' (left) or 'R' (right).                                                    |
| `--sample_rate`       |       | `48000`            | Sampling rate in Hz for signal generation, playback, and recording.                                        |
| `--window`            |       | `hann`             | FFT window type for analysis (e.g., `hann`, `hamming`, `blackmanharris`).                                  |
| `--output_csv`        |       | `None`             | Optional. Filename to save the frequency response results as a CSV file (e.g., `response_data.csv`).       |
| `--output_plot_amp`   |       | `None`             | Optional. Filename to save the amplitude response plot as an image (e.g., `amp_response.png`).             |
| `--output_plot_phase` |       | `None`             | Optional. Filename to save the phase response plot as an image (e.g., `phase_response.png`).               |
| `--no_plot_display`   |       | `False` (StoreTrue)| Suppress displaying plots interactively using Matplotlib's `plt.show()`. Plots are still saved if filenames are given. |
| `--help`              | `-h`  |                    | Show this help message and exit.                                                                           |

### Example Command

Measure frequency response from 20 Hz to 20 kHz, 3 points per octave, -20 dBFS amplitude, using default device (will prompt), output channel R, input channel L, and save results to CSV and plots:
```bash
python audio_freq_response_analyzer/audio_freq_response_analyzer.py \
    --points_per_octave 3 \
    --output_csv freq_response.csv \
    --output_plot_amp amp_plot.png \
    --output_plot_phase phase_plot.png
```

## Output Description

1.  **Device Selection**: If `--device` is not specified, a table of available audio devices is displayed, and the user is prompted to select one.
2.  **Progress Indication**: For each frequency step, a message like "Measuring at X.XX Hz..." is printed. If a step fails (e.g., no data recorded), a warning is shown.
3.  **Summary Table (Sample)**: After the measurement loop, a sample table (first 5 points) of the results is printed to the console, showing:
    *   Target Frequency (Hz)
    *   Actual Detected Frequency (Hz)
    *   Amplitude (dBFS)
    *   Raw Phase (degrees)
    *   Unwrapped Phase (degrees)
4.  **CSV Output**: If `--output_csv` is specified, a CSV file is created with columns: `Target Frequency (Hz)`, `Actual Frequency (Hz)`, `Amplitude (dBFS)`, `Phase (degrees)` (uses unwrapped phase).
5.  **Plot Output**:
    *   If `--output_plot_amp` is specified, an image file with the amplitude response plot (dBFS vs. Frequency on a log scale) is saved.
    *   If `--output_plot_phase` is specified, an image file with the phase response plot (unwrapped degrees vs. Frequency on a log scale) is saved.
    *   If plots are generated and `--no_plot_display` is NOT used, Matplotlib will attempt to display the plots interactively.

## Important Notes

-   **Loopback Configuration**: For accurate self-testing of an audio interface's frequency response, a physical or virtual loopback connection is essential (output connected to input). For testing a DUT, the signal path should go from the interface output, through the DUT, and back to the interface input.
-   **Audio Interface Quality**: The characteristics of the audio interface used will directly influence the measurement results.
-   **Signal Level**: The `--amplitude` should be set to a level that does not overload the DUT or the audio interface inputs.
