# Wow and Flutter Analyzer

## Overview

The Wow and Flutter Analyzer is a Python tool designed to measure speed variations in audio playback systems. It analyzes an audio recording of a stable test tone (typically 3.15 kHz or 3 kHz) to quantify wow and flutter, which are undesirable pitch variations.

This tool is primarily used for assessing the performance of analog audio playback equipment such as:
*   Turntables (vinyl record players)
*   Tape decks (cassette, reel-to-reel)
*   Other analog recording and playback devices

By analyzing the frequency variations of the recorded test tone, the script can determine the extent of these speed irregularities.

## Features

*   **Calculates Peak and RMS Wow and Flutter**: Provides both peak and root mean square (RMS) values for wow and flutter, offering a comprehensive view of the speed variations.
*   **Separate Wow and Flutter Components**: Distinguishes between:
    *   **Wow**: Slow-frequency variations (e.g., 0.5 Hz - 6 Hz)
    *   **Flutter**: Higher-frequency variations (e.g., 6 Hz - 200 Hz)
*   **Frequency Drift Calculation**: Measures the overall drift of the playback speed compared to the reference frequency over the duration of the recording.
*   **DIN Weighting**: Offers an option to apply DIN weighting (a common standard for wow and flutter measurement) to the frequency deviation signal. Unweighted analysis is also supported.
*   **Console Output**: Displays a clear, tabular summary of all calculated metrics (Wow, Flutter, Combined W&F, Drift) directly in the console.
*   **Graphical Plots**: Generates two key visualizations:
    *   **Frequency Deviation over Time**: A plot showing the instantaneous frequency deviation from the reference. This includes subplots detailing the isolated wow and flutter components over time. Both Hz and % deviation are shown.
    *   **Spectrum of Frequency Deviation**: A plot of the power spectral density of the frequency deviation signal. This helps visualize the frequencies at which wow and flutter are most prominent.
*   **Configurable Analysis**: Allows users to specify:
    *   The input audio file.
    *   The precise reference frequency of the test tone.
    *   Frequency ranges for defining wow and flutter.
    *   An output directory for saving generated plots. If not specified, plots are displayed interactively.

## Dependencies

The tool requires the following Python libraries:

*   `numpy`: For numerical operations, especially array manipulation.
*   `scipy`: For scientific computing, including signal processing functions (filters, Hilbert transform, spectral analysis).
*   `soundfile`: For reading audio files.
*   `matplotlib`: For generating plots.
*   `rich`: For formatted console output (tables, styled text).

You can install these dependencies using pip:

```bash
pip install numpy scipy soundfile matplotlib rich
```

Alternatively, if a `requirements.txt` file is provided with the tool:

```bash
pip install -r requirements.txt
```

## Usage

The script is run from the command line.

### Basic Syntax:

```bash
python wow_flutter_analyzer/wow_flutter_analyzer.py <input_file> [options]
```

### Command-Line Options:

*   `input_file`: (Required) Path to the input audio file (e.g., a WAV file of a recorded test tone).
*   `--ref_freq HZ`: Reference frequency of the test tone in Hz. Default: `3150.0`.
*   `--weighting TYPE`: Type of weighting filter to use. Choices: `unweighted`, `din`. Default: `din`.
*   `--min_wow_freq HZ`: Minimum frequency for the wow component in Hz. Default: `0.5`.
*   `--max_wow_freq HZ`: Maximum frequency for the wow component in Hz. Default: `6.0`.
*   `--min_flutter_freq HZ`: Minimum frequency for the flutter component in Hz. Default: `6.0`.
*   `--max_flutter_freq HZ`: Maximum frequency for the flutter component in Hz. Default: `200.0`.
*   `--output_dir DIR`: Directory to save plots and any other results. If not provided, plots are displayed interactively on screen.

### Example:

```bash
python wow_flutter_analyzer/wow_flutter_analyzer.py tests/test_tones/3150Hz_wow_flutter_example.wav --ref_freq 3150 --output_dir ./analysis_results
```
*(Note: The example audio file path `tests/test_tones/3150Hz_wow_flutter_example.wav` is illustrative and may not exist in your setup.)*

## Output

The tool provides results in two forms:

### 1. Console Output

Numerical results are printed to the console in a formatted table. This includes:
*   **Wow**: Peak and RMS values (in Hz and % deviation from reference frequency).
*   **Flutter**: Peak and RMS values (in Hz and % deviation).
*   **Combined Wow & Flutter**: Peak and RMS values (in Hz and %, indicating whether weighted or unweighted).
*   **Drift**:
    *   Average frequency of the recording.
    *   Overall drift compared to the reference frequency (Hz and %).
    *   Initial and final frequency of the recording (average over ~0.5s).
    *   Drift observed over the measurement duration (final - initial, in Hz and %).

### 2. Plots

If an `--output_dir` is specified, plots are saved as PNG files in that directory. Otherwise, they are displayed on screen.

*   **Frequency Deviation over Time (`frequency_deviation_plot.png`)**:
    *   The main plot shows the overall frequency deviation of the test tone over the duration of the recording. The Y-axis shows deviation in Hz, with a secondary Y-axis showing deviation in percent.
    *   A second subplot breaks down this deviation into its constituent Wow and Flutter components, plotted over time. This helps in observing the nature and magnitude of these variations.

*   **Spectrum of Frequency Deviation (`deviation_spectrum_plot.png`)**:
    *   This plot displays the power spectral density (PSD) of the (combined) frequency deviation signal.
    *   The X-axis is frequency (logarithmic scale), and the Y-axis is power (in dB/Hz).
    *   This spectrum is crucial for identifying the specific frequencies at which wow and flutter occur. For example, a peak at 0.55 Hz in this spectrum would indicate wow related to the turntable's rotation speed (33.3 RPM / 60 = ~0.55 Hz). The plot highlights the defined wow and flutter frequency bands.

## Test Tones

For accurate measurements, a high-quality recording of a stable frequency test tone is essential.
*   **Frequency**: Commonly 3000 Hz or 3150 Hz. The chosen frequency should be accurately known and specified with `--ref_freq`.
*   **Stability**: The tone itself should be free of significant intrinsic frequency variations.
*   **Recording Quality**: The recording chain (microphone, preamp, ADC if applicable) should be of good quality. The playback system being tested is the primary variable of interest.
*   **Sources**:
    *   **Software Generation**: Test tones can be generated as audio files using software (e.g., Audacity, REW, or custom scripts).
    *   **Test Records/Tapes**: Specialized test LPs or tapes provide pre-recorded test tones. Ensure these are clean and in good condition.

The quality and stability of the test tone and the recording process directly impact the accuracy of the wow and flutter analysis.

## License

This software is released under the **Unlicense**.

This means it is free and unencumbered software released into the public domain. Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means. For more information, please refer to <http://unlicense.org/>.
