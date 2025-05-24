# Audio IMD Analyzer

## Overview

This Python script, `audio_imd_analyzer.py`, is a command-line tool designed to generate a dual-tone test signal, play it through an audio output device, record it via an input device (typically in a loopback configuration or through a Device Under Test - DUT), and then analyze the recorded audio for Intermodulation Distortion (IMD). The primary analysis method implemented is based on the SMPTE RP120-1994 standard.

The tool allows users to specify various parameters for signal generation, audio device selection, and IMD analysis, providing detailed results including IMD percentage, IMD in dB, and a breakdown of individual intermodulation products.

## SMPTE RP120-1994 Standard

The SMPTE RP120-1994 standard is a common method for measuring intermodulation distortion in audio systems. It specifies a dual-tone test signal consisting of:
- **f1**: A low-frequency tone, typically **60 Hz**.
- **f2**: A high-frequency tone, typically **7000 Hz**.
- **Amplitude Ratio**: The amplitude of f1 is typically four times the amplitude of f2 (a 4:1 linear ratio, or +12 dB difference).

IMD is then calculated based on the amplitudes of the sideband frequencies (e.g., f2 ± f1, f2 ± 2*f1) relative to the amplitude of the f2 tone in the recorded signal.

## Dependencies

The script requires Python 3.8+ and the following Python libraries:
- **NumPy**: For numerical operations, especially array manipulation and FFT.
- **SoundDevice**: For audio playback and recording.
- **SciPy**: For signal processing functions, particularly FFT windowing (`scipy.signal.get_window`).
- **Rich**: For enhanced terminal output, including tables and styled text.

These dependencies can generally be installed using pip:
```bash
pip install numpy sounddevice scipy rich
```

## Usage

The script is run from the command line:
```bash
python audio_imd_analyzer.py [OPTIONS]
```

### Main Options

| Option                | Alias | Default Value      | Description                                                                                                |
|-----------------------|-------|--------------------|------------------------------------------------------------------------------------------------------------|
| `--f1`                |       | `60.0`             | Frequency of the first (low-frequency) tone in Hz.                                                         |
| `--f2`                |       | `7000.0`           | Frequency of the second (high-frequency) tone in Hz.                                                       |
| `--amplitude`         |       | `-12.0`            | Amplitude of the first tone (f1) in dBFS. The amplitude of f2 is derived from this and the `--ratio`.        |
| `--ratio`             |       | `4.0`              | Linear amplitude ratio of f1/f2 (e.g., 4 for a 4:1 ratio where f1 is stronger).                             |
| `--duration`          |       | `1.0`              | Duration of the generated test signal and recording in seconds.                                              |
| `--sample_rate`       |       | `48000`            | Sampling rate in Hz for signal generation, playback, and recording.                                        |
| `--device`            |       | Prompts user       | Integer ID of the audio device to use. If not provided, a list of available devices will be shown.         |
| `--output_channel`    | `-oc` | `R`                | Output channel for playback: 'L' (left) or 'R' (right).                                                    |
| `--input_channel`     | `-ic` | `L`                | Input channel for recording: 'L' (left) or 'R' (right).                                                    |
| `--window`            |       | `blackmanharris`   | FFT window type for analysis (e.g., `hann`, `hamming`, `blackmanharris`).                                  |
| `--num_sidebands`     |       | `3`                | Number of sideband pairs (e.g., f2 ± n*f1) to analyze for IMD calculation.                                 |
| `--output_csv`        |       | `None`             | **(Placeholder for future implementation)** Path to save IMD product details as a CSV file.                |
| `--help`              | `-h`  |                    | Show this help message and exit.                                                                           |

### Example Command

This command generates the standard SMPTE signal, plays it on the right channel of the default (or user-selected) audio device, records from the left channel, and analyzes for 3 sideband pairs using a Blackman-Harris window.

```bash
python audio_imd_analyzer.py --f1 60 --f2 7000 --amplitude -12 --ratio 4 --oc R --ic L
```
If no `--device` ID is provided, the script will list available devices and prompt for a selection.

## Output Description

The script provides the following output:
1.  **Device Selection**: If `--device` is not specified, a table of available audio devices is shown, and the user is prompted to select one.
2.  **Signal Generation Info**: Parameters used for generating the dual-tone signal and the max/min values of the generated signal.
3.  **Playback & Recording Info**: Details of the device and channels used for playback and recording, and confirmation of successful recording with shape and max/min values of the recorded audio.
4.  **IMD Analysis Results**:
    *   Parameters used for the analysis (f1, f2, window, number of sidebands).
    *   The measured amplitude of the reference tone f2 (nominal and actual detected frequency, dBFS, and linear amplitude).
    *   The overall IMD percentage and IMD in dB.
    *   A table detailing the detected intermodulation products, including:
        *   **Order (n)**: The order of the sideband (e.g., 1 for f2 ± f1, 2 for f2 ± 2*f1).
        *   **Type**: '+' for upper sidebands (f2 + n*f1), '-' for lower sidebands (f2 - n*f1).
        *   **Nom. Freq (Hz)**: The nominal (expected) frequency of the IMD product.
        *   **Act. Freq (Hz)**: The actual frequency at which the peak was detected in the spectrum.
        *   **Amplitude (Lin)**: The linear amplitude of the detected IMD product.
        *   **Level (dBr f2)**: The level of the IMD product in dB relative to the amplitude of the f2 reference tone.

**Example IMD Product Details Table:**
```
                                        IMD Product Details
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Order (n) ┃ Type   ┃ Nom. Freq (Hz)   ┃ Act. Freq (Hz)   ┃ Amplitude (Lin)   ┃ Level (dBr f2)   ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ 1         │ +      │           7060.0 │           7060.1 │          1.23e-05 │           -40.12 │
│ 1         │ -      │           6940.0 │           6939.8 │          1.19e-05 │           -40.55 │
│ 2         │ +      │           7120.0 │           7120.3 │          5.01e-07 │           -67.92 │
│ 2         │ -      │           6880.0 │           6879.9 │          4.88e-07 │           -68.21 │
└───────────┴────────┴──────────────────┴──────────────────┴───────────────────┴──────────────────┘
```
*(Note: Actual values will vary based on the audio interface and loopback/DUT characteristics.)*

## Important Notes

-   **Loopback Configuration / Device Under Test (DUT)**: For accurate self-testing of an audio interface, a physical loopback connection is typically required (connecting the output of the interface to its own input). Alternatively, the script can be used to test an external DUT by routing the output signal through the DUT and then into the input of the audio interface.
-   **Audio Interface Quality**: The quality of the audio interface (sound card) used for playback and recording significantly impacts the IMD results. High-quality interfaces with low noise and distortion are crucial for obtaining meaningful measurements. The script itself does not introduce IMD; any measured distortion comes from the audio hardware path.
-   **Error Handling**: The script includes error handling for device selection, signal generation issues, and audio stream problems. Error messages are printed to `stderr`.
-   **Amplitude Calibration**: The script operates with digital signal levels (dBFS). For absolute acoustic measurements, calibration of input and output levels would be necessary.
```
