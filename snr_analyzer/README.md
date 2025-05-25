# SNR Analyzer

## Overview

The SNR Analyzer is a command-line tool designed to measure the Signal-to-Noise Ratio (SNR) of an audio signal path. It works by playing a test signal (a generated sine wave) through a specified output audio device and recording it through a specified input audio device. It then records the noise floor from the same input device. From these recordings, it calculates the RMS values of the signal (estimated) and the noise, and then computes the SNR in decibels (dB).

## Features

-   Lists available audio input and output devices.
-   Generates a sine wave test signal of specified frequency and amplitude.
-   Plays the test signal and records it (signal + noise).
-   Records the ambient noise floor.
-   Calculates SNR using the formula: `SNR (dB) = 20 * log10(RMS_signal_only / RMS_noise)`, where `RMS_signal_only` is estimated from `sqrt(max(0, RMS_signal_plus_noise^2 - RMS_noise^2))`.
-   Displays results (SNR, RMS Signal, RMS Noise) in a formatted table on the console.
-   Allows user to specify audio devices, channels, signal parameters, and recording durations.

## Dependencies

-   Python 3.8+
-   **Libraries:**
    -   `numpy`: For numerical operations and array manipulation.
    -   `sounddevice`: For audio playback and recording.
    -   `scipy`: (Used by sounddevice, potentially for signal processing features if extended).
    -   `argparse`: For parsing command-line arguments.
    -   `rich`: For rich text and formatted table display in the console.

### Installation of Dependencies

1.  **Python:** Ensure Python 3.8 or newer is installed.
2.  **System-level audio libraries (for `sounddevice`):**
    On Debian/Ubuntu Linux, `sounddevice` requires PortAudio. Install it using:
    ```bash
    sudo apt-get update
    sudo apt-get install libportaudio2
    ```
    For other operating systems (Windows, macOS), PortAudio is often bundled or installed differently. Refer to the `sounddevice` documentation if you encounter issues.
3.  **Python libraries:**
    Install the required Python libraries using pip:
    ```bash
    pip install numpy sounddevice scipy argparse rich
    ```
    It's recommended to use a virtual environment.

## Usage

The tool is run from the command line.

**1. List Available Audio Devices:**
   To see a list of available audio devices and their IDs:
   ```bash
   python snr_analyzer/snr_analyzer.py --list_devices
   ```

**2. Run SNR Measurement:**
   Specify the output and input device IDs, and optionally other parameters.
   ```bash
   python snr_analyzer/snr_analyzer.py --output_device <OUT_ID> --input_device <IN_ID> [options]
   ```

**Command-Line Options:**

*   `--list_devices`: (Optional) List available audio devices and exit.
*   `--output_device OUTPUT_DEVICE`: (Required unless listing devices) Integer ID of the output audio device.
*   `--input_device INPUT_DEVICE`: (Required unless listing devices) Integer ID of the input audio device.
*   `--output_channel OUTPUT_CHANNEL`: (Optional) Output channel number (1-based). Default: 1.
*   `--input_channel INPUT_CHANNEL`: (Optional) Input channel number (1-based). Default: 1.
*   `--samplerate SAMPLERATE`: (Optional) Samplerate in Hz (e.g., 44100, 48000). Default: 48000.
*   `--frequency FREQUENCY`: (Optional) Frequency of the test sine wave in Hz. Default: 1000.0.
*   `--amplitude AMPLITUDE`: (Optional) Amplitude of the test sine wave (0.0 to 1.0). Default: 0.8.
*   `--signal_duration SIGNAL_DURATION`: (Optional) Duration of signal playback/recording in seconds. Default: 5.0.
*   `--noise_duration NOISE_DURATION`: (Optional) Duration of noise floor recording in seconds. Default: 5.0.
*   `--help`: Show this help message and exit.

**Example Command:**

```bash
python snr_analyzer/snr_analyzer.py --output_device 1 --input_device 3 --frequency 1000 --amplitude 0.7 --signal_duration 3 --noise_duration 3
```

## Output Example

The tool will print the results in a table:

```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric          ┃ Value         ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ RMS Signal      │ 0.070 Vrms    │
│ RMS Noise       │ 0.001 Vrms    │
│ SNR             │ 36.88 dB      │
└─────────────────┴───────────────┘
```
*(Note: RMS values are illustrative and depend on calibration and actual signal levels. "Vrms" unit is indicative if the system were calibrated; otherwise, it's a relative unitless value based on digital signal levels.)*

## Important Notes

*   **Audio Loopback:** For accurate self-testing of an audio interface, you'll typically need to physically connect the output of the interface to its input (a loopback connection). Ensure levels are appropriate to avoid clipping and damage.
*   **Device Selection:** Use the `--list_devices` option to identify the correct device IDs for your system. Device IDs can change depending on the operating system and connected hardware.
*   **Channel Selection:** Ensure the selected channels for output and input are correct for your setup. For mono signals on stereo devices, typically channel 1 (left) or 2 (right) is used.
*   **Noise Floor:** For a good noise measurement, ensure the environment is quiet and the input chain is not introducing excessive noise. The measurement captures the noise of the entire chain (output device -> cable -> input device -> system noise).
*   **Amplitude:** Start with a moderate amplitude (e.g., 0.5) and ensure it does not cause clipping on the output or input.

## License

This tool is released under the Unlicense. See the main repository for details. (The code is public domain).
