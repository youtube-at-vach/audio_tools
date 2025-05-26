# SNR Analyzer

## Overview
The SNR Analyzer is a command-line tool designed to measure the Signal-to-Noise Ratio (SNR) of an audio system or component. It works by playing a known test signal and recording it, then recording the noise floor, and finally calculating the ratio of the signal power to the noise power.

## Dependencies
- Python 3.8+
- numpy
- sounddevice
- scipy
- rich
- argparse

These Python libraries can be installed using pip:
```bash
pip install numpy sounddevice scipy rich argparse
```
The `sounddevice` library may also require system-level audio libraries such as PortAudio. On Debian/Ubuntu, this can be installed with:
```bash
sudo apt-get install libportaudio2
```

## Usage
The tool is run from the command line.

**1. List Available Audio Devices:**
To see a list of available audio input and output devices with their IDs:
```bash
python -m snr_analyzer.snr_analyzer --list_devices
```

**2. Measure SNR:**
```bash
python -m snr_analyzer.snr_analyzer --output_device <OUTPUT_DEVICE_ID> --input_device <INPUT_DEVICE_ID> [OPTIONS]
```

**Command-Line Options:**
- `--list_devices`: (Optional) Action flag. Lists available audio devices and their IDs, then exits.
- `--output_device <ID>`: (Required for measurement) Integer ID of the audio output device for playing the test signal.
- `--input_device <ID>`: (Required for measurement) Integer ID of the audio input device for recording.
- `--output_channel <NUM>`: (Optional) Output channel number (1-based). Default: 1.
- `--input_channel <NUM>`: (Optional) Input channel number (1-based). Default: 1.
- `--samplerate <RATE>`: (Optional) Samplerate in Hz (e.g., 44100, 48000). Default: 48000.
- `--frequency <FREQ>`: (Optional) Frequency of the generated test sine wave in Hz. Default: 1000.0.
- `--amplitude <AMP>`: (Optional) Amplitude of the test sine wave (0.0 to 1.0). Default: 0.8.
- `--signal_duration <SEC>`: (Optional) Duration of signal playback/recording in seconds. Default: 5.0.
- `--noise_duration <SEC>`: (Optional) Duration of noise floor recording in seconds. Default: 5.0.

**Example:**
```bash
python -m snr_analyzer.snr_analyzer --output_device 2 --input_device 4 --frequency 1000 --signal_duration 3 --noise_duration 3
```

## How it Works
1.  A sine wave of the specified frequency and amplitude is generated.
2.  This signal is played through the selected output device/channel and simultaneously recorded via the selected input device/channel. This recording captures the "Signal + Noise".
3.  A second recording is made from the input device/channel with no signal being played. This captures the "Noise Floor".
4.  The RMS (Root Mean Square) power of the "Signal + Noise" and "Noise Floor" recordings are calculated.
5.  The power of the signal alone is estimated by subtracting the noise power from the "Signal + Noise" power.
6.  The SNR is then calculated as `20 * log10(RMS_signal_only / RMS_noise)` and displayed in decibels (dB).

## Output
The tool will print a table to the console with the following results:
- RMS of Signal (estimated, from Signal+Noise - Noise)
- RMS of Noise
- SNR (dB)

Example:
```
 SNR Measurement Results
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric                       ┃ Value         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ RMS Signal (Estimated)       │ 0.707 Vrms    │
│ RMS Noise                    │ 0.001 Vrms    │
│ SNR                          │ 56.9897 dB    │
└──────────────────────────────┴───────────────┘
```
*(Note: Vrms units are nominal if system is not calibrated)*

## Important Notes
- **Audio Loopback:** For testing an audio interface or system, you'll typically need to create a physical or virtual audio loopback from the output of the device to its input.
- **Clean Noise Recording:** Ensure a quiet environment and no unexpected audio signals when the noise floor is being recorded for accurate results.
- **Device Selection:** Use the `--list_devices` option to correctly identify the device IDs for your system.
- **Input/Output Levels:** Adjust your system's input and output levels appropriately. Clipping the signal will invalidate results. The generated signal has an amplitude between 0.0 and 1.0 (peak).

## License
This tool is released under the Unlicense. See the main repository license for more details.
```
