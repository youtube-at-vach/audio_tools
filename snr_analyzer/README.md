# SNR Analyzer

## Overview
The SNR Analyzer is a command-line tool designed to measure the Signal-to-Noise Ratio (SNR) of an audio system or component. It works by playing a known test signal and recording it, then recording the noise floor, and finally calculating the ratio of the signal power to the noise power. It uses a single audio device for both input and output, targeting specific channels ('L' or 'R') on that device.

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
python -m snr_analyzer.snr_analyzer --device <DEVICE_ID> [OPTIONS]
```

**Command-Line Options:**
- `--list_devices`: (Optional) Action flag. Lists available audio devices and their IDs, then exits.
- `--device <ID>`: (Required for measurement) Integer ID of the audio device to be used for both input and output.
- `--output_channel <CHAN>`: (Optional) Output channel ('L' or 'R'). Default: 'R'.
- `--input_channel <CHAN>`: (Optional) Input channel ('L' or 'R'). Default: 'L'.
- `--samplerate <RATE>`: (Optional) Samplerate in Hz (e.g., 44100, 48000). Default: 48000.
- `--frequency <FREQ>`: (Optional) Frequency of the generated test sine wave in Hz. Default: 1000.0.
- `--amplitude <AMP>`: (Optional) Amplitude of the test sine wave (0.0 to 1.0). Default: 0.8.
- `--signal_duration <SEC>`: (Optional) Duration of signal playback/recording in seconds. Default: 5.0.
- `--noise_duration <SEC>`: (Optional) Duration of noise floor recording in seconds. Default: 5.0.

**Example:**
```bash
python -m snr_analyzer.snr_analyzer --device 3 --output_channel R --input_channel L --frequency 1000 --signal_duration 3 --noise_duration 3
```

## How it Works
1.  A sine wave of the specified frequency and amplitude is generated.
2.  This signal is played through the selected channel ('L' or 'R') of the specified output device and simultaneously recorded via the selected channel ('L' or 'R') of the same input device. This recording captures the "Signal + Noise".
3.  A second recording is made from the selected input channel of the device with no signal being played. This captures the "Noise Floor".
4.  The RMS (Root Mean Square) power of the "Signal + Noise" and "Noise Floor" recordings are calculated.
5.  The power of the signal alone is estimated by subtracting the measured noise power from the measured "Signal + Noise" power (i.e., Signal Power = Total Power - Noise Power). This common estimation method assumes that the signal and noise components are uncorrelated and that the statistical characteristics of the noise are consistent (stationary) during both measurement phases.
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
- **Audio Loopback:** For testing an audio interface or system, you'll typically need to create a physical or virtual audio loopback from the output of the device to its input (e.g., Right Output Channel to Left Input Channel).
- **Clean Noise Recording:** Ensure a quiet environment and no unexpected audio signals when the noise floor is being recorded for accurate results.
- **Device Selection:** Use the `--list_devices` option to correctly identify the device ID for your system. Ensure the chosen device supports both input and output, and has at least two channels if using 'L' and 'R' distinctly.
- **Input/Output Levels:** Adjust your system's input and output levels appropriately. Clipping the signal will invalidate results. The generated signal has an amplitude between 0.0 and 1.0 (peak).
- **Low Recorded Signal:** If the tool warns about a very low recorded signal level (i.e., the "signal + noise" measurement is unexpectedly quiet compared to the generated signal amplitude), double-check your audio interface's input/output gains, physical connections, and selected loopback configuration. The test signal may not be reaching the input correctly or could be heavily attenuated.
- **Estimation Accuracy:** The accuracy of the SNR, especially at very low values, depends on the validity of the assumptions mentioned in "How it Works" (uncorrelated signal/noise, stationary noise). Significant changes in background noise between the 'signal+noise' and 'noise' measurements can affect the result.

## License
This tool is released under the Unlicense. See the main repository license for more details.
```
