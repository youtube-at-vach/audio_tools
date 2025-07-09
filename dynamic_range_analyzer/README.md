# Unweighted Dynamic Range Analyzer

## Overview

This tool measures the unweighted dynamic range (DR) of an audio device. It works by playing a low-level test tone (-60 dBFS at 997 Hz), recording the output of the device via a loopback, and then measuring the level of the residual noise floor after the test tone has been filtered out.

Dynamic range is a fundamental measure of an audio device's performance, representing the ratio between the loudest possible undistorted signal and the quietest signal (the noise floor).

**Note:** This tool measures **unweighted** dynamic range. It does not apply A-weighting or any other frequency weighting curve.

## Dependencies

This tool relies on common libraries listed in the main project's `requirements.txt` file.

- `numpy`
- `sounddevice`
- `scipy`
- `rich`

Ensure you have installed them before running the tool:
```bash
# Navigate to the repository root
cd ..

# Install common dependencies
pip install -r requirements.txt
```

## Usage

1.  **Connect your hardware**: Set up an audio loopback on your audio interface, connecting an output channel to an input channel.
2.  **List devices**: Run the tool with `--list_devices` to see the available audio devices and their IDs.
    ```bash
    python3 dynamic_range_analyzer.py --list_devices
    ```
3.  **Run the measurement**: Execute the script with the device ID and the chosen input/output channels.
    ```bash
    # Example: Use device ID 3, with output 1 and input 1
    python3 dynamic_range_analyzer.py --device 3 --output_channel 1 --input_channel 1
    ```

### Command-Line Arguments

-   `--list_devices`: (Flag) List available audio devices and exit.
-   `--device ID`: (Required) The integer ID of the audio device to use.
-   `--output_channel CH`: (Optional) 1-based index of the output channel. Defaults to `1`.
-   `--input_channel CH`: (Optional) 1-based index of the input channel. Defaults to `1`.
-   `--samplerate RATE`: (Optional) Samplerate in Hz. Defaults to `48000`.
-   `--duration SEC`: (Optional) Duration of the test signal in seconds. Defaults to `5.0`.

## Example Output

```
 Unweighted Dynamic Range Measurement Results
+------------------------------+----------------+
| Metric                       | Value          |
|------------------------------+----------------|
| Dynamic Range (Unweighted)   | 98.55 dB       |
| Noise Level (Unweighted)     | -98.55 dBFS    |
| RMS Noise (Unweighted)       | 1.18e-05       |
+------------------------------+----------------+
```
