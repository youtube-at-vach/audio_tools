# Audio Phase Analyzer

## Overview and Purpose

The Audio Phase Analyzer is a command-line tool designed to measure and visualize the phase difference between two channels of a stereo audio signal. It generates a mono sine wave test tone, plays it through specified output channels, simultaneously records from specified input channels, and then calculates the phase difference between the recorded stereo channels.

This tool can be useful for:
- **Checking speaker wiring polarity:** Ensuring speakers are wired correctly (in-phase or out-of-phase).
- **Verifying stereo equipment phase integrity:** Checking if audio interfaces, mixers, or other stereo equipment maintain phase coherence.
- **Analyzing effects of audio processing:** Understanding how certain audio plugins or processing chains affect the phase relationship of a stereo signal.
- **Educational purposes:** Demonstrating concepts of phase, stereo imaging, and Lissajous figures.

## Features

- Measures phase difference in degrees (between -180° and +180°).
- Command-line interface for easy operation and scripting.
- User-selectable test tone parameters:
    - Frequency (`--frequency`)
    - Duration (`--duration`)
    - Amplitude (`--amplitude`)
- User-selectable audio input/output devices by ID or name (`--input_device`, `--output_device`).
- User-configurable mapping for stereo output and input channels (`--output_channels`, `--input_channels`).
- Optional Lissajous figure plotting for visual phase analysis (`--plot`).
- Clear and informative console output using the `rich` library for tables and styled text.
- Ability to list available audio devices and their properties (`--list_devices`).

## Dependencies

This tool requires Python 3 and the following Python libraries:
- `numpy`
- `sounddevice`
- `scipy`
- `rich`
- `matplotlib` (for plotting)

You can install these dependencies using pip:
```bash
pip install numpy sounddevice scipy rich matplotlib
```

**System Dependencies:**
- `sounddevice` relies on the PortAudio library. On Debian-based Linux systems (like Ubuntu), you can install it with:
  ```bash
  sudo apt-get update
  sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev
  ```
  For other operating systems, PortAudio might be bundled or require separate installation. Please refer to the `sounddevice` documentation for more details.

## Usage (Command-Line Options)

The script is typically run as a module from the parent directory of `audio_phase_analyzer`.

**Basic Usage Example:**
```bash
python -m audio_phase_analyzer.audio_phase_analyzer --frequency 1000 --duration 2
```
This will run the analyzer with a 1000 Hz test tone for 2 seconds using default devices and channel mappings.

**Command-Line Options:**

The script accepts various arguments to customize its behavior:

| Argument                      | Default               | Description                                                                                                |
|-------------------------------|-----------------------|------------------------------------------------------------------------------------------------------------|
| `--list_devices`              | N/A (action)          | List available audio devices and their details, then exit.                                                 |
| `-f, --frequency HZ`          | `1000.0` Hz           | Test frequency in Hz for the sine wave.                                                                    |
| `-d, --duration SECONDS`      | `2.0` s               | Duration of the test signal in seconds.                                                                    |
| `-sr, --samplerate RATE`      | `48000` Hz            | Sample rate in Hz.                                                                                         |
| `-a, --amplitude DBFS`        | `-6.0` dBFS           | Amplitude of the test tone in dBFS (0 dBFS = 1.0 linear).                                                  |
| `--input_device ID_OR_NAME`   | System Default        | Input device ID (integer) or name (string).                                                                |
| `--output_device ID_OR_NAME`  | System Default        | Output device ID (integer) or name (string).                                                               |
    | `--output_channels CHANNELS`  | `"1,2"`               | Comma-separated physical output channels (1-based) for the stereo signal (e.g., "1,2"). **Note: Due to `sounddevice` API changes (e.g., v0.5.2), explicit output channel mapping for `sd.playrec` is not directly supported. The signal will be played on the default output channels of the selected device, and this argument primarily serves for documentation/future compatibility.** |
| `--input_channels CHANNELS`   | `"1,2"`               | Comma-separated physical input channels (1-based) to record from (e.g., "1,2"). These are used for phase comparison. |
| `--plot`                      | N/A (action)          | Display a Lissajous figure (X-Y plot) of the recorded stereo channels using matplotlib.                      |

**Important Note on `sounddevice` API:**
This tool has been updated to be compatible with `sounddevice` versions where `sd.check_hostapi()` and `mapping` arguments for `sd.playrec`/`sd.rec` are deprecated or removed (e.g., v0.5.2 and later). If you encounter issues, ensure your `sounddevice` library is up-to-date and refer to its official documentation for API changes. Specifically, direct physical output channel mapping via `output_mapping` in `sd.playrec` is no longer supported in these versions; the signal will be routed to the device's default output channels.

## Example Output

When run, the tool first displays a table of the measurement parameters being used:
```
Measurement Parameters
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Parameter                    ┃ Value                                                ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Test Frequency               │ 1000.0 Hz                                            │
│ Duration                     │ 2.00 s                                               │
│ Sample Rate                  │ 48000 Hz                                             │
│ Amplitude                    │ -6.0 dBFS                                            │
│ Output Device                │ System Default                                       │
│ Input Device                 │ System Default                                       │
│ Output Channels (1-based)    │ [1, 2]                                               │
│ Input Channels (1-based)     │ 1 (Ref), 2 (DUT)                                     │
└──────────────────────────────┴──────────────────────────────────────────────────────┘
------------------------------
```
(Note: Device names will vary based on your system.)

After playback and recording, it shows the calculated phase difference in a panel:
```
Phase Analysis Result
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Calculated Phase Difference: 0.00°                                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
Subtitle: Input Ch 2 relative to Ch 1 @ 1000.0 Hz
```
The color of the phase value changes based on its value (green for near 0°, yellow for moderate phase, red for near +/-180°).

**Lissajous Figure:**
If `--plot` is specified, a matplotlib window will appear showing the Lissajous figure. This is an X-Y plot where the X-axis is the amplitude of the first input channel and the Y-axis is the amplitude of the second input channel for a short segment of the recorded audio.

## Interpreting Results

**Phase Difference (Degrees):**
The primary output is the phase difference between the two selected input channels, measured in degrees. The value represents the phase of the *first* input channel relative to the *second* input channel (as per the `--input_channels` argument, e.g., Ch1 relative to Ch2).
*   **0°:** The signals are perfectly in-phase. For audio, this typically means correct wiring and positive polarity for both channels.
*   **+/-180°:** The signals are perfectly out-of-phase (polarity inverted on one channel). This can lead to signal cancellation, especially for mono content.
*   **+/-90°:** The signals are in quadrature. One signal leads or lags the other by a quarter of a cycle.
*   **Other values:** Indicate varying degrees of phase shift. Small deviations from 0° might be normal, but large deviations in systems expected to be phase-coherent could indicate issues.

**Lissajous Figures (when `--plot` is used):**
Lissajous figures provide a visual representation of the phase relationship:
*   **Diagonal Line (positive slope, bottom-left to top-right):** Indicates 0° phase difference (in-phase).
*   **Diagonal Line (negative slope, top-left to bottom-right):** Indicates +/-180° phase difference (out-of-phase).
*   **Circle:** Indicates +/-90° phase difference. The direction of "rotation" (if viewed as an animation, which this plot is not) would distinguish between +90° and -90°.
*   **Ellipse:** Indicates other phase differences. The shape and orientation of the ellipse depend on the specific phase angle and amplitude relationship.

## License

This project is licensed under the Unlicense. This software is provided 'as-is', without any express or implied warranty.
