# Audio Crosstalk Analyzer

## Overview

The `audio_crosstalk_analyzer.py` script is a command-line tool designed to measure audio crosstalk between channels of an audio device. Crosstalk, also known as channel separation, is the undesired leakage of a signal from one channel into another. This tool plays a test signal (a sine wave) on a specified output channel and simultaneously records from multiple input channels. It then analyzes the recorded signals to determine the level of the test frequency on the driven (reference) input channel and its leakage into other (undriven) input channels.

This tool can be useful for:
-   Testing the performance of audio interfaces (sound cards).
-   Evaluating the quality of audio cables and connectors.
-   Assessing the electrical isolation between channels in a device under test (DUT).

## Features

-   **Test Modes**: Supports both single frequency tests and frequency sweep tests.
-   **Adjustable Signal Parameters**: Users can define the frequency (for single mode), frequency range and density (for sweep mode), and amplitude (dBFS) of the test signal.
-   **Flexible Channel Selection**: Allows specification of the output channel and multiple input channels for recording. The first specified input channel serves as the reference (driven channel).
-   **Informative Console Output**: Results are displayed in a clear, tabular format in the console using Rich.
-   **CSV Export**: Test results can be saved to a CSV file for further analysis or record-keeping.
-   **Plot Generation**: For frequency sweep tests, the script can generate and save a plot of crosstalk (dB) versus frequency (logarithmic scale) using Matplotlib. Plot display can also be suppressed.

## Dependencies

The script requires Python 3.8+ and the following Python libraries:
-   **NumPy**: For numerical operations, especially array manipulation and FFT.
-   **SoundDevice**: For audio playback and recording via PortAudio.
-   **SciPy**: For signal processing functions, particularly FFT windowing (`scipy.signal.get_window`).
-   **Rich**: For enhanced terminal output, including tables and styled text.
-   **Matplotlib**: For generating plots of crosstalk versus frequency (used in sweep mode).

These dependencies can be installed using pip:
```bash
pip install numpy sounddevice scipy rich matplotlib
```

## Usage

The script is run from the command line:
```bash
python audio_crosstalk_analyzer/audio_crosstalk_analyzer.py [OPTIONS]
```

## Main Options

| Option                      | Alias | Default Value    | Description                                                                                                                               |
|-----------------------------|-------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `--frequency HZ`            |       | `1000.0`         | Frequency for single mode test (Hz). Must be positive.                                                                                    |
| `--sweep`                   |       | `False`          | Enable sweep mode. Overrides `--frequency`.                                                                                               |
| `--start-frequency HZ`    |       | `20.0`           | Start frequency for sweep mode (Hz). Must be positive.                                                                                    |
| `--end-frequency HZ`      |       | `20000.0`        | End frequency for sweep mode (Hz). Must be positive.                                                                                      |
| `--points-per-octave N`   | `-ppo`| `3`              | Number of test points per octave in sweep mode. Must be positive.                                                                         |
| `--amplitude DBFS`          |       | `-12.0`          | Amplitude of the test signal (dBFS). Must be <= 0.                                                                                        |
| `--device ID`               |       | Prompts user     | Integer ID of the audio device for playback and recording. If not provided, a list of available devices will be shown for selection.      |
| `--sample-rate HZ`        |       | `48000`          | Sampling rate in Hz for signal generation, playback, and recording.                                                                       |
| `--output-channel CH_SPEC`| `-oc` | `L`              | Output channel for test signal (e.g., 'L', 'R', or numeric 0-based index '0', '1', ...).                                                  |
| `--input-channels CH_SPEC [...]` | `-ic` | **Required**   | List of input channels to record (e.g., 'L' 'R' or '0' '1' ...). The first channel is the reference for crosstalk calculation (i.e., the channel receiving the direct signal or loopback of the output channel). All specified input channels must be unique. At least two must be specified. |
| `--window WINDOW_TYPE`      |       | `hann`           | FFT window type for analysis (e.g., `hann`, `blackmanharris`).                                                                              |
| `--duration-per-step SECS`|       | `0.5`            | Duration of tone generation and recording for each frequency step (seconds). Must be positive.                                            |
| `--output-csv FILENAME.csv` |       | `None`           | Path to save results in CSV format (e.g., `results.csv`).                                                                                 |
| `--output-plot FILENAME.png`|       | `None`           | Path to save crosstalk plot as an image (e.g., `plot.png`). Plot is generated for sweep mode only.                                        |
| `--no-plot-display`       |       | `False`          | Suppress interactive display of the plot. The plot will still be saved if `--output_plot` is specified.                                     |
| `--help`                    | `-h`  |                  | Show this help message and exit.                                                                                                          |

*CH_SPEC refers to a channel specifier, which can be 'L' (left), 'R' (right), or a numeric 0-based index (e.g., '0', '1').*

## Example Commands

### Single Frequency Test

This command performs a crosstalk test at 1000 Hz. The test signal at -12 dBFS is played on the Left output channel of device 0. The script records from the Left input channel (as reference) and the Right input channel (as undriven) of device 0.

```bash
python audio_crosstalk_analyzer/audio_crosstalk_analyzer.py --frequency 1000 --amplitude -12 --output_channel L --input_channels L R --device 0
```

### Frequency Sweep Test with CSV and Plot Output

This command performs a frequency sweep from 20 Hz to 20000 Hz with 6 points per octave. The test signal is -12 dBFS, played on output channel 0 of device 0. Input channels 0 (reference) and 1 (undriven) are recorded. Results are saved to `crosstalk_results.csv` and a plot is saved to `crosstalk_plot.png`.

```bash
python audio_crosstalk_analyzer/audio_crosstalk_analyzer.py --sweep --start_freq 20 --end_freq 20000 --points_per_octave 6 --amplitude -12 --output_channel 0 --input_channels 0 1 --output_csv crosstalk_results.csv --output_plot crosstalk_plot.png --device 0
```

## Output Description

### Console Output

The script first prints details about the selected device and test parameters. During the test, it shows the progress for each frequency. Finally, it displays a summary table of the results. For each frequency tested:
-   **Freq (Hz)**: The nominal frequency of the test tone.
-   **Ref Ch ('X') Lvl (dBFS)**: The measured level of the test tone on the reference input channel (specified as the first channel in `--input_channels`). 'X' is the channel specifier.
-   **Ch 'Y' Lvl (dBFS)**: The measured level of the test tone on an undriven input channel 'Y'.
-   **Ch 'Y' Crosstalk (dB)**: The calculated crosstalk from the output channel to the undriven input channel 'Y', relative to the level measured on the reference input channel. More negative values indicate better isolation (less crosstalk).

Example table snippet:
```
                             Crosstalk Measurement Summary (Output Ch: 'L')
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Freq (Hz)    ┃ Ref Ch ('L') Lvl (dBFS)              ┃ Ch 'R' Lvl (dBFS)                    ┃ Ch 'R' Crosstalk (dB)                   ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      20.00   │                             -12.05   │                             -75.50   │                                  -63.45   │
│      25.19   │                             -12.03   │                             -78.12   │                                  -66.09   │
│      ...     │                                ...   │                                ...   │                                     ...   │
└──────────────┴──────────────────────────────────────┴──────────────────────────────────────┴─────────────────────────────────────────┘
```

### CSV File

If `--output_csv` is specified, a CSV file is generated with columns corresponding to the console output table, allowing for easy import into spreadsheet software or other analysis tools.

### Plot Image

If `--output_plot` is specified (and the test is in sweep mode), an image file (e.g., PNG) is generated. The plot shows:
-   **X-axis**: Frequency (Hz) on a logarithmic scale.
-   **Y-axis**: Crosstalk (dB).
-   Each line on the plot represents the crosstalk from the driven output channel to one of the specified undriven input channels across the frequency range. A legend identifies each line.

## Important Notes

-   **Loopback Configuration**: For accurate measurement of an audio interface's own crosstalk or for testing a Device Under Test (DUT), a proper loopback configuration is essential.
    -   **Interface Self-Test**: Connect a cable from the specified output channel (e.g., Line Out L) directly to the reference input channel (e.g., Line In L). Connect another cable from the same output channel to the undriven input channel you wish to measure crosstalk *into* (e.g., Line In R). This is not standard; typically, you play on one output, measure its loopback on the corresponding input, and measure the leakage on *other* inputs that are *not* directly fed.
    -   **Corrected Loopback for Typical Crosstalk**: Play signal on Output L. Input L (reference) should be connected to Output L. Input R (undriven) should be terminated appropriately (e.g., with its characteristic impedance or left open, depending on test standard, though this script assumes it's just another input of the interface). The script measures signal on Input L and Input R. Crosstalk is then Input R level relative to Input L level.
    -   **DUT Testing**: Route the signal from the interface's output channel, through the DUT, and then into the reference and undriven input channels of the interface.
-   **Audio Interface Quality**: The measured crosstalk will be limited by the performance of the audio interface itself (its own internal crosstalk and noise floor). Using a high-quality interface is crucial for measuring low levels of crosstalk accurately.
-   **Signal Levels**: Choose the test signal `--amplitude` carefully. It should be high enough to be well above the noise floor but low enough to avoid clipping the output stage of the playback device or the input stage of the recording device/DUT. Check the interface's specifications and use its level meters if available.
-   **Grounding and Shielding**: Ensure proper grounding and use shielded cables to minimize external interference, which can be misinterpreted as crosstalk.

## License

This software is released into the public domain via the Unlicense. You are free to use, modify, and distribute the code as you see fit. For more details, see <http://unlicense.org/>.
```
