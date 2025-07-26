# Audio Analyzer

**Version:** 1.4.5  
**Authors:** ChatGPT, vach, Jules, Gemini  
**Date:** 2024-10-16

This tool is provided for experimental and educational purposes. Feel free to use it without restriction.

---

## Overview

This script is a command-line tool for harmonic analysis of audio signals. It can measure Total Harmonic Distortion (THD), THD + Noise (THD+N), and Signal-to-Noise Ratio (SNR). The tool also supports continuous measurements with frequency and amplitude sweeps, test tone generation, and a mapping feature to visualize distortion across a range of frequencies and amplitudes.

It is a useful utility for evaluating the performance of audio equipment and for initial investigations in acoustic analysis.

Part of the [Audio Tools collection](../README.md).

---

## Features

-   Peak detection and harmonic analysis of audio signals.
-   Measurement of **THD** (Total Harmonic Distortion) and **THD+N** (Total Harmonic Distortion plus Noise).
-   Measurement of **SINAD** (Signal-to-Noise and Distortion Ratio), calculated as the inverse of THD+N in dB (`-(THD+N in dB)`).
-   Measurement of **SNR** (Signal-to-Noise Ratio).
-   Calculation and display of **gain** by comparing input and measured amplitudes.
-   Continuous measurement capabilities using **frequency sweeps** or **amplitude sweeps**.
-   A **mapping mode** (`--map`) to measure and visualize distortion across a 2D grid of frequencies and amplitudes.
-   A **test tone** output function for simple signal generation.
-   Displays harmonic analysis results after each measurement.
-   Calculates and displays the **average and standard deviation** from multiple measurements.

---

## File Structure

-   `audio_analyzer.py`: The main program for signal analysis and measurement.
-   `distorsion_visualizer.py`: A tool to visualize distortion data from CSV files.
-   `aligner.py`: A utility for signal timing alignment.
-   `audiocalc.py`: Handles audio-related calculations.
-   `requirement.txt`: Lists the required Python packages.

---

## Requirements

-   Python 3.8 or later.
-   The required Python packages are listed in `requirement.txt`.

Install the dependencies using pip:
```bash
pip install -r requirement.txt
```

---

## Usage

### Main Program: `audio_analyzer.py`

This is the primary script for performing measurements and harmonic analysis.

#### Basic Command

```bash
python audio_analyzer.py --frequency 1000 --amplitude -6 --duration 5.0
```

#### Main Options

| Option                 | Alias | Default         | Description                                                                 |
|------------------------|-------|-----------------|-----------------------------------------------------------------------------|
| `--frequency HZ`       | `-f`  | `1000`          | The fundamental frequency for the measurement (in Hz).                      |
| `--amplitude DBFS`     | `-a`  | `-6`            | The amplitude of the test tone (in dBFS).                                   |
| `--duration SECS`      |       | `5.0`           | The duration of the measurement (in seconds).                               |
| `--window {name}`      | `-w`  | `blackmanharris`| The window function to use for FFT analysis.                                |
| `--bandpass`           |       | `False`         | Apply a bandpass filter around the fundamental frequency.                   |
| `--sample-rate HZ`     | `-sr` | `48000`         | The sampling rate for playback and recording (in Hz).                       |
| `--output-channel CH`  | `-oc` | `R`             | The output channel to use ('L' or 'R').                                     |
| `--input-channel CH`   | `-ic` | (Opposite of oc)| The input channel to use ('L' or 'R').                                      |
| `--device ID`          | `-d`  | (User prompt)   | The numeric ID of the audio device to use.                                  |
| `--num-measurements N` | `-n`  | `2`             | The number of measurements to perform and average.                          |
| `--output-csv FILE`    |       | `None`          | The filename for saving measurement results to a CSV file.                  |

#### Mode Options (choose one)

| Option                | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `--sweep-amplitude`   | Enables amplitude sweep mode.                                            |
| `--sweep-frequency`   | Enables frequency sweep mode.                                            |
| `--map`               | Enables mapping mode to test across a grid of frequencies and amplitudes.|
| `--test`              | Enables test tone output mode.                                           |
| `--calib`             | Enables calibration mode (currently in testing).                         |

---

### Visualizer: `distorsion_visualizer.py`

This script reads distortion data from a CSV file (generated by `audio_analyzer.py` with the `--map` option) and creates a 2D or 3D plot.

#### Options

| Option                    | Alias | Default     | Description                                                                                             |
|---------------------------|-------|-------------|---------------------------------------------------------------------------------------------------------|
| `csv_file`                |       | (Required)  | Path to the CSV file containing distortion data.                                                        |
| `--device_name NAME`      | `-d`  | `None`      | An optional device name to include in the plot title.                                                   |
| `--amplitude_type {type}` | `-a`  | `Output`    | The amplitude data to use for the axis: 'Output(dBFS)' or 'Input(dBFS)'.                                |
| `--convert_to_dBVrms`     | `-c`  | `False`     | Convert the amplitude axis from dBFS to dBVrms.                                                         |
| `--plot_type {type}`      | `-p`  | `contour`   | The type of plot to generate: `contour` or `3d` (surface).                                              |
| `--color`                 |       | `False`     | Use a color map for the contour plot instead of monochrome.                                             |
| `--rotate`                |       | `False`     | Automatically rotate the 3D plot (creates an animation).                                                |

---

## Example Output

The measurement results are displayed in a table format in the console:

```
=== Measurement Results ===
┏━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Index ┃ Freq (Hz) ┃ Ampl (dBFS)┃ Out (dBFS)  ┃ In (dBFS)   ┃ THD (%)  ┃ THD+N (%)   ┃ SINAD (dB)  ┃ SNR (dB)  ┃ Gain (dB) ┃
┡━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 1     │ 1000.0    │ -6.00      │ -6.00       │ -6.05       │ 0.0012   │ 0.0100      │ 80.00       │ 80.05     │ -0.05     │
└───────┴───────────┴────────────┴─────────────┴─────────────┴──────────┴─────────────┴─────────────┴───────────┴───────────┘
```

---

## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

See the [UNLICENSE](https://unlicense.org/) for more details.
