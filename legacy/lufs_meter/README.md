# LUFS Meter

## Overview and Purpose

This tool measures the perceived loudness of audio files according to international standards. It calculates Integrated Loudness (I), Momentary Loudness (M), Short-term Loudness (S), Loudness Range (LRA), and True Peak (TP) levels. These metrics are crucial for audio mastering, broadcast, and ensuring consistent loudness across different materials.

The calculations are based on the methodologies specified in:
-   **ITU-R BS.1770-4**: Algorithms to measure audio programme loudness and true-peak audio level.
-   **EBU R128**: Loudness normalisation and permitted maximum level of audio signals.

## Features

-   Calculation of Integrated Loudness (LUFS)
-   Calculation of Momentary Loudness (Max LUFS)
-   Calculation of Short-term Loudness (Max LUFS)
-   Calculation of Loudness Range (LRA) in LU
-   Calculation of True Peak level (dBTP)
-   File-based analysis (WAV, FLAC, etc. supported by SoundFile).
-   Command-line interface for easy operation.
-   Optional output of results to a CSV file.
-   Optional informational target loudness display.

## Dependencies

The tool requires Python 3.8+ and the following libraries:

-   `numpy`
-   `scipy`
-   `soundfile`
-   `sounddevice` (Primarily for audio I/O backend, real-time input not yet implemented)
-   `rich`

These can be installed from the `requirements.txt` file located in this directory:

```bash
pip install -r requirements.txt
```
Make sure you have the necessary system libraries for `SoundFile` (like `libsndfile`) and `PortAudio` if you intend to use `sounddevice` capabilities in the future.

## Usage

The script is run from the command line.

**Basic command:**

```bash
python lufs_meter.py <audio_filepath>
```

**Command-line Options:**

```text
 usage: lufs_meter.py [-h] [-o OUTPUT_FILE] [-t TARGET_LOUDNESS] [-v] filepath

LUFS Meter: Analyze audio files for loudness according to ITU-R BS.1770-4 /
EBU R128.

positional arguments:
  filepath              Path to the input audio file.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Optional path to a CSV file to save the main loudness
                        results. (default: None)
  -t TARGET_LOUDNESS, --target_loudness TARGET_LOUDNESS
                        Optional target integrated loudness in LUFS. For
                        informational comparison. (default: None)
  -v, --verbose         Print full arrays of momentary and short-term LUFS
                        values. (default: False)
```

## Example Output (Console)

The console output, formatted using `rich`, will look something like this:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━ Loudness Analysis: sample.wav ━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                 │ Value                                                       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╪━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Integrated Loudness    │ -23.5 LUFS (-0.5 vs target)                                 │
│ Loudness Range (LRA)   │ 8.2 LU                                                      │
│ Max Momentary Loudness │ -18.1 LUFS                                                  │
│ Max Short-Term Loudness│ -20.3 LUFS                                                  │
│ True Peak              │ -1.5 dBTP                                                   │
│ Target Loudness        │ -23.0 LUFS                                                  │
└────────────────────────┴─────────────────────────────────────────────────────────────┘
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Target Comparison ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Measured Integrated Loudness is 0.5 LUFS below target.                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

*(Note: The actual table and panel appearance might vary slightly based on terminal width and specific `rich` version. The example above shows a possible layout where the target comparison appears as a panel after the main table if a target is specified.)*


## Output CSV File

If the `--output_file` (or `-o`) option is used, the results will be saved in a CSV file with the following headers:
`Filepath,Integrated LUFS,Loudness Range LU,Max Momentary LUFS,Max Short-Term LUFS,True Peak dBTP,Target Loudness LUFS`

## License

This software is released under the **Unlicense**. You are free to use, modify, distribute, and sell this software, for any purpose, with or without attribution. See the [Unlicense website](http://unlicense.org/) for more details.
