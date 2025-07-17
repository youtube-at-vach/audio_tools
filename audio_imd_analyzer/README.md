# Audio IMD Analyzer

## Overview

This Python script, `audio_imd_analyzer.py`, is a command-line tool designed to generate a dual-tone test signal, play it through an audio output device, record it via an input device (typically in a loopback configuration or through a Device Under Test - DUT), and then analyze the recorded audio for Intermodulation Distortion (IMD). The script supports two common IMD measurement standards:
- **SMPTE RP120-1994**: Utilizes a low-frequency tone and a high-frequency tone with a 4:1 amplitude ratio.
- **CCIF (ITU-R Recommendation BS.559-2, Method 2)**: Employs two closely spaced high-frequency tones of equal amplitude (e.g., 19kHz and 20kHz) to measure distortion products such as the 1kHz difference tone (f2-f1) and third-order products (e.g., 2*f1-f2 at 18kHz and 2*f2-f1 at 21kHz).

The tool allows users to specify various parameters for signal generation, audio device selection, and IMD analysis, providing detailed results including IMD percentage, IMD in dB, and a breakdown of individual intermodulation products according to the selected standard.

## SMPTE RP120-1994 Standard

The SMPTE RP120-1994 standard is a common method for measuring intermodulation distortion in audio systems. It specifies a dual-tone test signal consisting of:
- **f1**: A low-frequency tone, typically **60 Hz**.
- **f2**: A high-frequency tone, typically **7000 Hz**.
- **Amplitude Ratio**: The amplitude of f1 is typically four times the amplitude of f2 (a 4:1 linear ratio, or +12 dB difference).

IMD is then calculated based on the amplitudes of the sideband frequencies (e.g., f2 ± f1, f2 ± 2*f1) relative to the amplitude of the f2 tone in the recorded signal.

## CCIF (ITU-R) IMD Standard

The CCIF method for measuring intermodulation distortion, often associated with ITU-R Recommendation BS.559-2 (Method 2), uses two high-frequency tones of equal amplitude that are closely spaced.
- **Test Frequencies (f1, f2)**: Typical frequencies are **19 kHz** and **20 kHz**. Other pairs like 13kHz & 14kHz or 18kHz & 19kHz can also be used. The script defaults to 19kHz and 20kHz when CCIF mode is selected and frequencies are not user-specified.
- **Amplitude Ratio**: The two tones (f1 and f2) have an equal amplitude (a **1:1 linear ratio**). The script defaults to this ratio for CCIF mode if not specified by the user.

Key distortion products analyzed for CCIF IMD include:
- **d2 (Second-order difference tone)**: `|f2 - f1|`. For 19kHz and 20kHz tones, this is 1kHz.
- **d3 (Third-order products)**: 
    - `2*f1 - f2`. For 19kHz and 20kHz, this is 18kHz.
    - `2*f2 - f1`. For 19kHz and 20kHz, this is 21kHz.

The IMD percentage and dB value are calculated based on the RMS sum of these distortion products relative to the sum of the amplitudes of the two primary tones (f1 + f2).

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
python audio_imd_analyzer/audio_imd_analyzer.py [OPTIONS]
```

The tool supports two main modes: a **single-shot measurement** at fixed frequencies, and a **sweep measurement** where one frequency is varied across a range.

### Main Options

| Option                | Alias | Default Value      | Description                                                                                                |
|-----------------------|-------|--------------------|------------------------------------------------------------------------------------------------------------|
| `--f1`                |       | `60.0`             | Frequency of the first (low-frequency) tone in Hz. In sweep mode, this is the fixed frequency if `f2` is swept. |
| `--f2`                |       | `7000.0`           | Frequency of the second (high-frequency) tone in Hz. In sweep mode, this is the fixed frequency if `f1` is swept. |
| `--amplitude`         |       | `-12.0`            | Amplitude of the first tone (f1) in dBFS. The amplitude of f2 is derived from this and the `--ratio`.        |
| `--ratio`             |       | `4.0`              | Linear amplitude ratio of f1/f2 (e.g., 4 for a 4:1 ratio where f1 is stronger).                             |
| `--duration`          |       | `1.0`              | Duration of the test signal and recording for **each step** in seconds.                                    |
| `--sample_rate`       |       | `48000`            | Sampling rate in Hz for signal generation, playback, and recording.                                        |
| `--device`            |       | Prompts user       | Integer ID of the audio device to use. If not provided, a list of available devices will be shown.         |
| `--output_channel`    | `-oc` | `R`                | Output channel for playback: 'L' (left) or 'R' (right).                                                    |
| `--input_channel`     | `-ic` | `L`                | Input channel for recording: 'L' (left) or 'R' (right).                                                    |
| `--window`            |       | `blackmanharris`   | FFT window type for analysis (e.g., `hann`, `hamming`, `blackmanharris`).                                  |
| `--num_sidebands`     |       | `3`                | Number of sideband pairs (e.g., f2 ± n*f1) to analyze for SMPTE IMD calculation.                           |
| `--standard`          | `-std`| `smpte`            | IMD standard to use: `smpte` or `ccif`. If `ccif` is chosen and `--f1`, `--f2`, `--ratio` are not specified, they default to 19kHz, 20kHz, and 1.0 respectively. |
| `--output_csv`        |       | `None`             | Path to save results to a CSV file. In sweep mode, this saves the results of every step.                   |
| `--help`              | `-h`  |                    | Show this help message and exit.                                                                           |

### Sweep Measurement Options

To perform a sweep measurement, you must specify `--sweep-mode`. This will vary one of the test frequencies (`f1` or `f2`) across a defined range, performing an IMD measurement at each step.

| Option                | Default Value      | Description                                                                                                |
|-----------------------|--------------------|------------------------------------------------------------------------------------------------------------|
| `--sweep-mode`        | `None`             | Enables sweep mode. Set to `f1` or `f2` to specify which frequency to sweep. **Required for sweep.**         |
| `--sweep-start`       | `None`             | The starting frequency for the sweep in Hz. **Required for sweep.**                                        |
| `--sweep-end`         | `None`             | The ending frequency for the sweep in Hz. **Required for sweep.**                                          |
| `--sweep-steps`       | `10`               | The number of frequency steps to measure within the sweep range.                                           |
| `--sweep-scale`       | `linear`           | The scale of the sweep steps. Can be `linear` for evenly spaced steps or `log` for logarithmically spaced steps. |
| `--plot-file`         | `None`             | Path to save a plot of the sweep results (e.g., `sweep_results.png`). IMD (%) vs. Frequency is plotted.     |

### Example: Single-Shot Measurement

This command generates the standard SMPTE signal, plays it on the right channel of the default (or user-selected) audio device, records from the left channel, and analyzes for 3 sideband pairs using a Blackman-Harris window.

```bash
python audio_imd_analyzer/audio_imd_analyzer.py --f1 60 --f2 7000 --amplitude -12 --ratio 4 --oc R --ic L
```

### Example: CCIF Single-Shot Measurement

This command generates a CCIF signal with f1=19kHz and f2=20kHz (implicitly, as defaults for `--standard ccif`), each at -18dBFS (due to `--amplitude -18` and implicit 1:1 ratio for CCIF), plays it on the right channel, records from the left, and analyzes.

```bash
python audio_imd_analyzer/audio_imd_analyzer.py --standard ccif --amplitude -18 --oc R --ic L
```

### Example: Sweep Measurement

This command performs an SMPTE IMD sweep. It sweeps the low-frequency tone (`f1`) from 20 Hz to 200 Hz in 15 logarithmic steps, with the high-frequency tone (`f2`) fixed at 7000 Hz. For each step, it performs a measurement and then saves the aggregated results to a CSV file and generates a plot.

```bash
python audio_imd_analyzer/audio_imd_analyzer.py \
    --standard smpte \
    --sweep-mode f1 \
    --sweep-start 20 \
    --sweep-end 200 \
    --sweep-steps 15 \
    --sweep-scale log \
    --f2 7000 \
    --output-csv smpte_f1_sweep.csv \
    --plot-file smpte_f1_sweep.png
```

If no `--device` ID is provided, the script will list available devices and prompt for a selection.

## Output Description

The script provides the following output:
1.  **Device Selection**: If `--device` is not specified, a table of available audio devices is shown, and the user is prompted to select one.
2.  **Signal Generation Info**: Parameters used for generating the dual-tone signal (reflecting chosen standard and any user overrides) and the max/min values of the generated signal.
3.  **Playback & Recording Info**: Details of the device and channels used for playback and recording, and confirmation of successful recording with shape and max/min values of the recorded audio.
4.  **IMD Analysis Results (common for both standards)**:
    *   The script first indicates which IMD standard (`SMPTE` or `CCIF`) is being used for analysis.
    *   Parameters used for the analysis (e.g., f1, f2, window).
    *   The overall IMD percentage and IMD in dB, calculated according to the selected standard.
    *   A table detailing the detected intermodulation products. The content and reference for levels in this table depend on the standard:

    **For SMPTE:**
    *   The measured amplitude of the reference tone f2 (nominal and actual detected frequency, dBFS, and linear amplitude) is displayed.
    *   The IMD Product Details table includes:
        *   **Order (n)**: The order of the sideband (e.g., 1 for f2 ± f1, 2 for f2 ± 2*f1).
        *   **Type**: '+' for upper sidebands (f2 + n*f1), '-' for lower sidebands (f2 - n*f1).
        *   **Nom. Freq (Hz)**: The nominal (expected) frequency of the IMD product.
        *   **Act. Freq (Hz)**: The actual frequency at which the peak was detected in the spectrum.
        *   **Amplitude (Lin)**: The linear amplitude of the detected IMD product.
        *   **Level (dBr f2)**: The level of the IMD product in dB relative to the amplitude of the f2 reference tone.

**Example SMPTE IMD Product Details Table:**
```
                                     SMPTE IMD Product Details
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Order (n) ┃ Type   ┃ Nom. Freq (Hz)   ┃ Act. Freq (Hz)   ┃ Amplitude (Lin)   ┃ Level (dBr f2)   ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ 1         │ +      │           7060.0 │           7060.1 │          1.23e-05 │           -40.12 │
│ 1         │ -      │           6940.0 │           6939.8 │          1.19e-05 │           -40.55 │
│ 2         │ +      │           7120.0 │           7120.3 │          5.01e-07 │           -67.92 │
│ 2         │ -      │           6880.0 │           6879.9 │          4.88e-07 │           -68.21 │
└───────────┴────────┴──────────────────┴──────────────────┴───────────────────┴──────────────────┘
```

    **For CCIF:**
    *   The measured amplitudes of both reference tones, f1 and f2 (nominal and actual detected frequencies, dBFS, and linear amplitudes), are displayed.
    *   The IMD Product Details table includes:
        *   **Product Type**: Describes the IMD product (e.g., "d2 (f2-f1)", "d3 (2f1-f2)").
        *   **Nom. Freq (Hz)**: The nominal (expected) frequency of the IMD product.
        *   **Act. Freq (Hz)**: The actual frequency at which the peak was detected.
        *   **Amplitude (Lin)**: The linear amplitude of the detected IMD product.
        *   **Level (dBr f1+f2)**: The level of the IMD product in dB relative to the sum of the amplitudes of the f1 and f2 reference tones.

**Example CCIF IMD Product Details Table:**
```
                                     CCIF IMD Product Details
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Product Type ┃ Nom. Freq (Hz)   ┃ Act. Freq (Hz)   ┃ Amplitude (Lin)   ┃ Level (dBr f1+f2) ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ d2 (f2-f1)   │           1000.0 │            999.8 │          2.50e-05 │            -50.00 │
│ d3 (2f1-f2)  │          18000.0 │          17999.5 │          1.25e-06 │            -66.02 │
│ d3 (2f2-f1)  │          21000.0 │          21000.3 │          1.30e-06 │            -65.70 │
└──────────────┴──────────────────┴──────────────────┴───────────────────┴───────────────────┘
```
*(Note: Actual values will vary based on the audio interface and loopback/DUT characteristics.)*

## Important Notes

-   **Loopback Configuration / Device Under Test (DUT)**: For accurate self-testing of an audio interface, a physical loopback connection is typically required (connecting the output of the interface to its own input). Alternatively, the script can be used to test an external DUT by routing the output signal through the DUT and then into the input of the audio interface.
-   **Audio Interface Quality**: The quality of the audio interface (sound card) used for playback and recording significantly impacts the IMD results. High-quality interfaces with low noise and distortion are crucial for obtaining meaningful measurements. The script itself does not introduce IMD; any measured distortion comes from the audio hardware path.
-   **Error Handling**: The script includes error handling for device selection, signal generation issues, and audio stream problems. Error messages are printed to `stderr`.
-   **Amplitude Calibration**: The script operates with digital signal levels (dBFS). For absolute acoustic measurements, calibration of input and output levels would be necessary.
-   **Testing Status**: Due to hardware interaction complexities within virtualized environments, the automated test suite for the sweep functionality is currently incomplete. The core analysis functions are unit-tested, and the sweep feature has been verified manually.

## License

This software is released into the public domain via the Unlicense.
```
