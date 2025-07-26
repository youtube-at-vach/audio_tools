# Audio Transient Analyzer

## Overview

This tool measures the transient response of audio devices by playing a test signal through an output channel and recording it via an input channel. It supports two types of test signals: a simple impulse or a configurable tone burst. After playback and recording, it analyzes the recorded audio to determine key performance metrics related to transient response:

*   **Rise Time:** How quickly the system responds to a sudden change.
*   **Overshoot:** The extent to which the signal exceeds its final steady-state value.
*   **Settling Time:** How long it takes for the signal to stabilize after a transient.

## Dependencies

*   Python 3.8+
*   NumPy
*   SoundDevice
*   Rich
*   SciPy (specifically `scipy.signal.windows` for Tukey window in tone burst generation)

These can typically be installed via pip:
```bash
pip install numpy sounddevice rich scipy
```
Additionally, the `sounddevice` library requires PortAudio to be installed on your system.
On Debian/Ubuntu-based systems, this can be installed using:
```bash
sudo apt-get update && sudo apt-get install -y libportaudio2 portaudio19-dev
```

## Command-Line Interface (CLI) Options

The script is controlled via command-line arguments:

```
usage: audio_transient_analyzer.py [-h] [--signal_type {impulse,tone_burst}]
                                   [--amplitude AMPLITUDE] [--device DEVICE]
                                   [--sample_rate SAMPLE_RATE]
                                   [--output_channel OUTPUT_CHANNEL]
                                   [--input_channel INPUT_CHANNEL]
                                   [--duration DURATION]
                                   [--burst_freq BURST_FREQ]
                                   [--burst_cycles BURST_CYCLES]
                                   [--burst_envelope {hann,rectangular,tukey}]
                                   [--output_csv OUTPUT_CSV]

Audio Transient Analyzer

options:
  -h, --help            show this help message and exit
  --signal_type {impulse,tone_burst}
                        Type of transient signal to generate (default: impulse)
  --amplitude AMPLITUDE
                        Amplitude of the test signal in dBFS. Must be <= 0 (default: -6.0)
  --device DEVICE       Audio device ID. If not provided, the script will list
                        available devices and prompt for selection.
  --sample_rate SAMPLE_RATE
                        Sampling rate in Hz (default: 48000)
  --output_channel OUTPUT_CHANNEL
                        Output channel for test signal (e.g., 'L', 'R', or
                        numeric 1-based index, default: 'L')
  --input_channel INPUT_CHANNEL
                        Input channel for recording (e.g., 'L', 'R', or
                        numeric 1-based index, default: 'L')
  --duration DURATION   Duration of the recording in seconds. Must be positive (default: 0.1)
  --burst_freq BURST_FREQ
                        Frequency of the tone burst in Hz (for signal_type
                        'tone_burst'). Must be positive (default: 1000.0)
  --burst_cycles BURST_CYCLES
                        Number of cycles in the tone burst (for signal_type
                        'tone_burst'). Must be positive (default: 10)
  --burst_envelope {hann,rectangular,tukey}
                        Envelope for the tone burst (for signal_type
                        'tone_burst', default: 'hann')
  --output_csv OUTPUT_CSV
                        Path to save results in CSV format (e.g., results.csv)
```

## Output Values Explained

The analysis provides the following metrics:

*   **Peak Amplitude (Recorded):** The maximum absolute linear amplitude of the recorded signal, measured from the detected start of the transient. This gives an indication of the recorded signal level.
*   **Rise Time (s):** The time taken for the recorded signal to rise from 10% to 90% of its peak value (after the detected start). This is a common measure of how quickly a system can respond to an input.
*   **Overshoot (%):** The percentage by which the signal's peak value exceeds its estimated steady-state value. A high overshoot can indicate ringing or instability. This metric might be less meaningful for pure impulse responses that are expected to decay to zero.
*   **Settling Time (s):** The time taken from the signal's peak until it settles and remains within +/- 5% (default) of its estimated final value. This indicates how long it takes for the system to stabilize after the initial transient.

## Example Usage

1.  **Basic impulse response measurement using device ID 1, Left output, Left input:**
    ```bash
    python audio_transient_analyzer.py --device 1 --output_channel L --input_channel L
    ```
    (If `--device` is omitted, you will be prompted to select from a list of available devices.)

2.  **Tone burst response with a 1kHz, 20-cycle Hann-windowed burst at -10 dBFS, saving results:**
    ```bash
    python audio_transient_analyzer.py --signal_type tone_burst --burst_freq 1000 --burst_cycles 20 --burst_envelope hann --amplitude -10 --device 1 --output_csv transient_results.csv
    ```

## License

This software is released under the Unlicense. See the main repository license for more details.
