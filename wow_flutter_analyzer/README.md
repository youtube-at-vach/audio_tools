# Wow and Flutter Analyzer

## Overview and Purpose

**Wow and Flutter** are terms used to describe undesirable, audible frequency variations in audio playback, typically caused by mechanical inaccuracies in analog audio equipment like tape recorders, record players, or film projectors.

*   **Wow**: Refers to slow cyclical frequency variations, typically below 4 Hz. It can sound like the audio is slowly "warping" or "bending" in pitch.
*   **Flutter**: Refers to faster cyclical frequency variations, typically above 4 Hz. It can sound like a "roughness" or "graininess" in the audio, or in severe cases, like the sound is "gargling".

This tool, `wow_flutter_analyzer`, measures these frequency variations from a recording of a test tone (typically 3000 Hz or 3150 Hz). It analyzes the input audio file to quantify:
*   **Peak Wow**: The maximum slow frequency deviation, typically measured according to DIN standards (frequencies below a cutoff, often around 4 Hz).
*   **RMS Flutter (Unweighted)**: The root mean square of faster frequency deviations, without any specific frequency weighting applied.
*   **WRMS Flutter (Placeholder)**: A weighted root mean square flutter measurement. **Currently, this tool uses a placeholder bandpass filter (0.5 Hz - 200 Hz) and does not implement a specific standard weighting curve (e.g., NAB, IEC, DIN).** The results for WRMS Flutter should therefore be considered indicative and not directly comparable to measurements made with standard-compliant equipment.

The tool provides command-line output, can save results to text and CSV files, and can display a plot of the frequency deviation over time.

## Dependencies

The tool requires the following Python libraries:

*   `numpy`
*   `scipy`
*   `soundfile`
*   `rich` (for enhanced console output)
*   `matplotlib` (for plotting)

You can install these dependencies using pip:

```bash
pip install numpy scipy soundfile rich matplotlib
```

Alternatively, if a `requirements.txt` file is provided with the tool (as it is in this directory), you can install them using:

```bash
pip install -r requirements.txt
```

## Command-Line Usage

The script is run from the command line.

```bash
python wow_flutter_analyzer.py [OPTIONS]
```

**Required Arguments:**

*   `-i INPUT_FILE`, `--input_file INPUT_FILE`
    *   Path to the input audio file (e.g., WAV, FLAC) containing the test tone.
*   `-f FREQUENCY`, `--frequency FREQUENCY`
    *   The expected test frequency in Hertz (e.g., 3000, 3150).

**Optional Arguments:**

*   `-ot OUTPUT_TXT`, `--output_txt OUTPUT_TXT`
    *   Path to save the analysis results as a human-readable text file.
*   `-oc OUTPUT_CSV`, `--output_csv OUTPUT_CSV`
    *   Path to save/append the analysis results as a CSV file. If the file exists, results are appended.
*   `-p`, `--plot_deviation`
    *   Display a plot of the frequency deviation over time using `matplotlib`.
*   `-h`, `--help`
    *   Show a help message detailing all available command-line options and exit.

**Example Usage:**

```bash
# Analyze a test tone of 3150 Hz from 'test_tone.wav'
python wow_flutter_analyzer.py -i test_tone.wav -f 3150

# Analyze, save results to text and CSV, and show plot
python wow_flutter_analyzer.py -i my_capture.flac -f 3000 -ot results.txt -oc measurements.csv -p
```

## Output Explanation

The tool provides the following key measurements:

*   **Input File**: The path to the audio file being analyzed.
*   **Nominal Test Frequency**: The expected frequency of the test tone provided by the user.
*   **Detected Fundamental Frequency**: The actual dominant frequency found in the audio file near the nominal test frequency. This is used as the reference for deviation calculations.
*   **Peak Wow (DIN, <4 Hz)**:
    *   This value represents the peak-to-peak frequency deviation for slow variations (typically below 4 Hz, though the exact cutoff can be adjusted in the code).
    *   It's expressed as a percentage of the detected fundamental frequency.
    *   Higher values indicate more severe slow pitch variations.
*   **RMS Flutter (Unweighted, >4 Hz)**:
    *   This is the Root Mean Square (RMS) of the frequency deviations for faster variations (typically above the Wow cutoff).
    *   "Unweighted" means that all flutter frequencies contribute equally to the RMS calculation, without regard to human psychoacoustic perception.
    *   It's expressed as a percentage of the detected fundamental frequency.
    *   Higher values indicate more noticeable rapid pitch variations or roughness.
*   **WRMS Flutter (NAB Placeholder)**:
    *   This is a Weighted Root Mean Square flutter measurement. Weighting aims to emphasize frequencies where human hearing is most sensitive to flutter.
    *   **Important**: The current implementation uses a **placeholder bandpass filter (0.5 Hz to 200 Hz)**. This is **NOT** a standard-compliant NAB, IEC, or DIN weighting filter.
    *   Therefore, this value is indicative and useful for relative comparisons with this tool but should not be directly compared to measurements from equipment using standardized weighting curves.
    *   It's expressed as a percentage of the detected fundamental frequency.

These results are displayed in a table on the console (if `rich` is available) and can be saved to text and CSV files. The CSV file includes these values with more descriptive headers (e.g., `PeakWowPercent_DIN`).

## Measurement Considerations

To obtain meaningful and accurate Wow and Flutter measurements:

*   **Test Tone Quality**: Use a high-quality, stable test tone recording. The tone should be as pure a sine wave as possible, with minimal noise, distortion, and no existing frequency modulation.
*   **Recording Quality**: Ensure the recording of the playback device is clean. Avoid clipping, excessive noise, or hum during recording.
*   **Typical Frequencies**: Standard test frequencies are typically 3000 Hz or 3150 Hz. This tool is optimized around these values, but other frequencies can be used if specified.
*   **Duration**: A few seconds of stable test tone is usually sufficient (e.g., 5-10 seconds). The tool uses STFT (Short-Time Fourier Transform) and analyzes segments of the audio.
*   **No Other Audio**: The recording should only contain the test tone. Other sounds or music will interfere with the measurement.
*   **Consistent Setup**: For comparative measurements (e.g., before and after servicing a device), ensure the recording setup (microphone, levels, distance) is as consistent as possible.

## License

This tool is released under the **Unlicense**. For more details, please refer to the `LICENSE` file in the main repository or visit [https://unlicense.org/](https://unlicense.org/).
