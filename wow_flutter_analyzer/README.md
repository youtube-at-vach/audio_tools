# Wow and Flutter Analyzer

This tool measures speed variations (wow and flutter) in analog playback systems like turntables or tape decks. It analyzes a recording of a stable test tone (typically 3.15 kHz or 3 kHz) to detect frequency modulation caused by inconsistent playback speed.

## Dependencies

This tool requires the following Python libraries:

- `numpy`
- `scipy`
- `soundfile`
- `matplotlib`

These can be installed from the project root's `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

## Usage

The script is run from the command line, specifying the path to the audio file to be analyzed.

```bash
python3 wow_flutter_analyzer.py [options] <audio_file>
```

**Arguments:**

- `audio_file`: Path to the audio file (WAV, FLAC, etc.) containing the test tone.

**Options:**

- `--target_freq <freq>`: The target frequency of the test tone in Hz. Defaults to `3150.0`.
- `--plot`: If specified, displays a plot of the frequency deviation over time.

## Example Output

```
Unweighted Peak Wow & Flutter: 0.9987%
```

*(A plot window will also appear if `--plot` is used.)*

## Notes

- For accurate results, use a high-quality recording of a stable test tone (e.g., 3.15 kHz for NAB, or 3.0 kHz for DIN standards).
- The analysis is performed on the first audio channel if a multi-channel file is provided.
- The current implementation provides an unweighted peak measurement.

## License

This software is released into the public domain under the Unlicense. See the project's main license file for more details.