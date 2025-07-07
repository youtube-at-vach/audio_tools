# RT60 Analyzer

This tool measures the reverberation time (RT60) of an acoustic space by analyzing a recorded impulse response. RT60 is the time it takes for the sound pressure level to decay by 60 dB after the sound source is stopped.

## Dependencies

This tool requires the following Python libraries:

- `numpy`
- `scipy`
- `sounddevice`
- `matplotlib`

These can be installed from the project root's `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

## Usage

The script is run from the command line. It will start recording audio to capture the impulse response.

```bash
python3 rt60_analyzer.py [options]
```

**Options:**

- `--duration <seconds>`: The recording duration in seconds. Defaults to `5.0`.
- `--samplerate <Hz>`: The sample rate in Hz. Defaults to `48000`.
- `--device <ID>`: The numeric ID of the input audio device to use.
- `--plot`: If specified, displays a plot of the energy decay curve.

## Example Output

```
Recording for impulse response...
Recording finished.
RT60 (T20): 0.48 seconds
```

*(A plot window will also appear if `--plot` is used.)*

## Notes

- To get a good measurement, you need to create a loud, sharp, and broadband sound to act as the impulse. The classic methods are popping a balloon or firing a starter pistol. A loud clap can also work.
- The calculation is based on the T20 method (using the decay from -5 dB to -25 dB) and extrapolating to find the time to decay 60 dB. This is a common practice as the tail of the decay is often lost in the background noise.

## License

This software is released into the public domain under the Unlicense. See the project's main license file for more details.