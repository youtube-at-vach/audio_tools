# Audio Lissajous Analyzer

This tool visualizes the phase relationship of a stereo audio signal in real-time using a Lissajous figure (also known as an X-Y plot). It is useful for quickly assessing the phase, stereo width, and mono compatibility of audio signals.

- A vertical line indicates a mono signal (L and R are identical).
- A circle indicates a 90-degree phase difference between channels.
- A line at a 45-degree angle indicates a signal that is in-phase.
- A line at a -45-degree angle indicates a signal that is out-of-phase.

## Dependencies

The tool requires the following Python libraries:
- `sounddevice`: For audio I/O.
- `numpy`: For numerical operations.
- `matplotlib`: For plotting the figure.
- `rich`: For formatted terminal output.

You can install them using the included `requirements.txt`:
```bash
pip install -r requirements.txt
```
On some systems (like Debian/Ubuntu), you may also need to install the PortAudio library, which `sounddevice` depends on:
```bash
sudo apt-get install libportaudio2
```

Note: Depending on your `sounddevice` version, explicit `sd.check_hostapi()` calls might not be necessary or might cause errors if the function has been deprecated/removed. The current script is designed to work without this explicit call, as the check is often handled internally by `sounddevice` during stream initialization.

## Usage

Run the script from the command line. You must have an audio source connected to your input device. A loopback configuration is recommended for analyzing playback.

```bash
# To run with default settings (uses default input device, channels 1 & 2)
python3 audio_lissajous_analyzer/audio_lissajous_analyzer.py

# To list available audio devices
python3 audio_lissajous_analyzer/audio_lissajous_analyzer.py --list-devices

# To specify a device and channels
python3 audio_lissajous_analyzer/audio_lissajous_analyzer.py --device 1 --channels 3 4
```

### Command-Line Options

- `-l`, `--list-devices`: List all available audio devices and their IDs, then exit.
- `-d ID`, `--device ID`: Specify the input device ID to use. Defaults to the system's default input device.
- `-c L R`, `--channels L R`: Specify the two 1-based input channel indices to use for the X and Y axes of the plot. Default is `1 2`.
- `-r HZ`, `--samplerate HZ`: Set the sample rate in Hz. Default is `48000`.
- `-b N`, `--blocksize N`: The number of audio frames to read at a time. A smaller blocksize provides a more "real-time" feel. Default is calculated from `--block-duration`.
- `--block-duration MS`: Duration of each audio block in milliseconds. Default is `50`.
- `--update-interval MS`: How frequently the plot is updated, in milliseconds. Default is `20`.

