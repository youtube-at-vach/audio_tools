# Noise Generator

This tool generates various types of noise for audio testing purposes.

## Dependencies

- numpy

## Usage

```bash
python3 noise_generator.py <noise_type> [options]
```

### Noise Types

- `white`: White noise.
- `pink`: Pink noise.
- `brown`: Brown noise.
- `gaussian`: Gaussian noise.
- `violet`: Violet noise.
- `periodic`: Periodic noise.
- `mls`: Maximum Length Sequence (MLS) noise.

### Options

- `-d, --duration`: Duration in seconds (default: 5.0).
- `-r, --rate`: Sample rate in Hz (default: 44100).
- `-a, --amplitude`: Amplitude (0.0 to 1.0) (default: 0.5).
- `-o, --output`: Path to save the generated noise as a WAV file.
- `--list_devices`: List available audio devices and exit.
- `--device`: Output device ID. See --list_devices.
