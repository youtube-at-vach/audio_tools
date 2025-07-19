# Wavelet Noise Reducer

This tool reduces noise from an audio file using wavelet transform.

## Dependencies

- numpy
- scipy
- soundfile

## Usage

```bash
python3 wavelet_noise_reducer.py <input_file> <output_file> [options]
```

### Options

- `--wavelet`: Wavelet name (e.g., db8) (default: db8).
- `--levels`: Number of decomposition levels (default: 8).
- `--thresholds`: Threshold levels for each decomposition level (0.0 to 1.0). A list of float values. (default: [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
