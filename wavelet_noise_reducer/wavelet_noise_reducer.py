
import numpy as np
from scipy import signal
import soundfile as sf
import argparse
import pywt

class WaveletNoiseReducer:
    def __init__(self, wavelet_name='db8', levels=8):
        self.wavelet = wavelet_name
        self.levels = levels

    def reduce_noise(self, data, threshold_mask):
        if pywt is None:
            raise ImportError("pywt library is required for noise reduction.")

        original_length = len(data)
        coeffs = pywt.wavedec(data, self.wavelet, level=self.levels)

        # Apply the threshold_mask to detail coefficients
        # coeffs[0] is approximation, coeffs[1:] are detail coefficients
        # threshold_mask should have length 'levels'
        if len(threshold_mask) != self.levels:
            raise ValueError(f"threshold_mask must have {self.levels} elements, but got {len(threshold_mask)}")

        modified_coeffs = [coeffs[0]] # Start with approximation coefficient
        for i in range(self.levels):
            if threshold_mask[i] == 0.0: # If mask is 0, zero out the detail coefficient
                modified_coeffs.append(np.zeros_like(coeffs[i+1]))
            else: # Otherwise, keep it as is
                modified_coeffs.append(coeffs[i+1])

        denoised_data = pywt.waverec(modified_coeffs, self.wavelet)
        return signal.resample(denoised_data, original_length)


def main():
    parser = argparse.ArgumentParser(description='Wavelet Noise Reducer')
    parser.add_argument('input_file', help='Input WAV file')
    parser.add_argument('output_file', help='Output WAV file')
    parser.add_argument('--wavelet', default='db8', help='Wavelet name (e.g., db8)')
    parser.add_argument('--levels', type=int, default=8, help='Number of decomposition levels')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], help='Threshold levels for each decomposition level (0.0 to 1.0)')
    args = parser.parse_args()

    try:
        data, samplerate = sf.read(args.input_file)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    reducer = WaveletNoiseReducer(wavelet_name=args.wavelet, levels=args.levels)
    denoised_data = reducer.reduce_noise(data, args.thresholds)

    try:
        sf.write(args.output_file, denoised_data, samplerate)
        print(f"Successfully saved denoised file to {args.output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == '__main__':
    main()
