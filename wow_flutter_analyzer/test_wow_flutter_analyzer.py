
import unittest
import numpy as np
import soundfile as sf
from wow_flutter_analyzer import analyze_wow_flutter

class TestWowFlutterAnalyzer(unittest.TestCase):

    def test_analyze_wow_flutter_with_generated_signal(self):
        samplerate = 48000
        duration = 5
        target_freq = 3150.0
        # Introduce a 1% frequency deviation
        deviation_freq = 4.0 # 4 Hz flutter
        t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
        
        # Create a signal with frequency modulation
        instantaneous_freq = target_freq * (1 + 0.01 * np.sin(2 * np.pi * deviation_freq * t))
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / samplerate
        test_signal = np.sin(phase)

        # Save the generated signal to a temporary WAV file
        test_audio_file = 'test_tone.wav'
        sf.write(test_audio_file, test_signal, samplerate)

        # Analyze the generated file
        peak_wow_flutter, _, _ = analyze_wow_flutter(test_audio_file, target_freq)

        # The peak deviation should be close to 1%
        self.assertIsNotNone(peak_wow_flutter)
        self.assertAlmostEqual(peak_wow_flutter, 1.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
