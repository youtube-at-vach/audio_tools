# audiocalc.py
import numpy as np
from scipy.signal import butter, sosfiltfilt, get_window

class AudioCalc:
    @staticmethod
    def bandpass_filter(signal, sampling_rate, lowcut=20.0, highcut=20000.0):
        """
        20Hzから20kHzのバンドパスフィルターを適用します。

        Parameters:
            signal (np.ndarray): フィルタリング対象の信号。
            sampling_rate (float): サンプリングレート（Hz）。
            lowcut (float): フィルターの下限周波数（Hz）。
            highcut (float): フィルターの上限周波数（Hz）。

        Returns:
            np.ndarray: フィルタリング後の信号。
        """
        nyquist = 0.5 * sampling_rate
        sos = butter(8, [lowcut / nyquist, highcut / nyquist], btype='bandpass', output='sos')
        return sosfiltfilt(sos, signal)
    
    @staticmethod
    def notch_filter(signal, sampling_rate, target_frequency, quality_factor=30):
        """
        指定周波数のノッチフィルターを適用します。

        Parameters:
            signal (np.ndarray): フィルタリング対象の信号。
            sampling_rate (float): サンプリングレート（Hz）。
            target_frequency (float): ノッチフィルターを適用する周波数（Hz）。
            quality_factor (float): 品質係数。

        Returns:
            np.ndarray: フィルタリング後の信号。
        """
        nyquist = 0.5 * sampling_rate
        w0 = target_frequency / nyquist
        bandwidth = w0 / quality_factor
        sos = butter(2, [w0 - bandwidth/2, w0 + bandwidth/2], btype='bandstop', output='sos')
        return sosfiltfilt(sos, signal)
    
    @staticmethod
    def calculate_thdn(signal, sampling_rate, target_frequency, min_db=-140.0):
        """
        THD+N（Total Harmonic Distortion plus Noise）を計算します。

        Parameters:
            signal (np.ndarray): 入力信号。
            sampling_rate (float): サンプリングレート（Hz）。
            target_frequency (float): 基本周波数（Hz）。
            min_db (float, optional): 振幅が0の場合や無効な場合に返す最小dB値。デフォルトは-140.0 dB。

        Returns:
            float: THD+N値（dB）。
        """
        if np.max(np.abs(signal)) == 0:
            return min_db  # 振幅が0の場合のデフォルト値
        signal = signal / np.max(np.abs(signal))
        signal *= get_window('hann', len(signal))
        filtered_signal = AudioCalc.notch_filter(signal, sampling_rate, target_frequency)
        fundamental_rms = np.sqrt(np.mean(signal**2))
        filtered_rms = np.sqrt(np.mean(filtered_signal**2))
        thdn_value = 20 * np.log10(filtered_rms / fundamental_rms) if fundamental_rms > 0 else min_db
        return thdn_value
    
    @staticmethod
    def analyze_harmonics(audio_data, fundamental_freq, window_name, sampling_rate, min_db=-140.0):
        """
        高調波解析とTHD、THD+Nの計算を行います。

        Parameters:
            audio_data (np.ndarray): 音声データ。
            fundamental_freq (float): 基本周波数（Hz）。
            window_name (str): 使用する窓関数の名前。
            sampling_rate (float): サンプリングレート（Hz）。
            min_db (float, optional): 振幅が0の場合や無効な場合に返す最小dB値。デフォルトは-140.0 dB。

        Returns:
            dict: 解析結果を含む辞書。
        """
        
        window = get_window(window_name, len(audio_data))
        windowed_data = audio_data * window
        fft_result = np.fft.fft(windowed_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/sampling_rate)[:len(audio_data)//2]
        amplitude_spectrum = (2.0 / len(audio_data)) * np.abs(fft_result[:len(audio_data)//2])
        phase_spectrum = np.angle(fft_result[:len(audio_data)//2], deg=True)
    
        # 基本波解析
        index = np.argmin(np.abs(freqs - fundamental_freq))
        max_freq = freqs[index]
        max_amplitude = amplitude_spectrum[index]
        max_phase = phase_spectrum[index]
        amp_dbfs = 20 * np.log10(max_amplitude) if max_amplitude > 0 else min_db
    
        # 結果の整理
        result = {
            'frequency': max_freq,
            'amplitude_dbfs': amp_dbfs,
            'phase_deg': ((max_phase + 180) % 360) - 180,
            'max_amplitude': max_amplitude  # SNR計算用に追加
        }
    
        # 高調波解析
        harmonic_results = []
        for i in range(2, 10):
            harmonic_freq = fundamental_freq * i
            idx = np.argmin(np.abs(freqs - harmonic_freq))
            df = sampling_rate / len(audio_data)
            if harmonic_freq < sampling_rate / 2 and np.abs(freqs[idx] - harmonic_freq) <= df / 2:
                harmonic_amplitude = amplitude_spectrum[idx]
                relative_amp = harmonic_amplitude / max_amplitude if max_amplitude != 0 else 0
                amp_db = 20 * np.log10(relative_amp) if relative_amp > 0 else min_db
                phase_deg = phase_spectrum[idx]
                harmonic_results.append({
                    'order': i,
                    'frequency': freqs[idx],
                    'amplitude_dbr': amp_db,
                    'phase_deg': phase_deg
                })
            else:
                harmonic_results.append({
                    'order': i,
                    'frequency': None,
                    'amplitude_dbr': None,
                    'phase_deg': None
                })
    
        # THD計算
        harmonic_amplitudes_linear = [10 ** (h['amplitude_dbr'] / 20) for h in harmonic_results if h['amplitude_dbr'] is not None]
        thd = np.sqrt(sum(a ** 2 for a in harmonic_amplitudes_linear)) * 100
        thd_db = 20 * np.log10(thd / 100) if thd > 0 else min_db
    
        # THD+N計算
        thdn = AudioCalc.calculate_thdn(audio_data, sampling_rate, fundamental_freq, min_db=min_db)
        thdn_percent = 10 ** (thdn / 20) * 100 if thdn > min_db else 0.0
    
        return {
            'basic_wave': result,
            'harmonics': harmonic_results,
            'thd_percent': thd,
            'thd_db': thd_db,
            'thdn_percent': thdn_percent,
            'thdn_db': thdn
        }
