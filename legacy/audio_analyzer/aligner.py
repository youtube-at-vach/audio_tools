import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate
import soundfile as sf

class SignalAligner:
    def __init__(self, trim_time_sec=0.2, samplerate=48000):
        """
        SignalAligner クラスの初期化
        :param trim_time_sec: 信号の先頭をトリムする時間（秒）
        :param samplerate: サンプリングレート（Hz）
        """
        self.trim_time_sec = trim_time_sec
        self.samplerate = samplerate
        self.start_sample = None  # 開始サンプルのインデックス
        self.end_sample = None  # 終了サンプルのインデックス
        self._shift_time = None  # 信号をシフトする時間

    def generate_reference_signal(self, frequency, amplitude, phase, duration):
        """
        基準となる正弦波信号を生成
        :param frequency: 信号の周波数（Hz）
        :param amplitude: 信号の振幅
        :param phase: 信号の初期位相（ラジアン）
        :param duration: 信号の持続時間（秒）
        :return: 生成された正弦波信号
        """
        t = np.linspace(0, duration, int(duration * self.samplerate), endpoint=False)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

    def align_signal(self, data, frequency, amplitude, phase, reference_duration):
        """
        入力信号を基準信号に合わせてシフト・整列する
        :param data: 入力信号データ
        :param frequency: 基準信号の周波数（Hz）
        :param amplitude: 基準信号の振幅
        :param phase: 基準信号の初期位相（ラジアン）
        :param reference_duration: 基準信号の持続時間（秒）
        :return: 整列された信号と相関係数
        """
        # 基準信号を生成
        reference_signal = self.generate_reference_signal(frequency, amplitude, phase, reference_duration)
        # 入力信号の先頭をトリム
        trimmed_data = self.trim_start(data)
        # 信号を基準信号に合わせてシフトする時間を計算
        self._shift_time = self.cross_correlate_shift(reference_signal, trimmed_data)
        # 信号をフラクショナル（小数点以下の）シフトを行う
        shifted_data = self.fractional_shift(trimmed_data, self._shift_time)
        # 基準信号とシフト後の信号の長さを一致させる
        min_length = min(len(shifted_data), len(reference_signal))
        aligned_data = shifted_data[:min_length]
        # 開始サンプルと終了サンプルのインデックスを設定
        self.start_sample = int(self.trim_time_sec * self.samplerate) + max(0, int(self._shift_time * self.samplerate))
        self.end_sample = self.start_sample + min_length
        # 整列された信号と基準信号の相関係数を計算して返す
        return aligned_data, self.calculate_correlation(aligned_data, reference_signal)

    def trim_start(self, data):
        """
        信号の先頭をトリムする
        :param data: 入力信号データ
        :return: トリムされた信号データ
        """
        trim_samples = int(self.trim_time_sec * self.samplerate)
        if trim_samples >= len(data):
            raise ValueError("トリム時間がデータの長さを超えています。")
        return data[trim_samples:]

    def cross_correlate_shift(self, ref, target):
        """
        相互相関によりシフト量（時間）を計算
        :param ref: 基準信号
        :param target: 対象信号
        :return: シフト時間（秒）
        """
        # 相互相関を計算
        correlation = correlate(ref, target, mode='full')
        # 相関に対応するラグ（シフト量）を計算
        lags = np.arange(-len(target) + 1, len(ref))
        # 相関が最大となるラグを取得
        peak_lag = lags[np.argmax(correlation)]
        # ラグからシフト時間（秒）を計算
        shift_seconds = -peak_lag / self.samplerate
        return shift_seconds

    def fractional_shift(self, data, shift_seconds):
        """
        信号を指定した時間だけフラクショナル（小数点以下の）シフト
        :param data: シフトする信号
        :param shift_seconds: シフト時間（秒）
        :return: シフトされた信号
        """
        total_samples = len(data)
        # 元のタイムスタンプとシフト後のタイムスタンプを計算
        t_original = np.arange(total_samples) / self.samplerate
        t_shifted = t_original + shift_seconds
        # cubic補間を使用してシフトされた信号を生成
        interp_func = interp1d(t_original, data, kind='cubic', bounds_error=False, fill_value=0)
        return interp_func(t_shifted)[:total_samples]

    def calculate_correlation(self, data, reference_signal):
        """
        信号と基準信号の相関係数を計算
        :param data: 入力信号データ
        :param reference_signal: 基準信号データ
        :return: 相関係数
        """
        min_length = min(len(data), len(reference_signal))
        return np.corrcoef(reference_signal[:min_length], data[:min_length])[0, 1]

    @property
    def shift_time(self):
        """
        信号をシフトした時間を取得
        :return: シフト時間（秒）
        """
        return self._shift_time

def main():
    # サウンドファイルの読み込み
    file_path = "200Hz.wav"
    data, samplerate = sf.read(file_path)
    # SignalAlignerクラスを初期化
    processor = SignalAligner(trim_time_sec=0.20, samplerate=samplerate)
    # 信号を基準信号に整列
    aligned_data, correlation = processor.align_signal(data, frequency=200, amplitude=1, phase=np.pi/2, reference_duration=3)
    # 結果を出力
    print(f"相関係数: {correlation:.4f}")
    print(f"開始サンプルインデックス: {processor.start_sample}")
    print(f"終了サンプルインデックス: {processor.end_sample}")
    print(f"シフト時間: {processor.shift_time * 1e3:.2f} ms")
    # 整列された信号をファイルに保存
    sf.write("aligned.wav", aligned_data, samplerate)

if __name__ == "__main__":
    main()
