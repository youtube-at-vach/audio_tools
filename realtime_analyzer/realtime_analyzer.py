#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime Audio Analyzer v1.1.2
---------------------------------
リアルタイムで音声信号の高調波解析と可視化を行うスクリプトです。
基本周波数、高調波、ノイズレベル、THD、THD+N、SNRの測定をサポートします。

特徴:
- 自動振幅調整
- オプション付きのリアルタイムスペクトルプロット
- ピーク検出と高調波解析
- THDおよびTHD+Nの測定
- SNRの測定

作成者: ChatGPT および vach
日付: 2024-10-06
"""

import argparse
import sys
import threading
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import find_peaks, get_window, butter, sosfiltfilt, windows

# デフォルト設定
SAMPLE_RATE = 48000  # サンプリングレート (Hz)
BUFFER_SIZE = 8192    # 表示サンプル数
AVERAGE_FRAMES = 5    # スペクトル平均化フレーム数

# トーン生成設定
tone_settings = {
    'frequency': 1000,   # 基本周波数 (Hz)
    'amplitude': 0.5,    # 振幅
    'phase': 0           # 位相 (度)
}

# 出力チャンネル設定（0: 左, 1: 右）
output_channel = 1  # デフォルトは右チャンネル

# トーンジェネレーターの初期化
tone_generator = None

# 録音データ保存用
recorded_data = []

# リアルタイムプロット用バッファ
input_buffer = deque(maxlen=BUFFER_SIZE)

# FFT平均化バッファ
spectrum_buffer = deque(maxlen=AVERAGE_FRAMES)

class ToneGenerator:
    """シングルトーンを生成するクラス"""
    def __init__(self):
        self.frame = 0  # フレームカウンター

    def generate(self, frames):
        """指定フレーム数のシングルトーンを生成"""
        global SAMPLE_RATE
        t = (np.arange(frames) + self.frame) / SAMPLE_RATE
        freq = tone_settings['frequency']
        amp = tone_settings['amplitude']
        phase = tone_settings['phase']
        signal = amp * np.sin(2 * np.pi * freq * t + phase)
        self.frame += frames
        return np.clip(signal, -1.0, 1.0)  # 振幅を[-1.0, 1.0]に制限

def audio_callback(indata, outdata, frames, time, status):
    """音声ストリームのコールバック関数"""
    global tone_generator, output_channel
    if status:
        print(f"ストリームステータス: {status}")  # エラーステータス表示

    # トーン生成と出力
    outdata[:] = 0
    if outdata.shape[1] >= 2:
        if output_channel == 0:
            outdata[:, 0] = tone_generator.generate(frames)  # 左チャンネル出力
        else:
            outdata[:, 1] = tone_generator.generate(frames)  # 右チャンネル出力

    # 録音データをバッファに追加
    if indata.shape[1] >= 1:
        input_buffer.extend(indata[:, 0])             # 左チャンネル録音
        recorded_data.append(indata[:, 0].copy())    # 解析用に保存

def bandpass_filter(signal, sampling_rate, lowcut=20.0, highcut=20000.0):
    """20Hzから20kHzのバンドパスフィルターを適用"""
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(8, [low, high], btype='bandpass', output='sos')  # 8次フィルター
    return sosfiltfilt(sos, signal)  # 遅延補正付きフィルタリング

def notch_filter(signal, sampling_rate, target_frequency, quality_factor=30):
    """指定周波数のノッチフィルターを適用"""
    nyquist = 0.5 * sampling_rate
    w0 = target_frequency / nyquist
    bandwidth = w0 / quality_factor
    sos = butter(2, [w0 - bandwidth/2, w0 + bandwidth/2], btype='bandstop', output='sos')  # 2次フィルター
    return sosfiltfilt(sos, signal)  # フィルタリング

def calculate_thdn(signal, sampling_rate, target_frequency, window=True):
    """THD+Nを計算"""
    # 正規化
    signal = signal / np.max(np.abs(signal))

    # ウィンドウ適用
    if window:
        signal *= windows.hann(len(signal))

    # 基本波除去
    filtered_signal = notch_filter(signal, sampling_rate, target_frequency)

    # RMS計算
    fundamental_rms = np.sqrt(np.mean(signal**2))
    filtered_rms = np.sqrt(np.mean(filtered_signal**2))

    # THD+N計算
    thdn_value = 20 * np.log10(filtered_rms / fundamental_rms) if fundamental_rms > 0 else -140.00
    return thdn_value

def analyze_harmonics(audio_data, fundamental_freq, window_name, apply_bandpass=False):
    """高調波解析とTHD、THD+Nの計算"""
    global SAMPLE_RATE
    N = len(audio_data)

    if apply_bandpass:
        audio_data = bandpass_filter(audio_data, SAMPLE_RATE)  # バンドパス適用

    try:
        window = get_window(window_name, N)  # ウィンドウ生成
    except ValueError:
        print(f"無効な窓関数: {window_name}")
        print("有効な例: 'blackmanharris', 'hamming', 'hann', 'bartlett', 'flattop' など")
        return None

    windowed_data = audio_data * window
    fft_result = np.fft.fft(windowed_data, n=N)
    freqs = np.fft.fftfreq(N, 1/SAMPLE_RATE)[:N//2]
    amplitude_spectrum = (2.0 / N) * np.abs(fft_result[:N//2])
    phase_spectrum = np.angle(fft_result[:N//2], deg=True)

    # 基本波解析
    index = np.argmin(np.abs(freqs - fundamental_freq))
    max_freq = freqs[index]
    max_amplitude = amplitude_spectrum[index]
    max_phase = phase_spectrum[index]

    # 振幅をdBFSに変換
    amp_dbfs = 20 * np.log10(max_amplitude) if max_amplitude > 0 else -140.00

    print(f"\n### 基本波 ###")
    print(f"周波数 : {max_freq:.2f} Hz")
    print(f"振幅   : {amp_dbfs:.2f} dBFS")
    print(f"位相   : {((max_phase + 180) % 360) - 180:.2f} °")

    # ゲイン計算
    if tone_settings['amplitude'] != 0:
        gain = amp_dbfs - 20 * np.log10(tone_settings['amplitude'])
        print(f"ループゲイン : {gain:.2f} dB")
    print()

    # 基本波のRMSを計算
    if max_amplitude > 0:
        fundamental_rms = max_amplitude / np.sqrt(2)
    else:
        fundamental_rms = 0.0

    harmonic_frequencies = []
    harmonic_amplitudes = []
    harmonic_phases = []

    print("### 高調波 ###")
    for i in range(2, 10):  # 2次から9次
        harmonic_freq = fundamental_freq * i
        index = np.argmin(np.abs(freqs - harmonic_freq))
        df = SAMPLE_RATE / N
        if harmonic_freq < SAMPLE_RATE / 2 and np.abs(freqs[index] - harmonic_freq) <= df / 2:
            harmonic_amplitude = amplitude_spectrum[index]
            harmonic_phase = phase_spectrum[index]

            # 相対振幅
            relative_amp = harmonic_amplitude / max_amplitude if max_amplitude != 0 else 0
            harmonic_amplitudes.append(relative_amp)

            # 位相をラジアンに変換
            harmonic_phases.append(np.radians(harmonic_phase))

            # 振幅をdBrに変換
            amp_db = 20 * np.log10(relative_amp) if relative_amp > 0 else -140.00
            print(f"{i}次    : {freqs[index]:.2f} Hz | 振幅: {amp_db:.2f} dBr | 位相: {((harmonic_phase + 180) % 360) - 180:.2f} °")
            harmonic_frequencies.append(freqs[index])
        else:
            print(f"{i}次高調波の周波数: {harmonic_freq:.2f} Hz は検出されませんでした。")
            harmonic_amplitudes.append(0.0)
            harmonic_phases.append(0.0)
            harmonic_frequencies.append(harmonic_freq)

    # THD計算
    if max_amplitude != 0:
        thd = np.sqrt(sum(a ** 2 for a in harmonic_amplitudes)) * 100  # %
        thd_db = 20 * np.log10(thd / 100) if thd > 0 else -140.00  # dB
        print(f"\n全高調波歪 (THD): {thd:.4f}% / {thd_db:.2f} dB\n")
    else:
        thd = None
        thd_db = None
        print("全高調波歪 (THD): 基本波の振幅が0のため計算できません。\n")

    # THD+N計算
    thdn = calculate_thdn(audio_data, SAMPLE_RATE, fundamental_freq)
    thdn_db = thdn  # 既にdBで計算
    thdn_percent = 10 ** (thdn / 20) * 100 if thdn > -140 else 0.0  # %

    print(f"全高調波歪およびノイズ (THD+N): {thdn_percent:.4f}% / {thdn_db:.2f} dB\n")

    return {
        'fundamental_rms': fundamental_rms,
        'max_freq': max_freq,
        'max_amplitude': max_amplitude,
        'max_phase': max_phase,
        'harmonic_frequencies': harmonic_frequencies,
        'harmonic_amplitudes': harmonic_amplitudes,
        'harmonic_phases': harmonic_phases,
        'thd_percent': thd,
        'thd_db': thd_db,
        'thdn_percent': thdn_percent,
        'thdn_db': thdn_db
    }

def analyze(recorded_data, measurement_number, window_name, apply_bandpass=False):
    """録音データを解析し、高調波解析結果を表示"""
    if not recorded_data:
        print("録音データがありません。")
        return None

    recorded_audio = np.concatenate(recorded_data)  # データ結合

    save_recorded_audio(recorded_audio, measurement_number)

    # 基本周波数取得
    fundamental_freq = tone_settings["frequency"]

    # 高調波解析とTHD、THD+N計算
    measurement_result = analyze_harmonics(recorded_audio, fundamental_freq, window_name, apply_bandpass)

    if measurement_result is None:
        return None

    # RMS計算
    rms = np.sqrt(np.mean(recorded_audio ** 2))
    measurement_result['rms'] = rms

    return measurement_result  # 結果返却

def save_recorded_audio(audio_data, measurement_number):
    """録音データをWAVファイルに保存"""
    if measurement_number != "auto":
        filename = f"recorded_audio_{measurement_number}.wav"
        sf.write(filename, audio_data, SAMPLE_RATE, subtype='PCM_32')
        print(f"録音データを {filename} に保存しました。")

def compute_average_thd(thd_values, thd_values_db):
    """THDの平均値を計算 (%およびdB)"""
    if thd_values and thd_values_db:
        average_thd_percent = np.mean(thd_values)
        average_thd_db = np.mean([db for db in thd_values_db if db > -140.00])  # 有効値のみ
        print(f"=== 平均全高調波歪 (THD) ===")
        print(f"平均THD: {average_thd_percent:.4f}% / {average_thd_db:.2f} dB\n")
    else:
        print("THDのデータが不足しているため、平均THDを計算できません。\n")

def compute_average_thdn(thdn_values_percent, thdn_values_db):
    """THD+Nの平均値を計算 (%およびdB)"""
    if thdn_values_percent and thdn_values_db:
        average_thdn_percent = np.mean(thdn_values_percent)
        average_thdn_db = np.mean([db for db in thdn_values_db if db > -140.00])  # 有効値のみ
        print(f"=== 平均全高調波歪およびノイズ (THD+N) ===")
        print(f"平均THD+N: {average_thdn_percent:.4f}% / {average_thdn_db:.2f} dB\n")
    else:
        print("THD+Nのデータが不足しているため、平均THD+Nを計算できません。\n")

def compute_average_snr(snr_values):
    """SNRの平均値を計算"""
    if snr_values:
        valid_snr = [snr for snr in snr_values if snr is not None]
        if valid_snr:
            average_snr = np.mean(valid_snr)
            print(f"=== 平均SNR ===")
            print(f"平均SNR: {average_snr:.2f} dB\n")
        else:
            print("有効なSNRデータが不足しているため、平均SNRを計算できません。\n")
    else:
        print("SNRデータが不足しているため、平均SNRを計算できません。\n")

def compute_average_harmonics(harmonic_values_list):
    """高調波振幅の平均値を計算"""
    if harmonic_values_list:
        max_harmonics = max(len(amps) for amps in harmonic_values_list)
        avg_harmonic_amplitudes_linear = []

        for i in range(max_harmonics):
            amps = [amps[i] for amps in harmonic_values_list if len(amps) > i]
            avg_amp_linear = np.mean(amps) if amps else 0  # 平均振幅
            amp_db_correct = 20 * np.log10(avg_amp_linear) if avg_amp_linear > 0 else -140.00
            avg_harmonic_amplitudes_linear.append(amp_db_correct)

        print(f"=== 平均高調波振幅 ===")
        for i, amp_db in enumerate(avg_harmonic_amplitudes_linear, start=2):
            print(f"{i}次高調波: {amp_db:.2f} dBr")
    else:
        print("高調波データが不足しているため、平均振幅を計算できません。\n")

def compute_average_phases(harmonic_phases_list):
    """高調波位相の平均値をベクトル平均で計算"""
    if harmonic_phases_list:
        harmonic_phases_array = np.array(harmonic_phases_list)  # 配列化
        # ベクトル平均
        mean_phases = np.arctan2(
            np.mean(np.sin(harmonic_phases_array), axis=0),
            np.mean(np.cos(harmonic_phases_array), axis=0)
        ) * (180 / np.pi)  # 度に変換
        # 位相を-180～+180度に調整
        mean_phases = (mean_phases + 180) % 360 - 180
        print(f"=== 平均高調波位相 ===")
        for i, mean_phase in enumerate(mean_phases, start=2):
            print(f"{i}次高調波: {mean_phase:.2f} 度")
    else:
        print("高調波位相データが不足しているため、平均位相を計算できません。\n")

def select_device():
    """使用する音声デバイスを選択"""
    print("使用する音声デバイスを選択してください。\n")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (入力: {device['max_input_channels']}, 出力: {device['max_output_channels']})")
        # レイテンシーをミリ秒表示
        input_latency_ms = device['default_low_input_latency'] * 1000
        output_latency_ms = device['default_low_output_latency'] * 1000
        print(f"    サンプリングレート: {device['default_samplerate']} Hz")
        print(f"    レイテンシ: {input_latency_ms:.2f} ms (入力), {output_latency_ms:.2f} ms (出力)")
    while True:
        try:
            device_num = int(input("使用するデバイスの番号を入力してください: "))
            if 0 <= device_num < len(devices):
                return device_num
            else:
                print("有効なデバイス番号を入力してください。")
        except ValueError:
            print("数値を入力してください。")

def measure(device_index, input_channel, output_channel, measurement_number, duration=10, window_name='blackmanharris', apply_bandpass=False):
    """音声測定を実行し、結果を解析"""
    global recorded_data
    recorded_data = []

    with sd.Stream(callback=audio_callback, channels=2, samplerate=SAMPLE_RATE, device=device_index):
        print(f"\n{tone_settings['frequency']} Hz のトーンを再生中... 測定 {measurement_number}")
        sd.sleep(int(duration * 1000))  # 測定時間待機
        return analyze(recorded_data, measurement_number, window_name, apply_bandpass)  # 解析結果返却

def measure_noise(device_index, input_channel, output_channel, duration=2, window_name='blackmanharris'):
    """ノイズレベルを測定（トーン再生なし）"""
    print("\nノイズ測定を開始します。トーンは再生されません。")
    original_amp = tone_settings['amplitude']    # 現在の振幅保存
    tone_settings['amplitude'] = 0.0             # 振幅を0に設定
    noise_result = measure(device_index, input_channel, output_channel, measurement_number="noise", duration=duration, window_name=window_name)
    tone_settings['amplitude'] = original_amp    # 振幅を復元

    if noise_result and 'rms' in noise_result:
        noise_rms = noise_result['rms']
        print(f"ノイズ RMS: {noise_rms:.6f}\n")
        return noise_rms
    else:
        print("ノイズ測定に失敗しました。")
        return None

def measure_multiple(num_measurements, device_index, input_channel, output_channel, fundamental_freq, window_name, noise_rms, apply_bandpass=False):
    """複数回の測定を実行"""
    thd_values_percent = []
    thd_values_db = []
    thdn_values_percent = []
    thdn_values_db = []
    harmonic_amplitudes_list = []
    harmonic_phases_list = []
    snr_values = []

    for i in range(num_measurements):
        measurement_result = measure(device_index, input_channel, output_channel, i + 1, window_name=window_name, apply_bandpass=apply_bandpass)
        if measurement_result:
            # THD追加
            if measurement_result['thd_percent'] is not None and measurement_result['thd_db'] is not None:
                thd_values_percent.append(measurement_result['thd_percent'])
                thd_values_db.append(measurement_result['thd_db'])

            # THD+N追加
            if measurement_result['thdn_percent'] is not None and measurement_result['thdn_db'] is not None:
                thdn_values_percent.append(measurement_result['thdn_percent'])
                thdn_values_db.append(measurement_result['thdn_db'])

            harmonic_amplitudes_list.append(measurement_result['harmonic_amplitudes'])
            harmonic_phases_list.append(measurement_result['harmonic_phases'])

            # SNR計算の修正
            fundamental_rms = measurement_result.get('fundamental_rms', 0.0)
            if noise_rms is not None and fundamental_rms > 0:
                snr = 20 * np.log10(fundamental_rms / noise_rms)
                snr_values.append(snr)
                print(f"SNR: {snr:.2f} dB")
            else:
                snr_values.append(None)
                print("SNR: N/A (基本波RMSが0またはノイズRMSが不明です。)")

            print("")  # 空行

    return thd_values_percent, thd_values_db, thdn_values_percent, thdn_values_db, harmonic_amplitudes_list, harmonic_phases_list, snr_values

def auto_set_amplitude(device_index, input_channel, output_channel, target_dbfs=-6.0, tolerance=1.0, max_iterations=10, window_name='blackmanharris'):
    """自動でトーンの音量を調整し、入力が目標dBFSになるように設定"""
    current_amp = 0.5  # 初期振幅
    for iteration in range(1, max_iterations + 1):
        tone_settings['amplitude'] = current_amp
        print(f"振幅自動設定 (Iteration {iteration}): {current_amp:.4f}")

        # 短時間トーン再生と録音
        measurement_result = measure(device_index, input_channel, output_channel, measurement_number="auto", duration=1.0, window_name=window_name, apply_bandpass=False)

        if measurement_result is None:
            print("自動設定に失敗しました: 測定結果なし")
            return False

        # 測定振幅取得
        input_amplitude = measurement_result['max_amplitude']
        input_dbfs = 20 * np.log10(input_amplitude) if input_amplitude > 0 else -140.00

        print(f"測定入力振幅: {input_dbfs:.2f} dBFS")

        # 目標範囲内か確認
        if abs(input_dbfs - target_dbfs) <= tolerance:
            print("自動設定成功しました。")
            return True

        # 振幅調整
        error = target_dbfs - input_dbfs
        adjustment_factor = 10 ** (error / 20)
        current_amp *= adjustment_factor

        # 振幅を0.0～1.0に制限
        current_amp = max(0.0, min(current_amp, 1.0))

    print("自動設定に失敗しました: 最大反復回数に到達")
    return False

def start_realtime_plot(window_name):
    """リアルタイムスペクトルプロットを開始"""
    fig, ax = plt.subplots()
    freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1/SAMPLE_RATE)
    line, = ax.plot(freqs, np.zeros(len(freqs)), color='red')  # 赤色ライン
    ax.set_xlim(0, 5000)    # 0～5kHz表示
    ax.set_ylim(-140, 0)    # -140～0 dB
    ax.set_title('Real Time Spectrum')
    ax.set_xlabel('Frequency(Hz)')
    ax.set_ylabel('Amplitude (dB)')

    # ピーク検出設定
    peak_height_threshold = -80  # ピーク最低高さ (dB)
    peak_prominence = 10         # ピーク顕著性

    # ピークマーカーとラベル保存用
    peak_markers = []
    peak_labels = []

    def update_plot(frame):
        nonlocal peak_markers, peak_labels
        if len(input_buffer) < BUFFER_SIZE:
            return [line]

        # 最新データ取得
        data = np.array(input_buffer)[-BUFFER_SIZE:]
        try:
            window = get_window(window_name, len(data))  # ウィンドウ適用
        except ValueError:
            print(f"無効な窓関数: {window_name}")
            print("有効な例: 'blackmanharris', 'hamming', 'hann', 'bartlett', 'flattop' など")
            return [line]
        
        windowed_data = data * window
        fft_result = np.fft.rfft(windowed_data)
        amplitude_spectrum = (2.0 / len(data)) * np.abs(fft_result)
        amplitude_spectrum_dB = 20 * np.log10(amplitude_spectrum + 1e-6)  # ログ変換

        # FFT結果を平均化
        spectrum_buffer.append(amplitude_spectrum_dB)
        average_spectrum_dB = np.mean(spectrum_buffer, axis=0)

        # プロット更新
        line.set_ydata(average_spectrum_dB)

        # ピーク検出
        peaks, properties = find_peaks(average_spectrum_dB, height=peak_height_threshold, prominence=peak_prominence)

        # 既存ピーク削除
        for marker in peak_markers:
            marker.remove()
        for label in peak_labels:
            label.remove()
        peak_markers = []
        peak_labels = []

        # 新ピーク追加
        if len(peaks) > 0:
            peak_markers = ax.plot(freqs[peaks], average_spectrum_dB[peaks], "x", color='black')
            for peak in peaks:
                label = ax.text(freqs[peak], average_spectrum_dB[peak] + 1, f"{freqs[peak]:.1f} Hz", color='black', fontsize=8, rotation=45)
                peak_labels.append(label)

        return [line] + peak_markers + peak_labels

    ani = animation.FuncAnimation(fig, update_plot, interval=100, blit=True)
    plt.show()

def main():
    """メイン処理"""
    global SAMPLE_RATE, output_channel, tone_generator
    parser = argparse.ArgumentParser(description="音声信号の測定と高調波解析を行うプログラム")
    parser.add_argument('-n', '--num_measurements', type=int, default=4, help='測定回数 (デフォルト: 4)')
    parser.add_argument('-f', '--frequency', type=float, default=1000, help='基本周波数 (デフォルト: 1000 Hz)')
    parser.add_argument('-a', '--amplitude', type=float, default=-6, help='トーン振幅 (dBFS) (デフォルト: -6 dBFS)')
    parser.add_argument('-w', '--window', type=str, default='blackmanharris', help='窓関数種類 (デフォルト: blackmanharris)')
    parser.add_argument('--auto', action='store_true', help='自動振幅調整オプション')
    parser.add_argument('--noise_duration', type=float, default=2.0, help='ノイズ測定時間 (秒) (デフォルト: 2.0秒)')
    parser.add_argument('--bandpass', action='store_true', help='バンドパスフィルター適用オプション')
    parser.add_argument('--no_plot', action='store_true', help='リアルタイムプロットを表示しない')
    parser.add_argument('-sr', '--sample_rate', type=int, default=48000, help='サンプリングレート (デフォルト: 48000 Hz)')
    parser.add_argument('-oc', '--output_channel', type=str, choices=['L', 'R'], default='R', help='出力チャンネル (LまたはR, デフォルト: R)')

    args = parser.parse_args()

    # 窓関数の確認
    try:
        get_window(args.window, BUFFER_SIZE)
    except ValueError:
        print(f"無効な窓関数: {args.window}")
        print("有効な例: 'blackmanharris', 'hamming', 'hann', 'bartlett', 'flattop' など")
        sys.exit(1)

    # トーン設定
    tone_settings['frequency'] = args.frequency
    tone_settings['amplitude'] = 10 ** (args.amplitude / 20)  # dBFSを線形振幅に変換

    # サンプリングレート設定
    SAMPLE_RATE = args.sample_rate
    print(f"サンプリングレートを {SAMPLE_RATE} Hz に設定しました。")

    # 出力チャンネル設定と入力チャンネルのクロス設定
    if args.output_channel.upper() == 'L':
        output_channel = 0  # 左チャンネル出力
        input_channel = 1   # 右チャンネル入力
        print("出力チャンネルを左 (L) に設定しました。")
        print("入力チャンネルを右 (R) に設定しました。")
    else:
        output_channel = 1  # 右チャンネル出力
        input_channel = 0   # 左チャンネル入力
        print("出力チャンネルを右 (R) に設定しました。")
        print("入力チャンネルを左 (L) に設定しました。")

    device_index = select_device()

    # トーンジェネレーター初期化
    tone_generator = ToneGenerator()

    stop_event = threading.Event()  # スレッド停止用

    def measurement_thread_func():
        """音声測定スレッドの実行関数"""
        if args.auto:
            print("自動振幅調整を開始します。")
            success = auto_set_amplitude(device_index, input_channel, output_channel, target_dbfs=args.amplitude, window_name=args.window)
            if not success:
                print("自動振幅調整に失敗しました。プログラムを終了します。")
                stop_event.set()
                return

            loopback_gain_db = args.amplitude - 20 * np.log10(tone_settings['amplitude'])
            print(f"ループバックゲイン: {loopback_gain_db:.2f} dB")
            
            # レイテンシー低減のため待機
            sd.sleep(1000)

        try:
            # ノイズ測定
            noise_rms = measure_noise(device_index, input_channel, output_channel, duration=args.noise_duration, window_name=args.window)
            if noise_rms is None:
                print("ノイズ測定に失敗したため、プログラムを終了します。")
                stop_event.set()
                return

            # 測定実行
            thd_values_percent, thd_values_db, thdn_values_percent, thdn_values_db, harmonic_amplitudes_list, harmonic_phases_list, snr_values = measure_multiple(
                args.num_measurements,
                device_index,
                input_channel,
                output_channel,
                tone_settings['frequency'],
                args.window,
                noise_rms,
                apply_bandpass=args.bandpass
            )

            # 平均計算
            compute_average_thd(thd_values_percent, thd_values_db)
            compute_average_thdn(thdn_values_percent, thdn_values_db)
            compute_average_snr(snr_values)
            compute_average_harmonics(harmonic_amplitudes_list)
            compute_average_phases(harmonic_phases_list)

        except Exception as e:
            print(f"測定中にエラーが発生しました: {e}")
        finally:
            stop_event.set()  # スレッド停止

    # 測定スレッド開始
    measurement_thread = threading.Thread(target=measurement_thread_func)
    measurement_thread.start()

    # プロット表示設定
    if not args.no_plot:
        # リアルタイムプロットを表示
        try:
            start_realtime_plot(args.window)
        except KeyboardInterrupt:
            print("\nプログラムを中断しました。")
        finally:
            stop_event.set()            # 測定スレッド停止
            measurement_thread.join()  # 測定スレッド待機
            plt.close('all')           # プロット閉鎖
            print("プログラムを正常に終了しました。")
    else:
        # プロット非表示モード
        try:
            measurement_thread.join()
        except KeyboardInterrupt:
            print("\nプログラムを中断しました。")
        finally:
            print("プログラムを正常に終了しました。")

if __name__ == '__main__':
    main()
