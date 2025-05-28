"""
audio_signal_generator.py - Version 1.0.2

A Python script to generate various audio signals including tones, noise, and frequency sweeps. 
Supports customizable options for sample rate, bit depth, fade in/out, and more.

Usage:
    python audio_signal_generator.py -f 440 -d 5 -o output.wav
    python audio_signal_generator.py --sweep --start_freq 20 --end_freq 20000 -d 10 -o sweep.wav

Author: ChatGPT and vach
Date: 2024/9/26
copyright pass
"""

import numpy as np
import soundfile as sf
import argparse

# Common Audio Library Imports
from common_audio_lib.audio_io_utils import generate_sine_wave as common_generate_sine_wave
from common_audio_lib.signal_processing_utils import dbfs_to_linear


# 指定された周波数のサイン波を生成する関数 (Refactored to use common_audio_lib)
def generate_tone(frequency, duration, sample_rate):
    # Option B: Call common_generate_sine_wave with a high reference amplitude (e.g., -0.1 dBFS).
    # save_tone_to_wav will then normalize this and scale to the user-specified args.dbfs.
    # The phase argument for common_generate_sine_wave defaults to 0.0, which is fine.
    # Amplitude is specified in dBFS.
    # common_generate_sine_wave(frequency: float, amplitude_dbfs: float, duration: float, sample_rate: float, phase: float = 0.0)
    reference_dbfs_for_generation = -0.1 # High amplitude, but with a tiny bit of headroom
    tone = common_generate_sine_wave(
        frequency=frequency,
        amplitude_dbfs=reference_dbfs_for_generation,
        duration=duration,
        sample_rate=sample_rate
    )
    return tone

#  
def generate_square_wave(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 符号関数を使用して矩形波を生成
    square_wave = np.sign(np.sin(2 * np.pi * frequency * t))
    return square_wave

# 指定された周波数の三角波を生成する関数
def generate_triangle_wave(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 三角波を生成 (振幅0.5)
    triangle_wave = (2 * np.abs(2 * ((t * frequency) % 1) - 1) - 1)
    return triangle_wave

# 指定された周波数のノコギリ波を生成する関数 (上昇/下降を指定可能)
def generate_sawtooth_wave(frequency, duration, sample_rate, ramp_type='ramp+'):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if ramp_type == 'ramp+':  # 上昇ノコギリ波
        sawtooth_wave = 2 * ((t * frequency) % 1) - 1
    elif ramp_type == 'ramp-':  # 下降ノコギリ波
        sawtooth_wave = -2 * ((t * frequency) % 1) + 1
    else:
        raise ValueError("ramp_typeは'ramp+'または'ramp-'でなければなりません。")
    return sawtooth_wave

# 周波数スイープを生成する関数
def generate_sweep(start_freq, end_freq, duration, sample_rate, logarithmic=False, wave_type='sine'):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if logarithmic: # 対数スイープ
        k = (end_freq / start_freq)**(1 / duration)
        frequencies = start_freq * (end_freq / start_freq) ** (t / duration)
    else: # 線形スイープ
        k = (end_freq - start_freq) / duration
        frequencies = start_freq + k * t

    if logarithmic:
        sweep = np.sin(2 * np.pi * start_freq * ((k**t - 1) / np.log(k)))
    else:
        sweep = np.sin(2 * np.pi * (start_freq * t + (k / 2) * t * t))
    
    if wave_type == 'square': # 矩形波スイープ
        sweep = np.sign(sweep) 
    
    return sweep

# A-weightingカーブの値を計算する関数
def a_weight(f):
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    ra = f4**2 * f**4 / ((f**2 + f1**2) * np.sqrt((f**2 + f2**2) * (f**2 + f3**2)) * (f**2 + f4**2))
    return ra

# B-weightingカーブの値を計算する関数
def b_weighting(f):
    f1 = 20.598997
    f2 = 158.48932
    f3 = 737.86223
    f4 = 12194.217
    rb = f4 * f**3 / ((f**2 + f1**2) * np.sqrt(f**2 + f2**2) * (f**2 + f4**2))
    return rb

# C-weightingカーブの値を計算する関数
def c_weighting(f):
    f1 = 20.598997
    f4 = 12194.217
    rc = f4**2 * f**2 / ((f**2 + f1**2) * (f**2 + f4**2))
    return rc

# A, B, Cの重み付けを適用する関数
def apply_weighting(tone, sample_rate, weighting_type):
    num_samples = len(tone)
    frequencies = np.fft.rfftfreq(num_samples, d=1/sample_rate)

    if weighting_type == 'A':
        weighting_curve = a_weight(frequencies)
    elif weighting_type == 'B':
        weighting_curve = b_weighting(frequencies)
    elif weighting_type == 'C':
        weighting_curve = c_weighting(frequencies)
    else:
        raise ValueError("Weighting type must be 'A', 'B', or 'C'.")

    freq_spectrum = np.fft.rfft(tone)
    freq_spectrum *= weighting_curve
    weighted_tone = np.fft.irfft(freq_spectrum)

    return weighted_tone - np.mean(weighted_tone)  # DCオフセットを除去


# カラードノイズを生成する関数
def generate_noise(duration, sample_rate, color='white'):
    num_samples = int(duration * sample_rate)
    noise = np.random.randn(num_samples)

    def apply_filter(noise, frequencies, scaling_factors): # フィルタ適用関数 (共通処理)
        freq_spectrum = np.fft.rfft(noise)
        freq_spectrum *= scaling_factors
        filtered_noise = np.fft.irfft(freq_spectrum)
        return filtered_noise - np.mean(filtered_noise) # DCオフセットを除去

    frequencies = np.fft.rfftfreq(num_samples, d=1/sample_rate)

    if color == 'pink':
        scaling_factors = 1 / np.sqrt(frequencies + 1e-10) # ピンクノイズ
        noise = apply_filter(noise, frequencies, scaling_factors)
    elif color == 'grey': # グレーノイズ (A-weightingの逆特性)
        scaling_factors = 1 / a_weight(frequencies)
        scaling_factors[scaling_factors > 1e5] = 1  # 無限大を避けるクリッピング
        noise = apply_filter(noise, frequencies, scaling_factors)
    elif color == 'brown' or color == 'red': # ブラウンノイズ/レッドノイズ
        scaling_factors = 1 / (frequencies + 1e-10)  # ゼロ除算防止
        noise = apply_filter(noise, frequencies, scaling_factors)
    elif color == 'blue': # ブルーノイズ
        scaling_factors = np.sqrt(frequencies)
        noise = apply_filter(noise, frequencies, scaling_factors)
    elif color == 'purple' or color == 'violet': # パープルノイズ/バイオレットノイズ
        scaling_factors = frequencies
        noise = apply_filter(noise, frequencies, scaling_factors)

    return noise / np.max(np.abs(noise))  # 正規化


# フェードイン・フェードアウトを適用する関数
def apply_fade(tone, sample_rate, fade_duration):
    num_samples = len(tone)
    fade_in_samples = int(sample_rate * fade_duration)
    fade_out_samples = int(sample_rate * fade_duration)

    if num_samples < fade_in_samples + fade_out_samples:
        raise ValueError("音声の長さがフェード時間に対して短すぎます。")

    # フェードカーブを作成
    fade_in = np.linspace(0, 1, fade_in_samples)
    fade_out = np.linspace(1, 0, fade_out_samples)

    # フェードを適用
    tone[:fade_in_samples] *= fade_in
    tone[-fade_out_samples:] *= fade_out

    return tone


# 音量スイープを生成する関数
def generate_volume_sweep(tone, sample_rate, duration, logarithmic=False):
    num_samples = len(tone)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    if logarithmic: # 対数スイープ
        volume_envelope = np.logspace(-3, np.log10(1), num_samples) # 0.001から0.5まで
    else: # 線形スイープ
        volume_envelope = np.linspace(0, 1, num_samples) # 0から0.5まで

    return tone * volume_envelope

def save_tone_to_wav(filename, tone, sample_rate, bit_depth, fade_duration=None, dbfs=-3):
    if fade_duration:
        tone = apply_fade(tone, sample_rate, fade_duration)

    # dBFS値に基づいて振幅を調整
    # The input 'tone' from the refactored generate_tone is already scaled (e.g. to -0.1 dBFS).
    # We still normalize it to ensure its peak is 1.0 before applying the final dbfs scaling.
    # This makes the behavior consistent regardless of the exact peak of the input 'tone'.
    max_abs_val = np.max(np.abs(tone))
    if max_abs_val > 1e-9: # Avoid division by zero or by very small numbers if signal is silent
        normalized_tone = tone / max_abs_val
    else: # Signal is effectively silent
        normalized_tone = tone # Keep it as is (all zeros or very small values)

    # Convert target dbfs to linear scale using common_audio_lib function
    target_linear_amplitude = dbfs_to_linear(dbfs)
    
    # Scale the normalized tone to the target linear amplitude
    scaled_tone = normalized_tone * target_linear_amplitude
    
    # Ensure the final scaled tone does not exceed [-1.0, 1.0] due to any potential float precision issues
    # or if dbfs > 0 was requested (though typically dbfs <= 0).
    # common_generate_sine_wave already clips, but this is a final safety for this function's output.
    final_tone_to_save = np.clip(scaled_tone, -1.0, 1.0)


    # ビット深度に応じてデータを変換
    if bit_depth == '16':
        audio_data = np.int16(final_tone_to_save * 32767)
    elif bit_depth == '24':
        # For 24-bit, soundfile expects data to be scaled as if it were int32,
        # but it will be written as 24-bit.
        # Scaling to int32 range then letting soundfile handle PCM_24 subtype is correct.
        audio_data = np.int32(final_tone_to_save * 2147483647) 
    elif bit_depth == '32': # PCM_32 (integer)
        audio_data = np.int32(final_tone_to_save * 2147483647)
    elif bit_depth == 'float': # FLOAT (32-bit float)
        audio_data = final_tone_to_save.astype(np.float32)
    else:
        raise ValueError("ビット深度は16, 24, 32, または'float'で指定してください。")

    sf.write(filename, audio_data, sample_rate, subtype=get_wav_subtype(bit_depth))


# WAVファイルのサブタイプを取得する関数
def get_wav_subtype(bit_depth):
    if bit_depth == '16':
        return 'PCM_16'
    elif bit_depth == '24':
        return 'PCM_24'
    elif bit_depth == '32':
        return 'PCM_32'
    elif bit_depth == 'float':
        return 'FLOAT'
    else:
        raise ValueError("無効なビット深度です。16, 24, 32, または'float'から選択してください。")


def main():
    parser = argparse.ArgumentParser(description='トーンやノイズ、スイープなどを生成するツール')
    
    # 引数の定義
    parser.add_argument('-i', '--input', type=str, help='入力WAVファイル (オプション)')
    parser.add_argument('-f', '--frequency', type=float, default=1000, help='トーンの周波数 (Hz、デフォルト: 1kHz)')
    parser.add_argument('--sweep', action='store_true', help='周波数スイープを生成する')
    parser.add_argument('--log_sweep', action='store_true', help='対数スイープを使用する')
    parser.add_argument('--start_freq', type=float, default=20, help='スイープ開始周波数 (Hz、デフォルト: 20Hz)')
    parser.add_argument('--end_freq', type=float, default=20000, help='スイープ終了周波数 (Hz、デフォルト: 20kHz)')
    parser.add_argument('--sweep_type', choices=['sine', 'square'], default='sine', help='スイープの波形タイプ (sine, square, デフォルト: sine)')

    parser.add_argument('--noise', choices=['white', 'pink', 'grey', 'brown', 'red', 'blue', 'purple', 'violet'], help='ノイズの色を指定 (例: --noise pink)')
    parser.add_argument('--square', action='store_true', help='矩形波を生成する')
    parser.add_argument('--triangle', action='store_true', help='三角波を生成する')
    parser.add_argument('--sawtooth', action='store_true', help='ノコギリ波を生成する')
    parser.add_argument('--ramp_type', choices=['ramp+', 'ramp-'], default='ramp+', help='ノコギリ波の上昇/下降タイプ (ramp+ または ramp-, デフォルト: ramp+)')


    parser.add_argument('--volume_sweep', action='store_true', help='音量スイープを生成する')
    parser.add_argument('--log_volume_sweep', action='store_true', help='対数音量スイープを使用する')
    parser.add_argument('-s', '--sample_rate', type=int, default=48000, help='サンプルレート (Hz、デフォルト: 48000Hz)')
    parser.add_argument('-b', '--bit_depth', type=str, default='float', choices=['16', '24', '32', 'float'], help='ビット深度 (デフォルト: float)')
    parser.add_argument('-d', '--duration', type=float, default=5, help='持続時間 (秒、デフォルト: 5秒)')
    parser.add_argument('-o', '--output', type=str, default='tone.wav', help='出力ファイル名 (デフォルト: tone.wav)')
    parser.add_argument('--fade_duration', type=float, default=None, help='フェードイン・フェードアウトの持続時間 (秒)')
    parser.add_argument('--weighting', choices=['A', 'B', 'C'], help='重み付けの種類 (A, B, C)')
    parser.add_argument('--dbfs', type=float, default=-3, help='出力音量 (dBFS, デフォルト: -3dBFS)')


    args = parser.parse_args()

    # 入力ファイル、スイープ、ノイズ、波形に応じて音声を生成
    if args.input: # 入力ファイルあり
        try:
            tone, args.sample_rate = sf.read(args.input)
            args.duration = len(tone) / args.sample_rate  # 入力ファイルの長さに基づいて再生時間を設定
        except Exception as e:
            print(f"入力ファイルの読み込みエラー: {e}")
            return
    elif args.sweep: # スイープ生成
        tone = generate_sweep(args.start_freq, args.end_freq, args.duration, args.sample_rate, args.log_sweep, args.sweep_type)
    elif args.noise: # ノイズ生成
        tone = generate_noise(args.duration, args.sample_rate, args.noise)
    elif args.square: # 矩形波生成
        tone = generate_square_wave(args.frequency, args.duration, args.sample_rate)
    elif args.triangle: # 三角波生成
        tone = generate_triangle_wave(args.frequency, args.duration, args.sample_rate)
    elif args.sawtooth: # ノコギリ波生成
        tone = generate_sawtooth_wave(args.frequency, args.duration, args.sample_rate, args.ramp_type)
    else: # サイン波生成 (デフォルト)
        tone = generate_tone(args.frequency, args.duration, args.sample_rate)

    # 音量スイープを適用
    if args.volume_sweep or args.log_volume_sweep:
        tone = generate_volume_sweep(tone, args.sample_rate, args.duration, args.log_volume_sweep)

    # 重み付けを適用
    if args.weighting:
        tone = apply_weighting(tone, args.sample_rate, args.weighting)

    # WAVファイルに保存
    save_tone_to_wav(args.output, tone, args.sample_rate, args.bit_depth, args.fade_duration, args.dbfs)


if __name__ == '__main__':
    main()