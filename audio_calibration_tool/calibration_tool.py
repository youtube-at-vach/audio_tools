import sounddevice as sd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
import time
import configparser
from datetime import datetime
import os

# --- デフォルト設定 ---
DEVICE_ID = 2
SAMPLERATE = 44100
FREQUENCY = 1000
DURATION = 5
AMPLITUDE = 0.5 # 基準振幅
TEST_FILENAME = "loopback_test.wav"

def dbfs(signal):
    """信号のRMSからdBFS値を計算する"""
    if np.max(np.abs(signal)) == 0:
        return -np.inf
    rms = np.sqrt(np.mean(signal**2))
    return 20 * np.log10(rms)

def peak_dbfs(signal):
    """信号のピークからdBFS値を計算する"""
    if np.max(np.abs(signal)) == 0:
        return -np.inf
    peak = np.max(np.abs(signal))
    return 20 * np.log10(peak)

def save_calibration_settings(physical_gain, filepath):
    """校正データをINIファイルに保存します。"""
    config = configparser.ConfigParser()
    config['Calibration'] = {
        'physical_gain': physical_gain,
        'last_calibration_date': datetime.now().isoformat()  # 現在の日付をISOフォーマットで保存
    }
    
    with open(filepath, 'w') as configfile:
        config.write(configfile)
    
    print(f"設定が {filepath} に保存されました。")

def load_calibration_settings(filepath):
    """INIファイルから校正データを読み込みます。"""
    config = configparser.ConfigParser()
    
    try:
        config.read(filepath)
        physical_gain = config.getfloat('Calibration', 'physical_gain')
        last_calibration_date = config.get('Calibration', 'last_calibration_date')
        return physical_gain, last_calibration_date
    except (FileNotFoundError, configparser.NoSectionError, configparser.NoOptionError):
        print(f"{filepath} が見つかりません。")
        return None, None

def list_devices():
    """利用可能なオーディオデバイスを一覧表示する"""
    print("利用可能なオーディオデバイス:")
    print(sd.query_devices())

def run_calibration(device_id, samplerate, frequency, amplitude):
    """
    キャリブレーションモード: 基準信号を再生し、入力レベルをリアルタイムで表示する。
    ユーザーは表示を見ながら出力ノブを調整する。
    """
    print("--- キャリブレーション開始 ---")
    print(f"デバイスID: {device_id}")
    print(f"基準信号: {frequency} Hz サイン波")
    print("目標入力レベル: 約 -6 dBFS (振幅 {amplitude} の場合)")
    print("Ctrl+C を押すと終了します。")

    # 出力信号 (右チャンネルのみ)
    blocksize = samplerate # 1秒ごとに更新
    output_signal = np.zeros((blocksize, 2), dtype='float32')
    t = np.arange(blocksize) / samplerate
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    output_signal[:, 1] = sine_wave

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        outdata[:] = output_signal[:frames]
        # 入力レベルを計算して表示
        volume = dbfs(indata[:, 0]) # 左チャンネルの入力
        # プログレスバーのように表示
        bar_length = 40
        level = int((volume + 60) / 60 * bar_length) # -60dBFSを0とする
        bar = '#' * level + '-' * (bar_length - level)
        print(f"\r入力レベル: [{bar}] {volume:+.2f} dBFS", end='')

    try:
        with sd.Stream(device=device_id, samplerate=samplerate, channels=2, callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nキャリブレーションを終了しました。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")

def run_loopback_test(device_id, samplerate, frequency, duration, amplitude, filename):
    """ループバックテストを実行し、結果をWAVファイルに保存する"""
    print("--- ループバックテスト開始 ---")
    try:
        device_info = sd.query_devices(device_id)
        print(f"テストデバイス: {device_info['name']}")

        num_samples = int(duration * samplerate)
        t = np.linspace(0., duration, num_samples, endpoint=False)
        sine_wave = amplitude * np.sin(2. * np.pi * frequency * t)

        output_signal = np.zeros((num_samples, 2), dtype='float32')
        output_signal[:, 1] = sine_wave

        print(f"{duration}秒間、右チャンネルからサイン波を再生し、左チャンネルで録音します...")
        recorded_signal = sd.playrec(
            output_signal, samplerate=samplerate, channels=1, device=device_id, blocking=True
        )
        sf.write(filename, recorded_signal, samplerate)
        print(f"テスト完了。録音データを '{filename}' に保存しました。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

def analyze_wav_file(filename):
    """WAVファイルをFFT分析し、周波数スペクトルをプロットする"""
    print("--- WAVファイル分析開始 ---")
    try:
        data, samplerate = sf.read(filename)
        print(f"'{filename}' を読み込みました (サンプルレート: {samplerate} Hz)")

        signal = data[:, 0] if data.ndim > 1 else data
        n = len(signal)
        # Apply a Hann window to the signal to reduce spectral leakage
        window = np.hanning(n)
        windowed_signal = signal * window
        fft_result = np.abs(np.fft.fft(windowed_signal)[:n//2])
        fft_freq = np.fft.fftfreq(n, d=1/samplerate)[:n//2]

        peak_index = np.argmax(fft_result)
        peak_frequency = fft_freq[peak_index]
        peak_amplitude = 20 * np.log10(fft_result[peak_index])

        print(f"ピーク周波数: {peak_frequency:.2f} Hz")
        print(f"ピーク振幅: {peak_amplitude:.2f} dB")

        plt.figure(figsize=(12, 6))
        plt.plot(fft_freq, 20 * np.log10(fft_result))
        plt.title(f'Frequency Spectrum of {filename}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.grid(True)
        plt.xlim(0, samplerate / 2)
        plt.ylim(np.max(20 * np.log10(fft_result)) - 100, np.max(20 * np.log10(fft_result)) + 10)
        plt.axvline(x=peak_frequency, color='r', linestyle='--', label=f'Peak: {peak_frequency:.2f} Hz')
        plt.legend()
        
        output_plot = 'frequency_spectrum.png'
        plt.savefig(output_plot)
        print(f"周波数スペクトルグラフを '{output_plot}' に保存しました。")

    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。先に 'test' モードを実行してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def run_check_mode(device_id, samplerate, frequency, amplitude):
    """
    チェックモード: 基準信号を再生し、入力レベルを測定してキャリブレーションの成否を判定する。
    """
    print("--- キャリブレーションチェック開始 ---")
    print(f"デバイスID: {device_id}")
    print(f"基準信号: {frequency} Hz サイン波")

    recorded_data = []
    frame_counter = 0
    duration = 1.0 # 短時間で測定

    def callback(indata, outdata, frames, time, status):
        nonlocal frame_counter
        if status:
            print(status)

        # 出力信号 (右チャンネルのみ)
        t = (np.arange(frames) + frame_counter) / samplerate
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
        outdata[:, 1] = sine_wave
        outdata[:, 0] = 0 # 左チャンネルは0
        frame_counter += frames

        # 入力信号 (左チャンネル)
        if indata.shape[1] >= 1:
            recorded_data.append(indata[:, 0].copy())

    try:
        with sd.Stream(device=device_id, samplerate=samplerate, channels=2, callback=callback):
            sd.sleep(int(duration * 1000))

        if not recorded_data:
            print("録音データがありません。")
            return False

        audio = np.concatenate(recorded_data)
        volume = peak_dbfs(audio) # 左チャンネルの入力 (ピークdBFS)

        # キャリブレーション成功の判定基準 (例: -7 dBFS から -5 dBFS の範囲内)
        # 基準振幅が0.5の場合、目標は-6dBFS
        # 許容範囲を±1dBとする
        target_dbfs = peak_dbfs(np.array([amplitude]))
        lower_bound = target_dbfs - 1.0
        upper_bound = target_dbfs + 1.0

        if lower_bound <= volume <= upper_bound:
            print(f"キャリブレーション成功: {volume:+.2f} dBFS (目標ピーク: {target_dbfs:+.2f} dBFS, 範囲: {lower_bound:+.2f} ~ {upper_bound:+.2f} dBFS)")
            return volume - target_dbfs # 物理ゲインを返す
        else:
            print(f"キャリブレーション失敗: {volume:+.2f} dBFS (目標ピーク: {target_dbfs:+.2f} dBFS, 範囲: {lower_bound:+.2f} ~ {upper_bound:+.2f} dBFS)")
            return volume - target_dbfs # 物理ゲインを返す

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None # エラー時はNoneを返す


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="オーディオデバイスのキャリブレーションとテストを行うツール")
    parser.add_argument(
        'mode',
        choices=['list', 'calibrate', 'test', 'analyze', 'check'],
        help='''実行するモードを選択: 
'list': デバイス一覧表示, 
'calibrate': 出力レベルキャリブレーション, 
'test': ループバックテスト実行, 
'analyze': テスト結果の分析, 
'check': 入力レベルのリアルタイム確認'''
    )
    parser.add_argument("-d", "--device", type=int, default=DEVICE_ID, help=f"オーディオデバイスID (デフォルト: {DEVICE_ID})")
    parser.add_argument("-sr", "--samplerate", type=int, default=SAMPLERATE, help=f"サンプルレート (デフォルト: {SAMPLERATE})")
    parser.add_argument("-f", "--frequency", type=int, default=FREQUENCY, help=f"周波数 (Hz) (デフォルト: {FREQUENCY})")
    parser.add_argument("-a", "--amplitude", type=float, default=AMPLITUDE, help=f"振幅 (0.0-1.0) (デフォルト: {AMPLITUDE})")
    parser.add_argument("--duration", type=float, default=DURATION, help=f"テスト時間 (秒) (デフォルト: {DURATION})")
    parser.add_argument("--file", type=str, default=TEST_FILENAME, help=f"テスト用ファイル名 (デフォルト: {TEST_FILENAME})")

    args = parser.parse_args()

    # モードに応じて実行
    if args.mode == 'list':
        list_devices()
    elif args.mode == 'calibrate':
        run_calibration(args.device, args.samplerate, args.frequency, args.amplitude)
    elif args.mode == 'test':
        run_loopback_test(args.device, args.samplerate, args.frequency, args.duration, args.amplitude, args.file)
    elif args.mode == 'analyze':
        analyze_wav_file(args.file)
    elif args.mode == 'check':
        physical_gain = run_check_mode(args.device, args.samplerate, args.frequency, args.amplitude)
        if physical_gain is not None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '..')) # audio_tools/audio_calibration_tool から audio_tools/ へ
            calibration_filepath = os.path.join(project_root, 'calibration_settings.ini')
            save_calibration_settings(physical_gain, calibration_filepath)
            print(f"物理ゲイン {physical_gain:+.2f} dB を {calibration_filepath} に保存しました。")