import sounddevice as sd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
import time

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
        fft_result = np.abs(np.fft.fft(signal)[:n//2])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="オーディオデバイスのキャリブレーションとテストを行うツール")
    parser.add_argument(
        'mode',
        choices=['list', 'calibrate', 'test', 'analyze'],
        help="実行するモードを選択: \n'list': デバイス一覧表示, \n'calibrate': 出力レベルキャリブレーション, \n'test': ループバックテスト実行, \n'analyze': テスト結果の分析"
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
