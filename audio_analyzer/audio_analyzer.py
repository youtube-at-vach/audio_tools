#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Analyzer v1.4.5
---------------------------------
音声信号の高調波解析を行うスクリプトです。
基本周波数、高調波、THD、THD+N、SNR、ゲインの測定をサポートします。

特徴:
- ピーク検出と高調波解析
- THDおよびTHD+Nの測定
- SNRの測定
- 入力振幅と測定振幅の比較によるゲイン表示
- 各測定終了時に高調波解析結果の表示
- 平均測定結果に標準偏差の表示
- スイープモード（周波数スイープ、振幅スイープ）による連続測定
- テストトーンの出力機能
- **新機能: --mapによる周波数・振幅のマッピング測定**

作成者: ChatGPT および vach
日付: 2024-10-16
"""
import configparser
import argparse
import sys
from datetime import datetime
import csv
import re

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import get_window
from rich import print
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt

from audiocalc import AudioCalc  # 計算モジュールをインポート
from aligner import SignalAligner  # SignalAlignerをインポート

console = Console()

def save_calibration_settings(output_conversion, input_conversion, filename='calibration_settings.ini'):
    """校正データをINIファイルに保存します。"""
    config = configparser.ConfigParser()
    config['Calibration'] = {
        'output_conversion': output_conversion,
        'input_conversion': input_conversion,
        'last_calibration_date': datetime.now().isoformat()  # 現在の日付をISOフォーマットで保存
    }
    
    with open(filename, 'w') as configfile:
        config.write(configfile)
    
    print(f"設定が {filename} に保存されました。")

def load_calibration_settings(filename='calibration_settings.ini'):
    """INIファイルから校正データを読み込みます。"""
    config = configparser.ConfigParser()
    
    try:
        config.read(filename)
        output_conversion = config.getfloat('Calibration', 'output_conversion')
        input_conversion = config.getfloat('Calibration', 'input_conversion')
        last_calibration_date = config.get('Calibration', 'last_calibration_date')
        return output_conversion, input_conversion, last_calibration_date
    except (FileNotFoundError, configparser.NoSectionError, configparser.NoOptionError):
        print(f"{filename} が見つかりません。")
        return None

def generate_tone(frames, frequency, amplitude, phase, sample_rate, frame_counter):
    """指定フレーム数のシングルトーンを生成"""
    t = (np.arange(frames) + frame_counter) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return np.clip(signal, -1.0, 1.0), frame_counter + frames

def parse_input(input_string):
    """
    入力された文字列から数値と単位をパースします。

    :param input_string: 数値と単位を含む文字列（例: "5.5 V", "200 mV"）
    :return: タプル (数値, 単位)
    """
    # 正規表現パターンを定義
    pattern = r'([\d.-]+)\s*([mnu]?V|dBV)'
    match = re.match(pattern, input_string.strip())
    
    if match:
        value = float(match.group(1))  # 数値部分を取得
        unit = match.group(2)           # 単位部分を取得
        return value, unit
    else:
        raise ValueError("無効な入力形式です。数値と単位を含む文字列を入力してください。")

def convert_to_volts(value, unit):
    """
    与えられた値を指定された単位からボルト（V）に変換します。

    :param value: 変換する電圧の値（数値）
    :param unit: 単位（'V', 'mV', 'uV', 'nV', 'dBV' のいずれか）
    :return: ボルト（V）に変換された値
    """
    if unit == 'V':
        return value
    elif unit == 'mV':
        return value / 1000  # ミリボルトをボルトに変換
    elif unit == 'uV':
        return value / 1_000_000  # マイクロボルトをボルトに変換
    elif unit == 'nV':
        return value / 1_000_000_000  # ナノボルトをボルトに変換
    elif unit == 'dBV':
        return 10 ** (value / 20)  # dBVをボルトに変換
    else:
        raise ValueError("無効な単位です。'V', 'mV', 'uV', 'nV', 'dBV' のいずれかを指定してください。")

def test_tone(frequency, amplitude, sample_rate, device_index, output_channel, wait=True):
    """テストトーンを出力し、エンターキーで停止"""
    if wait:
        console.print("[green]テストトーンを出力します。停止するにはエンターキーを押してください。[/green]")
    frame_counter = 0

    def callback(outdata, frames, time, status):
        nonlocal frame_counter
        if status:
            console.print(f"[yellow]ストリームステータス: {status}[/yellow]")
        tone, frame_counter = generate_tone(frames, frequency, amplitude, 0, sample_rate, frame_counter)
        outdata[:] = tone.reshape(-1, 1)  # モノラル出力

    stream = sd.OutputStream(callback=callback, channels=output_channel, samplerate=sample_rate, device=device_index)
    stream.start()

    if wait:
        try:
            input()  # エンターキーを待つ
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()
            console.print("[green]テストトーンを停止しました。[/green]")
    return stream

def get_voltage_input(prompt_message):
    """電圧と単位を入力させ、無効な単位の場合は再入力を促す"""
    while True:
        voltage_input = Prompt.ask(prompt_message)
        try:
            measured_voltage, unit = parse_input(voltage_input) 
            measured_voltage = float(measured_voltage)  # 電圧を数値に変換
            convert_to_volts(measured_voltage, unit)  # 単位の検証
            return measured_voltage, unit  # 有効な入力の場合、値を返す
        except (ValueError, IndexError):
            console.print("[red]無効な入力です。'V', 'mV', 'uV', 'nV', 'dBV' のいずれかの単位を含む形式で再度入力してください。[/red]")

def calibration_mode(sample_rate, device_index, output_channel, amplitude_dbfs):
    """校正モードのメイン関数"""
    
    # -6dBFSのテストトーンを出力
    amplitude = 10 ** (amplitude_dbfs / 20)
    console.print(f"ピーク振幅が{amplitude_dbfs:.2f}dBFSの正弦波でキャリブレーションを行います")

    # テストトーンを出力、ストリームを開いたままにして入力を待つ
    stream = test_tone(1000, amplitude, sample_rate, device_index, output_channel, wait=False)
    measured_voltage, unit = get_voltage_input("AC電圧計で測定したRMS電圧と単位を入力してください (例: -12dBV, 25mV)")

    # テストトーンを止める
    stream.stop()
    stream.close()

    # 実測値をVrmsに変換したもの
    volt = convert_to_volts(measured_voltage, unit)

    # dBFSとdBVrmsのdB差を計算し表示
    output_factor = volt / amplitude # Vrms/dBFS
    output_conversion = 20 * np.log10(output_factor)
    console.print(f"出力のdBFSに対する換算量: 0dBFS = {output_conversion:+.2f} dBVrms")

    # ループバック測定準備のための停止
    Prompt.ask("[green]出力を入力に接続してください。用意ができたらエンターキーで続行します。[/green]")

    # テストトーンを出力して測定
    result = measure(
        device_index=device_index,
        output_channel=output_channel,
        frequency=1000,
        amplitude=amplitude,
        phase=0,
        duration=3,
        sample_rate=sample_rate,
        window_name='flattop', # 正確な振幅測定のためのwindow
        apply_bandpass=False,
        aligner=SignalAligner(trim_time_sec=0.20, samplerate=sample_rate) # 位相同期測定によりノイズを減らす
    )

    # 基本波の振幅FSと実測値Vの比を計算
    input_factor = volt / result['basic_wave']['max_amplitude']  # Vrms/dBFS
    input_conversion = 20 * np.log10(input_factor)
    console.print(f"入力のdBFSに対する換算量: 0dBFS = {input_conversion:+.2f} dBVrms")

    # 設定値を保存
    save_calibration_settings(output_conversion, input_conversion)

def select_device():
    """使用する音声デバイスを選択"""
    console.print("使用する音声デバイスを選択してください。\n")
    devices = sd.query_devices()
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("番号", justify="right")
    table.add_column("デバイス名")
    table.add_column("入力チャンネル")
    table.add_column("出力チャンネル")
    table.add_column("サンプリングレート")
    table.add_column("入力レイテンシ(ms)")
    table.add_column("出力レイテンシ(ms)")

    for i, device in enumerate(devices):
        table.add_row(
            str(i),
            device['name'],
            str(device['max_input_channels']),
            str(device['max_output_channels']),
            f"{device['default_samplerate']}",
            f"{device['default_low_input_latency'] * 1000:.2f}",
            f"{device['default_low_output_latency'] * 1000:.2f}"
        )
    console.print(table)

    while True:
        try:
            device_num = int(Prompt.ask("使用するデバイスの番号を入力してください"))
            if 0 <= device_num < len(devices):
                return device_num
            else:
                console.print("[red]有効なデバイス番号を入力してください。[/red]")
        except ValueError:
            console.print("[red]数値を入力してください。[/red]")

def print_harmonic_analysis(harmonics):
    """高調波解析結果を表形式で表示"""
    table = Table(title="ハーモニック解析結果", show_header=True, header_style="bold blue")
    table.add_column("次数", justify="right")
    table.add_column("周波数(Hz)", justify="right")
    table.add_column("振幅(dBr)", justify="right")
    table.add_column("位相差(deg)", justify="right")

    for h in harmonics:
        table.add_row(
            str(h.get('order', 'N/A')),
            f"{h.get('frequency', 'N/A'):.1f}" if h.get('frequency') is not None else "N/A",
            f"{h.get('amplitude_dbr', 'N/A'):.2f}" if h.get('amplitude_dbr') is not None else "N/A",
            f"{h.get('phase_deg', 'N/A'):.2f}" if h.get('phase_deg') is not None else "N/A"
        )
    console.print(table)

def measure(device_index, output_channel, frequency, amplitude, phase, duration, sample_rate, window_name, apply_bandpass, aligner):
    """音声測定を実行し、結果を解析"""
    recorded_data = []
    frame_counter = 0

    def callback(indata, outdata, frames, time, status):
        nonlocal frame_counter
        if status:
            console.print(f"[yellow]ストリームステータス: {status}[/yellow]")
        outdata.fill(0)
        if outdata.shape[1] >= 2:
            tone, frame_counter = generate_tone(frames, frequency, amplitude, phase, sample_rate, frame_counter)
            outdata[:, output_channel] = tone
        if indata.shape[1] >= 1:
            recorded_data.append(indata[:, 0].copy())

    with sd.Stream(callback=callback, channels=2, samplerate=sample_rate, device=device_index):
        console.print(f"[green]{frequency} Hz のトーンを再生中... 測定時間: {duration}秒[/green]")
        sd.sleep(int(duration * 1000))

    if not recorded_data:
        console.print("[red]録音データがありません。[/red]")
        return None

    # 録音データを結合
    audio = np.concatenate(recorded_data)

    # バンドパスフィルターを適用
    if apply_bandpass:
        audio = AudioCalc.bandpass_filter(audio, sample_rate)

    # 位相を考慮して正確にカット
    if aligner:
        align_audio, coef = aligner.align_signal(audio,
                            frequency=frequency, amplitude=1,
                            phase=np.pi/2, reference_duration=duration-1)

        if coef > 0.683:
            audio  = align_audio
            console.print(f"[green]位相同期: 成功, 相関係数:{coef:.2f}[/green]")
        else:
            console.print(f"[red]位相同期: 失敗, 相関係数:{coef:.2f}[/red]")
    else:
        console.print(f"位相同期: 非同期(位相差確度なし)")

    return AudioCalc.analyze_harmonics(
        audio_data=audio,
        fundamental_freq=frequency,
        window_name=window_name,
        sampling_rate=sample_rate,
        min_db=-140.0
    )

def measure_noise(device_index, output_channel, duration, sample_rate, apply_bandpass):
    """ノイズレベルを測定（トーン再生なし）"""
    console.print("\n[green]ノイズ測定を開始します。トーンは再生されません。[/green]")
    noise_data = []

    def callback(indata, outdata, frames, time, status):
        if status:
            console.print(f"[yellow]ストリームステータス: {status}[/yellow]")
        outdata.fill(0)
        if indata.shape[1] >= 1:
            noise_data.append(indata[:, 0].copy())

    with sd.Stream(callback=callback, channels=2, samplerate=sample_rate, device=device_index):
        sd.sleep(int(duration * 1000))

    if not noise_data:
        console.print("[red]ノイズ測定データがありません。[/red]")
        return None

    noise_audio = np.concatenate(noise_data)

    if apply_bandpass:
        noise_audio = AudioCalc.bandpass_filter(noise_audio, sample_rate)

    noise_rms = np.sqrt(np.mean(noise_audio ** 2))
    console.print(f"ノイズ RMS(dBFS): {20 * np.log10(noise_rms):.2f}\n")
    return noise_rms

def display_measurements(measurements):
    """測定結果をテーブル形式で表示"""
    console.print("[bold]=== 測定結果一覧 ===[/bold]")
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("測定", justify="right")
    table.add_column("周波数(Hz)", justify="right")
    table.add_column("振幅(dBFS)", justify="right")
    table.add_column("出力(dBFS)", justify="right")
    table.add_column("入力(dBFS)", justify="right")
    table.add_column("THD(%)", justify="right")
    table.add_column("THD+N(%)", justify="right")
    table.add_column("SNR(dB)", justify="right")
    table.add_column("ゲイン(dB)", justify="right")
    for i, m in enumerate(measurements):
        table.add_row(
            str(i + 1),
            f"{m.get('周波数(Hz)', 'N/A'):.1f}" if '周波数(Hz)' in m else "N/A",
            f"{m.get('振幅(dBFS)', 'N/A'):.2f}" if '振幅(dBFS)' in m else "N/A",
            f"{m['出力(dBFS)']:.2f}",
            f"{m['入力(dBFS)']:.2f}" if m['入力(dBFS)'] is not None else "N/A",
            f"{m['THD(%)']:.4f}" if m['THD(%)'] is not None else "N/A",
            f"{m['THD+N(%)']:.4f}" if m['THD+N(%)'] is not None else "N/A",
            f"{m['SNR(dB)']:.2f}" if m['SNR(dB)'] is not None else "N/A",
            f"{m['ゲイン(dB)']:+.2f}" if m['ゲイン(dB)'] is not None else "N/A"
        )
    console.print(table)

def display_statics(measurements):
    """平均測定結果と標準偏差を表示"""
    # 平均計算と表示
    thd_list = [m['THD(%)'] for m in measurements if m['THD(%)'] is not None]
    thdn_list = [m['THD+N(%)'] for m in measurements if m['THD+N(%)'] is not None]
    snr_list = [m['SNR(dB)'] for m in measurements if m['SNR(dB)'] is not None]
    gain_list = [m['ゲイン(dB)'] for m in measurements if m['ゲイン(dB)'] is not None]

    console.print("[bold]=== 平均測定結果 ===[/bold]")
    avg_table = Table(show_header=True, header_style="bold blue")
    avg_table.add_column("指標", justify="left")
    avg_table.add_column("平均値 ± 標準偏差", justify="right")

    if thd_list:
        avg_thd = np.mean(thd_list)
        std_thd = np.std(thd_list, ddof=1) if len(thd_list) > 1 else 0.0
        avg_table.add_row("全高調波歪 (THD)", f"{avg_thd:.4f}% ± {std_thd:.4f}%")
    else:
        avg_table.add_row("全高調波歪 (THD)", "N/A")

    if thdn_list:
        avg_thdn = np.mean(thdn_list)
        std_thdn = np.std(thdn_list, ddof=1) if len(thdn_list) > 1 else 0.0
        avg_table.add_row("全高調波歪およびノイズ (THD+N)", f"{avg_thdn:.4f}% ± {std_thdn:.4f}%")
    else:
        avg_table.add_row("全高調波歪およびノイズ (THD+N)", "N/A")

    if snr_list:
        avg_snr = np.mean(snr_list)
        std_snr = np.std(snr_list, ddof=1) if len(snr_list) > 1 else 0.0
        avg_table.add_row("SNR(dB)", f"{avg_snr:.2f} dB ± {std_snr:.2f} dB")
    else:
        avg_table.add_row("SNR(dB)", "N/A")

    if gain_list:
        avg_gain = np.mean(gain_list)
        std_gain = np.std(gain_list, ddof=1) if len(gain_list) > 1 else 0.0
        avg_table.add_row("ゲイン(dB)", f"{avg_gain:+.2f} dB ± {std_gain:+.2f} dB")
    else:
        avg_table.add_row("ゲイン(dB)", "N/A")

    console.print(avg_table)

def generate_octave_frequencies(step=1):
    """1/3オクターブの中心周波数を生成"""
    freqs = [20, 25, 31.5, 40, 50, 63, 80, 100,
           125, 160, 200, 250, 315, 400, 500,
           630, 800, 1000, 1250, 1600, 2000,
           2500, 3150, 4000, 5000, 6300, 8000,
           10000, 12500, 16000, 20000]
    return freqs[::step]

def results_to_measurement(result, noise_rms, output_dbfs, freq=None, amp_dbfs=None):
    """測定結果を辞書形式に整形"""
    measurement = {}
    if freq is not None:
        measurement['周波数(Hz)'] = freq
    if amp_dbfs is not None:
        measurement['振幅(dBFS)'] = amp_dbfs

    if result:
        thd = result['thd_percent']
        thd_db = result['thd_db']
        thdn = result['thdn_percent']
        thdn_db = result['thdn_db']
        signal_rms = result['basic_wave']['max_amplitude'] / np.sqrt(2)
        snr = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else -140.00
        gain_db = result['basic_wave']['amplitude_dbfs'] - output_dbfs

        measurement.update({
            '出力(dBFS)': output_dbfs,   # 数値型
            '入力(dBFS)': result['basic_wave']['amplitude_dbfs'],  # 数値型
            'THD(%)': thd,                  # 数値型
            'THD(dBr)': thd_db,            
            'THD+N(%)': thdn,               # 数値型
            'THD+N(dBr)':thdn_db,
            'SNR(dB)': snr,                 # 数値型
            'ゲイン(dB)': gain_db           # 数値型
        })

        console.print("[cyan]測定結果:[/cyan]")
        console.print(f"出力(dBFS): {output_dbfs:.2f}")
        console.print(f"入力(dBFS): {result['basic_wave']['amplitude_dbfs']:.2f}")
        console.print(f"THD(%, dBr): {thd:.4f} / {thd_db:.2f}")
        console.print(f"THD+N(%, dBr): {thdn:.4f} / {thdn_db:.2f}")
        console.print(f"SNR(dB): {snr:.2f}")
        console.print(f"ゲイン(dB): {gain_db:+.2f}\n")

        print_harmonic_analysis(result['harmonics'])
    else:
        measurement.update({
            '出力(dBFS)': output_dbfs,   # 数値型
            '入力(dBFS)': None,
            'THD(%)': None,
            'THD(dBr)': None,
            'THD+N(%)': None,
            'THD+N(dBr)': None,
            'SNR(dB)': None,
            'ゲイン(dB)': None
        })
        console.print("[yellow]測定結果:[/yellow]")
        console.print(f"出力(dBFS): {output_dbfs:.2f}")
        console.print("入力(dBFS): N/A")
        console.print("THD(%, dBr): N/A")
        console.print("THD+N(%, dBr): N/A")
        console.print("SNR(dB): N/A")
        console.print("ゲイン(dB): N/A\n")

    return measurement

def perform_measurements(device_index, output_channel, sample_rate, apply_bandpass, frequency_list, amplitude_list, duration, window_name, aligner, output_csv=None):
    """共通の測定処理を実行"""
    measurements = []

    # ノイズ測定
    noise_rms = measure_noise(
        device_index=device_index,
        output_channel=output_channel,
        duration=duration,
        sample_rate=sample_rate,
        apply_bandpass=apply_bandpass
    )
    if noise_rms is None:
        console.print("[red]ノイズ測定に失敗したため、プログラムを終了します。[/red]")
        sys.exit(1)

    # CSVファイルが指定されている場合、書き込み準備
    if output_csv:
        csvfile = open(output_csv, mode='w', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(csvfile, fieldnames=['Measurement_Number', 'Frequency(Hz)', 'Amplitude(dBFS)', 'Output(dBFS)', 'Input(dBFS)', 'THD(%)', 'THD(dBr)', 'THD+N(%)', 'THD+N(dBr)', 'SNR(dB)', 'Gain(dB)'])
        csv_writer.writeheader()
    else:
        csv_writer = None

    try:
        for idx, (freq, amp_dbfs) in enumerate(zip(frequency_list, amplitude_list), start=1):
            console.print(f"\n[bold]--- 測定 {idx}: 周波数={freq} Hz, 振幅={amp_dbfs} dBFS ---[/bold]")
            amplitude = 10 ** (amp_dbfs / 20)
            result = measure(
                device_index=device_index,
                output_channel=output_channel,
                frequency=freq,
                amplitude=amplitude,
                phase=0,
                duration=duration,
                sample_rate=sample_rate,
                window_name=window_name,
                apply_bandpass=apply_bandpass,
                aligner=aligner
            )

            measurement = results_to_measurement(result, noise_rms, amp_dbfs, freq=freq, amp_dbfs=amp_dbfs)
            measurements.append(measurement)

            if csv_writer:
                # CSVの見出しを英語にしています。matplotlibが日本語だと文字化けするためです。
                csv_writer.writerow({
                    'Measurement_Number': idx,
                    'Frequency(Hz)': freq,
                    'Amplitude(dBFS)': amp_dbfs,
                    'Output(dBFS)': f"{measurement['出力(dBFS)']:.2f}",
                    'Input(dBFS)': f"{measurement['入力(dBFS)']:.2f}" if measurement['入力(dBFS)'] is not None else '',
                    'THD(%)': f"{measurement['THD(%)']:.4f}" if measurement['THD(%)'] is not None else '',
                    'THD(dBr)': f"{measurement['THD(dBr)']:.4f}" if measurement['THD(dBr)'] is not None else '',
                    'THD+N(%)': f"{measurement['THD+N(%)']:.4f}" if measurement['THD+N(%)'] is not None else '',
                    'THD+N(dBr)': f"{measurement['THD+N(dBr)']:.4f}" if measurement['THD+N(dBr)'] is not None else '',
                    'SNR(dB)': f"{measurement['SNR(dB)']:.2f}" if measurement['SNR(dB)'] is not None else '',
                    'Gain(dB)': f"{measurement['ゲイン(dB)']:+.2f}" if measurement['ゲイン(dB)'] is not None else ''
                })
    finally:
        if csv_writer:
            csvfile.close()

    return measurements

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="音声信号の測定と高調波解析を行うプログラム")

    # 基本オプション
    parser.add_argument('-f', '--frequency', type=float, default=1000, help='基本周波数 (デフォルト: 1000 Hz)')
    parser.add_argument('-a', '--amplitude', type=float, default=-6, help='トーン振幅 (dBFS) (デフォルト: -6 dBFS)')
    parser.add_argument('-w', '--window', type=str, default='blackmanharris', help='窓関数種類 (デフォルト: blackmanharris)')
    parser.add_argument('--duration', type=float, default=5.0, help='測定時間 (秒) (デフォルト: 5.0秒)')
    parser.add_argument('--bandpass', action='store_true', help='バンドパスフィルター適用オプション')
    parser.add_argument('-sr', '--sample_rate', type=int, default=48000, help='サンプリングレート (デフォルト: 48000 Hz)')
    parser.add_argument('-oc', '--output_channel', type=str, choices=['L', 'R'], default='R', help='出力チャンネル (LまたはR, デフォルト: R)')

    # 測定回数オプション（通常モード用）
    parser.add_argument('-n', '--num_measurements', type=int, default=2, help='測定回数 (デフォルト: 2)')

    # モード選択オプション
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--sweep-amplitude', action='store_true', help='振幅スイープモードを有効化')
    group.add_argument('--sweep-frequency', action='store_true', help='周波数スイープモードを有効化')
    group.add_argument('--map', action='store_true', help='マッピングモードを有効化')
    group.add_argument('--test', action='store_true', help='テストトーンを出力')
    group.add_argument('--calib', action='store_true', help='キャリブレーションモード')
    # 追加オプション
    parser.add_argument('--output_csv', type=str, help='測定結果を保存するCSVファイル名')

    args = parser.parse_args()

    # デバイス選択
    device_index = select_device()

    # サンプリングレート設定
    sample_rate = args.sample_rate
    console.print(f"サンプリングレートを [bold]{sample_rate}[/bold] Hz に設定しました。\n")

    # 出力チャンネル設定と入力チャンネルの決定
    output_channel = 0 if args.output_channel.upper() == 'L' else 1
    input_channel = 1 - output_channel
    console.print(f"出力チャンネル: [bold]{'左 (L)' if output_channel == 0 else '右 (R)'}[/bold]")
    console.print(f"入力チャンネル: [bold]{'右 (R)' if input_channel == 1 else '左 (L)'}[/bold]\n")

    if args.test:
        # テストトーンモード
        amplitude = 10 ** (args.amplitude / 20)  # dBFSを線形振幅に変換
        test_tone(args.frequency, amplitude, args.sample_rate, device_index, output_channel)
        sys.exit(0)

    if args.calib:
        # キャリブレーションモード
        calibration_mode(args.sample_rate, device_index, output_channel, args.amplitude)
        sys.exit(0)

    # 測定時間の検証
    if args.duration <= 1.0 and not args.map:
        console.print("[red]測定時間は1秒以上に設定してください。[/red]")
        sys.exit(1)

    # トーン設定
    amplitude = 10 ** (args.amplitude / 20)  # dBFSを線形振幅に変換
    phase = 0  # 位相は固定

    # モードに応じた周波数と振幅のリストを作成
    frequency_list = []
    amplitude_list = []
    aligner = None

    if args.map:
        # --mapモード
        # 周波数と振幅の測定する組み合わせを生成
        frequencies = generate_octave_frequencies(step=2)
        amplitudes = np.arange(-48, 3, 3)
        for freq in frequencies:
            for amp in amplitudes:
                frequency_list.append(freq)
                amplitude_list.append(amp)

        console.print(f"マッピングモード: 周波数リスト={frequencies}")
        console.print(f"振幅リスト(dBFS)={amplitudes}")
        console.print(f"測定回数(回): {len(frequency_list)}")
        # --mapモードではalignerを使用しない
    elif args.sweep_frequency:
        # 周波数スイープモード
        frequency_list = generate_octave_frequencies(step=1)  # すべての周波数
        amplitude_list = [args.amplitude] * len(frequency_list)  # 振幅は固定
        console.print(f"周波数スイープモード: 周波数リスト={frequency_list}")
        console.print(f"振幅は固定: {args.amplitude} dBFS")
        aligner = SignalAligner(trim_time_sec=0.20, samplerate=args.sample_rate)
    elif args.sweep_amplitude:
        # 振幅スイープモード
        frequency_list = [args.frequency] * len(np.arange(-99, 3, 3))  # 周波数は固定
        amplitude_list = np.arange(-99, 3, 3)  # -99dBFSから0dBFSまで指定ステップ
        console.print(f"振幅スイープモード: 振幅リスト(dBFS)={amplitude_list}")
        console.print(f"周波数は固定: {args.frequency} Hz")
        aligner = SignalAligner(trim_time_sec=0.20, samplerate=args.sample_rate)
    else:
        # 通常モード
        frequency_list = [args.frequency] * args.num_measurements
        amplitude_list = [args.amplitude] * args.num_measurements
        console.print(f"通常モード: 周波数={args.frequency} Hz, 振幅={args.amplitude} dBFS")
        # 通常モードではalignerを使用
        aligner = SignalAligner(trim_time_sec=0.20, samplerate=args.sample_rate)

    try:
        # --sweep-frequency または --sweep-amplitude または 通常モード
        measurements = perform_measurements(
            device_index=device_index,
            output_channel=output_channel,
            sample_rate=sample_rate,
            apply_bandpass=args.bandpass,
            frequency_list=frequency_list,
            amplitude_list=amplitude_list,
            duration=args.duration,
            window_name=args.window,
            aligner=aligner,
            output_csv=args.output_csv  # CSV保存
        )

        # 結果の表示
        display_measurements(measurements)
        display_statics(measurements)

        if args.output_csv:
            console.print(f"[green]測定結果が '{args.output_csv}' に保存されました。[/green]")

    except KeyboardInterrupt:
        console.print("\n[red]プログラムを中断しました。[/red]")
    finally:
        console.print("[green]プログラムを正常に終了しました。[/green]")

if __name__ == '__main__':
    main()
