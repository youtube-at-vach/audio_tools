# Audio Calibration Tool

このツールは、オーディオデバイスのキャリブレーションとテストを支援します。

## 特徴

- **デバイス一覧表示**: システムに接続されているオーディオデバイスを一覧表示します。
- **出力レベルキャリブレーション**: 基準信号を再生し、リアルタイムで入力レベルを表示することで、ユーザーがオーディオ出力レベルを調整するのを支援します。
- **ループバックテスト**: 指定された周波数と振幅のテストトーンを再生し、それを録音してWAVファイルに保存します。
- **WAVファイル分析**: 録音されたWAVファイルをFFT分析し、周波数スペクトルをプロットして保存します。

## 使用方法

`calibration_tool.py` を実行する際には、以下のモードのいずれかを指定します。

### デバイス一覧表示モード (`list`)

利用可能なオーディオデバイスを一覧表示します。

```bash
python calibration_tool.py list
```

### キャリブレーションモード (`calibrate`)

基準信号を再生し、リアルタイムで入力レベルを表示します。ユーザーはこれを見ながら出力ノブを調整します。`Ctrl+C` で終了します。

```bash
python calibration_tool.py calibrate [-d DEVICE_ID] [-sr SAMPLERATE] [-f FREQUENCY] [-a AMPLITUDE]
```

### ループバックテストモード (`test`)

指定されたテストトーンを再生し、録音してWAVファイルに保存します。

```bash
python calibration_tool.py test [-d DEVICE_ID] [-sr SAMPLERATE] [-f FREQUENCY] [-a AMPLITUDE] [--duration DURATION] [--file FILENAME]
```

### WAVファイル分析モード (`analyze`)

指定されたWAVファイルをFFT分析し、周波数スペクトルをプロットしてPNGファイルとして保存します。

```bash
python calibration_tool.py analyze [--file FILENAME]
```

## オプション

| オプション | 説明 | デフォルト値 |
|---|---|---|
| `-d, --device` | オーディオデバイスID | `2` |
| `-sr, --samplerate` | サンプルレート (Hz) | `44100` |
| `-f, --frequency` | 周波数 (Hz) | `1000` |
| `-a, --amplitude` | 振幅 (0.0-1.0) | `0.5` |
| `--duration` | テスト時間 (秒) | `5` |
| `--file` | テスト用ファイル名 | `loopback_test.wav` |

## 必要な環境

- Python 3.x
- 以下のPythonパッケージが必要です。
  - `sounddevice`
  - `numpy`
  - `soundfile`
  - `matplotlib`

これらのパッケージは `pip` を使用してインストールできます。

```bash
pip install sounddevice numpy soundfile matplotlib
```