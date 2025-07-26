# ノイズ分離スペクトラム解析プログラム (Noise Spectrum Analyzer)

## 概要

このツールは、オーディオ信号（WAVファイル）を解析し、その周波数スペクトルを複数のノイズ成分に自動的に分類・定量化します。
`scipy.optimize.curve_fit` を用いて、スペクトルを「1/fノイズ + ホワイトノイズ」のモデルにフィッティングし、各成分の寄与を分離します。また、商用電源ハムとその高調波も検出・定量化します。

## 依存ライブラリ

このツールは、以下のPythonライブラリに依存しています。

-   numpy
-   scipy

リポジトリのルートから、以下のコマンドで共通ライブラリをインストールできます。

```bash
pip install -r requirements.txt
```

## 使用方法

基本的なコマンドラインの実行例は以下の通りです。

```bash
python3 noise_spectrum_analyzer/noise_spectrum_analyzer.py <path_to_your_wav_file>
```

### オプション

-   `wav_file`: 解析対象のWAVファイルのパス。（必須）
-   `--max_harmonics`: 検出する商用電源ノイズの高調波の最大次数。（デフォルト: 10）

## 出力結果の例

```
Total Noise Voltage: 543.21 uV RMS
Frequency Range: 0.5 Hz to 24.0 kHz
Breakdown:
  - 1/f Noise (alpha=1.02): 498.76 uV (84.2%)
  - White Noise: 199.87 uV (13.8%)
  - Power Line Noise (60Hz): 78.90 uV (2.0%)
```

## 注意事項

-   正確な解析のため、入力WAVファイルは10秒以上の長さを持つことが推奨されます。
-   ノイズ分類は、スペクトル形状に基づく便宜的なものであり、物理的なノイズ源を厳密に特定するものではありません。
-   `curve_fit` が失敗した場合、プログラムは簡易的な分類にフォールバックすることがあります。