# Audio Signal Generator

このプログラムは、トーン、ノイズ、スイープ信号などの音声信号を生成してWAVファイルとして保存するPythonツールです。

著作権は放棄されています。これはインターネット上の情報として、誰でも自由に使用、改変、配布できます。

## 📦 特徴

- サイン波、矩形波、三角波、ノコギリ波、パルス波の生成
- ホワイトノイズ、ピンクノイズ、グレーノイズ、ブラウンノイズ、レッドノイズ、ブルーノイズ、バイオレットノイズなど各種ノイズ生成
- 線形／対数スイープ信号の生成（周波数・音量）
- 音声ファイルの読み込みと再利用
- A/B/C 重み付けの適用
- フェードイン・フェードアウト機能
- 任意のサンプルレート・ビット深度（16, 24, 32bit, float）に対応

## 🔧 使用方法

```bash
python audio_signal_generator.py [オプション]
```

### 主なオプション

| オプション               | 説明                                                                 |
|------------------------|----------------------------------------------------------------------|
| `-f`, `--frequency`     | トーンの周波数 (Hz, デフォルト: 1000)                                |
| `--sweep`               | 線形周波数スイープを生成                                            |
| `--log_sweep`           | 対数周波数スイープに切り替え                                        |
| `--start_freq`          | スイープ開始周波数 (デフォルト: 20Hz)                               |
| `--end_freq`            | スイープ終了周波数 (デフォルト: 20kHz)                              |
| `--noise`               | ノイズ色を指定: white, pink, grey, brown, red, blue, violet, etc.  |
| `--sweep_type`          | スイープの波形タイプ: `sine`, `square`（デフォルト: `sine`）         |
| `--square`              | 矩形波を生成                                                        |
| `--triangle`            | 三角波を生成                                                        |
| `--pulse`               | パルス波を生成                                                      |
| `--duty_cycle`          | パルス波のデューティサイクル (0.0-1.0, デフォルト: 0.5)             |
| `--sawtooth`            | ノコギリ波を生成                                                    |
| `--ramp_type`           | ノコギリ波の方向: `ramp+`, `ramp-`（デフォルト: `ramp+`）            |
| `--volume_sweep`        | 線形音量スイープを適用                                              |
| `--log_volume_sweep`    | 対数音量スイープを適用                                              |
| `--weighting`           | 重み付け: A, B, C                                                   |
| `-s`, `--sample_rate`   | サンプルレート (Hz, デフォルト: 48000)                              |
| `-b`, `--bit_depth`     | ビット深度: 16, 24, 32, float（デフォルト: float）                   |
| `-d`, `--duration`      | 持続時間（秒, デフォルト: 5秒）                                     |
| `-o`, `--output`        | 出力ファイル名（デフォルト: `tone.wav`）                            |
| `--fade_duration`       | フェードイン・アウト時間（秒）                                     |
| `-i`, `--input`         | 入力WAVファイルから読み込み                                        |
| `--dbfs`                | 出力音量 (dBFS, デフォルト: -3)                                     |

## 📝 ライセンス

This is free and unencumbered software released into the public domain.

You may use it for any purpose, commercial or non-commercial, without restriction.

See [`UNLICENSE`](https://unlicense.org/) for more details.

---

## 💡 例

```bash
# 1kHzサイン波を3秒生成してtone.wavに保存
python audio_signal_generator.py -f 1000 -d 3 -o tone.wav

# 20Hz〜20kHzの対数スイープを5秒生成
python audio_signal_generator.py --sweep --log_sweep --start_freq 20 --end_freq 20000 -d 5 -o sweep.wav

# ピンクノイズを10秒生成して出力
python audio_signal_generator.py --noise pink -d 10 -o pink_noise.wav

# 矩形波とA特性の適用
python audio_signal_generator.py --square --weighting A -o square_A.wav
```

## 🔗 関連プロジェクト

- [`audio_analyzer`](../audio_analyzer): 本ツールと併用できる、歪や特性を分析するツールです。
