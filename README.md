# 🚀 **MeasureLab (Audio Measurement Suite)** 🎶

A collection of DIY audio measurement and analysis tools, grown organically as needed.
「必要に応じて作り足しながら育ててきた DIY のオーディオ測定・解析ツール集」です。

**MeasureLab** は、これらのツールを1つの GUI アプリにまとめた形で提供します。Python と PyQt6 製で、高精度な信号生成・解析・測定を直感的に扱えます。

## Quick glance (English)

- PyQt6 desktop app bundling 25+ DIY modules: signal generator, spectrum/PSD analyzer, sound level & LUFS meters, lock-in/FRA, network/impedance analyzers, oscilloscope, spectrogram, distortion/IMD tools, recorder/player, inverse filter, frequency counter, lock-in frequency counter, sound quality analyzer, noise profiler, boxcar averager, goniometer, and more.
- Built for hobbyists and engineers: device routing, calibration (input/output/SPL), multi-language UI, light/dark themes.
- Runs on Windows/Linux; grab the AppImage/ZIP or `python main_gui.py` from source (Python 3.10+).

## ✨ 主な機能 (Features)

### 🛠️ ウィジット / 測定モジュール
以下のモジュール/ウィジットが統合されています:

1.  **Welcome**: 起動時のウェルカム画面で主要機能を案内。
2.  **Signal Generator**: 正弦波、矩形波、三角波、ノコギリ波(立ち上がり/立ち下がり)、ホワイト/ピンクノイズ、周波数スイープ信号を生成。位相制御、振幅制御、ステレオ出力に対応。
3.  **Spectrum Analyzer**: 高速FFTによるリアルタイムスペクトル解析。PSD/RMS表示、SI単位表示、周波数範囲制限、メモリ機能、カーソル測定に対応。
4.  **Sound Level Meter**: A/C/Z 周波数重み付け、FAST/SLOW/IMPULSE/10ms 時間重み付け、20Hz–20k/12.5k/8k 帯域選択に対応した高機能騒音計。Lp/Leq/LE/Lmax/Lmin/Lpeak表示、キャリブレーションオフセット適用に対応。
5.  **LUFS Meter**: ラウドネス (LUFS/LKFS) のリアルタイム測定。クレストファクター、ダイナミックレンジ表示。
6.  **Loopback Finder**: オーディオインターフェースのレイテンシー(遅延)測定ツール。
7.  **Distortion Analyzer**: THD、THD+N、SINAD、IMD (SMPTE/CCIF) の測定。内蔵信号発生器、周波数スイープ、ハーモニクスバーグラフ、平均化機能搭載。
8.  **Advanced Distortion Meter**: MIM (Multi-tone Intermodulation)、SPDR (Spurious-free Dynamic Range)、PIM (Passive Intermodulation) 測定を含む高度な歪み解析。
9.  **Network Analyzer**: 周波数特性(ゲイン・位相・群遅延)の測定。スイープ測定、複数トレース表示、周波数範囲制限対応。
10. **Oscilloscope**: 2チャンネル波形表示、トリガー機能、カーソル測定、演算波形(A+B, A-B)、リアルタイムローパス/ハイパスフィルタリング対応。
11. **Raw Time Series**: 長時間スパンをリングバッファで保持する2chスクロール波形モニタ。
12. **Transient Analyzer**: トリガ収録＋CWT で過渡解析、解析帯域/スケールを柔軟に指定。
13. **Lock-in Amplifier**: 位相敏感検波 (PSD) による微小信号測定。周波数応答解析 (FRA) モード、ハーモニクス復調(2次〜10次)、キャリブレーション機能搭載。
14. **Lock-in THD+N Analyzer**: ロックイン検波を用いた THD/THD+N 測定専用モジュール。整数周期ウィンドウと平均化、残差履歴・プロット表示、ハーモニクス/残差バーグラフで歪み成分を可視化。
15. **Impedance Analyzer**: インピーダンス測定とOSL (Open/Short/Load) キャリブレーション。複数プロットモード(Z/θ、R/X、Q、C/L、Nyquist、Smith Chart)、スイープ測定、キャリブレーション補間対応。
16. **Inverse Filter**: キャリブレーションマップから逆特性FIRを設計し、音声ファイルへ適用するデコンボリューションツール。ゲイン上限による正則化、タップ数/スムージング指定、応答プレビュー、出力ピーク正規化付きのバッチ処理に対応。
17. **Frequency Counter**: 高精度な周波数・周期測定。アラン分散プロット、ジッターヒストグラムおよび統計、キャリブレーション機能搭載。
18. **Lock-in Frequency Counter**: ロックイン検波 (PSD) による高精度な周波数・位相偏差のトラッキング。微小な偏差の可視化と安定性の評価に対応。
19. **Spectrogram**: 時間-周波数表示のスペクトログラム。周波数範囲制限、カラーマップ選択対応。
20. **Boxcar Averager**: ボックスカー平均によるノイズ低減と過渡応答解析。内部パルス/ステップ生成、外部リファレンス同期(立ち上がり/立ち下がりエッジ)対応。
21. **Goniometer**: ステレオ信号の位相相関と空間分布の可視化。Lissajous表示、フォスファー表示モード(残光効果)、カスタムカラーパレット対応。
22. **Noise Profiler**: ノイズ特性の詳細解析ツール。1/fノイズ、ハムノイズ、ホワイトノイズの自動検出と定量化。平均化モード、LNAゲイン補正、熱雑音限界表示、等価抵抗表示対応。
23. **Recorder & Player**: オーディオファイル(WAV/MP3/FLAC/OGG等)の録音・再生。リサンプリング、ループ再生、ソフトウェアループバック機能搭載。
24. **Sound Quality Analyzer**: 音質評価指標 (Integrated/Momentary Loudness, Zwicker Sharpness, Roughness, Tonality) の数値およびグラフ表示。
25. **Timecode Monitor & Generator**: LTC タイムコードのエンコード/デコードとリアルタイム監視。フレームベース計算、ドロップフレーム率、複数FPS表示、タイムゾーン/オフセット、JAMメモリ付きジェネレーターを備える。
26. **Detachable Wrapper**: 任意ウィジットを独立ウィンドウとして切り離し・再接続できるUIユーティリティ。
27. **Settings**: デバイス設定、キャリブレーション、テーマ選択、多言語切り替えなど。

### 🌍 多言語対応 (Localization)
世界中の主要な言語をサポートしています。設定画面から切り替え可能です。
*   英語 (English)
*   日本語 (Japanese)
*   中国語 (Chinese)
*   スペイン語 (Spanish)
*   フランス語 (French)
*   ドイツ語 (German)
*   ポルトガル語 (Portuguese)
*   ロシア語 (Russian)
*   韓国語 (Korean)

### ⚙️ 高度な設定
*   **入出力設定**: デバイス選択、サンプリングレート (44.1kHz - 192kHz)、バッファーサイズ変更。
*   **キャリブレーション**: 入力感度と出力ゲインの補正ウィザードを搭載し、電圧 (Vrms, Vpeak, dBu, dBV) での正確な読み取りが可能。
*   **チャンネルルーティング**: 入力・出力チャンネルの個別割り当てに対応。
*   **テーマ設定**: ライト/ダーク/システムテーマの切り替えが可能。

---

## 🚀 インストールと実行 (Installation & Usage)

### 📦 ビルド済みパッケージを使用する場合
**Releases** ページから最新のバージョンをダウンロードしてください。

*   **Windows**: `MeasureLab-<version>-windows-x64-onefile.zip`（または `MeasureLab-<version>-windows-x64-onedir.zip`）をダウンロードして解凍し、`MeasureLab.exe` を実行します。
*   **Linux**: `MeasureLab-<version>-linux-x86_64.AppImage` をダウンロードし、実行権限を付与して起動します。
    ```bash
    chmod +x MeasureLab-*-linux-x86_64.AppImage
    ./MeasureLab-*-linux-x86_64.AppImage
    ```

#### Linux（任意）: JACK / PipeWire を使う場合の注意

Linux ではそのまま **PortAudio** バックエンドでも通常利用できますが、環境によっては **バッファ境界で位相が飛ぶ**（位相連続性が崩れる）ことがあります。
位相の連続性が重要な測定（位相・群遅延・ロックイン等）を行う場合は、入出力先として **JACK** もしくは **PipeWire** を指定して使うことを推奨します。

ただし JACK / PipeWire 経由にすると、起動後に音が出ない・入出力がつながらない場合があります。その際は **QJackCtl** などでルーティング（接続）を確認・設定してください。

※この項目はあくまでオプションです。PortAudio のままでも普通に使えます。

### 🐍 ソースコードから実行する場合

**必要条件**: Python 3.10 以上

1.  リポジトリをクローンします。
2.  依存関係をインストールします（再現性のため constraints を利用）：
    ```bash
    pip install -c constraints.txt -r requirements.txt
    ```
3.  アプリケーションを起動します：
    ```bash
    python main_gui.py
    ```

### 🛠️ 開発向けセットアップ

テストやLint/型チェックを実行する場合は開発ツールもインストールしてください。

```bash
pip install -c constraints.txt -e .[dev]
```

- Lint: `ruff check src scripts tests`
- Type check: `mypy src`
- Tests: `pytest`（ハードウェア/GUI依存テストは環境変数が必要; CIではデフォルトでスキップ）

---

## 📜 ライセンス (License)

このプロジェクトは **The Unlicense** の下でパブリックドメインとして公開されています。
営利・非営利を問わず、自由にコピー、変更、配布、使用することができます。

> **Note**: This is free and unencumbered software released into the public domain.