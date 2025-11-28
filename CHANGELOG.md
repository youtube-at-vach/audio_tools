# Changelog

## [v0.1.4] - 2025-11-28

### Added
- **Impedance Analyzer**: インピーダンス測定モジュールを追加。
    - デュアルロックイン計算による高精度測定。
    - 周波数スイープ機能。
    - OSL (Open/Short/Load) キャリブレーション機能。
    - 既知の負荷抵抗値を指定可能なUI。
- **Goniometer**: ステレオ信号の位相相関と空間分布を可視化するウィジェットを追加。
- **Signal Generator Phase Control**: 信号生成器に位相制御機能を追加。
- **Boxcar Averager External Sync**: ボックスカー平均器に外部リファレンス同期モードを追加。
- **Distortion Analyzer Harmonics Bar Graph**: 歪み解析器に高調波の棒グラフ表示を追加。

### Changed
- **Theme-Aware Styling**: すべての測定モジュールウィジェットのボタンとラベルにテーマ対応スタイリングを実装。
    - ライト/ダークテーマで適切な視認性を確保。
    - システムテーマの自動検出と適用。

### Fixed
- **Localization**: Goniometer と Advanced Distortion Meter の多言語翻訳を追加。

## [v0.1.3] - 2025-11-27

### Added
- **Advanced Distortion Meter**: New widget for advanced distortion analysis.
    - Added MIM (Multitone Intermodulation) measurement (TD+N).
    - Added SPDR (Spurious-Free Dynamic Range) measurement.
    - Added PIM (Passive Intermodulation) measurement.
- **Oscilloscope Filters**: Added real-time filtering capabilities to the Oscilloscope.
    - Low Pass Filter (LPF)
    - High Pass Filter (HPF)
    - Band Pass Filter (BPF)

### Changed
- **Oscilloscope UI**: Refactored Oscilloscope interface.
    - Split into Display and Control panels.
    - Added dedicated Vertical and Trigger control groups.
    - Added Math channel (A+B, A-B, etc.).
    - Added Cursors for time and voltage measurement.
- **Localization**: Updated translation strings for new modules and settings.

## [v0.1.2] - 2025-11-26
...
