# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-11-26

### Added
- **Theme Management**: システム、ライト、ダークモードのテーマ管理機能を追加。設定画面から選択可能
- **Boxcar Averager**: 信号解析用のBoxcar Averager測定モジュールとGUIウィジェットを追加
- **Multi-tone Signal Generation**: マルチトーン信号生成とTD+N（Total Distortion + Noise）解析機能を追加
- **Power Spectral Density (PSD)**: V/√Hz単位でのPSD解析モードとアベレージング機能を追加
- **Oscilloscope Enhancements**: 
  - 微分・積分の数学モードを追加
  - 測定カーソルと情報表示機能を追加
  - 右チャンネル表示のトグル機能を追加
- **LUFS Meter Enhancement**: クレストファクター表示を追加
- **Audio Callback Status**: `sounddevice`コールバックのステータスフラグ報告とGUI表示機能を追加
- **Welcome Screen**: 新しいオーディオ解析ツールを機能リストに追加

### Changed
- **Network Analyzer**: 
  - GUI内の振幅単位変換を再設計
  - Fast Chirpモードで絶対レベル表示を有効化
  - 位相減算機能をリファレンスプロットに追加
- **Output Amplitude Control**: 複数の単位に対応した柔軟な出力振幅制御を実装
- **Spectrogram**: フォルマント強調機能を削除
- **Oscilloscope**: トリガーレベルライン表示を無効化

### Fixed
- Welcome画面のデバッグプリントを削除

### Documentation
- GEMINI.mdドキュメントを更新
- README.mdを更新（ChatGPT-OSSワーク）

## [0.1.1] - Previous Release

初期リリース版

