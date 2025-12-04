# Changelog

## [v0.1.7] - 2025-12-04

### Added

* Recorder & Player 機能を追加し、ソフトウェアループバックに対応
* 非同期オーディオ読み込み（リサンプリング＋進捗表示）を追加
* Noise Profiler に平均化モードを追加
* Phosphor（残光）表示モードおよびカラーパレット設定を追加
* Signal Generator に鋸歯状波の上昇／下降タイプ選択機能を追加
* Frequency Counter ウィジェットを追加
* 多数のUIコンポーネント（Spectrum Analyzer / Spectrogram / Network Analyzer / Lock-in / Distortion Analyzer / Impedance Analyzer など）に新しい表示・制御機能を追加
* 各ウィジェットの新機能に対応する翻訳キーを追加

### Changed

* 多数のUI文字列を `tr()` 化し、多言語化を大規模に強化
* プロットの周波数制限機能を追加し、表示・計算処理を改善
* Distortion Analyzer・Lock-in・Network Analyzer 等でコンボボックスの値処理を index / itemData ベースに変更
* Spectrogram・Noise Profiler・Network Analyzer などの翻訳ファイルを更新

### Fixed

* グループディレイの x 軸スケーリングを修正（ログ周波数に対応）
* 不必要な ViewBox 追加による描画問題を修正
* 1/f ノイズ解析のハム除外帯域を 5 Hz に拡大し判定精度を改善
* その他、細かな UI 表記や計算ロジックの修正

### Removed

* Noise Profiler の 1/f コーナー周波数手動指定機能を削除
* PyInstaller ワークフローでの不要な scipy サブモジュール除外設定を削除
* 不要な再現テストスクリプト（pyqtgraph legend など）を削除

### Documentation

* GEMINI.md を更新
* CHANGELOG エントリを整理

## [v0.1.6] - 2025-12-03

### 新機能
- ロックインアンプのキャリブレーションシステムを追加 (周波数応答マッピングと絶対ゲイン補正)
- ヒルベルト変換に基づく参照周波数とコヒーレンスの推定を実装
- インピーダンス解析のための動的プロットモードと凡例の更新
- Windows Nuitkaビルド用のGitHub Actionsワークフローを追加

### 改善・変更
- PyInstallerビルドサイズを削減 (未使用のscipyサブモジュールを除外)
- 不要なデータファイルの削除と.gitignoreの更新

## [v0.1.5] - 2025-11-29

Windows10対応のためのテストリリース

## [v0.1.4] - 2025-11-26
...
