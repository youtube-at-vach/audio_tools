## 目的

このリポジトリで作業するエージェント向けのメモです。
「実際に確認できたこと」を中心に、セットアップ・起動・テスト・構成・開発時の注意点をまとめます。

## 言語 / ローカライズ方針

- GUI 表示文字列は **`tr()` で囲って** 多言語対応を前提に実装してください。
- まずは **英語キー（`tr("...")` の引数）** を入れて実装 → 仕様が固まった段階で翻訳 JSON を更新、という進め方を想定しています。
- 翻訳ファイルは `src/assets/lang/*.json` にあります（`en.json` が基本）。
- 翻訳キーの整合チェックは `scripts/check_trn_keys.py` を使えます。

## 実行環境（確認できた範囲）

- OS: Linux
- Python: 3.10+ 想定（README 記載）
- この作業環境では `./.venv/bin/python`（Python 3.12.3）で動作確認しました。

## セットアップ（venv）

リポジトリ直下で:

1) venv 作成

- Linux:
	- `python3 -m venv .venv`
	- `./.venv/bin/python -m pip install -U pip`

2) 依存導入

- `./.venv/bin/python -m pip install -r requirements.txt`

補足:

- `PyWavelets` は pip パッケージ名で、Python の import 名は `pywt` です。
- この環境では VS Code のツール連携の都合で `python` コマンド解決が失敗するケースがあったため、以降は `./.venv/bin/python` を明示して実行しました。

## 起動方法

### GUI（MeasureLab 本体）

- `./.venv/bin/python main_gui.py`

起動時の流れ（コード確認ベース）:

- `main_gui.py`:
	- `ConfigManager` で言語設定を読み、スプラッシュ表示（`welcome.png`）中にモジュールを事前ロード
	- 必要に応じて起動時のウィンドウ挙動ログを出せます（後述）
- `src/gui/main_window.py`:
	- 左サイドバーで Welcome / Settings / 各測定モジュールに切替
	- Settings と各モジュールは **遅延 import / 遅延生成** されます（重い依存の起動時コスト低減）
	- `preload_all_modules()` はスプラッシュ中に Settings + 全モジュールをロードします

### CLI（任意）

`src/main.py` は対話式 CLI（`inquirer`）を使いますが、`requirements.txt` には `inquirer` が含まれていないため、そのままだと実行できません。

- 使う場合: `./.venv/bin/python -m pip install inquirer`

## テスト

最小スモーク（この環境で実行して 5 tests pass を確認）:

- `./.venv/bin/python -m pytest -q tests/test_config.py tests/test_si_formatting.py`

全体:

- `./.venv/bin/python -m pytest -q`

注意:

- オーディオ I/O を伴うテストや、環境依存（デバイスが無い等）のテストが混ざる可能性があります。CI/実機前提のものは適宜切り分けて実行してください。

## 主要ディレクトリ / コンポーネント

- `main_gui.py`: GUI エントリポイント（スプラッシュ + preload）。
- `src/gui/main_window.py`: 画面全体（サイドバー・遅延ロード・モジュール切替・ステータス表示）。
- `src/gui/widgets/`: 各測定モジュールのウィジェット実装。
- `src/core/audio_engine.py`: `sounddevice` ベースの Audio I/O。
	- 複数クライアントをミックスするコールバック・レジストリ方式
	- ソフトウェアループバック（Internal Loopback）やミュート制御
	- PipeWire/JACK 向けにストリーム常駐モード（resident）をサポート
- `src/core/config_manager.py`: `config.json` のロード/保存（デバイス・SR・ブロックサイズ・言語・テーマ等）。
- `src/core/localization.py`: `LocalizationManager` と `tr()`。
- `src/core/theme_manager.py`: light/dark/system テーマ切替（Qt 6.5+ では OS の theme change を検出）。

## Linux オーディオ（注意点）

README にもある通り、Linux では PortAudio バックエンドのままでも動きますが、位相連続性が重要な測定では JACK / PipeWire の利用が推奨される場合があります。

- `ConfigManager` に `pipewire_jack_resident` 設定があり、`AudioEngine.set_pipewire_jack_resident()` で制御されます。

## デバッグ用の環境変数（確認できた範囲）

- `MEASURELAB_DEBUG_WINDOWS=1`
	- 起動時の「一瞬出るトップレベルウィンドウ（flash）」調査用に、表示/サイズ変化等をログします。
- `MEASURELAB_DEBUG_WINDOWS_TRACE=1`
	- 条件に合う “怪しい” 小さな無題ウィンドウの出現時にスタックトレースを出します。

## 変更時のガイド

- UI 文言は `tr()` を必ず通し、キーはまず英語で入れる。
- 新規モジュール追加は `src/gui/main_window.py` の `_module_keys` と `_load_module_class()` を更新する（キー名は一致させる）。
- 依存追加が必要になったら `requirements.txt` と、PyInstaller/配布手順（必要であれば）も合わせて見直す。

---

このファイルは「推測」ではなく「確認できた事実」に寄せて更新してください（環境差分が出やすい領域のため）。