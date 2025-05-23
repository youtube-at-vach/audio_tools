# 新規測定プログラム製作手順 (New Measurement Program Production Procedure)

このドキュメントは、本リポジトリに新しいオーディオ測定プログラムを追加する際の標準的な手順を概説するものです。

## 1. 企画・設計 (Planning and Design)

-   **目的の明確化**: 新しい測定プログラムが解決する課題、測定する具体的なオーディオ特性（例: THD+N, 周波数特性, IMDなど）を定義します。
-   **既存ツールの確認**: リポジトリ内に類似の機能を持つツールがないか確認します。既存ツールを拡張できる場合は、新規作成よりも優先します。
-   **必要な機能のリストアップ**:
    -   入力信号の種類（例: サイン波、矩形波、ホワイトノイズ、外部ファイル）
    -   ユーザーが設定可能なパラメータ（例: 周波数、振幅、テスト時間、FFTサイズ）
    -   出力形式（例: コンソールへのテキスト表示、グラフ表示、CSVファイルへの保存）
    -   対応するオーディオ規格や測定方法（例: SMPTE, DIN, CCIF for IMD）
-   **使用ライブラリの選定**: `numpy`, `scipy`, `sounddevice`, `matplotlib`, `rich`など、必要なライブラリを検討します。
-   **設計段階で、新たに追加が必要となりそうなサードパーティライブラリ（例：データプロットのための `matplotlib`）を特定し、依存関係リストに追加する準備をします。(In the design phase, identify any new third-party libraries that may need to be added (e.g., `matplotlib` for data plotting) and prepare to add them to the dependency list.)**

## 2. 開発環境の準備 (Development Environment Setup)

-   Pythonの適切なバージョン（本リポジトリでは Python 3.8+ を推奨）を準備します。
-   必要なライブラリをインストールします (`pip install -r requirements.txt` もしくは個別に `pip install <library_name>`)。
-   Gitリポジトリをクローンまたは最新の状態に更新します。

## 3. 実装 (Implementation)

-   **モジュール構成**:
    -   プログラムの主たる機能は、リポジトリルート直下に作成する専用のディレクトリ（例: `new_tool_analyzer/`）に配置します。
    -   メインスクリプトは `new_tool_analyzer/new_tool_analyzer.py` のように命名します。
    -   ディレクトリ内には `__init__.py` を配置し、Pythonパッケージとして認識できるようにします。
-   **信号生成**:
    -   必要なテスト信号を生成する関数を実装します。可能であれば既存の信号生成ツール（例: `audio_signal_generator`）の機能を活用します。
-   **オーディオ入出力**:
    -   `sounddevice` ライブラリを使用して、オーディオデバイスの選択、再生、録音機能を実装します。
    -   ユーザーが入力/出力チャンネル（L/R）を選択できるようにします。
    -   オーディオストリーミングの要求（連続的な長時間処理か、断続的な短時間処理の繰り返し等）に応じて、`sounddevice`ライブラリの適切な利用方法（コールバック方式の`sd.Stream`か、ブロッキング方式の`sd.playrec`等）を選択します。また、`sd.playrec` のチャンネルマッピングなど、API特有の仕様にも注意します。(Select the appropriate usage method of the `sounddevice` library (e.g., callback-based `sd.Stream` or blocking `sd.playrec`) according to the audio streaming requirements (continuous long-term processing, intermittent repetition of short-term processing, etc.). Also, pay attention to API-specific specifications such as channel mapping for `sd.playrec`.)
-   **解析処理**:
    -   FFT、ウィンドウ関数、フィルタリングなど、測定に必要な信号処理を実装します。
    -   `numpy` や `scipy.signal` を活用します。
    -   FFTを用いた解析では、正確な振幅スペクトルを得るための正規化（例：ウィンドウ関数の総和で除算）や、既知の周波数成分を正確に捉えるための周波数ビン選択/補間処理に注意を払います。(In analysis using FFT, pay attention to normalization for obtaining accurate amplitude spectra (e.g., dividing by the sum of the window function) and to frequency bin selection/interpolation processing for accurately capturing known frequency components.)
-   **結果表示**:
    -   `rich` ライブラリを使用して、結果を整形してコンソールに表示します。
    -   必要に応じて、`matplotlib` を用いたグラフ表示機能や、CSVファイルへの出力機能を実装します。
-   **コマンドラインインターフェース (CLI)**:
    -   `argparse` を使用して、ユーザーがパラメータを指定できるCLIを設計します。
    -   ヘルプメッセージ (`--help`) を充実させます。
-   **エラーハンドリング**:
    -   デバイスが見つからない場合、不正なパラメータが指定された場合など、予期されるエラーを適切に処理し、ユーザーフレンドリーなエラーメッセージを表示します。
-   **周波数スイープのように反復的な測定を行うツールでは、全データ点を収集した後に実行する後処理（例：位相データのアンラップ処理）の順序とデータフローを考慮して統合します。(For tools that perform iterative measurements like frequency sweeps, integrate by considering the order and data flow of post-processing (e.g., unwrapping phase data) to be executed after collecting all data points.)**
-   **コーディング規約**:
    -   PEP 8 に準拠したコーディングスタイルを心がけます。
    -   コメントやdocstringを適切に追加します。

## 4. テスト (Testing)

-   **単体テスト**:
    -   主要な関数（信号生成、解析処理など）に対して単体テストを作成します。
    -   Pythonの `unittest` フレームワークを使用し、テストスクリプトは `new_tool_analyzer/test_new_tool_analyzer.py` のように命名します。
    -   テストケースでは、既知の入力に対する期待される出力を検証します。
-   **統合テスト**:
    -   実際にオーディオデバイスを使用してループバックテストなどを行い、プログラム全体が意図した通りに動作することを確認します。
    -   様々なパラメータの組み合わせでテストします。
-   **テストの実行**:
    -   リポジトリルートから `python -m unittest new_tool_analyzer/test_new_tool_analyzer.py` のようにしてテストを実行します。
-   **テストコードが依存するライブラリ（特にオーディオ処理やプロット用ライブラリ）がテスト環境で利用可能であること、また、必要なシステムレベルの依存関係（例：PortAudio）が満たされていることを確認します。(Ensure that libraries depended upon by the test code (especially audio processing and plotting libraries) are available in the test environment, and that necessary system-level dependencies (e.g., PortAudio) are satisfied.)**

## 5. ドキュメント作成 (Documentation)

-   **README.md**:
    -   `new_tool_analyzer/README.md` を作成し、以下の情報を記載します:
        -   ツールの概要と目的
        -   関連するオーディオ規格（もしあれば）
        -   依存ライブラリ（新たに追加したものも含む、例: `matplotlib`）と、その標準的なインストール方法を明記します。(Clearly state the dependent libraries (including newly added ones, e.g., `matplotlib`) and their standard installation method.)
        -   使用方法（コマンドラインオプション、設定可能なパラメータの説明）
        -   出力結果の例と説明
        -   注意事項（例: ループバック設定、オーディオインターフェースの品質など）
-   **本体リポジトリのREADME.md更新**:
    -   リポジトリルートの `README.md` に、新しいツールへのリンクと簡単な説明を追加します。
-   **サンプルファイル**:
    -   必要であれば、設定ファイル例や出力結果のサンプルを提供します。

## 6. プルリクエストとレビュー (Pull Request and Review)

-   変更内容をコミットし、GitHub上でプルリクエストを作成します。
-   （もし協力者がいれば）レビューを受け、フィードバックに基づいて修正を行います。
-   全てのチェックが通ったら、メインブランチにマージします。

## 7. その他 (Miscellaneous)

-   **ライセンス**: 本リポジトリの他のツールと同様に、Unlicense を適用し、著作権を放棄します。READMEにその旨を記載します。
-   **ファイルエンコーディング**: すべてのテキストファイル（.py, .md, .txt など）は UTF-8 で保存します。

---

この手順はあくまでガイドラインであり、開発するツールの特性や規模に応じて適宜調整してください。 (This procedure is merely a guideline; please adjust it as appropriate according to the characteristics and scale of the tool being developed.)
