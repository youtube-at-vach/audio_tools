# 新規測定プログラム製作手順 (New Measurement Program Production Procedure)

このドキュメントは、本リポジトリに新しいオーディオ測定プログラムを追加する際の標準的な手順を概説するものです。

## 1. 企画・設計 (Planning and Design)

-   **将来のアイデア確認**: まず `misc/future_tool_ideas.md` を確認し、既存の将来的なツール構想やアイデアに合致するものがないか、または参考にできる点がないかを確認します。(Check Future Ideas: First, review `misc/future_tool_ideas.md` to see if there are existing future tool concepts or ideas that align with the current task or can provide inspiration.)
-   **目的の明確化**: 新しい測定プログラムが解決する課題、測定する具体的なオーディオ特性（例: THD+N, 周波数特性, IMDなど）を定義します。
-   **既存ツールの確認**: リポジトリ内に類似の機能を持つツールがないか確認します。既存ツールを拡張できる場合は、新規作成よりも優先します。
    -   **未カバーの測定カテゴリ特定**: 基本的なオーディオ測定カテゴリ（例：高調波歪み、周波数特性、相互変調歪み、ノイズ、ステレオ特性、過渡応答など）をリストアップし、未カバーの領域を特定します。例えば `audio_crosstalk_analyzer` はステレオ特性の未カバー領域（信号漏れ）に焦点を当てています。(Identify uncovered basic audio measurement categories by listing them (e.g., harmonic distortion, frequency response, IMD, noise, stereo characteristics, transient response) and target an uncovered area. For example, `audio_crosstalk_analyzer` focuses on an uncovered area of stereo characteristics - signal leakage.)
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
    -   必要なライブラリのインストールに加え、`sounddevice`のようなライブラリが依存する可能性のあるシステムレベルのオーディオライブラリ（例: Linux上のPortAudio (`libportaudio2`など)）がシステムにインストールされていることを確認します。これらは通常、Pythonライブラリのインストール時に自動的に解決されませんが、実行時エラーの原因となることがあります。(In addition to installing necessary libraries, ensure that system-level audio libraries that libraries like `sounddevice` might depend on (e.g., PortAudio on Linux (`libportaudio2`, etc.)) are installed on the system. These are typically not resolved automatically during Python library installation but can cause runtime errors.)
-   Gitリポジトリをクローンまたは最新の状態に更新します。

## 3. 実装 (Implementation)

-   **モジュール構成**:
    -   プログラムの主たる機能は、リポジトリルート直下に作成する専用のディレクトリ（例: `new_tool_analyzer/`）に配置します。
    -   メインスクリプトは `new_tool_analyzer/new_tool_analyzer.py` のように命名します。
    -   ディレクトリ内には `__init__.py` を配置し、Pythonパッケージとして認識できるようにします。
-   **信号生成**:
    -   必要なテスト信号を生成する関数を実装します。可能であれば既存の信号生成ツール（例: `audio_signal_generator`）の機能を活用します。
-   **オーディオ入出力 (Audio I/O)**:
    -   `sounddevice` ライブラリを使用して、オーディオデバイスの選択、再生、録音機能を実装します。
    -   ユーザーが入力/出力チャンネル（L/R）を選択できるようにします。
    -   オーディオストリーミングの要求（連続的な長時間処理か、断続的な短時間処理の繰り返し等）に応じて、`sounddevice`ライブラリの適切な利用方法（コールバック方式の`sd.Stream`か、ブロッキング方式の`sd.playrec`等）を選択します。また、`sd.playrec` のチャンネルマッピングなど、API特有の仕様にも注意します。(Select the appropriate usage method of the `sounddevice` library (e.g., callback-based `sd.Stream` or blocking `sd.playrec`) according to the audio streaming requirements (continuous long-term processing, intermittent repetition of short-term processing, etc.). Also, pay attention to API-specific specifications such as channel mapping for `sd.playrec`.)
    -   `sounddevice`でオーディオデバイスを指定する際、多くの環境でデバイスの整数IDだけでなく、デバイス名（部分的な文字列でも可）も使用できます。これにより、CLIでのユーザーエクスペリエンスが向上する場合があります。(When specifying audio devices with `sounddevice`, device names (even partial strings) can often be used in addition to integer IDs in many environments. This can improve user experience in CLIs.)
    -   **複数チャンネル同時再生録音時の注意点 (Notes on simultaneous multi-channel playback and recording)**:
        -   特定の出力チャンネルでモノラル信号を再生する場合、`sd.playrec()` に渡す出力バッファをデバイスの最大出力チャンネル数で初期化し、対象チャンネルに信号を配置します（例： `output_buffer = np.zeros((len(mono_signal), device_max_out_ch)); output_buffer[:, target_output_ch_idx] = mono_signal`）。(When playing a mono signal on a specific output channel, initialize the output buffer passed to `sd.playrec()` with the device's maximum number of output channels and place the signal in the target channel (e.g., `output_buffer = np.zeros((len(mono_signal), device_max_out_ch)); output_buffer[:, target_output_ch_idx] = mono_signal`).)
        -   `sd.playrec()` の `input_mapping` 引数を使用して、録音する物理入力チャンネルを1ベースのインデックスで指定します（例： `sd.playrec(..., input_mapping=[1, 2])` で物理チャンネル1と2を録音）。(Use the `input_mapping` argument of `sd.playrec()` with 1-based indices to specify the physical input channels to record from (e.g., `sd.playrec(..., input_mapping=[1, 2])` to record from physical channels 1 and 2).)
        -   モノラル信号を特定の2チャンネル（例：ステレオ左右）から同一内容で出力する場合（デュアルモノ出力）、`np.tile(mono_signal.reshape(-1, 1), (1, 2))`のようにして2チャンネルのバッファを準備し、`sd.playrec`の`output_mapping`引数で物理出力チャンネルを指定します（例: `output_mapping=[1, 2]`）。(When outputting a mono signal with identical content from two specific channels (e.g., stereo left and right) for dual-mono output, prepare a 2-channel buffer, for example, using `np.tile(mono_signal.reshape(-1, 1), (1, 2))`, and specify the physical output channels using the `output_mapping` argument of `sd.playrec` (e.g., `output_mapping=[1, 2]`)).
-   **解析処理 (Analysis Processing)**:
    -   FFT、ウィンドウ関数、フィルタリングなど、測定に必要な信号処理を実装します。
    -   `numpy` や `scipy.signal` を活用します。
    -   FFTを用いた解析では、正確な振幅スペクトルを得るための正規化（例：ウィンドウ関数の総和で除算）や、既知の周波数成分を正確に捉えるための周波数ビン選択/補間処理に注意を払います。(In analysis using FFT, pay attention to normalization for obtaining accurate amplitude spectra (e.g., dividing by the sum of the window function) and to frequency bin selection/interpolation processing for accurately capturing known frequency components.)
    -   **相対測定におけるレベル比較 (Level comparison in relative measurements)**:
        -   クロストーク測定のように、基準チャンネルの信号振幅 (A_ref) と測定対象チャンネルの信号振幅 (A_measured) を比較して相対的なdB値を算出する場合（例： `20 * np.log10(A_measured / A_ref)`）、両振幅が有効な値であることを確認し、ゼロ除算や非常に小さい値の対数処理を避けます。(When comparing signal amplitudes from a reference channel (A_ref) and a measured channel (A_measured) to calculate a relative dB value, as in crosstalk measurements (e.g., `20 * np.log10(A_measured / A_ref)`), ensure both amplitudes are valid and avoid division by zero or taking logarithms of very small values.)
    -   位相解析などの相対的な測定を行う関数では、入力信号の特性（例：無音、非常に短い信号）に関するエッジケースを考慮し、エラー終了する代わりに警告を出すか、定義された値（例：位相0度）を返すなど、堅牢な処理を実装します。(For functions performing relative measurements like phase analysis, consider edge cases related to input signal characteristics (e.g., silence, very short signals) and implement robust handling, such as issuing a warning or returning a defined value (e.g., 0 degrees phase) instead of erroring out.)
-   **結果表示 (Results Display)**:
    -   `rich` ライブラリを使用して、結果を整形してコンソールに表示します。
    -   必要に応じて、`matplotlib` を用いたグラフ表示機能や、CSVファイルへの出力機能を実装します。
    -   特定の種類のプロット（例：リサージュ図形）では、正確な視覚的解釈のために特有のmatplotlib設定が重要になる場合があります（例：リサージュ図形における `plt.axis('equal')`）。ツール開発者は、生成するプロットの種類に応じて適切な設定を調査・適用すべきです。(For certain types of plots (e.g., Lissajous figures), specific matplotlib settings may be crucial for accurate visual interpretation (e.g., `plt.axis('equal')` for Lissajous figures). Tool developers should investigate and apply appropriate settings depending on the type of plot being generated.)
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

-   **単体テスト (Unit Tests)**:
    -   主要な関数（信号生成、解析処理など）に対して単体テストを作成します。
    -   Pythonの `unittest` フレームワークを使用し、テストスクリプトは `new_tool_analyzer/test_new_tool_analyzer.py` のように命名します。
    -   テストケースでは、既知の入力に対する期待される出力を検証します。
    -   位相差計算のような方向性を持つ測定（例：Ch1に対するCh2の位相）をテストする場合、テストケース設計時に期待される符号（+/-）や規約（例：Ch2-Ch1かCh1-Ch2か）を明確に定義し、それに基づいてアサーションを行います。(When testing directional measurements like phase difference calculation (e.g., phase of Ch2 relative to Ch1), clearly define the expected sign (+/-) and convention (e.g., Ch2-Ch1 or Ch1-Ch2) during test case design and make assertions accordingly.)
    -   **複数チャンネル処理のテストデータ生成 (Test data generation for multi-channel processing)**:
        -   複数チャンネルのオーディオデータを扱う関数をテストする際は、各チャンネルで特性の異なる合成信号を含む2次元NumPy配列をテスト入力として作成します（例： `test_data = np.array([ch1_signal, ch2_signal]).T`）。(When testing functions that handle multi-channel audio data, create 2D NumPy arrays as test input, synthesizing signals with different characteristics for each channel (e.g., `test_data = np.array([ch1_signal, ch2_signal]).T`).)
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

## 8. 知見の記録と将来構想へのフィードバック (Logging Knowledge and Feedback to Future Concepts)

-   **新たな知見・アイデアの記録**: 作業完了後、開発プロセスを通じて得られた新たな知見、または将来的に有用と思われる新しい測定ツールのアイデアが生まれた場合は、`misc/future_tool_ideas.md` に追記して下さい。これは将来のツール開発のための貴重なリソースとなります。(Record New Insights/Ideas: After completing the work, if new insights were gained through the development process, or if new ideas for potentially useful measurement tools emerged, append them to `misc/future_tool_ideas.md`. This serves as a valuable resource for future tool development.)

---

この手順はあくまでガイドラインであり、開発するツールの特性や規模に応じて適宜調整してください。 (This procedure is merely a guideline; please adjust it as appropriate according to the characteristics and scale of the tool being developed.)
