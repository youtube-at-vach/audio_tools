このプログラムは現在制作中です。バグが含まれている可能性ｆ非常に高いです。
実験やプログラムの参考程度にお使い下さい。断り無く自由に使って下さい。

Audio Analyzer v1.4.5

概要

このスクリプトは、音声信号の高調波解析を行うツールです。主な機能には、THD（全高調波歪）、THD+N、SNRの測定、音声信号のゲイン表示、周波数スイープや振幅スイープによる連続測定、テストトーンの出力、および新機能としてマッピング測定が含まれます。

Copyright: pass
作成者: ChatGPT および Vach
日付: 2024年10月16日

特徴

    ピーク検出と高調波解析
    THD（全高調波歪）および THD+N の測定
    SNR（信号対雑音比）の測定
    入力振幅と測定振幅の比較によるゲイン表示
    各測定終了時に高調波解析結果を表示
    複数回の測定から得た平均結果と標準偏差を表示
    周波数スイープや振幅スイープによる連続測定
    テストトーンの出力機能
    新機能: --map オプションによる周波数・振幅のマッピング測定

ファイル構成

    audio_analyzer.py: メインプログラム。音声信号の解析と測定を行います。
    distorsion_visualizer.py: 歪みデータを視覚化します。
    aligner.py: 信号のタイミング整列を行うユーティリティ。
    audiocalc.py: オーディオ関連の計算処理を担当。
    200Hz.wav: Alignerモジュールの実証用の録音サンプルオーディオファイル。
    requirement.txt: 必要なPythonパッケージを記載したファイル。

必要な環境

    Python 3.8以降が必要です。
    必要なパッケージは requirement.txt に記載されています。インストールには以下のコマンドを使用します。

pip install -r requirement.txt

使用方法

    audio_analyzer.py

このプログラムは、音声信号の測定および高調波解析を行います。周波数や振幅の設定を指定し、測定結果を取得できます。

基本的なコマンド:
python audio_analyzer.py --frequency 1000 --amplitude -6 --duration 5.0

オプション:

    -f, --frequency: 測定する基本周波数（デフォルト: 1000 Hz）
    -a, --amplitude: トーンの振幅（デフォルト: -6 dBFS）
    -w, --window: 窓関数の種類（デフォルト: blackmanharris）
    --duration: 測定時間（デフォルト: 5秒）
    --bandpass: バンドパスフィルターの適用
    -sr, --sample_rate: サンプリングレート（デフォルト: 48000 Hz）
    -oc, --output_channel: 出力チャンネル（LまたはR、デフォルト: R）
    -n, --num_measurements: 測定回数（デフォルト: 2回）

モードオプション:

1つのモードを選択して使用できます。

    --sweep-amplitude: 振幅スイープモード
    --sweep-frequency: 周波数スイープモード
    --map: マッピング測定モード
    --test: テストトーンの出力
    --calib: キャリブレーションモード
            (現在テスト中です。AC電圧計でdBFSの補正値を作ることができます。)

追加オプション:

    --output_csv: 測定結果をCSVファイルに保存するためのファイル名

    distorsion_visualizer.py

このプログラムは、CSVファイルから歪みデータを読み取り、視覚化します。

基本的なコマンド:
python distorsion_visualizer.py sample.csv --device_name 'Test Device'

オプション:

    csv_file: 歪みデータを含むCSVファイルのパス
    -d, --device_name: タイトルに含めるデバイス名（デフォルトは空白）
    -a, --amplitude_type: 振幅データとして Output(dBFS) または Input(dBFS) を選択（デフォルト: Output(dBFS)）
    -c, --convert_to_dBVrms: 振幅をdBVrmsに変換するオプション



Audio Analyzerプログラムの中の関数の詳細

    テストトーンの生成
    generate_tone 関数は、指定された周波数、振幅、位相、サンプルレートに基づいてシングルトーン信号を生成します。

    テストトーンの再生
    test_tone 関数は、ユーザーが指定した周波数でテストトーンを再生します。エンターキーで終了。

    音声デバイスの選択
    select_device 関数は、使用可能な音声デバイスをリスト表示し、選択させます。

    高調波解析結果の表示
    print_harmonic_analysis 関数で、解析された高調波データを表形式で表示します。

    音声測定
    measure 関数で、指定された周波数と振幅で音声測定を行います。

    ノイズ測定
    measure_noise 関数は、環境ノイズを測定し、RMSレベルを計算します。

    測定結果の表示
    display_measurements および display_statics 関数で、測定結果をテーブル形式で表示します。

    周波数・振幅スイープ
    perform_measurements 関数で、複数の周波数や振幅で連続測定を行い、結果をCSV形式で保存可能です。