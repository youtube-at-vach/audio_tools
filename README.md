# Audio Tools

このリポジトリには、オーディオ解析と信号生成に関する小さなプログラムが含まれています。

- オーディオアナライザー ([`audio_analyzer`](./audio_analyzer/README.md)): 音声信号の基本的な特性（レベル、周波数スペクトル、波形など）を分析し、歪み（THD、THD+N、SINAD）を測定するツールです。オーディオ機器の性能評価や音響解析の初期調査に適しています。
- オーディオシグナルジェネレーター ([`audio_signal_generator`](./audio_signal_generator/README.md)): 正弦波、矩形波、のこぎり波、三角波、ホワイトノイズ、ピンクノイズなど、様々な種類のテストトーンや任意波形を生成するツールです。オーディオ機器のテストや音響測定の信号源として利用できます。
- 相互変調歪アナライザー ([`audio_imd_analyzer`](./audio_imd_analyzer/README.md)): オーディオ信号の相互変調歪（IMD）を測定するツールです。SMPTE (RP120-1994) および CCIF (ITU-R) 規格に基づいた測定に対応しており、アンプやスピーカーなどの非線形歪の評価に役立ちます。(詳細は [`audio_imd_analyzer/README.md`](./audio_imd_analyzer/README.md) を参照)
- 周波数応答アナライザー ([`audio_freq_response_analyzer`](./audio_freq_response_analyzer/README.md)): オーディオデバイスの周波数応答（振幅特性および位相特性）を測定し、グラフとしてプロットするツールです。スピーカー、マイク、アンプなどの周波数特性評価やイコライザー調整の参考に利用できます。(詳細は [`audio_freq_response_analyzer/README.md`](./audio_freq_response_analyzer/README.md) を参照)
- クロストークアナライザー ([`audio_crosstalk_analyzer`](./audio_crosstalk_analyzer/README.md)): ステレオオーディオチャンネル間のクロストーク（信号漏れ）の量を測定するツールです。オーディオケーブルやミキサー、インターフェースなどのチャンネルセパレーション性能の評価に使用します。(詳細は [`audio_crosstalk_analyzer/README.md`](./audio_crosstalk_analyzer/README.md) を参照)
- 過渡応答アナライザー ([`audio_transient_analyzer`](./audio_transient_analyzer/README.md)): オーディオデバイスの過渡応答特性（インパルス応答、ステップ応答における立ち上がり時間、オーバーシュート、セトリング時間など）を測定するツールです。スピーカーやアンプの動特性評価に有用です。(詳細は [`audio_transient_analyzer/README.md`](./audio_transient_analyzer/README.md) を参照)
- 位相アナライザー ([`audio_phase_analyzer`](./audio_phase_analyzer/README.md)): ステレオオーディオチャンネル間の位相差を測定し、リサージュ図形などで視覚化するツールです。スピーカーの極性チェック、ステレオ機器の位相整合性の確認、サラウンドシステムの位相管理、マイクロフォンの位相アライメント、位相効果（フェイザー、フランジャーなど）の分析に役立ちます。(詳細は [`audio_phase_analyzer/README.md`](./audio_phase_analyzer/README.md) を参照)
- SNRアナライザー ([`snr_analyzer`](./snr_analyzer/README.md)): テスト信号の再生と録音を通じて、オーディオシステムのS/N比（シグナル対ノイズ比）を測定するツールです。オーディオインターフェースやプリアンプなどのノイズ性能評価に利用できます。
- LUFSメーター ([`lufs_meter`](./lufs_meter/README.md)): ITU-R BS.1770 および EBU R128規格に準拠したラウドネス測定を行うツールです。統合ラウドネス（Integrated）、モーメンタリラウドネス（Momentary）、ショートタームラウドネス（Short-term）、ラウドネスレンジ（LRA）、トゥルーピークレベルを測定し、放送コンテンツや音楽制作におけるラウドネス管理に不可欠です。

---

## ライセンスについて

ここに置かれたプログラムは著作権を放棄していますので、**誰でも自由に使用・改変・再配布・商用利用**することができます。  
これはインターネット上の「情報」として提供されるものであり、**誰のものでもありません**。

---

## The Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

In jurisdictions that recognize copyright laws, the author has dedicated any and all copyright interest in the software to the public domain.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

---

どうぞご自由にお使いください。