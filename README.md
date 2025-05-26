# Audio Tools

このリポジトリには、オーディオ解析と信号生成に関する小さなプログラムが含まれています。

- `audio_analyzer`: 音声信号の歪や特性を解析するツール
- `audio_signal_generator`: 任意波形やテストトーンを生成するためのツール
- `audio_imd_analyzer`: オーディオ信号の相互変調歪（IMD）を測定するツール。SMPTE (RP120-1994) および CCIF (ITU-R) 規格をサポートしています。(詳細は `audio_imd_analyzer/README.md` を参照)
- `audio_freq_response_analyzer`: オーディオデバイスの周波数応答（振幅および位相）を測定・プロットするツール (詳細は `audio_freq_response_analyzer/README.md` を参照)
- `audio_crosstalk_analyzer`: チャンネル間のオーディオクロストーク（信号漏れ）を測定するツール。(詳細は `audio_crosstalk_analyzer/README.md` を参照)
- `audio_transient_analyzer`: オーディオデバイスの過渡応答（立ち上がり時間、オーバーシュート、セトリング時間など）を測定するツール。(詳細は `audio_transient_analyzer/README.md` を参照)
- `audio_phase_analyzer`: ステレオオーディオチャンネル間の位相差を測定および視覚化するツール。スピーカーの極性チェック、ステレオ機器の位相整合性の確認、位相効果の分析に役立ちます。(詳細は `audio_phase_analyzer/README.md` を参照)
- **[SNR Analyzer (`snr_analyzer`)](./snr_analyzer/README.md)**: Measures the Signal-to-Noise Ratio (SNR) of an audio system by playing and recording test signals.
- **[LUFS Meter (`lufs_meter`)](./lufs_meter/README.md)**: Measures audio loudness (Integrated, Momentary, Short-term, LRA, True Peak) according to ITU-R BS.1770 and EBU R128 standards.

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