# Recording & Playback Widget Implementation

## Changes Made

### New Widget: Recorder & Player
**Feature**: Added a new widget for recording and playing audio files.
- **File Support**: Supports WAV, MP3, FLAC, OGG, etc. via `soundfile`.
- **Playback**:
    - Load audio files.
    - **Automatic Resampling**: Automatically resamples files to match the audio engine's sample rate using `scipy.signal.resample` to prevent speed/pitch issues.
    - Play/Stop control.
    - Loop playback mode.
    - Output routing (Stereo, Left, Right, **Mono**).
    - **Mono Mode**: Mixes all input channels to mono and outputs to both Left and Right channels. Useful for single-channel recordings in stereo files.
- **Recording**:
    - Record from input.
    - Input selection (Stereo, Left, Right).
    - Save recording to file (WAV, FLAC, OGG).
    - **Internal Loopback**: Added "Internal Loopback (Software)" checkbox.
        - When enabled, the audio engine feeds the mixed output back into the input.
        - Allows recording the output of other modules (e.g., Signal Generator).
        - Allows other modules (e.g., Spectrum Analyzer) to analyze the playback from the Recorder.
- **Integration**: Added to the main window sidebar.

### Dependencies
- Added `soundfile` to `requirements.txt`.
- Updated GitHub Actions workflows (`build_appimage.yml`, `build_windows.yml`) to include `libsndfile1` (Linux) and collect `soundfile` binaries (PyInstaller).

## Verification Results

### Code Verification
- **Dependency Management**: Verified `soundfile` is added and CI workflows install necessary system libraries.
- **Widget Logic**: Verified `RecorderPlayer` class implements `MeasurementModule` interface and handles audio callbacks correctly.
- **Resampling**: Verified `load_file` logic detects sample rate mismatch and uses `scipy.signal.resample`.
- **Mono Output**: Verified `audio_callback` implements mixing logic for 'Mono' mode.
- **Internal Loopback**: Verified `AudioEngine` implements feedback loop using `last_output_buffer` and `RecorderPlayer` controls it.
- **UI Integration**: Verified `MainWindow` registers the new widget.

## Next Steps
- User should verify that the widget appears in the sidebar.
- Test loading and playing various audio formats.
- Test recording and saving files.
- Verify loop playback works as expected.
