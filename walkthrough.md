# Lock-in Amplifier Phase and Unit Fixes

## Changes Implemented

### 1. Phase Correction in Lock-in Amplifier
- **Issue**: The measured phase was offset by -90 degrees in the Lock-in Amplifier compared to manual measurement.
- **Cause**: The signal generator was producing a Sine wave ($\sin(\omega t)$), while the Lock-in detection uses a Cosine-aligned reference (via Hilbert transform or implicit assumption).
- **Fix**: Changed the signal generator in `LockInAmplifier.start_analysis` to produce a Cosine wave (`np.cos`). This aligns the generated signal with the analysis reference, resulting in 0 degrees phase for a perfect loopback.

### 2. Y-Axis Unit Selection
- **Issue**: The FRA plot was fixed to **dBFS**, but the user requested units like **dBV**, **dBu**, etc.
- **Fix**: Added a "Plot Unit" dropdown to the "Frequency Response" tab in the Lock-in Amplifier widget.
- **Supported Units**:
    - **dBFS**: Default (Relative to Digital Full Scale)
    - **dBV**: Decibels relative to 1 Volt RMS ($20 \log_{10}(V_{rms})$)
    - **dBu**: Decibels relative to 0.775 Volts RMS ($20 \log_{10}(V_{rms} / 0.7746)$)
    - **Vrms**: Volts RMS (Linear scale)
    - **Vpeak**: Volts Peak (Linear scale)
- **Implementation**: The conversion uses the `input_sensitivity` from the Audio Engine's calibration data.

## Verification
- **Phase**: The Lock-in Amplifier sweep should now show 0 degrees (flat) for a loopback cable.
- **Units**: The Y-axis label and values will update according to the selected unit during the sweep.
