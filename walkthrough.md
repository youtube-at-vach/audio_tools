# Network Analyzer Fast Chirp Amplitude Fix Walkthrough

## Changes Made

### Fast Chirp Normalization
**Issue**: "Fast Chirp" mode was displaying the normalized System Gain (0dB for loopback) regardless of the output amplitude. This meant that lowering the output level did not change the displayed result, which was confusing when measuring absolute levels.
**Fix**: Updated `_execute_fast_sweep` in `network_analyzer.py` to add the output amplitude (in dB) to the calculated magnitude when operating in Single Channel mode (Left/Right).
- **Single Channel Mode**: Now displays the **Absolute Signal Level** (e.g., -20 dBFS).
- **XFER Mode**: Continues to display the **Relative Transfer Function** (Gain), which is correct for that mode.

## Verification Results

### Manual Verification Checklist
- [x] **Code Logic**: Verified that `mag_db` is adjusted by `20 * log10(output_amp)` only when `input_mode != 'XFER'`.

## Next Steps
- User should verify that running Fast Chirp with different output amplitudes (e.g., -20 dBFS, -6 dBFS) now results in plots that reflect those levels.
