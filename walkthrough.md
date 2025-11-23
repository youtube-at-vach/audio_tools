# Lock-in Amplifier Fixes

## Changes Implemented

### 1. Phase Correction in Lock-in Amplifier
- **Fix**: Changed signal generator to use Cosine wave (`np.cos`) for 0-degree phase alignment.

### 2. Y-Axis Unit Selection
- **Fix**: Added "Plot Unit" dropdown to FRA tab. Supports dBFS, dBV, dBu, Vrms, Vpeak.

### 3. Plot Reset & Auto-Range
- **Fix**: Forced `enableAutoRange()` at the start of each sweep and `autoRange()` at the end.

### 4. Output Settings Unification
- **Fix**: Removed redundant FRA Amplitude settings. The sweep now uses the Manual Control settings.

### 5. Measurement Loop Reliability (New)
- **Issue**: Measurements were failing (returning 0 or noise) when the Manual Control timer was not running, because the data processing logic (`process_data`) was tied to the GUI timer.
- **Fix**: Updated `FRASweepWorker` to explicitly trigger `process_data()` inside the sweep loop.
- **Logic**:
    1. Set Frequency.
    2. Wait `Settling Time`.
    3. Clear averaging history.
    4. Loop `Averaging Count` times:
        - Wait for `Buffer Duration` (e.g. ~0.1s).
        - Call `process_data()` to analyze the current buffer.
    5. Record the averaged result.
- **Benefit**: This ensures reliable measurements even with heavy processing or when the GUI timer is inactive, and guarantees that the data used for the measurement corresponds to the current frequency.

## Verification
- **Reliability**: Run an FRA sweep without starting the Manual Control. The plot should populate with valid data.
- **Averaging**: Increasing "Averaging" count in Manual settings should slow down the sweep slightly per point but reduce noise.
