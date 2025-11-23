# Fix FRA Measurement Loop

## Goal
Fix the issue where FRA measurements fail (return 0 or noise) when using an active filter or when the Manual Control timer is not running. The root cause is that the `process_data` method, which calculates magnitude and phase, is only called by the GUI timer, which may not be running or synchronized during an FRA sweep.

## Proposed Changes

### `src/gui/widgets/lock_in_amplifier.py`

#### [MODIFY] `FRASweepWorker` class
- **Update `run` method**:
    - Inside the sweep loop, after setting the frequency and sleeping for `settle_time`:
        - Clear the averaging history (`self.module.history.clear()`).
        - Explicitly call `self.module.process_data()` multiple times (equal to `averaging_count` + buffer flush) to ensure fresh data is processed and averaged.
        - Add a small sleep between `process_data` calls to allow the audio callback to fill the buffer (though `process_data` works on the current buffer state, we need to wait for *new* samples to arrive).
        - Actually, `process_data` just reads the *current* buffer. We need to wait for the buffer to be updated by the audio callback.
    - **Better Approach**:
        - The audio callback runs continuously.
        - We need to wait for `averaging_count` *new* chunks of data.
        - Instead of complex synchronization, we can just sleep for a duration that ensures enough new data has arrived, then call `process_data`.
        - `settle_time` handles the initial transient.
        - After settling, we need to capture `averaging_count` samples for the lock-in filter.
        - The `process_data` method appends to `history`.
        - We should call `process_data` periodically.
        - **Revised Logic**:
            1. Set Freq.
            2. Sleep `settle_time`.
            3. Clear `history`.
            4. Loop `averaging_count` times:
                - Sleep `buffer_duration` (e.g. 0.1s for 4096 samples).
                - Call `self.module.process_data()`.
            5. Read result.

## Verification Plan

### Manual Verification
1. Open Lock-in Amplifier.
2. Go to "Frequency Response" tab.
3. **Do NOT** start "Start Output & Measure" in Manual tab.
4. Start FRA Sweep.
5. Verify that the plot updates and shows reasonable values (not 0).
6. Verify that the "Reference Status" in Manual tab (if visible) or the internal state updates.
