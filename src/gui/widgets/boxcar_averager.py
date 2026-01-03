import argparse

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.audio_engine import AudioEngine
from src.core.localization import tr
from src.measurement_modules.base import MeasurementModule


class BoxcarAverager(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False

        # Parameters
        self.mode = 'Internal Pulse' # 'Internal Pulse', 'Internal Step', 'External Rising', 'External Falling'
        self.period_samples = 4800 # 100ms at 48k
        self.pulse_width_samples = 48 # 1ms
        self.trigger_level = 0.0
        self.trigger_channel = 0 # 0: Left, 1: Right
        self.input_channel = 'Stereo' # 'Left', 'Right', 'Stereo'

        # External Sync Parameters
        self.ref_channel = 1 # 0: Left, 1: Right
        self.trigger_edge = 'Rising' # 'Rising', 'Falling'

        # Internal Pulse gate (reduces noise/compute by integrating only within window)
        # Gate coordinates are in samples within the period, relative to window_origin_sample.
        self.gate_enabled = False
        self.gate_start_samples = 0  # can be negative; will be mod period
        self.gate_length_samples = int(round(0.010 * self.audio_engine.sample_rate))  # default 10ms

        # Internal PRBS/MLS generator cache
        self._mls_cache = None
        self._mls_cache_period = None
        self._mls_seed = 0xACE1

        # State
        self.accumulator = None
        self.count = 0
        self.max_averages = 0 # 0 = Infinite

        # Buffers
        self.input_ring_buffer = None
        self.sample_index_ring = None
        self.input_write_pos = 0
        self.input_read_pos = 0

        # Absolute sample tracking
        self.global_sample_counter = 0
        # Defines the 0-phase reference for (sample % period) folding.
        # Kept stable across resets to avoid integration-window drift.
        self.window_origin_sample = 0

        # Reset handling: when True, accumulation restarts at a stable boundary.
        self.reset_pending = False

        # External trigger state
        self.last_ref_sample = None
        self.last_ref_sample_index = None
        self.capture_active = False
        self.capture_idx = 0

        self.callback_id = None

    @property
    def name(self) -> str:
        return "Boxcar Averager"

    @property
    def description(self) -> str:
        return "High-precision signal averaging for transient analysis."

    def run(self, args: argparse.Namespace):
        print("Boxcar Averager CLI not implemented")

    def get_widget(self):
        return BoxcarAveragerWidget(self)

    def start_analysis(self):
        if self.is_running: return
        self.is_running = True

        # Ring buffers for raw input + absolute sample indices.
        # 2 seconds buffer
        self.input_ring_buffer = np.zeros((self.audio_engine.sample_rate * 2, 2), dtype=float)
        self.sample_index_ring = np.zeros((len(self.input_ring_buffer),), dtype=np.int64)
        self.input_write_pos = 0
        self.input_read_pos = 0

        # Absolute coordinate system for this run
        self.global_sample_counter = 0
        self.window_origin_sample = 0

        # Reset accumulator (but keep origin stable)
        self.reset_average()

        self.callback_id = self.audio_engine.register_callback(self._callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def reset_average(self):
        self.accumulator = np.zeros((self.period_samples, 2))
        self.count = 0
        # Keep window_origin_sample stable so the integration window doesn't drift.
        # We also restart accumulation at a stable boundary when possible.
        self.reset_pending = True
        self.last_ref_sample = None
        self.last_ref_sample_index = None
        self.capture_active = False
        self.capture_idx = 0

    def _gate_segments_in_period(self, period: int):
        """Return list of (start,end) segments in [0,period) for the active gate.

        If gate is disabled or invalid, returns empty list.
        """
        if not self.gate_enabled:
            return []

        gate_len = int(self.gate_length_samples)
        if gate_len <= 0 or gate_len >= period:
            return []

        gate_start_mod = int(self.gate_start_samples) % period
        gate_end = gate_start_mod + gate_len
        if gate_end <= period:
            return [(gate_start_mod, gate_end)]
        return [(gate_start_mod, period), (0, gate_end - period)]

    def _ensure_mls_cache(self, period: int) -> np.ndarray:
        """Return MLS/PRBS sequence of length == period (values in {-1, +1})."""
        if self._mls_cache is not None and self._mls_cache_period == period:
            return self._mls_cache

        # 16-bit maximal-length LFSR (polynomial taps 16,14,13,11 => 0xB400)
        reg = int(self._mls_seed) & 0xFFFF
        if reg == 0:
            reg = 1
        seq = np.empty((period,), dtype=np.float64)
        for i in range(period):
            lsb = reg & 1
            seq[i] = 1.0 if lsb else -1.0
            reg >>= 1
            if lsb:
                reg ^= 0xB400

        self._mls_cache = seq
        self._mls_cache_period = period
        return seq

    def _callback(self, indata, outdata, frames, time, status):
        if status: print(status)

        abs_start = int(self.global_sample_counter)

        # 1. Store Input
        # Handle Ring Buffer Wrap
        buf_len = len(self.input_ring_buffer)
        write_idx = self.input_write_pos

        if write_idx + frames <= buf_len:
            if indata.shape[1] >= 2:
                self.input_ring_buffer[write_idx:write_idx+frames] = indata[:, :2]
            else:
                self.input_ring_buffer[write_idx:write_idx+frames, 0] = indata[:, 0]
                self.input_ring_buffer[write_idx:write_idx+frames, 1] = indata[:, 0]

            self.sample_index_ring[write_idx:write_idx+frames] = np.arange(abs_start, abs_start + frames, dtype=np.int64)
        else:
            # Wrap around
            first_part = buf_len - write_idx
            second_part = frames - first_part

            if indata.shape[1] >= 2:
                self.input_ring_buffer[write_idx:] = indata[:first_part, :2]
                self.input_ring_buffer[:second_part] = indata[first_part:, :2]
            else:
                self.input_ring_buffer[write_idx:, 0] = indata[:first_part, 0]
                self.input_ring_buffer[write_idx:, 1] = indata[:first_part, 0]
                self.input_ring_buffer[:second_part, 0] = indata[first_part:, 0]
                self.input_ring_buffer[:second_part, 1] = indata[first_part:, 0]

            self.sample_index_ring[write_idx:] = np.arange(abs_start, abs_start + first_part, dtype=np.int64)
            self.sample_index_ring[:second_part] = np.arange(abs_start + first_part, abs_start + frames, dtype=np.int64)

        self.input_write_pos = (write_idx + frames) % buf_len
        self.global_sample_counter += int(frames)

        # 2. Generate Output (Internal Mode)
        outdata.fill(0)
        if 'Internal' in self.mode:
            # Generate Pulse/Step
            # Use absolute coordinates so resets / timer jitter don't shift the phase.
            t = (np.arange(frames, dtype=np.int64) + abs_start) - int(self.window_origin_sample)
            t_mod = t % int(self.period_samples)

            signal = np.zeros(frames)

            if self.mode == 'Internal Pulse':
                # High for pulse_width
                signal = np.where(t_mod < self.pulse_width_samples, 1.0, 0.0)
            elif self.mode == 'Internal Step':
                # High for half period
                signal = np.where(t_mod < (self.period_samples // 2), 1.0, -1.0)
            elif self.mode == 'Internal Impulse':
                # One-sample impulse at t_mod == 0
                signal = np.where(t_mod == 0, 1.0, 0.0)
            elif self.mode == 'Internal PRBS/MLS':
                mls = self._ensure_mls_cache(int(self.period_samples))
                signal = mls[t_mod.astype(np.intp, copy=False)]

            # Output to both channels? Or just Left? Let's do Left.
            # Output to selected channel(s)
            if outdata.shape[1] >= 1 and self.input_channel in ['Left', 'Stereo']:
                outdata[:, 0] = signal * 0.5 # -6dB
            if outdata.shape[1] >= 2 and self.input_channel in ['Right', 'Stereo']:
                outdata[:, 1] = signal * 0.5

    def process(self):
        """
        Called periodically to process input buffer and update average.
        """
        if not self.is_running: return

        # Determine available data
        buf_len = len(self.input_ring_buffer)
        write_pos = self.input_write_pos
        read_pos = self.input_read_pos

        if write_pos >= read_pos:
            available = write_pos - read_pos
        else:
            available = buf_len - read_pos + write_pos

        if available == 0: return

        # Extract data + absolute indices (linearized)
        if write_pos >= read_pos:
            data = self.input_ring_buffer[read_pos:write_pos]
            idxs = self.sample_index_ring[read_pos:write_pos]
        else:
            data = np.concatenate((self.input_ring_buffer[read_pos:], self.input_ring_buffer[:write_pos]))
            idxs = np.concatenate((self.sample_index_ring[read_pos:], self.sample_index_ring[:write_pos]))

        self.input_read_pos = write_pos

        if len(data) == 0:
            return

        # Optional reset alignment: restart accumulation on a stable window boundary.
        # This keeps the integration window position consistent in absolute coordinates.
        if self.reset_pending and 'Internal' in self.mode:
            origin = int(self.window_origin_sample)
            period = int(self.period_samples)
            start_mod = int((int(idxs[0]) - origin) % period)
            skip = (period - start_mod) % period
            if skip >= len(data):
                return
            if skip > 0:
                data = data[skip:]
                idxs = idxs[skip:]
        if self.reset_pending:
            # After we have aligned/skipped (if needed), start accumulating.
            self.reset_pending = False

        # Process Data
        if 'Internal' in self.mode:
            # Synchronous folding based on absolute sample indices.
            origin = int(self.window_origin_sample)
            period = int(self.period_samples)

            gate_segments = self._gate_segments_in_period(period)

            if self.accumulator.shape[0] != period:
                self.reset_average()
                return

            fold_idx = int((int(idxs[0]) - origin) % period)
            num_samples = len(data)
            current_idx = 0

            while current_idx < num_samples:
                remaining_in_period = period - fold_idx
                chunk_size = min(num_samples - current_idx, remaining_in_period)

                if gate_segments:
                    # Accumulate only where chunk overlaps the gate window.
                    chunk_start = fold_idx
                    chunk_end = fold_idx + chunk_size
                    for seg_start, seg_end in gate_segments:
                        inter_start = max(chunk_start, seg_start)
                        inter_end = min(chunk_end, seg_end)
                        if inter_end <= inter_start:
                            continue
                        data_off0 = current_idx + (inter_start - chunk_start)
                        data_off1 = current_idx + (inter_end - chunk_start)
                        self.accumulator[inter_start:inter_end] += data[data_off0:data_off1]
                else:
                    self.accumulator[fold_idx : fold_idx + chunk_size] += data[current_idx : current_idx + chunk_size]
                fold_idx += chunk_size
                current_idx += chunk_size

                if fold_idx >= period:
                    fold_idx = 0
                    self.count += 1

        else:
            # External Trigger (Reference Sync)
            # We need to find triggers in 'data'
            # We scan the reference channel for edge crossings.

            ref_idx = self.ref_channel
            if data.shape[1] <= ref_idx: return # Safety

            ref_sig = data[:, ref_idx]

            # Simple Edge Detection
            # We need state from previous chunk to detect edge across boundary?
            # For simplicity, we just look inside current chunk.
            # Ideally we should keep last sample.

            # Create a shifted array including the last sample
            if self.last_ref_sample is None:
                # Prevent a false trigger right at the first sample after start/reset
                self.last_ref_sample = float(ref_sig[0])
                self.last_ref_sample_index = int(idxs[0]) - 1
            extended_ref = np.concatenate(([self.last_ref_sample], ref_sig))
            self.last_ref_sample = float(ref_sig[-1])
            self.last_ref_sample_index = int(idxs[-1])

            # Detect Crossings
            # Rising: prev < level <= curr
            # Falling: prev > level >= curr
            level = self.trigger_level

            if self.trigger_edge == 'Rising':
                triggers = (extended_ref[:-1] < level) & (extended_ref[1:] >= level)
            else:
                triggers = (extended_ref[:-1] > level) & (extended_ref[1:] <= level)

            trigger_indices = np.where(triggers)[0]
            # Absolute trigger sample indices; trigger_indices maps directly to data indices.
            trigger_samples_abs = idxs[trigger_indices] if len(trigger_indices) > 0 else np.array([], dtype=np.int64)

            # State Machine:
            # We might be currently capturing a window.
            # Or waiting for a trigger.

            # Non-retriggerable capture windows, pinned to absolute trigger samples.
            period = int(self.period_samples)
            if self.accumulator.shape[0] != period:
                self.reset_average()
                return

            abs_start = int(idxs[0])
            abs_end = int(idxs[-1]) + 1
            abs_ptr = abs_start

            # Helper: find next trigger >= abs_ptr
            def _next_trigger(at_or_after: int):
                if trigger_samples_abs.size == 0:
                    return None
                pos = np.searchsorted(trigger_samples_abs, at_or_after, side='left')
                if pos >= trigger_samples_abs.size:
                    return None
                return int(trigger_samples_abs[pos])

            while abs_ptr < abs_end:
                if self.capture_active:
                    # Continue capturing until window full.
                    take = min(period - self.capture_idx, abs_end - abs_ptr)
                    if take <= 0:
                        break
                    rel_start = abs_ptr - abs_start
                    rel_end = rel_start + take
                    self.accumulator[self.capture_idx : self.capture_idx + take] += data[rel_start:rel_end]
                    self.capture_idx += take
                    abs_ptr += take

                    if self.capture_idx >= period:
                        self.capture_active = False
                        self.capture_idx = 0
                        self.count += 1
                else:
                    next_trig = _next_trigger(abs_ptr)
                    if next_trig is None or next_trig >= abs_end:
                        break
                    # Start capture exactly at the trigger sample.
                    self.capture_active = True
                    self.capture_idx = 0
                    abs_ptr = next_trig

class BoxcarAveragerWidget(QWidget):
    def __init__(self, module: BoxcarAverager):
        super().__init__()
        self.module = module
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.setInterval(50) # 20 FPS

    def init_ui(self):
        layout = QVBoxLayout()

        # Controls
        controls_group = QGroupBox(tr("Controls"))
        # Use a grid to avoid an overly wide single-row toolbar.
        controls_layout = QGridLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(4)

        self.toggle_btn = QPushButton(tr("Start"))
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        controls_layout.addWidget(self.toggle_btn, 0, 0)

        self.reset_btn = QPushButton(tr("Reset"))
        self.reset_btn.clicked.connect(self.on_reset)
        controls_layout.addWidget(self.reset_btn, 0, 1)

        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("Internal Pulse"), "Internal Pulse")
        self.mode_combo.addItem(tr("Internal Step"), "Internal Step")
        self.mode_combo.addItem(tr("Internal Impulse"), "Internal Impulse")
        self.mode_combo.addItem(tr("Internal PRBS/MLS"), "Internal PRBS/MLS")
        self.mode_combo.addItem(tr("External Reference"), "External Reference")
        mode_idx = self.mode_combo.findData(self.module.mode)
        if mode_idx >= 0:
            self.mode_combo.setCurrentIndex(mode_idx)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        controls_layout.addWidget(QLabel(tr("Mode:")), 0, 2)
        controls_layout.addWidget(self.mode_combo, 0, 3)

        # Period
        self.period_spin = QDoubleSpinBox()
        self.period_spin.setRange(1, 1000) # ms
        self.period_spin.setValue(100)
        self.period_spin.setSuffix(" ms")
        self.period_spin.valueChanged.connect(self.on_period_changed)
        controls_layout.addWidget(QLabel(tr("Period:")), 0, 4)
        controls_layout.addWidget(self.period_spin, 0, 5)

        # Channel
        self.channel_combo = QComboBox()
        self.channel_combo.addItem(tr("Stereo"), "Stereo")
        self.channel_combo.addItem(tr("Left"), "Left")
        self.channel_combo.addItem(tr("Right"), "Right")
        ch_idx = self.channel_combo.findData(self.module.input_channel)
        if ch_idx >= 0:
            self.channel_combo.setCurrentIndex(ch_idx)
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        controls_layout.addWidget(QLabel(tr("Channel:")), 0, 6)
        controls_layout.addWidget(self.channel_combo, 0, 7)

        # Gate Controls
        # Parent this container so calling .show() during init can't create a
        # transient top-level window (startup flash during module preload).
        self.gate_group = QWidget(controls_group)
        gate_layout = QHBoxLayout(self.gate_group)
        gate_layout.setContentsMargins(0,0,0,0)

        self.gate_enable_chk = QCheckBox(tr("Gate"))
        self.gate_enable_chk.setChecked(bool(self.module.gate_enabled))
        self.gate_enable_chk.toggled.connect(self.on_gate_enable_changed)
        gate_layout.addWidget(self.gate_enable_chk)

        self.gate_start_spin = QDoubleSpinBox()
        self.gate_start_spin.setRange(-1000.0, 1000.0)
        self.gate_start_spin.setDecimals(2)
        self.gate_start_spin.setSingleStep(0.1)
        self.gate_start_spin.setSuffix(" ms")
        self.gate_start_spin.setValue(float(self.module.gate_start_samples) / float(self.module.audio_engine.sample_rate) * 1000.0)
        self.gate_start_spin.valueChanged.connect(self.on_gate_start_changed)
        gate_layout.addWidget(QLabel(tr("Start:")))
        gate_layout.addWidget(self.gate_start_spin)

        self.gate_width_spin = QDoubleSpinBox()
        self.gate_width_spin.setRange(0.0, 1000.0)
        self.gate_width_spin.setDecimals(2)
        self.gate_width_spin.setSingleStep(0.1)
        self.gate_width_spin.setSuffix(" ms")
        self.gate_width_spin.setValue(float(self.module.gate_length_samples) / float(self.module.audio_engine.sample_rate) * 1000.0)
        self.gate_width_spin.valueChanged.connect(self.on_gate_width_changed)
        gate_layout.addWidget(QLabel(tr("Width:")))
        gate_layout.addWidget(self.gate_width_spin)

        # Gate controls on second row to keep width compact.
        controls_layout.addWidget(self.gate_group, 1, 0, 1, 4)

        # External Sync Controls (Hidden by default)
        # Same reasoning as gate_group: ensure a parent exists before hide/show.
        self.ext_group = QWidget(controls_group)
        ext_layout = QHBoxLayout(self.ext_group)
        ext_layout.setContentsMargins(0,0,0,0)

        self.ref_combo = QComboBox()
        self.ref_combo.addItems([tr("Left"), tr("Right")])
        self.ref_combo.setCurrentIndex(1) # Default Right
        self.ref_combo.currentIndexChanged.connect(self.on_ref_changed)
        ext_layout.addWidget(QLabel(tr("Ref:")))
        ext_layout.addWidget(self.ref_combo)

        self.edge_combo = QComboBox()
        self.edge_combo.addItem(tr("Rising"), "Rising")
        self.edge_combo.addItem(tr("Falling"), "Falling")
        edge_idx = self.edge_combo.findData(self.module.trigger_edge)
        if edge_idx >= 0:
            self.edge_combo.setCurrentIndex(edge_idx)
        self.edge_combo.currentIndexChanged.connect(self.on_edge_changed)
        ext_layout.addWidget(QLabel(tr("Edge:")))
        ext_layout.addWidget(self.edge_combo)

        self.trig_spin = QDoubleSpinBox()
        self.trig_spin.setRange(-1.0, 1.0)
        self.trig_spin.setSingleStep(0.1)
        self.trig_spin.setValue(0.0)
        self.trig_spin.valueChanged.connect(self.on_trig_changed)
        ext_layout.addWidget(QLabel(tr("Lvl:")))
        ext_layout.addWidget(self.trig_spin)

        # External sync controls on second row.
        controls_layout.addWidget(self.ext_group, 1, 4, 1, 4)
        self.ext_group.hide()
        # Gate controls are always visible (Pulse/Step/External).
        self.gate_group.show()

        # Let combo/spin fields take remaining space when window is narrow.
        controls_layout.setColumnStretch(3, 1)
        controls_layout.setColumnStretch(5, 1)
        controls_layout.setColumnStretch(7, 1)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Plot
        self.plot = pg.PlotWidget(title=tr("Averaged Signal"))
        self.plot.setLabel('left', tr("Amplitude"))
        self.plot.setLabel('bottom', tr("Time"), units='s')
        self.plot.showGrid(x=True, y=True)
        self.curve_l = self.plot.plot(pen='g', name=tr("Left"))
        self.curve_r = self.plot.plot(pen='r', name=tr("Right"))

        layout.addWidget(self.plot)
        self.setLayout(layout)

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText(tr("Stop"))
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText(tr("Start"))

    def on_reset(self):
        self.module.reset_average()

    def on_mode_changed(self, idx):
        val = self.mode_combo.itemData(idx)
        if val is None:
            return
        self.module.mode = val
        self.module.reset_average()

        if val == 'External Reference':
            self.ext_group.show()
        else:
            self.ext_group.hide()
        # Gate controls are always visible; no per-mode toggling.
        self.gate_group.show()

    def on_period_changed(self, val):
        # val is ms
        sr = self.module.audio_engine.sample_rate
        self.module.period_samples = int(val / 1000 * sr)
        self.module.reset_average()

    def on_channel_changed(self, idx):
        val = self.channel_combo.itemData(idx)
        if val is None:
            return
        self.module.input_channel = val
        self.module.reset_average()

    def on_ref_changed(self, idx):
        self.module.ref_channel = idx
        self.module.reset_average()

    def on_edge_changed(self, idx):
        val = self.edge_combo.itemData(idx)
        if val is None:
            return
        self.module.trigger_edge = val
        self.module.reset_average()

    def on_trig_changed(self, val):
        self.module.trigger_level = val
        self.module.reset_average()

    def on_gate_enable_changed(self, checked: bool):
        self.module.gate_enabled = bool(checked)
        self.module.reset_average()

    def on_gate_start_changed(self, start_ms: float):
        sr = self.module.audio_engine.sample_rate
        self.module.gate_start_samples = int(round(start_ms / 1000.0 * sr))
        self.module.reset_average()

    def on_gate_width_changed(self, width_ms: float):
        sr = self.module.audio_engine.sample_rate
        self.module.gate_length_samples = max(0, int(round(width_ms / 1000.0 * sr)))
        self.module.reset_average()

    def update_plot(self):
        if not self.module.is_running: return

        self.module.process()

        if self.module.count > 0:
            avg_full = self.module.accumulator / self.module.count
            sr = self.module.audio_engine.sample_rate
            period = int(self.module.period_samples)

            # If gate is enabled in Internal mode, show only the gated window.
            if 'Internal' in self.module.mode and self.module.gate_enabled:
                gate_len = int(self.module.gate_length_samples)
                if 0 < gate_len < period:
                    gate_start = int(self.module.gate_start_samples)
                    start_mod = gate_start % period
                    end_mod = start_mod + gate_len

                    if end_mod <= period:
                        avg = avg_full[start_mod:end_mod]
                    else:
                        tail = avg_full[start_mod:]
                        head = avg_full[:(end_mod - period)]
                        avg = np.vstack([tail, head])

                    t0 = gate_start / sr
                    t = (np.arange(len(avg), dtype=np.float64) / sr) + t0
                else:
                    avg = avg_full
                    t = np.linspace(0, period / sr, len(avg_full))
            else:
                avg = avg_full
                t = np.linspace(0, period / sr, len(avg_full))

            self.curve_l.setData(t, avg[:, 0])
            self.curve_r.setData(t, avg[:, 1])

            # Visibility
            ch = self.module.input_channel
            if ch == 'Left':
                self.curve_l.setVisible(True)
                self.curve_r.setVisible(False)
            elif ch == 'Right':
                self.curve_l.setVisible(False)
                self.curve_r.setVisible(True)
            else:
                self.curve_l.setVisible(True)
                self.curve_r.setVisible(True)

            self.plot.setTitle(tr("Averaged Signal (N={0})").format(self.module.count))
