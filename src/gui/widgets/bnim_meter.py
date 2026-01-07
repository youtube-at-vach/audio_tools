import threading
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter

from src.core.audio_engine import AudioEngine
from src.core.localization import tr
from src.measurement_modules.base import MeasurementModule


class BNIMMeter(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False

        # Settings
        self.fft_size = 2048
        self.overlap = 0.5
        self.max_itd_ms = 0.8 # Max Interaural Time Difference in ms (human range is ~0.7ms)
        self.num_itd_bins = 256
        self.freq_min = 20
        self.freq_max = 5000 # Primary localization range
        self.decay = 0.85
        self.gain = 1.0
        self.glow_sigma = 1.0

        # Optional ILD (LSO-like) weighting
        self.enable_ild = False
        self.ild_strength = 0.6
        self.ild_width_db = 6.0
        self.ild_freq_split_hz = 1500.0

        # Click-to-play test signal (optional)
        self.play_enable_click = False
        self.play_loop = False
        self.play_on_cycles = 10
        self.play_off_cycles = 900
        # Playback ILD: positive attenuation (dB) applied to the ITD-delayed ear.
        # Non-delayed ear stays at 0 dB.
        self.play_ild_atten_db = 0.0
        # Conservative default level (avoid accidental loud output)
        self.play_amplitude = 0.2

        self._play_lock = threading.Lock()
        self._play_buffer = None  # np.ndarray shape (N, 2)
        self._play_index = 0
        self._play_active = False
        self._last_click_freq_hz = None
        self._last_click_itd_ms = None

        # State
        self.callback_id = None
        self.sample_rate = self.audio_engine.sample_rate
        self._buffer_lock = threading.Lock()
        self.audio_buffer = np.zeros((4096, 2), dtype=np.float32)
        self._buffer_seq = 0
        self._last_processed_seq = -1

        # Neural Map State (Frequencies x ITD)
        # We'll determine the actual shape during first processing
        self.neural_map = None
        self.frequencies = None
        self.itd_axis = None
        self._phase_diff_model = None
        self._itd_axis_norm = None

    @property
    def name(self) -> str:
        return "BNIM Meter"

    @property
    def description(self) -> str:
        return "Binaural Neural Interferometric Meter - ITD/ILD Neural Map."

    def run(self, args):
        print("BNIM Meter CLI not implemented")

    def get_widget(self):
        return BNIMMeterWidget(self)

    def start_analysis(self):
        if self.is_running: return
        self.is_running = True
        self.sample_rate = self.audio_engine.sample_rate

        with self._buffer_lock:
            self.audio_buffer = np.zeros((self.fft_size * 2, 2), dtype=np.float32)
            self._buffer_seq = 0
            self._last_processed_seq = -1

        # Prepare axes
        self.itd_axis = np.linspace(-self.max_itd_ms, self.max_itd_ms, self.num_itd_bins)
        self._itd_axis_norm = (self.itd_axis / max(1e-9, float(self.max_itd_ms))).astype(np.float32, copy=False)

        # Frequency bins for RFFT
        freqs = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)
        # Select indices within range
        self.freq_indices = np.where((freqs >= self.freq_min) & (freqs <= self.freq_max))[0]
        self.frequencies = freqs[self.freq_indices]

        # Initialize Neural Map
        self.neural_map = np.zeros((len(self.frequencies), self.num_itd_bins))

        # Precompute ITD phase model: (freq, itd)
        delays_s = (self.itd_axis / 1000.0).astype(np.float32, copy=False)
        self._phase_diff_model = (-2.0 * np.pi * self.frequencies[:, None].astype(np.float32, copy=False) * delays_s[None, :]).astype(np.float32, copy=False)

        self.callback_id = self.audio_engine.register_callback(self._callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

        with self._play_lock:
            self._play_active = False
            self._play_buffer = None
            self._play_index = 0

    @staticmethod
    def _fractional_delay_zero_padded(x: np.ndarray, delay_samples: float) -> np.ndarray:
        """Delay signal by delay_samples (>=0) using linear interpolation, zero padding outside bounds."""
        if delay_samples <= 0.0:
            return x.astype(np.float32, copy=False)

        n = np.arange(len(x), dtype=np.float64)
        idx = n - float(delay_samples)

        i0 = np.floor(idx).astype(np.int64, copy=False)
        frac = (idx - i0).astype(np.float64, copy=False)

        y = np.zeros_like(x, dtype=np.float32)

        valid = (i0 >= 0) & (i0 + 1 < len(x))
        if not np.any(valid):
            return y

        i0v = i0[valid]
        i1v = i0v + 1
        fv = frac[valid].astype(np.float32, copy=False)

        xv0 = x[i0v].astype(np.float32, copy=False)
        xv1 = x[i1v].astype(np.float32, copy=False)
        y[valid] = (1.0 - fv) * xv0 + fv * xv1
        return y

    def build_click_test_burst(self, *, freq_hz: float, itd_ms: float, on_cycles: int, off_cycles: int, ild_atten_db: float) -> np.ndarray:
        """Build a stereo tone burst using (frequency, ITD, ILD attenuation). Returns float32 array (N,2)."""
        sr = float(self.sample_rate)
        f = float(np.clip(freq_hz, 1.0, sr / 2.0))

        on_cycles_i = int(max(1, on_cycles))
        off_cycles_i = int(max(0, off_cycles))

        total_cycles = on_cycles_i + off_cycles_i
        total_duration_s = float(total_cycles) / f
        n_base = int(max(1, int(np.round(total_duration_s * sr))))

        # ON segment length in samples
        n_on = int(np.clip(int(np.round((float(on_cycles_i) / f) * sr)), 0, n_base))

        t = (np.arange(n_base, dtype=np.float64) / sr)
        tone = np.sin(2.0 * np.pi * f * t).astype(np.float32, copy=False)

        env = np.zeros(n_base, dtype=np.float32)
        if n_on > 0:
            if n_on >= 2:
                env[:n_on] = np.hanning(n_on).astype(np.float32, copy=False)
            else:
                env[:n_on] = 1.0

        base = (tone * env) * float(self.play_amplitude)

        # Pad a little so small ITD doesn't truncate the delayed channel tail.
        itd_ms_c = float(np.clip(itd_ms, -float(self.max_itd_ms), float(self.max_itd_ms)))
        delay_samples = abs(itd_ms_c) * sr / 1000.0
        pad = int(np.ceil(delay_samples)) + 2
        x = np.zeros(n_base + pad, dtype=np.float32)
        x[:n_base] = base

        # UI convention: +ITD (right side of plot) should localize to the RIGHT.
        # For a right-side source, left ear arrives later (left delayed).
        if itd_ms_c >= 0:
            # Positive ITD: left ear delayed
            l = self._fractional_delay_zero_padded(x, delay_samples)
            r = x
        else:
            # Negative ITD: right ear delayed
            l = x
            r = self._fractional_delay_zero_padded(x, delay_samples)

        # Playback ILD rule:
        # - Determine which ear was delayed by ITD.
        # - Apply -atten_dB to THAT ear only.
        # - If ITD == 0 (no delay), apply no ILD attenuation.
        atten = float(np.clip(ild_atten_db, 0.0, 60.0))
        if delay_samples <= 1e-9 or atten <= 0.0:
            g_l = 1.0
            g_r = 1.0
        else:
            g_att = float(10.0 ** (-atten / 20.0))
            if itd_ms_c >= 0:
                # left is delayed
                g_l = g_att
                g_r = 1.0
            else:
                # right is delayed
                g_l = 1.0
                g_r = g_att

        y = np.column_stack((l * g_l, r * g_r)).astype(np.float32, copy=False)
        return y

    def trigger_click_test_playback(self, *, freq_hz: float, itd_ms: float, on_cycles: int, off_cycles: int, ild_atten_db: float):
        """Arm a one-shot playback buffer. Output occurs inside the audio callback."""
        if not self.is_running:
            return

        self._last_click_freq_hz = float(freq_hz)
        self._last_click_itd_ms = float(itd_ms)

        buf = self.build_click_test_burst(
            freq_hz=freq_hz,
            itd_ms=itd_ms,
            on_cycles=on_cycles,
            off_cycles=off_cycles,
            ild_atten_db=ild_atten_db,
        )

        with self._play_lock:
            self._play_buffer = buf
            self._play_index = 0
            self._play_active = True

    def refresh_click_test_playback_if_looping(self):
        """If loop playback is enabled and a click position exists, rebuild buffer with current settings."""
        if not self.is_running:
            return
        if not bool(self.play_enable_click and self.play_loop):
            return
        if self._last_click_freq_hz is None or self._last_click_itd_ms is None:
            return

        buf = self.build_click_test_burst(
            freq_hz=float(self._last_click_freq_hz),
            itd_ms=float(self._last_click_itd_ms),
            on_cycles=int(self.play_on_cycles),
            off_cycles=int(self.play_off_cycles),
            ild_atten_db=float(self.play_ild_atten_db),
        )

        with self._play_lock:
            # Keep playback active, but restart from beginning to reflect the new settings immediately.
            self._play_buffer = buf
            self._play_index = 0
            self._play_active = True

    def _callback(self, indata, outdata, frames, time, status):
        # Update local buffer (Roll)
        if indata is None:
            # Still allow test playback even if input stream is absent.
            self._render_playback(outdata, frames)
            return

        if indata.shape[1] >= 2:
            src = indata[:, :2]
        else:
            src = np.column_stack((indata[:, 0], indata[:, 0]))

        with self._buffer_lock:
            if frames >= len(self.audio_buffer):
                self.audio_buffer[:] = src[-len(self.audio_buffer):, :].astype(np.float32, copy=False)
            else:
                self.audio_buffer = np.roll(self.audio_buffer, -frames, axis=0)
                self.audio_buffer[-frames:] = src.astype(np.float32, copy=False)
            self._buffer_seq += 1

        self._render_playback(outdata, frames)

    def _render_playback(self, outdata, frames: int):
        """Write click-test playback into outdata if active; otherwise silence."""
        with self._play_lock:
            active = bool(self._play_active and self._play_buffer is not None)
            if not active:
                outdata.fill(0)
                return

            buf = self._play_buffer
            outdata.fill(0)
            if buf is None or len(buf) == 0:
                self._play_active = False
                self._play_buffer = None
                self._play_index = 0
                return

            loop = bool(self.play_loop)
            idx = int(self._play_index)
            written = 0

            while written < int(frames):
                remaining = int(frames) - written
                if idx >= len(buf):
                    if loop:
                        idx = 0
                    else:
                        break

                end = min(idx + remaining, len(buf))
                chunk = buf[idx:end]
                n = int(len(chunk))
                if n <= 0:
                    if loop:
                        idx = 0
                        continue
                    break

                if outdata.shape[1] >= 2:
                    outdata[written:written + n, 0] = chunk[:, 0]
                    outdata[written:written + n, 1] = chunk[:, 1]
                elif outdata.shape[1] == 1:
                    outdata[written:written + n, 0] = chunk[:, 0]

                written += n
                idx = end

            if loop:
                # Keep active and wrap index.
                self._play_index = int(idx % len(buf))
            else:
                # One-shot: stop once buffer is exhausted.
                if idx >= len(buf) or written < int(frames):
                    self._play_active = False
                    self._play_buffer = None
                    self._play_index = 0
                else:
                    self._play_index = int(idx)

    def process_buffer(self):
        """Perform the 'neural' processing: ITD/ILD extraction per frequency."""
        if not self.is_running: return

        # Extract last window
        with self._buffer_lock:
            # Avoid redundant FFTs if the GUI timer fires faster than audio callbacks.
            # Keep first-call behavior so unit tests that manually set audio_buffer still run.
            if self._buffer_seq == self._last_processed_seq and self._last_processed_seq != -1:
                return
            self._last_processed_seq = self._buffer_seq
            window_data = self.audio_buffer[-self.fft_size:].copy()
        L = window_data[:, 0]
        R = window_data[:, 1]

        # Apply window (Hann)
        win = np.hanning(self.fft_size)
        L_w = L * win
        R_w = R * win

        # FFT
        fft_L = np.fft.rfft(L_w)
        fft_R = np.fft.rfft(R_w)

        # Select active frequencies
        fft_L = fft_L[self.freq_indices]
        fft_R = fft_R[self.freq_indices]

        # Normalize by magnitude to focus on phase (ITD)
        # Mag L/R for ILD
        mag_L = np.abs(fft_L)
        mag_R = np.abs(fft_R)

        # Avoid div by zero
        eps = 1e-10
        mag_sum = mag_L + mag_R + eps

        # Jeffress-style coincidence map against a precomputed delay-line phase model.
        phase_diff_model = self._phase_diff_model

        phase_L = np.angle(fft_L)
        phase_R = np.angle(fft_R)

        # Broadcast
        phase_diff_signal = (phase_L - phase_R)[:, np.newaxis]

        # Coincidence = cos(phase_diff_signal - phase_diff_model)
        coincidence = 0.5 + 0.5 * np.cos(phase_diff_signal - phase_diff_model)

        # Band intensity (energy proxy)
        band_intensity = np.log1p(mag_sum * self.gain).astype(np.float32, copy=False)
        coincidence = coincidence.astype(np.float32, copy=False) * band_intensity[:, np.newaxis]

        # Optional ILD (LSO-like) lateral bias.
        # Positive ILD (L>R) boosts negative ITD side ("left"), and vice versa.
        if self.enable_ild:
            ild_db = 20.0 * np.log10((mag_L + eps) / (mag_R + eps))
            ild_db = np.clip(ild_db, -60.0, 60.0)

            # Smoothly activate ILD effect above a split frequency (mammalian heuristic).
            f = self.frequencies.astype(np.float32, copy=False)
            split = float(self.ild_freq_split_hz)
            ild_band_weight = np.clip((f - split) / max(1.0, split), 0.0, 1.0)

            ild_sign = np.tanh(ild_db / max(1e-6, float(self.ild_width_db))).astype(np.float32, copy=False)
            # Map ITD axis to [-1,1]; negative is left.
            itd_norm = self._itd_axis_norm
            # If ild_sign>0 (left louder), boost negative ITD (itd_norm<0) via (-itd_norm).
            lateral = (1.0 + (self.ild_strength * ild_band_weight)[:, None] * (-itd_norm[None, :]) * ild_sign[:, None]).astype(np.float32, copy=False)
            coincidence *= np.clip(lateral, 0.0, 5.0)

        # Update neural map with persistence
        self.neural_map = (self.neural_map * self.decay) + (coincidence * (1.0 - self.decay))

class BNIMMeterWidget(QWidget):
    def __init__(self, module: BNIMMeter):
        super().__init__()
        self.module = module
        self.init_ui()

        # Disable default plot interactions (pan/zoom) and use our own click/drag handling.
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.viewport().installEventFilter(self)

        self._dragging = False
        self._last_drag_update_t = 0.0
        self._drag_update_interval_s = 0.03  # ~33 Hz
        self._loop_before_drag = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.setInterval(40) # 25 FPS is usually enough for this kind of "neural" look

    def init_ui(self):
        layout = QHBoxLayout()

        # --- Left: Display ---
        display_layout = QVBoxLayout()

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#050505')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')

        # Label axes
        self.label_itd = QLabel(tr("ITD (Left <-> Right)"))
        self.label_itd.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_itd.setStyleSheet("color: #00ff00; font-size: 10px;")

        display_layout.addWidget(self.plot_widget, stretch=1)
        display_layout.addWidget(self.label_itd)
        self.img_item = pg.ImageItem()
        # Custom colormap: Black -> Dark Blue -> Green -> Yellow -> White
        pos = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        color = np.array([[0,0,0,255], [0,0,100,255], [0,255,0,255], [255,255,0,255], [255,255,255,255]], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(lut)

        self.plot_widget.addItem(self.img_item)

        layout.addLayout(display_layout, stretch=3)

        # --- Right: Controls ---
        controls_group = QGroupBox(tr("BNIM Controls"))
        controls_layout = QVBoxLayout()

        self.toggle_btn = QPushButton(tr("Start"))
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        controls_layout.addWidget(self.toggle_btn)

        # Persistence
        controls_layout.addWidget(QLabel(tr("Persistence:")))
        self.decay_slider = QSlider(Qt.Orientation.Horizontal)
        self.decay_slider.setRange(50, 99)
        self.decay_slider.setValue(int(self.module.decay * 100))
        self.decay_slider.valueChanged.connect(self.on_decay_changed)
        controls_layout.addWidget(self.decay_slider)

        # Gain
        controls_layout.addWidget(QLabel(tr("Gain:")))
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(1, 200)
        self.gain_slider.setValue(100)
        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        controls_layout.addWidget(self.gain_slider)

        # Glow
        controls_layout.addWidget(QLabel(tr("Neural Glow:")))
        self.glow_slider = QSlider(Qt.Orientation.Horizontal)
        self.glow_slider.setRange(0, 50)
        self.glow_slider.setValue(int(self.module.glow_sigma * 10))
        self.glow_slider.valueChanged.connect(self.on_glow_changed)
        controls_layout.addWidget(self.glow_slider)

        # ILD (LSO-like)
        self.ild_checkbox = QCheckBox(tr("Enable ILD weighting"))
        self.ild_checkbox.setChecked(bool(self.module.enable_ild))
        self.ild_checkbox.stateChanged.connect(self.on_ild_enabled_changed)
        controls_layout.addWidget(self.ild_checkbox)

        controls_layout.addWidget(QLabel(tr("ILD Strength:")))
        self.ild_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.ild_strength_slider.setRange(0, 200)
        self.ild_strength_slider.setValue(int(self.module.ild_strength * 100))
        self.ild_strength_slider.valueChanged.connect(self.on_ild_strength_changed)
        controls_layout.addWidget(self.ild_strength_slider)

        # Freq Range
        controls_layout.addWidget(QLabel(tr("Max Freq (Hz):")))
        self.freq_combo = QComboBox()
        self.freq_combo.addItems(["1000", "2000", "5000", "10000"])
        self.freq_combo.setCurrentText("5000")
        self.freq_combo.currentTextChanged.connect(self.on_freq_changed)
        controls_layout.addWidget(self.freq_combo)

        # --- Click-to-play test signal ---
        controls_layout.addWidget(QLabel(tr("Click-to-play Test Signal:")))

        self.play_click_checkbox = QCheckBox(tr("Enable click-to-play"))
        self.play_click_checkbox.setChecked(bool(self.module.play_enable_click))
        self.play_click_checkbox.stateChanged.connect(self.on_play_click_enabled_changed)
        controls_layout.addWidget(self.play_click_checkbox)

        self.play_loop_checkbox = QCheckBox(tr("Loop last click"))
        self.play_loop_checkbox.setChecked(bool(self.module.play_loop))
        self.play_loop_checkbox.stateChanged.connect(self.on_play_loop_changed)
        controls_layout.addWidget(self.play_loop_checkbox)

        play_row = QHBoxLayout()

        self.play_on_spin = QSpinBox()
        self.play_on_spin.setRange(1, 200)
        self.play_on_spin.setValue(int(self.module.play_on_cycles))
        self.play_on_spin.valueChanged.connect(self.on_play_on_cycles_changed)
        play_row.addWidget(QLabel(tr("On cycles")))
        play_row.addWidget(self.play_on_spin)

        self.play_off_spin = QSpinBox()
        self.play_off_spin.setRange(0, 5000)
        self.play_off_spin.setValue(int(self.module.play_off_cycles))
        self.play_off_spin.valueChanged.connect(self.on_play_off_cycles_changed)
        play_row.addWidget(QLabel(tr("Off cycles")))
        play_row.addWidget(self.play_off_spin)

        controls_layout.addLayout(play_row)

        self.play_ild_spin = QDoubleSpinBox()
        self.play_ild_spin.setRange(0.0, 30.0)
        self.play_ild_spin.setSingleStep(1.0)
        self.play_ild_spin.setDecimals(0)
        self.play_ild_spin.setValue(float(self.module.play_ild_atten_db))
        self.play_ild_spin.valueChanged.connect(self.on_play_ild_changed)
        controls_layout.addWidget(QLabel(tr("Playback ILD Attenuation (dB):")))
        controls_layout.addWidget(self.play_ild_spin)

        self.play_last_label = QLabel(tr("Last: (click plot)"))
        controls_layout.addWidget(self.play_last_label)

        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group, stretch=1)

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

    def on_decay_changed(self, val):
        self.module.decay = val / 100.0

    def on_gain_changed(self, val):
        self.module.gain = val / 100.0

    def on_glow_changed(self, val):
        self.module.glow_sigma = val / 10.0

    def on_ild_enabled_changed(self, state):
        self.module.enable_ild = bool(state)

    def on_ild_strength_changed(self, val):
        self.module.ild_strength = val / 100.0

    def on_freq_changed(self, text):
        was_running = self.module.is_running
        if was_running: self.module.stop_analysis()
        self.module.freq_max = int(text)
        if was_running: self.module.start_analysis()

    def on_play_click_enabled_changed(self, state):
        self.module.play_enable_click = bool(state)

    def on_play_loop_changed(self, state):
        self.module.play_loop = bool(state)

    def on_play_on_cycles_changed(self, val):
        self.module.play_on_cycles = int(val)
        self.module.refresh_click_test_playback_if_looping()

    def on_play_off_cycles_changed(self, val):
        self.module.play_off_cycles = int(val)
        self.module.refresh_click_test_playback_if_looping()

    def on_play_ild_changed(self, val):
        self.module.play_ild_atten_db = float(val)
        self.module.refresh_click_test_playback_if_looping()

    def on_plot_clicked(self, event):
        if not self.module.is_running:
            return
        if not bool(self.module.play_enable_click):
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        # Convert scene click to data coordinates.
        p = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
        itd_ms = float(p.x())
        freq_hz = float(p.y())

        # Clamp to display range.
        itd_ms = float(np.clip(itd_ms, -float(self.module.max_itd_ms), float(self.module.max_itd_ms)))
        freq_hz = float(np.clip(freq_hz, float(self.module.freq_min), float(self.module.freq_max)))

        self.module.trigger_click_test_playback(
            freq_hz=freq_hz,
            itd_ms=itd_ms,
            on_cycles=int(self.module.play_on_cycles),
            off_cycles=int(self.module.play_off_cycles),
            ild_atten_db=float(self.module.play_ild_atten_db),
        )
        self.play_last_label.setText(tr("Last: {freq:.0f} Hz, {itd:+.3f} ms").format(freq=freq_hz, itd=itd_ms))

    def _handle_plot_point(self, scene_pos, force: bool):
        if not self.module.is_running:
            return
        if not bool(self.module.play_enable_click):
            return

        # Convert scene position to data coordinates.
        p = self.plot_widget.plotItem.vb.mapSceneToView(scene_pos)
        itd_ms = float(p.x())
        freq_hz = float(p.y())

        # Clamp to display range.
        itd_ms = float(np.clip(itd_ms, -float(self.module.max_itd_ms), float(self.module.max_itd_ms)))
        freq_hz = float(np.clip(freq_hz, float(self.module.freq_min), float(self.module.freq_max)))

        self.module.trigger_click_test_playback(
            freq_hz=freq_hz,
            itd_ms=itd_ms,
            on_cycles=int(self.module.play_on_cycles),
            off_cycles=int(self.module.play_off_cycles),
            ild_atten_db=float(self.module.play_ild_atten_db),
        )
        self.play_last_label.setText(tr("Last: {freq:.0f} Hz, {itd:+.3f} ms").format(freq=freq_hz, itd=itd_ms))

    def eventFilter(self, obj, event):
        # Capture mouse events from the plot viewport so the plot doesn't pan.
        if obj is self.plot_widget.viewport():
            et = event.type()

            if et == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._dragging = True
                    self._last_drag_update_t = 0.0
                    # While dragging, force loop so parameters change continuously.
                    self._loop_before_drag = bool(self.module.play_loop)
                    self.module.play_loop = True
                    # Map viewport pos -> scene pos
                    scene_pos = self.plot_widget.mapToScene(event.pos())
                    self._handle_plot_point(scene_pos=scene_pos, force=True)
                    return True

            if et == QEvent.Type.MouseMove:
                if self._dragging:
                    now = time.monotonic()
                    if (now - self._last_drag_update_t) >= self._drag_update_interval_s:
                        self._last_drag_update_t = now
                        scene_pos = self.plot_widget.mapToScene(event.pos())
                        self._handle_plot_point(scene_pos=scene_pos, force=False)
                    return True

            if et == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton and self._dragging:
                    self._dragging = False
                    # Restore previous loop setting.
                    if self._loop_before_drag is not None:
                        self.module.play_loop = bool(self._loop_before_drag)
                        self._loop_before_drag = None
                    return True

        return super().eventFilter(obj, event)

    def update_display(self):
        if not self.module.is_running: return

        self.module.process_buffer()

        data = self.module.neural_map
        if data is None: return

        # Apply Glow
        if self.module.glow_sigma > 0:
            data = gaussian_filter(data, sigma=self.module.glow_sigma)

        # Display
        # Y axis is frequency, X axis is ITD
        # pyqtgraph ImageItem expects (X, Y)
        # Our neural_map is (Freqs, ITD) -> (Y, X)
        self.img_item.setImage(data.T, autoLevels=False)

        # Auto-adjust levels for better visibility
        # We want to see the peaks clearly
        mx = float(np.max(data)) if data.size else 0.0
        self.img_item.setLevels([0, mx * 0.8 + 0.1])

        # Set Rect
        # x: -max_itd to +max_itd
        # y: log scale or linear? Let's start linear for simplicity
        itd = self.module.max_itd_ms
        fmin = float(self.module.freq_min)
        fmax = float(self.module.freq_max)
        self.img_item.setRect(pg.QtCore.QRectF(-itd, fmin, 2*itd, max(1.0, fmax - fmin)))
        self.plot_widget.setXRange(-itd, itd)
        self.plot_widget.setYRange(fmin, fmax)
