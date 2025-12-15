import argparse
import time
import numpy as np
import pyqtgraph as pg
from scipy import signal
import threading
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QProgressBar, QGroupBox, QGridLayout, QFrame, QCheckBox)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class LufsMeter(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.sample_rate = 48000 # Default, updated on start

        # Use a deep floor so very low-noise devices don't collapse to -INF.
        # This affects only dBFS-related meters (RMS/Peak and C-weighted variants).
        self._db_floor = -200.0
        
        # Filter states (per-channel for strict BS.1770 energy summation)
        self.zi_shelf_l = None
        self.zi_shelf_r = None
        self.zi_hp_l = None
        self.zi_hp_r = None

        # C-weighting (for SPL calibration compatibility)
        self.c_b = None
        self.c_a = None
        self.c_zi_l = None
        self.c_zi_r = None
        
        # Buffers / windows
        self.momentary_window = 0.4 # 400ms
        self.short_term_window = 3.0 # 3s
        self.buffer_size_m = 0
        self.buffer_size_s = 0

        # Ring-buffers of per-sample energy (Lk^2 + Rk^2)
        self._p_ring_m = None
        self._p_ring_s = None
        self._p_pos_m = 0
        self._p_pos_s = 0
        self._p_filled_m = 0
        self._p_filled_s = 0
        self._p_sum_m = 0.0
        self._p_sum_s = 0.0
        
        # Values
        self.momentary_lufs = -100.0
        self.short_term_lufs = -100.0
        self.integrated_lufs = -100.0

        # Integrated loudness (BS.1770-style gating, streaming)
        self._i_started_at = None
        self._i_sample_count = 0
        self._i_block_step = 0
        self._i_since_last_block = 0
        self._i_block_ms = []  # per-block mean-square (Lk^2+Rk^2), 400 ms blocks, 75% overlap
        self._i_abs_gate_ms = float(10 ** ((-70.0 + 0.691) / 10.0))
        self._i_dirty = False
        self._i_lock = threading.Lock()
        
        # Stereo RMS & Peak
        self.rms_l = self._db_floor
        self.rms_r = self._db_floor
        self.peak_l = self._db_floor
        self.peak_r = self._db_floor
        self.peak_hold_l = self._db_floor
        self.peak_hold_r = self._db_floor
        self.crest_l = 0.0
        self.crest_r = 0.0

        # C-weighted RMS/Peak (dBFS_C) for SPL display
        self.rms_c_l = self._db_floor
        self.rms_c_r = self._db_floor
        self.peak_c_l = self._db_floor
        self.peak_c_r = self._db_floor
        self.peak_hold_c_l = self._db_floor
        self.peak_hold_c_r = self._db_floor
        self.crest_c_l = 0.0
        self.crest_c_r = 0.0
        
        self.callback_id = None

    @property
    def name(self) -> str:
        return "LUFS & Level Meter"

    @property
    def description(self) -> str:
        return "Real-time Loudness (LUFS) and Stereo Level Meter"

    def run(self, args: argparse.Namespace):
        print("LUFS Meter running from CLI (not fully implemented)")

    def get_widget(self):
        return LufsMeterWidget(self)

    def _init_filters(self):
        # K-weighting filter coefficients (ITU-R BS.1770-4)
        # Keep float32 to avoid per-block float64 upcasts in the audio callback.
        self.b0_shelf = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float32)
        self.a0_shelf = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float32)
        self.b1_hp = np.array([1.0, -2.0, 1.0], dtype=np.float32)
        self.a1_hp = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float32)
        
        # Initial filter states (per-channel)
        zi_shelf = signal.lfilter_zi(self.b0_shelf, self.a0_shelf).astype(np.float32, copy=False)
        zi_hp = signal.lfilter_zi(self.b1_hp, self.a1_hp).astype(np.float32, copy=False)
        self.zi_shelf_l = zi_shelf.copy()
        self.zi_shelf_r = zi_shelf.copy()
        self.zi_hp_l = zi_hp.copy()
        self.zi_hp_r = zi_hp.copy()

        # C-weighting (IEC 61672) for SPL calibration compatibility
        self.c_b, self.c_a = self._design_c_weighting(self.sample_rate)
        zi = signal.lfilter_zi(self.c_b, self.c_a).astype(np.float32, copy=False)
        self.c_zi_l = zi.copy()
        self.c_zi_r = zi.copy()

    def _design_c_weighting(self, sr: float):
        """Design digital C-weighting filter (IEC 61672) for sample rate sr.

        Matches the SPL calibration wizard's filter so that measured dBFS_C
        is compatible with the stored SPL offset.
        """
        sr = float(sr)
        if sr <= 0:
            raise ValueError("Invalid sample rate")

        w1 = 2 * np.pi * 20.6
        w2 = 2 * np.pi * 12194.0

        zeros = np.array([0.0, 0.0])
        poles = np.array([-w1, -w1, -w2, -w2])
        gain = 1.0

        # Normalize to 0 dB at 1 kHz
        s = 1j * 2 * np.pi * 1000.0
        h = gain * (s**2) / ((s + w1) ** 2 * (s + w2) ** 2)
        gain = 1.0 / np.abs(h)

        z, p, k = signal.bilinear_zpk(zeros, poles, gain, fs=sr)
        b, a = signal.zpk2tf(z, p, k)
        # Use float32 for callback efficiency; response accuracy remains sufficient for metering.
        return b.astype(np.float32), a.astype(np.float32)

    def reset_peaks(self):
        self.peak_hold_l = self._db_floor
        self.peak_hold_r = self._db_floor
        self.peak_hold_c_l = self._db_floor
        self.peak_hold_c_r = self._db_floor

    def reset_integration(self):
        self.integrated_lufs = -100.0
        self._i_started_at = time.perf_counter()
        self._i_sample_count = 0
        self._i_since_last_block = 0
        # 400 ms block with 75% overlap -> 100 ms step
        self._i_block_step = int(round(0.1 * float(self.sample_rate)))
        with self._i_lock:
            self._i_block_ms = []
            self._i_dirty = False

    def update_integrated_lufs_if_dirty(self):
        """Recompute gated integrated loudness (BS.1770) when new blocks arrive.

        Intended to be called from the GUI thread to keep the audio callback lean.
        """
        with self._i_lock:
            if not self._i_dirty:
                return
            blocks = np.asarray(self._i_block_ms, dtype=np.float64)
            self._i_dirty = False

        if blocks.size == 0:
            self.integrated_lufs = -100.0
            return

        mean_ms_ungated = float(blocks.mean())
        l_ungated = self._to_lufs(mean_ms_ungated)
        rel_gate_l = l_ungated - 10.0
        rel_gate_ms = float(10 ** ((rel_gate_l + 0.691) / 10.0))
        gate_ms = max(self._i_abs_gate_ms, rel_gate_ms)

        gated = blocks[blocks > gate_ms]
        if gated.size == 0:
            self.integrated_lufs = -100.0
            return

        self.integrated_lufs = self._to_lufs(float(gated.mean()))

    def reset_all_stats(self):
        self.reset_peaks()
        self.reset_integration()

    def start_meter(self):
        if self.is_running:
            return

        self.is_running = True
        self.sample_rate = self.audio_engine.sample_rate
        self._init_filters()

        # Reset session accumulators
        self.reset_integration()
        
        # Initialize buffers (ring of per-sample power)
        self.buffer_size_m = int(round(self.momentary_window * float(self.sample_rate)))
        self.buffer_size_s = int(round(self.short_term_window * float(self.sample_rate)))
        self._p_ring_m = np.zeros(self.buffer_size_m, dtype=np.float32)
        self._p_ring_s = np.zeros(self.buffer_size_s, dtype=np.float32)
        self._p_pos_m = 0
        self._p_pos_s = 0
        self._p_filled_m = 0
        self._p_filled_s = 0
        self._p_sum_m = 0.0
        self._p_sum_s = 0.0

        abs_gate_ms = self._i_abs_gate_ms

        def ring_update_power(ring: np.ndarray, pos: int, filled: int, sum_p: float, p_chunk: np.ndarray):
            """Write p_chunk into ring (overwrite) and update running sum in O(len(p_chunk))."""
            n = int(ring.shape[0])
            m = int(p_chunk.shape[0])
            if m <= 0:
                return pos, filled, sum_p

            end = pos + m
            if end <= n:
                old = ring[pos:end]
                sum_p -= float(np.sum(old, dtype=np.float64))
                ring[pos:end] = p_chunk
                sum_p += float(np.sum(p_chunk, dtype=np.float64))
            else:
                first = n - pos
                old1 = ring[pos:]
                old2 = ring[:(end - n)]
                sum_p -= float(np.sum(old1, dtype=np.float64))
                sum_p -= float(np.sum(old2, dtype=np.float64))

                ring[pos:] = p_chunk[:first]
                ring[:(end - n)] = p_chunk[first:]
                sum_p += float(np.sum(p_chunk[:first], dtype=np.float64))
                sum_p += float(np.sum(p_chunk[first:], dtype=np.float64))

            pos = end % n
            filled = min(n, filled + m)
            return pos, filled, sum_p
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
            # --- Stereo RMS & Peak Calculation ---
            # indata is (frames, channels)
            num_channels = indata.shape[1]
            
            if num_channels >= 2:
                l_channel = indata[:, 0]
                r_channel = indata[:, 1]
            elif num_channels == 1:
                l_channel = indata[:, 0]
                r_channel = indata[:, 0] # Duplicate mono
            else:
                # Should not happen if stream is active
                l_channel = np.zeros(frames)
                r_channel = np.zeros(frames)
            
            # RMS (Instantaneous for this block)
            # Use dot for low-allocation sumsq
            if frames > 0:
                rms_l_linear = float(np.sqrt(np.dot(l_channel, l_channel) / float(frames)))
                rms_r_linear = float(np.sqrt(np.dot(r_channel, r_channel) / float(frames)))
            else:
                rms_l_linear = 0.0
                rms_r_linear = 0.0
            self.rms_l = self._to_db(rms_l_linear)
            self.rms_r = self._to_db(rms_r_linear)

            # --- C-weighted RMS/Peak (for SPL calibration) ---
            # Calibration wizard measures dBFS_C using a C-weighting IIR filter and RMS.
            # We compute the same here so that SPL = dBFS_C + offset is consistent.
            if self.c_b is not None and self.c_a is not None and self.c_zi_l is not None and self.c_zi_r is not None:
                # Keep float32 throughout; input stream is float32.
                l_c, self.c_zi_l = signal.lfilter(self.c_b, self.c_a, l_channel, zi=self.c_zi_l)
                r_c, self.c_zi_r = signal.lfilter(self.c_b, self.c_a, r_channel, zi=self.c_zi_r)

                if frames > 0:
                    rms_c_l_linear = float(np.sqrt(np.dot(l_c, l_c) / float(frames) + 1e-24))
                    rms_c_r_linear = float(np.sqrt(np.dot(r_c, r_c) / float(frames) + 1e-24))
                else:
                    rms_c_l_linear = 0.0
                    rms_c_r_linear = 0.0
                self.rms_c_l = self._to_db(rms_c_l_linear)
                self.rms_c_r = self._to_db(rms_c_r_linear)

                peak_c_l_linear = float(np.max(np.abs(l_c))) if len(l_c) else 0.0
                peak_c_r_linear = float(np.max(np.abs(r_c))) if len(r_c) else 0.0
                self.peak_c_l = self._to_db(peak_c_l_linear)
                self.peak_c_r = self._to_db(peak_c_r_linear)

                self.peak_hold_c_l = max(self.peak_hold_c_l, self.peak_c_l)
                self.peak_hold_c_r = max(self.peak_hold_c_r, self.peak_c_r)

                self.crest_c_l = self.peak_c_l - self.rms_c_l
                self.crest_c_r = self.peak_c_r - self.rms_c_r
            
            # Peak (Instantaneous)
            peak_l_linear = np.max(np.abs(l_channel))
            peak_r_linear = np.max(np.abs(r_channel))
            self.peak_l = self._to_db(peak_l_linear)
            self.peak_r = self._to_db(peak_r_linear)
            
            # Peak Hold Update
            self.peak_hold_l = max(self.peak_hold_l, self.peak_l)
            self.peak_hold_r = max(self.peak_hold_r, self.peak_r)
            
            # Crest Factor (Peak dB - RMS dB)
            # Ensure we don't subtract -100 from -100 resulting in 0 if both are silence, which is fine.
            # But if RMS is -100 and Peak is -90, CF is 10.
            self.crest_l = self.peak_l - self.rms_l
            self.crest_r = self.peak_r - self.rms_r
            
            # --- LUFS Calculation (Strict stereo: per-channel K-weighting, sum energies) ---
            # For true mono input, avoid double-counting energy (would read +3 dB too hot).
            if num_channels == 1:
                l_lufs = l_channel
                r_lufs = np.zeros_like(l_channel)
            else:
                l_lufs = l_channel
                r_lufs = r_channel

            # Apply K-weighting per channel
            l_shelf, self.zi_shelf_l = signal.lfilter(self.b0_shelf, self.a0_shelf, l_lufs, zi=self.zi_shelf_l)
            r_shelf, self.zi_shelf_r = signal.lfilter(self.b0_shelf, self.a0_shelf, r_lufs, zi=self.zi_shelf_r)
            l_k, self.zi_hp_l = signal.lfilter(self.b1_hp, self.a1_hp, l_shelf, zi=self.zi_hp_l)
            r_k, self.zi_hp_r = signal.lfilter(self.b1_hp, self.a1_hp, r_shelf, zi=self.zi_hp_r)

            # Per-sample power (avoid rolling full windows)
            p_chunk = (l_k * l_k) + (r_k * r_k)
            self._p_pos_m, self._p_filled_m, self._p_sum_m = ring_update_power(
                self._p_ring_m, self._p_pos_m, self._p_filled_m, self._p_sum_m, p_chunk
            )
            self._p_pos_s, self._p_filled_s, self._p_sum_s = ring_update_power(
                self._p_ring_s, self._p_pos_s, self._p_filled_s, self._p_sum_s, p_chunk
            )

            # Track session time
            self._i_sample_count += int(frames)

            # Momentary (400 ms) and Short-term (3 s)
            n_m = float(max(1, self._p_filled_m))
            n_s = float(max(1, self._p_filled_s))
            ms_m = float(self._p_sum_m / n_m) if self._p_filled_m > 0 else 0.0
            ms_s = float(self._p_sum_s / n_s) if self._p_filled_s > 0 else 0.0
            self.momentary_lufs = self._to_lufs(ms_m)
            self.short_term_lufs = self._to_lufs(ms_s)

            # Integrated loudness with gating (400 ms blocks, 75% overlap)
            # Start once we have a full 400 ms window.
            if self._p_filled_m >= self.buffer_size_m and self._i_block_step > 0:
                self._i_since_last_block += int(frames)
                while self._i_since_last_block >= self._i_block_step:
                    self._i_since_last_block -= self._i_block_step
                    block_ms = float(self._p_sum_m / float(self.buffer_size_m))
                    if block_ms > abs_gate_ms:
                        with self._i_lock:
                            self._i_block_ms.append(block_ms)
                            self._i_dirty = True
            else:
                self._i_since_last_block += int(frames)
            
            # No output (meter is analysis-only). AudioEngine provides a fresh zeroed buffer.

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_meter(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def _to_db(self, value):
        if value <= 0 or not np.isfinite(value):
            return float(self._db_floor)
        if value <= 1e-20:
            return float(self._db_floor)
        return 20 * np.log10(value)

    def _to_lufs(self, mean_square):
        if mean_square <= 1e-10:
            return -100.0
        return -0.691 + 10 * np.log10(mean_square)

    def get_integrated_seconds(self) -> float:
        if self._i_sample_count <= 0 or self.sample_rate <= 0:
            return 0.0
        return self._i_sample_count / float(self.sample_rate)

class LufsMeterWidget(QWidget):
    def __init__(self, module: LufsMeter):
        super().__init__()
        self.module = module

        # Optional SPL display mode (requires SPL calibration)
        self._show_spl = False
        
        # History for plotting
        self.history_size = 400 # 20s at 50ms interval
        self.m_history = np.full(self.history_size, -100.0)
        self.s_history = np.full(self.history_size, -100.0)

        # Session stats (since last reset)
        self._reset_session_stats()
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.setInterval(50) # 20 FPS

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        self.toggle_btn = QPushButton(tr("Start Metering"))
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        controls_layout.addWidget(self.toggle_btn)

        self.spl_check = QCheckBox(tr("Show SPL"))
        self.spl_check.toggled.connect(self.on_spl_toggled)
        controls_layout.addWidget(self.spl_check)
        
        self.reset_btn = QPushButton(tr("Reset Peaks"))
        self.reset_btn.clicked.connect(self.module.reset_peaks)
        controls_layout.addWidget(self.reset_btn)

        self.reset_stats_btn = QPushButton(tr("Reset Stats"))
        self.reset_stats_btn.clicked.connect(self.on_reset_stats)
        controls_layout.addWidget(self.reset_stats_btn)
        layout.addLayout(controls_layout)

        self._sync_spl_checkbox()
        
        # --- Meters Area ---
        meters_group = QGroupBox(tr("Levels"))
        grid = QGridLayout()
        
        # 1. Stereo RMS / Peak Meters
        # Left
        grid.addWidget(QLabel(tr("L")), 0, 0)
        self.l_bar = QProgressBar()
        self.l_bar.setRange(-120, 0) # Extended range for low-noise devices
        self.l_bar.setTextVisible(False)
        self.l_bar.setOrientation(Qt.Orientation.Vertical)
        self.l_bar.setFixedSize(30, 200)
        grid.addWidget(self.l_bar, 0, 1, 2, 1)
        
        self.l_val_label = QLabel(tr("-INF"))
        grid.addWidget(self.l_val_label, 2, 1, Qt.AlignmentFlag.AlignHCenter)
        
        self.l_peak_label = QLabel(tr("Pk: -INF"))
        self.l_peak_label.setStyleSheet("color: red; font-size: 10px;")
        grid.addWidget(self.l_peak_label, 3, 1, Qt.AlignmentFlag.AlignHCenter)
        
        self.l_cf_label = QLabel(tr("CF: 0.0"))
        self.l_cf_label.setStyleSheet("color: cyan; font-size: 10px;")
        grid.addWidget(self.l_cf_label, 4, 1, Qt.AlignmentFlag.AlignHCenter)

        # Right
        grid.addWidget(QLabel(tr("R")), 0, 2)
        self.r_bar = QProgressBar()
        self.r_bar.setRange(-120, 0)
        self.r_bar.setTextVisible(False)
        self.r_bar.setOrientation(Qt.Orientation.Vertical)
        self.r_bar.setFixedSize(30, 200)
        grid.addWidget(self.r_bar, 0, 3, 2, 1)
        
        self.r_val_label = QLabel(tr("-INF"))
        grid.addWidget(self.r_val_label, 2, 3, Qt.AlignmentFlag.AlignHCenter)
        
        self.r_peak_label = QLabel(tr("Pk: -INF"))
        self.r_peak_label.setStyleSheet("color: red; font-size: 10px;")
        grid.addWidget(self.r_peak_label, 3, 3, Qt.AlignmentFlag.AlignHCenter)
        
        self.r_cf_label = QLabel(tr("CF: 0.0"))
        self.r_cf_label.setStyleSheet("color: cyan; font-size: 10px;")
        grid.addWidget(self.r_cf_label, 4, 3, Qt.AlignmentFlag.AlignHCenter)
        
        # Spacer
        grid.setColumnMinimumWidth(4, 30)
        
        # 2. LUFS Meters
        # Momentary
        grid.addWidget(QLabel(tr("M")), 0, 5)
        self.m_bar = QProgressBar()
        self.m_bar.setRange(-120, 0)
        self.m_bar.setTextVisible(False)
        self.m_bar.setOrientation(Qt.Orientation.Vertical)
        self.m_bar.setFixedSize(30, 200)
        grid.addWidget(self.m_bar, 0, 6, 2, 1)
        
        self.m_val_label = QLabel(tr("-INF"))
        grid.addWidget(self.m_val_label, 2, 6, Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(QLabel(tr("LUFS(M)")), 3, 6, Qt.AlignmentFlag.AlignHCenter)

        # Short-term
        grid.addWidget(QLabel(tr("S")), 0, 7)
        self.s_bar = QProgressBar()
        self.s_bar.setRange(-120, 0)
        self.s_bar.setTextVisible(False)
        self.s_bar.setOrientation(Qt.Orientation.Vertical)
        self.s_bar.setFixedSize(30, 200)
        grid.addWidget(self.s_bar, 0, 8, 2, 1)
        
        self.s_val_label = QLabel(tr("-INF"))
        grid.addWidget(self.s_val_label, 2, 8, Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(QLabel(tr("LUFS(S)")), 3, 8, Qt.AlignmentFlag.AlignHCenter)

        meters_group.setLayout(grid)
        layout.addWidget(meters_group)

        # --- Statistics Area ---
        stats_group = QGroupBox(tr("Statistics"))
        stats_grid = QGridLayout()

        # Headers
        stats_grid.addWidget(QLabel(tr("")), 0, 0)
        stats_grid.addWidget(QLabel(tr("Current")), 0, 1)
        stats_grid.addWidget(QLabel(tr("Min")), 0, 2)
        stats_grid.addWidget(QLabel(tr("Max")), 0, 3)
        stats_grid.addWidget(QLabel(tr("Avg")), 0, 4)

        # Momentary row
        stats_grid.addWidget(QLabel(tr("LUFS (M)")), 1, 0)
        self.stats_m_cur = QLabel(tr("-INF"))
        self.stats_m_min = QLabel(tr("-INF"))
        self.stats_m_max = QLabel(tr("-INF"))
        self.stats_m_avg = QLabel(tr("-INF"))
        stats_grid.addWidget(self.stats_m_cur, 1, 1)
        stats_grid.addWidget(self.stats_m_min, 1, 2)
        stats_grid.addWidget(self.stats_m_max, 1, 3)
        stats_grid.addWidget(self.stats_m_avg, 1, 4)

        # Short-term row
        stats_grid.addWidget(QLabel(tr("LUFS (S)")), 2, 0)
        self.stats_s_cur = QLabel(tr("-INF"))
        self.stats_s_min = QLabel(tr("-INF"))
        self.stats_s_max = QLabel(tr("-INF"))
        self.stats_s_avg = QLabel(tr("-INF"))
        stats_grid.addWidget(self.stats_s_cur, 2, 1)
        stats_grid.addWidget(self.stats_s_min, 2, 2)
        stats_grid.addWidget(self.stats_s_max, 2, 3)
        stats_grid.addWidget(self.stats_s_avg, 2, 4)

        # Integrated + duration row
        stats_grid.addWidget(QLabel(tr("LUFS (I)")), 3, 0)
        self.stats_i_val = QLabel(tr("-INF"))
        stats_grid.addWidget(self.stats_i_val, 3, 1)
        stats_grid.addWidget(QLabel(tr("Time")), 3, 2)
        self.stats_i_time = QLabel(tr("0.0 s"))
        stats_grid.addWidget(self.stats_i_time, 3, 3, 1, 2)

        stats_group.setLayout(stats_grid)
        layout.addWidget(stats_group)
        
        # --- Time Series Plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', tr('LUFS'), units='dB')
        self.plot_widget.setLabel('bottom', tr('Time'), units='s')
        self.plot_widget.setYRange(-60, 0)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('k')
        self.plot_widget.setFixedHeight(200)
        
        # Curves
        self.m_curve = self.plot_widget.plot(pen=pg.mkPen('c', width=1), name=tr('Momentary')) # Cyan
        self.s_curve = self.plot_widget.plot(pen=pg.mkPen('y', width=2), name=tr('Short-Term')) # Yellow
        
        # Target Line
        self.target_line = pg.InfiniteLine(angle=0, pos=-23, pen=pg.mkPen('g', style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.target_line)

        # Target band (-23 LUFS ±2) for quick visual alignment
        self.target_band = pg.LinearRegionItem(values=[-25, -21], orientation=pg.LinearRegionItem.Horizontal)
        self.target_band.setBrush(pg.mkBrush(0, 255, 0, 35))
        self.target_band.setMovable(False)
        self.target_band.setZValue(-10)
        self.plot_widget.addItem(self.target_band)
        
        layout.addWidget(self.plot_widget)
        
        layout.addStretch()
        self.setLayout(layout)

    def _reset_session_stats(self):
        self._m_min = None
        self._m_max = None
        self._m_sum = 0.0
        self._m_n = 0

        self._s_min = None
        self._s_max = None
        self._s_sum = 0.0
        self._s_n = 0

    def on_reset_stats(self):
        self._reset_session_stats()
        self.m_history[:] = -100.0
        self.s_history[:] = -100.0
        self.module.reset_all_stats()

    def on_toggle(self, checked):
        if checked:
            self.module.start_meter()
            self.timer.start()
            self.toggle_btn.setText(tr("Stop Metering"))
        else:
            self.module.stop_meter()
            self.timer.stop()
            self.toggle_btn.setText(tr("Start Metering"))

    def _get_spl_offset_db(self):
        try:
            return self.module.audio_engine.calibration.get_spl_offset_db()
        except Exception:
            return None

    def _sync_spl_checkbox(self):
        has_cal = self._get_spl_offset_db() is not None
        self.spl_check.setEnabled(has_cal)
        if not has_cal and self._show_spl:
            self._show_spl = False
            self.spl_check.blockSignals(True)
            self.spl_check.setChecked(False)
            self.spl_check.blockSignals(False)

    def on_spl_toggled(self, checked: bool):
        if checked and self._get_spl_offset_db() is None:
            self._show_spl = False
            self.spl_check.blockSignals(True)
            self.spl_check.setChecked(False)
            self.spl_check.blockSignals(False)
            return
        self._show_spl = bool(checked)

    def update_display(self):
        if not self.module.is_running:
            return

        # Keep integrated LUFS computation off the audio callback.
        self.module.update_integrated_lufs_if_dirty()

        # Allow calibration to appear/disappear without recreating the widget
        self._sync_spl_checkbox()
            
        # Update RMS/Peak
        if self._show_spl and self._get_spl_offset_db() is not None:
            # Use C-weighted dBFS values (compatible with SPL calibration wizard)
            rms_l = self.module.rms_c_l
            rms_r = self.module.rms_c_r
            peak_hold_l = self.module.peak_hold_c_l
            peak_hold_r = self.module.peak_hold_c_r
            crest_l = self.module.crest_c_l
            crest_r = self.module.crest_c_r

            spl_offset = float(self._get_spl_offset_db())

            def to_spl(dbfs_c: float) -> float:
                # Keep true silence as -INF, but allow very low-noise readings.
                if dbfs_c <= -199.9:
                    return -200.0
                return dbfs_c + spl_offset

            disp_rms_l = to_spl(rms_l)
            disp_rms_r = to_spl(rms_r)
            disp_peak_hold_l = to_spl(peak_hold_l)
            disp_peak_hold_r = to_spl(peak_hold_r)
            disp_unit = "dB SPL"
        else:
            rms_l = self.module.rms_l
            rms_r = self.module.rms_r
            peak_hold_l = self.module.peak_hold_l
            peak_hold_r = self.module.peak_hold_r
            crest_l = self.module.crest_l
            crest_r = self.module.crest_r

            disp_rms_l = rms_l
            disp_rms_r = rms_r
            disp_peak_hold_l = peak_hold_l
            disp_peak_hold_r = peak_hold_r
            disp_unit = "dBFS"
        
        l_min = int(self.l_bar.minimum())
        l_max = int(self.l_bar.maximum())
        r_min = int(self.r_bar.minimum())
        r_max = int(self.r_bar.maximum())
        self.l_bar.setValue(int(max(l_min, min(l_max, rms_l))))
        self.r_bar.setValue(int(max(r_min, min(r_max, rms_r))))
        
        self.l_val_label.setText(tr("{0} {1}").format(self._format_db(disp_rms_l), disp_unit))
        self.r_val_label.setText(tr("{0} {1}").format(self._format_db(disp_rms_r), disp_unit))
        
        self.l_peak_label.setText(tr("Pk: {0} {1}").format(self._format_db(disp_peak_hold_l), disp_unit))
        self.r_peak_label.setText(tr("Pk: {0} {1}").format(self._format_db(disp_peak_hold_r), disp_unit))
        
        self.l_cf_label.setText(tr("CF: {0:.1f}").format(crest_l))
        self.r_cf_label.setText(tr("CF: {0:.1f}").format(crest_r))
        
        # Update LUFS
        m_lufs = self.module.momentary_lufs
        s_lufs = self.module.short_term_lufs
        
        m_min = int(self.m_bar.minimum())
        m_max = int(self.m_bar.maximum())
        s_min = int(self.s_bar.minimum())
        s_max = int(self.s_bar.maximum())
        self.m_bar.setValue(int(max(m_min, min(m_max, m_lufs))))
        self.s_bar.setValue(int(max(s_min, min(s_max, s_lufs))))
        
        self.m_val_label.setText(tr("{0:.1f}").format(m_lufs))
        self.s_val_label.setText(tr("{0:.1f}").format(s_lufs))

        # Update session stats
        self._update_session_stats(m_lufs, s_lufs)
        self._update_stats_labels(m_lufs, s_lufs)
        
        # Color coding
        self._set_bar_color(self.l_bar, rms_l)
        self._set_bar_color(self.r_bar, rms_r)
        self._set_lufs_bar_color(self.m_bar, m_lufs)
        self._set_lufs_bar_color(self.s_bar, s_lufs)
        
        # Update Plot
        self.m_history = np.roll(self.m_history, -1)
        self.m_history[-1] = m_lufs
        
        self.s_history = np.roll(self.s_history, -1)
        self.s_history[-1] = s_lufs
        
        # X axis (time)
        # 0 to -20s
        x = np.linspace(-self.history_size * 0.05, 0, self.history_size)
        
        self.m_curve.setData(x, self.m_history)
        self.s_curve.setData(x, self.s_history)

    def _format_db(self, value: float) -> str:
        if value <= -199.9:
            return tr("-INF")
        return tr("{0:.1f}").format(value)

    def _format_seconds(self, seconds: float) -> str:
        if seconds < 0:
            seconds = 0.0
        if seconds < 60:
            return tr("{0:.1f} s").format(seconds)
        minutes = int(seconds // 60)
        rem = seconds - (minutes * 60)
        return tr("{0:d} m {1:.0f} s").format(minutes, rem)

    def _update_session_stats(self, m_lufs: float, s_lufs: float):
        # Momentary
        if m_lufs > -99.9:
            self._m_min = m_lufs if self._m_min is None else min(self._m_min, m_lufs)
            self._m_max = m_lufs if self._m_max is None else max(self._m_max, m_lufs)
            self._m_sum += float(m_lufs)
            self._m_n += 1

        # Short-term
        if s_lufs > -99.9:
            self._s_min = s_lufs if self._s_min is None else min(self._s_min, s_lufs)
            self._s_max = s_lufs if self._s_max is None else max(self._s_max, s_lufs)
            self._s_sum += float(s_lufs)
            self._s_n += 1

    def _update_stats_labels(self, m_lufs: float, s_lufs: float):
        self.stats_m_cur.setText(self._format_db(m_lufs))
        self.stats_s_cur.setText(self._format_db(s_lufs))

        self.stats_m_min.setText(self._format_db(self._m_min if self._m_min is not None else -100.0))
        self.stats_m_max.setText(self._format_db(self._m_max if self._m_max is not None else -100.0))
        m_avg = (self._m_sum / self._m_n) if self._m_n > 0 else -100.0
        self.stats_m_avg.setText(self._format_db(m_avg))

        self.stats_s_min.setText(self._format_db(self._s_min if self._s_min is not None else -100.0))
        self.stats_s_max.setText(self._format_db(self._s_max if self._s_max is not None else -100.0))
        s_avg = (self._s_sum / self._s_n) if self._s_n > 0 else -100.0
        self.stats_s_avg.setText(self._format_db(s_avg))

        self.stats_i_val.setText(self._format_db(self.module.integrated_lufs))
        self.stats_i_time.setText(self._format_seconds(self.module.get_integrated_seconds()))

    def _set_bar_color(self, bar, val):
        # Standard dBFS colors
        if val > -3:
            color = "red"
        elif val > -12:
            color = "#aaaa00" # Yellow
        else:
            color = "#00ff00" # Green
        bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

    def _set_lufs_bar_color(self, bar, lufs):
        # EBU R128 target is -23 LUFS
        if lufs > -21:
            color = "red"
        elif lufs > -25:
            color = "#00ff00" # Green (Target)
        else:
            color = "#aaaa00" # Yellow/Orange
        bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
