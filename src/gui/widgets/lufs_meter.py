import argparse
import time
import numpy as np
import pyqtgraph as pg
from scipy import signal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QProgressBar, QGroupBox, QGridLayout, QFrame)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class LufsMeter(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.sample_rate = 48000 # Default, updated on start
        
        # Filter states
        self.zi_shelf = None
        self.zi_hp = None
        
        # Buffers
        self.momentary_window = 0.4 # 400ms
        self.short_term_window = 3.0 # 3s
        self.buffer_size_m = 0
        self.buffer_size_s = 0
        self.audio_buffer = np.array([]) # K-weighted buffer for LUFS
        
        # Values
        self.momentary_lufs = -100.0
        self.short_term_lufs = -100.0
        self.integrated_lufs = -100.0

        # Integrated loudness accumulation (simple running mean square of K-weighted samples)
        self._i_sumsq = 0.0
        self._i_count = 0
        self._i_started_at = None
        
        # Stereo RMS & Peak
        self.rms_l = -100.0
        self.rms_r = -100.0
        self.peak_l = -100.0
        self.peak_r = -100.0
        self.peak_hold_l = -100.0
        self.peak_hold_r = -100.0
        self.crest_l = 0.0
        self.crest_r = 0.0
        
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
        self.b0_shelf = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
        self.a0_shelf = np.array([1.0, -1.69065929318241, 0.73248077421585])
        self.b1_hp = np.array([1.0, -2.0, 1.0])
        self.a1_hp = np.array([1.0, -1.99004745483398, 0.99007225036621])
        
        # Initial filter states
        self.zi_shelf = signal.lfilter_zi(self.b0_shelf, self.a0_shelf)
        self.zi_hp = signal.lfilter_zi(self.b1_hp, self.a1_hp)

    def reset_peaks(self):
        self.peak_hold_l = -100.0
        self.peak_hold_r = -100.0

    def reset_integration(self):
        self._i_sumsq = 0.0
        self._i_count = 0
        self.integrated_lufs = -100.0
        self._i_started_at = time.perf_counter()

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
        
        # Initialize buffers
        self.buffer_size_m = int(self.momentary_window * self.sample_rate)
        self.buffer_size_s = int(self.short_term_window * self.sample_rate)
        self.audio_buffer = np.zeros(self.buffer_size_s)
        
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
            rms_l_linear = np.sqrt(np.mean(l_channel**2))
            rms_r_linear = np.sqrt(np.mean(r_channel**2))
            self.rms_l = self._to_db(rms_l_linear)
            self.rms_r = self._to_db(rms_r_linear)
            
            # Peak (Instantaneous)
            peak_l_linear = np.max(np.abs(l_channel))
            peak_r_linear = np.max(np.abs(r_channel))
            self.peak_l = self._to_db(peak_l_linear)
            self.peak_r = self._to_db(peak_r_linear)
            
            # Peak Hold Update
            self.peak_hold_l = max(self.peak_hold_l, self.peak_l)
            self.peak_hold_r = max(self.peak_hold_r, self.peak_r)
            
            # Crest Factor (Peak dB - RMS dB)
            
            # Crest Factor (Peak dB - RMS dB)
            # Ensure we don't subtract -100 from -100 resulting in 0 if both are silence, which is fine.
            # But if RMS is -100 and Peak is -90, CF is 10.
            self.crest_l = self.peak_l - self.rms_l
            self.crest_r = self.peak_r - self.rms_r
            
            # --- LUFS Calculation (Mono K-weighted) ---
            # Average channels for mono K-weighting input (Simplified)
            mono_input = (l_channel + r_channel) / 2
            
            # Apply K-weighting
            filtered_shelf, self.zi_shelf = signal.lfilter(self.b0_shelf, self.a0_shelf, mono_input, zi=self.zi_shelf)
            k_weighted, self.zi_hp = signal.lfilter(self.b1_hp, self.a1_hp, filtered_shelf, zi=self.zi_hp)
            
            # Update buffer
            self.audio_buffer = np.roll(self.audio_buffer, -len(k_weighted))
            self.audio_buffer[-len(k_weighted):] = k_weighted

            # Integrated (simple running average over all K-weighted samples)
            # Note: This is not full BS.1770 gating; it is a useful session statistic.
            self._i_sumsq += float(np.sum(k_weighted**2))
            self._i_count += int(len(k_weighted))
            if self._i_count > 0:
                self.integrated_lufs = self._to_lufs(self._i_sumsq / self._i_count)
            
            # Momentary (last 400ms)
            m_data = self.audio_buffer[-self.buffer_size_m:]
            ms_m = np.mean(m_data**2)
            self.momentary_lufs = self._to_lufs(ms_m)
            
            # Short-term (last 3s)
            s_data = self.audio_buffer
            ms_s = np.mean(s_data**2)
            self.short_term_lufs = self._to_lufs(ms_s)
            
            outdata.fill(0)

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_meter(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def _to_db(self, value):
        if value <= 1e-10:
            return -100.0
        return 20 * np.log10(value)

    def _to_lufs(self, mean_square):
        if mean_square <= 1e-10:
            return -100.0
        return -0.691 + 10 * np.log10(mean_square)

    def get_integrated_seconds(self) -> float:
        if self._i_count <= 0 or self.sample_rate <= 0:
            return 0.0
        return self._i_count / float(self.sample_rate)

class LufsMeterWidget(QWidget):
    def __init__(self, module: LufsMeter):
        super().__init__()
        self.module = module
        
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
        
        self.reset_btn = QPushButton(tr("Reset Peaks"))
        self.reset_btn.clicked.connect(self.module.reset_peaks)
        controls_layout.addWidget(self.reset_btn)

        self.reset_stats_btn = QPushButton(tr("Reset Stats"))
        self.reset_stats_btn.clicked.connect(self.on_reset_stats)
        controls_layout.addWidget(self.reset_stats_btn)
        layout.addLayout(controls_layout)
        
        # --- Meters Area ---
        meters_group = QGroupBox(tr("Levels"))
        grid = QGridLayout()
        
        # 1. Stereo RMS / Peak Meters
        # Left
        grid.addWidget(QLabel(tr("L")), 0, 0)
        self.l_bar = QProgressBar()
        self.l_bar.setRange(-60, 0) # dBFS range
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
        self.r_bar.setRange(-60, 0)
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
        self.m_bar.setRange(-60, 0)
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
        self.s_bar.setRange(-60, 0)
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

    def update_display(self):
        if not self.module.is_running:
            return
            
        # Update RMS/Peak
        rms_l = self.module.rms_l
        rms_r = self.module.rms_r
        peak_hold_l = self.module.peak_hold_l
        peak_hold_r = self.module.peak_hold_r
        
        self.l_bar.setValue(int(max(-60, min(0, rms_l))))
        self.r_bar.setValue(int(max(-60, min(0, rms_r))))
        
        self.l_val_label.setText(tr("{0:.1f}").format(rms_l))
        self.r_val_label.setText(tr("{0:.1f}").format(rms_r))
        
        self.l_peak_label.setText(tr("Pk: {0:.1f}").format(peak_hold_l))
        self.r_peak_label.setText(tr("Pk: {0:.1f}").format(peak_hold_r))
        
        self.l_cf_label.setText(tr("CF: {0:.1f}").format(self.module.crest_l))
        self.r_cf_label.setText(tr("CF: {0:.1f}").format(self.module.crest_r))
        
        # Update LUFS
        m_lufs = self.module.momentary_lufs
        s_lufs = self.module.short_term_lufs
        
        self.m_bar.setValue(int(max(-60, min(0, m_lufs))))
        self.s_bar.setValue(int(max(-60, min(0, s_lufs))))
        
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
        if value <= -99.9:
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
