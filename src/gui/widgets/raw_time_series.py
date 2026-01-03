import argparse
import threading

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
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


class RawTimeSeries(MeasurementModule):
    """Continuous raw-ish time series monitor for long durations.

    Stores a decimated stream in a ring buffer to keep memory bounded.
    """

    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False

        # User controls
        self.time_span_s = 10.0  # 10 / 60 / 300
        self.vscale = 1.0
        self.paused = False

        # Storage settings
        self.max_span_s = 300.0
        self.storage_rate_hz = 2000.0  # decimated storage rate to keep memory reasonable

        # Ring buffer
        self._lock = threading.Lock()
        self._buf = np.zeros((int(self.max_span_s * self.storage_rate_hz) + 1, 2), dtype=np.float32)
        self._write_pos = 0
        self._filled = 0

        self.callback_id = None

    @property
    def name(self) -> str:
        return "Raw Time Series"

    @property
    def description(self) -> str:
        return "Long-span scrolling time series monitor."

    def run(self, args: argparse.Namespace):
        print("Raw Time Series running from CLI (not implemented)")

    def get_widget(self):
        return RawTimeSeriesWidget(self)

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        with self._lock:
            self._buf.fill(0)
            self._write_pos = 0
            self._filled = 0

        def callback(indata, outdata, frames, time, status):
            if indata is None:
                outdata.fill(0)
                return

            # Ensure 2ch logical input
            if indata.shape[1] >= 2:
                src = indata[:, :2]
            else:
                src = np.column_stack((indata[:, 0], indata[:, 0]))

            # Decimate for storage
            sr = float(self.audio_engine.sample_rate)
            target = float(self.storage_rate_hz)
            step = int(max(1, round(sr / target)))
            dec = src[::step].astype(np.float32, copy=False)

            if dec.size:
                with self._lock:
                    n = int(dec.shape[0])
                    buf_len = int(self._buf.shape[0])

                    end = self._write_pos + n
                    if end <= buf_len:
                        self._buf[self._write_pos:end, :] = dec
                    else:
                        first = buf_len - self._write_pos
                        self._buf[self._write_pos:buf_len, :] = dec[:first]
                        self._buf[0 : (n - first), :] = dec[first:]

                    self._write_pos = (self._write_pos + n) % buf_len
                    self._filled = min(buf_len, self._filled + n)

            outdata.fill(0)

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_analysis(self):
        if not self.is_running:
            return

        if self.callback_id is not None:
            self.audio_engine.unregister_callback(self.callback_id)
            self.callback_id = None

        self.is_running = False

    def _get_last_samples(self, n: int) -> np.ndarray:
        """Return last n samples (<= filled) as a contiguous array."""
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        with self._lock:
            filled = int(self._filled)
            if filled <= 0:
                return np.zeros((0, 2), dtype=np.float32)

            n = int(min(n, filled))
            buf_len = int(self._buf.shape[0])
            end = int(self._write_pos)
            start = (end - n) % buf_len

            if start < end:
                return self._buf[start:end, :].copy()

            part1 = self._buf[start:buf_len, :]
            part2 = self._buf[0:end, :]
            return np.vstack((part1, part2)).copy()

    def get_display_data(self, span_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (t, data) where t is seconds, data is (N,2)."""
        if span_s <= 0:
            return None

        n = int(round(float(span_s) * float(self.storage_rate_hz)))
        data = self._get_last_samples(n)
        if data.size == 0:
            return None

        # Time axis aligned to "now" at t=0
        t = (np.arange(-len(data) + 1, 1, dtype=np.float32) / float(self.storage_rate_hz))
        return t, data


class RawTimeSeriesWidget(QWidget):
    def __init__(self, module: RawTimeSeries):
        super().__init__()
        self.module = module

        self._last_frame = None  # (t, data)

        self._init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_plot)
        self.timer.setInterval(100)  # ~10 fps is enough for multi-minute scrolling

    def _init_ui(self):
        root = QHBoxLayout(self)

        # Left: plots
        left = QVBoxLayout()

        self.plots = pg.GraphicsLayoutWidget()

        self.plot_ch1 = self.plots.addPlot(row=0, col=0)
        self.plot_ch2 = self.plots.addPlot(row=1, col=0)

        self.plot_ch1.showGrid(x=True, y=True)
        self.plot_ch2.showGrid(x=True, y=True)

        self.plot_ch1.setLabel("left", tr("Amplitude"), units="V")
        self.plot_ch2.setLabel("left", tr("Amplitude"), units="V")
        self.plot_ch2.setLabel("bottom", tr("Time"), units="s")

        self.plot_ch1.setTitle(tr("CH1"))
        self.plot_ch2.setTitle(tr("CH2"))

        # Make CH1/CH2 plot areas visually equal.
        # CH2 has a bottom axis label, which otherwise shrinks its viewbox.
        bottom_axis_h = int(
            self.plot_ch2.getAxis("bottom").sizeHint(Qt.SizeHint.PreferredSize).height()
        )
        self.plot_ch1.getAxis("bottom").setHeight(bottom_axis_h)
        self.plot_ch2.getAxis("bottom").setHeight(bottom_axis_h)
        # Keep CH1 clean while reserving the same axis space.
        self.plot_ch1.getAxis("bottom").setStyle(showValues=False)

        # Same time axis and same vertical scale
        self.plot_ch2.setXLink(self.plot_ch1)
        self.plot_ch2.setYLink(self.plot_ch1)

        # Curves
        self.curve_ch1 = self.plot_ch1.plot(pen=pg.mkPen("#00ff00", width=1))
        self.curve_ch2 = self.plot_ch2.plot(pen=pg.mkPen("#ff0000", width=1))

        # Performance helpers
        for p in (self.plot_ch1, self.plot_ch2):
            p.setClipToView(True)
            p.setDownsampling(mode="peak")

        left.addWidget(self.plots)
        root.addLayout(left, stretch=1)

        # Right: controls
        right_widget = QWidget()
        right_widget.setFixedWidth(260)
        right = QVBoxLayout(right_widget)

        ctrl_group = QGroupBox(tr("General"))
        ctrl = QVBoxLayout(ctrl_group)

        # Start/Stop
        self.btn_start = QPushButton(tr("Start"))
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(self._on_start_toggled)
        ctrl.addWidget(self.btn_start)

        # Time span
        span_row = QHBoxLayout()
        span_row.addWidget(QLabel(tr("Time Span:") ))
        self.combo_span = QComboBox()
        self._span_options = {
            "10 s": 10.0,
            "60 s": 60.0,
            "300 s": 300.0,
        }
        self.combo_span.addItems(list(self._span_options.keys()))
        self.combo_span.setCurrentText("10 s")
        self.combo_span.currentTextChanged.connect(self._on_span_changed)
        span_row.addWidget(self.combo_span)
        ctrl.addLayout(span_row)

        # Vertical scale (oscilloscope-like)
        v_row = QHBoxLayout()
        v_row.addWidget(QLabel(tr("Scale:")))
        self.combo_v = QComboBox()
        self._vscale_options = {
            "0.01x": 0.01,
            "0.02x": 0.02,
            "0.05x": 0.05,
            "0.1x": 0.1,
            "0.2x": 0.2,
            "0.5x": 0.5,
            "1.0x": 1.0,
            "2.0x": 2.0,
            "5.0x": 5.0,
            "10.0x": 10.0,
            "20.0x": 20.0,
            "50.0x": 50.0,
            "100.0x": 100.0,
            "200.0x": 200.0,
            "500.0x": 500.0,
            "1000.0x": 1000.0,
        }
        self.combo_v.addItems(list(self._vscale_options.keys()))
        self.combo_v.setCurrentText("1.0x")
        self.combo_v.currentTextChanged.connect(self._on_vscale_changed)
        v_row.addWidget(self.combo_v)
        ctrl.addLayout(v_row)

        # Pause
        self.btn_pause = QPushButton(tr("Pause"))
        self.btn_pause.setCheckable(True)
        self.btn_pause.clicked.connect(self._on_pause_toggled)
        ctrl.addWidget(self.btn_pause)

        right.addWidget(ctrl_group)
        right.addStretch(1)

        root.addWidget(right_widget)

        # Apply defaults
        self._apply_view_ranges()

    def _on_start_toggled(self, checked: bool):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.btn_start.setText(tr("Stop"))
        else:
            self.timer.stop()
            self.module.stop_analysis()
            self.btn_start.setText(tr("Start"))

    def _on_span_changed(self, key: str):
        self.module.time_span_s = float(self._span_options.get(key, 10.0))
        self._apply_view_ranges()

    def _on_vscale_changed(self, key: str):
        self.module.vscale = float(self._vscale_options.get(key, 1.0))

    def _on_pause_toggled(self, checked: bool):
        self.module.paused = bool(checked)
        self.btn_pause.setText(tr("Pause") if not checked else tr("Pause"))
        # Keep capture running; only freeze display.

    def _apply_view_ranges(self):
        span = float(getattr(self.module, "time_span_s", 10.0))
        # Fixed vertical view range; amplitude scaling is applied to data.
        self.plot_ch1.setXRange(-span, 0)
        self.plot_ch1.setYRange(-1.1, 1.1)

    @staticmethod
    def _decimate_for_plot(t: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        if t is None or y is None or len(t) == 0:
            return np.asarray([]), np.asarray([])
        if len(t) <= max_points:
            return t, y
        step = int(max(1, len(t) // max_points))
        return t[::step], y[::step]

    def _update_plot(self):
        if not self.module.is_running:
            return

        if getattr(self.module, "paused", False):
            # Still keep last frame visible.
            return

        span = float(getattr(self.module, "time_span_s", 10.0))
        frame = self.module.get_display_data(span)
        if frame is None:
            return

        self._last_frame = frame
        t, data = frame

        scale = float(getattr(self.module, "vscale", 1.0))
        y1 = data[:, 0] * scale
        y2 = data[:, 1] * scale

        # Decimate for drawing load.
        max_points = 6000 if span <= 60 else 8000
        tt1, yy1 = self._decimate_for_plot(t, y1, max_points=max_points)
        tt2, yy2 = self._decimate_for_plot(t, y2, max_points=max_points)

        self.curve_ch1.setData(tt1, yy1)
        self.curve_ch2.setData(tt2, yy2)

        # Force scrolling view even if user drags/zooms (unless paused).
        self._apply_view_ranges()
