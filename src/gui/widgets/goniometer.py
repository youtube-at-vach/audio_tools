import argparse
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QGroupBox, QSlider, QProgressBar, QCheckBox)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class Goniometer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        
        # Settings
        self.buffer_size = 2048
        self.gain = 1.0
        self.auto_gain = False
        self.decay = 0.8 # Persistence factor (0-1)
        
        # State
        self.audio_buffer = np.zeros((self.buffer_size, 2))
        self.correlation = 0.0
        self.callback_id = None
        
    @property
    def name(self) -> str:
        return "Goniometer"

    @property
    def description(self) -> str:
        return "Stereo image visualizer (Lissajous) and Phase Correlation."

    def run(self, args: argparse.Namespace):
        print("Goniometer CLI not implemented")

    def get_widget(self):
        return GoniometerWidget(self)

    def start_analysis(self):
        if self.is_running: return
        self.is_running = True
        self.audio_buffer = np.zeros((self.buffer_size, 2))
        self.callback_id = self.audio_engine.register_callback(self._callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def _callback(self, indata, outdata, frames, time, status):
        if status: print(status)
        
        # Get stereo data
        if indata.shape[1] >= 2:
            new_data = indata[:, :2]
        else:
            # Mono input - duplicate to both channels? 
            # If mono, L=R, so it should be a vertical line.
            new_data = np.column_stack((indata[:, 0], indata[:, 0]))
            
        # Update buffer (Roll)
        if frames >= self.buffer_size:
            self.audio_buffer[:] = new_data[-self.buffer_size:]
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -frames, axis=0)
            self.audio_buffer[-frames:] = new_data
            
        # Calculate Correlation (Instantaneous for this block)
        # Avoid division by zero
        l = new_data[:, 0]
        r = new_data[:, 1]
        
        dot = np.sum(l * r)
        mag_l = np.sum(l**2)
        mag_r = np.sum(r**2)
        
        if mag_l > 1e-9 and mag_r > 1e-9:
            self.correlation = dot / np.sqrt(mag_l * mag_r)
        elif mag_l < 1e-9 and mag_r < 1e-9:
            self.correlation = 0.0 # Silence
        else:
            # One channel silent? Correlation is technically 0
            self.correlation = 0.0
            
        outdata.fill(0)

class GoniometerWidget(QWidget):
    def __init__(self, module: Goniometer):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.setInterval(30) # 30 FPS

    def init_ui(self):
        layout = QHBoxLayout()
        
        # --- Left: Display ---
        display_layout = QVBoxLayout()
        
        # 1. Goniometer Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setXRange(-1.1, 1.1)
        self.plot_widget.setYRange(-1.1, 1.1)
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add background lines for reference
        # Diagonal (L only, R only)
        # L only: M=S -> y=x
        # R only: M=-S -> y=-x
        self.plot_widget.addItem(pg.InfiniteLine(pos=(0,0), angle=45, pen=pg.mkPen('#444', style=Qt.PenStyle.DashLine)))
        self.plot_widget.addItem(pg.InfiniteLine(pos=(0,0), angle=-45, pen=pg.mkPen('#444', style=Qt.PenStyle.DashLine)))
        
        # Scatter for the cloud
        # We use a ScatterPlotItem or just a PlotCurveItem with dots?
        # PlotCurveItem is faster for connected lines.
        # For Goniometer, usually it's a connected trace (Lissajous).
        self.trace = self.plot_widget.plot(pen=pg.mkPen(color=(0, 255, 200, 150), width=1))
        
        display_layout.addWidget(self.plot_widget, stretch=1)
        
        # 2. Correlation Meter
        corr_layout = QHBoxLayout()
        corr_layout.addWidget(QLabel("-1"))
        
        self.corr_bar = QProgressBar()
        self.corr_bar.setRange(-100, 100) # Map -1.0..1.0 to -100..100
        self.corr_bar.setTextVisible(True)
        self.corr_bar.setFormat("%v") # Show value? Or custom text
        self.corr_bar.setFixedHeight(20)
        corr_layout.addWidget(self.corr_bar, stretch=1)
        
        corr_layout.addWidget(QLabel("+1"))
        display_layout.addLayout(corr_layout)
        
        layout.addLayout(display_layout, stretch=3)
        
        # --- Right: Controls ---
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        
        # Start/Stop
        self.toggle_btn = QPushButton("Start")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; } QPushButton:checked { background-color: #ffcccc; }")
        controls_layout.addWidget(self.toggle_btn)
        
        # Gain
        controls_layout.addWidget(QLabel("Gain:"))
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(1, 100) # 0.1x to 10.0x
        self.gain_slider.setValue(10) # 1.0x
        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        controls_layout.addWidget(self.gain_slider)
        
        self.gain_label = QLabel("1.0x")
        controls_layout.addWidget(self.gain_label, alignment=Qt.AlignmentFlag.AlignRight)
        
        # Auto Gain
        self.auto_gain_chk = QCheckBox("Auto Gain")
        self.auto_gain_chk.toggled.connect(lambda x: setattr(self.module, 'auto_gain', x))
        controls_layout.addWidget(self.auto_gain_chk)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        
        layout.addWidget(controls_group, stretch=1)
        
        self.setLayout(layout)

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText("Stop")
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText("Start")

    def on_gain_changed(self, val):
        # Logarithmic-ish feel
        # 1 -> 0.1
        # 10 -> 1.0
        # 50 -> 5.0
        # 100 -> 10.0
        gain = val / 10.0
        self.module.gain = gain
        self.gain_label.setText(f"{gain:.1f}x")

    def update_display(self):
        if not self.module.is_running: return
        
        # Get data
        data = self.module.audio_buffer
        l = data[:, 0]
        r = data[:, 1]
        
        # Apply Gain
        if self.module.auto_gain:
            peak = np.max(np.abs(data))
            if peak > 1e-6:
                gain = 0.8 / peak # Target 0.8 peak
                # Smooth gain change?
                self.module.gain = self.module.gain * 0.9 + gain * 0.1
        
        l = l * self.module.gain
        r = r * self.module.gain
        
        # Transform to M/S (Rotated 45 deg)
        # M (Y) = (L + R) * 0.707
        # S (X) = (L - R) * 0.707
        
        # Note: In some goniometers, Side is X, Mid is Y.
        # L = (M + S) / 1.414
        # R = (M - S) / 1.414
        
        # If L=1, R=0 -> M=0.7, S=0.7 -> Top-Right
        # If L=0, R=1 -> M=0.7, S=-0.7 -> Top-Left?
        # Wait, usually L is Top-Left (-X, +Y) and R is Top-Right (+X, +Y)?
        # Let's check standard.
        # Standard:
        # Vertical axis: M (L+R)
        # Horizontal axis: S (L-R)
        # But usually rotated so L is diagonal up-left, R is diagonal up-right.
        # That IS the M/S plot.
        
        # Let's stick to:
        # X = (L - R) / sqrt(2)  (Side)
        # Y = (L + R) / sqrt(2)  (Mid)
        # If L=1, R=0 -> X=0.7, Y=0.7. (Up-Right)
        # If L=0, R=1 -> X=-0.7, Y=0.7. (Up-Left)
        # Wait, I want L to be Left.
        # So maybe X = (R - L) / sqrt(2)? No, standard is Side = L-R.
        # If X = L-R:
        # L=1 -> X=1 (Right).
        # So L is on the Right side of the screen? That's confusing.
        # Usually Left channel is associated with Left side.
        # So let's flip X.
        # X = (R - L) / sqrt(2)?
        # L=1 -> X=-0.7 (Left). Correct.
        # R=1 -> X=0.7 (Right). Correct.
        
        # Let's use X = (R - L) * 0.707
        
        m = (l + r) * 0.707
        s = (r - l) * 0.707 # R - L to put L on left
        
        # Downsample for display performance if needed
        # 2048 points is fine for pyqtgraph
        
        self.trace.setData(s, m)
        
        # Update Correlation
        corr = self.module.correlation
        # Map -1..1 to -100..100
        val = int(corr * 100)
        self.corr_bar.setValue(val)
        self.corr_bar.setFormat(f"{corr:.2f}")
        
        # Color code
        # +1 to 0: Green
        # 0 to -1: Red
        
        if corr >= 0:
            color = "#00ff00"
        else:
            color = "#ff0000"
            
        self.corr_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
