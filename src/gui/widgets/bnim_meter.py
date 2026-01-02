import numpy as np
from typing import Tuple, Dict, Any
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QGroupBox, QSlider, QCheckBox, QApplication)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr
from scipy.ndimage import gaussian_filter

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
        
        # State
        self.callback_id = None
        self.sample_rate = self.audio_engine.sample_rate
        self.audio_buffer = np.zeros((4096, 2))
        
        # Neural Map State (Frequencies x ITD)
        # We'll determine the actual shape during first processing
        self.neural_map = None
        self.frequencies = None
        self.itd_axis = None

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
        self.audio_buffer = np.zeros((self.fft_size * 2, 2))
        
        # Prepare axes
        self.itd_axis = np.linspace(-self.max_itd_ms, self.max_itd_ms, self.num_itd_bins)
        
        # Frequency bins for RFFT
        freqs = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)
        # Select indices within range
        self.freq_indices = np.where((freqs >= self.freq_min) & (freqs <= self.freq_max))[0]
        self.frequencies = freqs[self.freq_indices]
        
        # Initialize Neural Map
        self.neural_map = np.zeros((len(self.frequencies), self.num_itd_bins))
        
        self.callback_id = self.audio_engine.register_callback(self._callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def _callback(self, indata, outdata, frames, time, status):
        # Update local buffer (Roll)
        if frames >= len(self.audio_buffer):
            self.audio_buffer[:] = indata[-len(self.audio_buffer):, :2]
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -frames, axis=0)
            self.audio_buffer[-frames:] = indata[:, :2]
        outdata.fill(0)

    def process_buffer(self):
        """Perform the 'neural' processing: ITD/ILD extraction per frequency."""
        if not self.is_running: return
        
        # Extract last window
        window_data = self.audio_buffer[-self.fft_size:]
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
        
        # Jeffress Model Approximation:
        # Peak occurs when phase(L) == phase(R * phase_shift)
        # delta_phi = angle(L) - (angle(R) + phase(phase_shift))
        # We want to find tau such that angle(L) - angle(R) == phase(phase_shift)
        
        delays_ms = self.itd_axis
        delays_s = delays_ms / 1000.0
        itd_grid, freq_grid = np.meshgrid(delays_s, self.frequencies)
        # w*tau
        phase_diff_model = -2 * np.pi * freq_grid * itd_grid 
        
        phase_L = np.angle(fft_L)
        phase_R = np.angle(fft_R)
        
        # Broadcast
        phase_diff_signal = (phase_L - phase_R)[:, np.newaxis]
        
        # Coincidence = cos(phase_diff_signal - phase_diff_model)
        coincidence = 0.5 + 0.5 * np.cos(phase_diff_signal - phase_diff_model)
        
        # Apply ILD weighting?
        # Or just use coincidence?
        # Let's use ILD to modulate the overall intensity of the frequency band
        band_intensity = np.log1p(mag_sum * self.gain)
        coincidence *= band_intensity[:, np.newaxis]
        
        # Update neural map with persistence
        self.neural_map = (self.neural_map * self.decay) + (coincidence * (1.0 - self.decay))

class BNIMMeterWidget(QWidget):
    def __init__(self, module: BNIMMeter):
        super().__init__()
        self.module = module
        self.init_ui()
        
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
        self.label_itd = QLabel(tr("ITD (Left <-> Right)"))
        self.label_itd.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        
        # Freq Range
        controls_layout.addWidget(QLabel(tr("Max Freq (Hz):")))
        self.freq_combo = QComboBox()
        self.freq_combo.addItems(["1000", "2000", "5000", "10000"])
        self.freq_combo.setCurrentText("5000")
        self.freq_combo.currentTextChanged.connect(self.on_freq_changed)
        controls_layout.addWidget(self.freq_combo)

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
        
    def on_freq_changed(self, text):
        was_running = self.module.is_running
        if was_running: self.module.stop_analysis()
        self.module.freq_max = int(text)
        if was_running: self.module.start_analysis()
        
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
        self.img_item.setLevels([0, np.max(data) * 0.8 + 0.1])
        
        # Set Rect
        # x: -max_itd to +max_itd
        # y: log scale or linear? Let's start linear for simplicity
        itd = self.module.max_itd_ms
        freq = self.module.freq_max
        self.img_item.setRect(pg.QtCore.QRectF(-itd, 0, 2*itd, freq))
        self.plot_widget.setXRange(-itd, itd)
        self.plot_widget.setYRange(0, freq)
