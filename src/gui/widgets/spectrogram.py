import argparse
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QCheckBox, QGroupBox, QSlider)
from PyQt6.QtGui import QTransform
from PyQt6.QtCore import QTimer, Qt
from scipy.signal import get_window, lfilter
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class Spectrogram(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        
        # Parameters
        self.fft_size = 2048
        self.overlap = 0.5
        self.window_type = 'hann'
        self.channel_mode = 'Left' # 'Left', 'Right', 'Average'
        self.history_length = 500 # Number of time steps to keep
        self.sweep_speed_index = 0 # 0: Fast, 1: Medium, 2: Slow, 3: Meteor
        
        # State
        self.input_buffer = np.zeros(self.fft_size) # For overlap processing
        self.spectrogram_data = np.full((self.history_length, self.fft_size // 2 + 1), -120.0)
        self.callback_id = None
        
        # Accumulator for Sweep Speed
        self.accumulator = None
        self.acc_count = 0
        
        # Ring buffer for incoming audio
        self.audio_buffer = np.zeros((self.fft_size * 2, 2)) # Keep enough for overlap
        self.audio_buffer_pos = 0
        
    @property
    def name(self) -> str:
        return "Spectrogram"

    @property
    def description(self) -> str:
        return "Time-frequency analysis (Spectrogram)."

    def run(self, args: argparse.Namespace):
        print("Spectrogram CLI not implemented")

    def get_widget(self):
        return SpectrogramWidget(self)

    def set_fft_size(self, size):
        self.fft_size = size
        self.reset_buffers()

    def reset_buffers(self):
        self.spectrogram_data = np.full((self.history_length, self.fft_size // 2 + 1), -120.0)
        self.audio_buffer = np.zeros((self.fft_size * 2, 2))
        self.audio_buffer_pos = 0
        self.accumulator = None
        self.acc_count = 0

    def start_analysis(self):
        if self.is_running: return
        self.is_running = True
        self.reset_buffers()
        self.callback_id = self.audio_engine.register_callback(self._callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def _callback(self, indata, outdata, frames, time, status):
        if status: print(status)
        
        # Append to ring buffer
        # We need to handle the case where frames > buffer space, but usually frames is small (e.g. 1024)
        # For simplicity, let's just roll and append.
        
        if frames > len(self.audio_buffer):
            # Should not happen with reasonable buffer sizes
            self.audio_buffer[:] = 0
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -frames, axis=0)
            if indata.shape[1] >= 2:
                self.audio_buffer[-frames:] = indata[:, :2]
            else:
                self.audio_buffer[-frames:, 0] = indata[:, 0]
                self.audio_buffer[-frames:, 1] = indata[:, 0]
                
        outdata.fill(0)

class SpectrogramWidget(QWidget):
    def __init__(self, module: Spectrogram):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_spectrogram)
        self.timer.setInterval(30) # ~30 FPS

    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- Controls ---
        controls_group = QGroupBox("Settings")
        controls_layout = QHBoxLayout()
        
        # Start/Stop
        self.toggle_btn = QPushButton("Start")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; } QPushButton:checked { background-color: #ffcccc; }")
        controls_layout.addWidget(self.toggle_btn)
        
        # Channel
        controls_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(['Left', 'Right', 'Average'])
        self.channel_combo.currentTextChanged.connect(self.on_channel_changed)
        controls_layout.addWidget(self.channel_combo)
        
        # FFT Size
        controls_layout.addWidget(QLabel("FFT Size:"))
        self.fft_combo = QComboBox()
        self.fft_combo.addItems(['512', '1024', '2048', '4096', '8192'])
        self.fft_combo.setCurrentText(str(self.module.fft_size))
        self.fft_combo.currentTextChanged.connect(self.on_fft_changed)
        controls_layout.addWidget(self.fft_combo)
        
        # Window
        controls_layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(['hann', 'hamming', 'blackman', 'bartlett', 'boxcar'])
        self.window_combo.currentTextChanged.connect(self.on_window_changed)
        controls_layout.addWidget(self.window_combo)
        
        # Colormap
        controls_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo'])
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        controls_layout.addWidget(self.cmap_combo)
        
        # Sweep Speed
        controls_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['Fast (Realtime)', 'Medium (1m)', 'Slow (5m)', 'Meteor (10m)'])
        self.speed_combo.currentIndexChanged.connect(self.on_speed_changed)
        controls_layout.addWidget(self.speed_combo)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # --- Plot ---
        # We use a GraphicsLayoutWidget to hold the Plot and Histogram
        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)
        
        # Plot Item
        self.plot = self.win.addPlot(title="Spectrogram")
        self.plot.setLabel('left', 'Frequency', units='Hz')
        self.plot.setLabel('bottom', 'Time', units='frames')
        
        # Image Item
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        
        # Histogram (Colormap Control)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.win.addItem(self.hist)
        
        # Set default colormap
        self.hist.gradient.loadPreset('viridis')
        self.hist.setLevels(-120, 0) # Default dB range
        
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

    def on_channel_changed(self, val):
        self.module.channel_mode = val

    def on_fft_changed(self, val):
        self.module.set_fft_size(int(val))
        # Update Image Transform if needed (scale)
        # We will handle scaling in update_spectrogram

    def on_window_changed(self, val):
        self.module.window_type = val

    def on_cmap_changed(self, val):
        self.hist.gradient.loadPreset(val)

    def on_speed_changed(self, idx):
        self.module.sweep_speed_index = idx
        # Reset accumulator on speed change to avoid mixing
        self.module.accumulator = None
        self.module.acc_count = 0

    def update_spectrogram(self):
        if not self.module.is_running: return
        
        # Get latest data from buffer
        # We take the last fft_size samples
        raw_data = self.module.audio_buffer[-self.module.fft_size:]
        
        # Select Channel
        if self.module.channel_mode == 'Left':
            sig = raw_data[:, 0]
        elif self.module.channel_mode == 'Right':
            sig = raw_data[:, 1]
        else:
            sig = np.mean(raw_data, axis=1)
            
        # Windowing
        window = get_window(self.module.window_type, len(sig))
        sig_win = sig * window
        
        # Window Correction Factor (Coherent Gain)
        win_correction = 1.0 / np.mean(window)
        
        # FFT
        fft_res = np.fft.rfft(sig_win)
        mag = np.abs(fft_res)
        
        # Normalize
        mag = mag / len(sig) * 2 * win_correction # Peak Amplitude
        
        # Convert to dB
        with np.errstate(divide='ignore'):
            mag_db = 20 * np.log10(mag + 1e-12)
            
        # --- Accumulation Logic ---
        if self.module.accumulator is None or self.module.accumulator.shape != mag_db.shape:
            self.module.accumulator = mag_db
            self.module.acc_count = 1
        else:
            # Max Hold Accumulation
            self.module.accumulator = np.maximum(self.module.accumulator, mag_db)
            self.module.acc_count += 1
            
        # Determine Target Frames based on Speed
        # Update rate is 30ms.
        # Fast: Update every frame (1)
        # Medium: 1 min = 60s. 500 pixels. 0.12s/pixel. 30ms -> 4 frames.
        # Slow: 5 min = 300s. 0.6s/pixel. 30ms -> 20 frames.
        # Meteor: 10 min = 600s. 1.2s/pixel. 30ms -> 40 frames.
        
        target_frames = 1
        if self.module.sweep_speed_index == 1: target_frames = 4
        elif self.module.sweep_speed_index == 2: target_frames = 20
        elif self.module.sweep_speed_index == 3: target_frames = 40
        
        if self.module.acc_count < target_frames:
            return # Wait for more data
            
        # Push to Spectrogram
        final_mag_db = self.module.accumulator
        self.module.accumulator = None # Reset
        self.module.acc_count = 0
            
        # Update Spectrogram Data
        # Roll history
        self.module.spectrogram_data = np.roll(self.module.spectrogram_data, -1, axis=0)
        self.module.spectrogram_data[-1] = final_mag_db
        
        # Update Image
        # ImageItem expects (width, height) where width is x-axis.
        # We want Time on X, Freq on Y? Or Time on Y?
        # Standard Spectrogram: Time on X, Freq on Y.
        # But scrolling waterfall is often Time on Y (scrolling down).
        # Let's do Time on X (scrolling left).
        
        # Data shape: (History, FreqBins) -> (Time, Freq)
        # So we can pass it directly.
        self.img.setImage(self.module.spectrogram_data, autoLevels=False)
        
        # Set Scale
        # X axis: Time (0 to History)
        # Y axis: Frequency (0 to Nyquist)
        sample_rate = self.module.audio_engine.sample_rate
        nyquist = sample_rate / 2
        
        # Scale Y to match Frequency
        # Image height is fft_size // 2 + 1
        # We want it to span 0 to Nyquist
        y_scale = nyquist / (self.module.spectrogram_data.shape[1])
        
        self.img.resetTransform()
        self.img.setTransform(QTransform().scale(1, y_scale))
        self.plot.setLimits(yMin=0, yMax=nyquist)
        self.plot.setYRange(0, nyquist)
