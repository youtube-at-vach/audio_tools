import numpy as np
import time
import pyqtgraph as pg
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
                             QComboBox, QGroupBox, QFormLayout, QFrame, QPushButton)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.analysis import AudioCalc

class FrequencyCounter(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.callback_id = None
        
        # Settings
        self.gate_threshold_db = -60.0
        self.update_interval_ms = 100 # Fast: 100ms, Slow: 500ms
        self.buffer_size = 8192 # Good resolution
        self.selected_channel = 0 # 0: Ch1, 1: Ch2
        
        # State
        self.input_buffer = np.zeros(self.buffer_size)
        self.history_len = 100
        self.freq_history = deque(maxlen=self.history_len)
        self.time_history = deque(maxlen=self.history_len)
        self.start_time = 0
        self.current_freq = 0.0
        self.current_amp_db = -100.0
        
    @property
    def name(self) -> str:
        return "Frequency Counter"

    @property
    def description(self) -> str:
        return "High-precision frequency measurement."

    def run(self, args):
        print("Frequency Counter running in CLI mode (not fully implemented)")
        self.start_analysis()
        try:
            while True:
                freq = self.process()
                if freq:
                    print(f"Frequency: {freq:.4f} Hz")
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_analysis()

    def get_widget(self):
        return FrequencyCounterWidget(self)

    def start_analysis(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.input_buffer = np.zeros(self.buffer_size)
        self.freq_history.clear()
        self.time_history.clear()
        self.start_time = 0 # Will be set on first update
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
            # Capture Selected Channel
            if indata.shape[1] > self.selected_channel:
                new_data = indata[:, self.selected_channel]
            elif indata.shape[1] > 0:
                 # Fallback to Ch 0 if selected not available
                new_data = indata[:, 0]
            else:
                new_data = np.zeros(frames)
                
            # Ring buffer
            if len(new_data) >= self.buffer_size:
                self.input_buffer[:] = new_data[-self.buffer_size:]
            else:
                self.input_buffer = np.roll(self.input_buffer, -len(new_data))
                self.input_buffer[-len(new_data):] = new_data
                
            outdata.fill(0)

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def process(self):
        if not self.is_running:
            return None
            
        data = self.input_buffer.copy()
        sr = self.audio_engine.sample_rate
        
        # 1. Check Amplitude (Gate)
        rms = np.sqrt(np.mean(data**2))
        db = 20 * np.log10(rms + 1e-12)
        self.current_amp_db = db
        
        if db < self.gate_threshold_db:
            return None # Signal too low
            
        # 2. Coarse Estimate (FFT)
        window = np.hamming(len(data))
        fft_res = np.fft.rfft(data * window)
        freqs = np.fft.rfftfreq(len(data), 1/sr)
        
        idx = np.argmax(np.abs(fft_res))
        coarse_freq = freqs[idx]
        
        # 3. Fine Estimate (Parabolic)
        # (Already implemented in AudioCalc.analyze_harmonics, but let's do a quick one here or skip to optimization)
        # Optimization is robust enough if coarse is close.
        
        # 4. Precision Estimate (Sine Fit)
        # Only run if we have a reasonable signal
        if coarse_freq > 10: # Avoid DC/VLF noise
            try:
                precise_freq = AudioCalc.optimize_frequency(data, sr, coarse_freq)
                self.current_freq = precise_freq
                return precise_freq
            except:
                return coarse_freq
        else:
            return coarse_freq

class FrequencyCounterWidget(QWidget):
    def __init__(self, module: FrequencyCounter):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.setInterval(self.module.update_interval_ms)
        
        # Start time tracking
        self.module.start_time = time.time()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- Display Area ---
        display_frame = QFrame()
        display_frame.setStyleSheet("background-color: #000; border: 2px solid #444; border-radius: 10px;")
        display_layout = QVBoxLayout(display_frame)
        
        self.freq_label = QLabel("0.00000 Hz")
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Use a monospaced font if available, or just a clean sans-serif
        font = QFont("Courier New", 72, QFont.Weight.Bold)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.freq_label.setFont(font)
        self.freq_label.setStyleSheet("color: #00ff00;") # Green LED style
        display_layout.addWidget(self.freq_label)
        
        self.amp_label = QLabel("-- dBFS")
        self.amp_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.amp_label.setFont(QFont("Arial", 16))
        self.amp_label.setStyleSheet("color: #888;")
        display_layout.addWidget(self.amp_label)
        
        layout.addWidget(display_frame)
        
        # --- Controls ---
        controls_layout = QHBoxLayout()
        
        # Gate
        gate_layout = QHBoxLayout()
        gate_layout.addWidget(QLabel("Gate (dB):"))
        self.gate_spin = QDoubleSpinBox()
        self.gate_spin.setRange(-120, 0)
        self.gate_spin.setValue(self.module.gate_threshold_db)
        self.gate_spin.valueChanged.connect(lambda v: setattr(self.module, 'gate_threshold_db', v))
        gate_layout.addWidget(self.gate_spin)
        controls_layout.addLayout(gate_layout)

        # Channel
        ch_layout = QHBoxLayout()
        ch_layout.addWidget(QLabel("Channel:"))
        self.ch_combo = QComboBox()
        self.ch_combo.addItems(["Ch 1", "Ch 2"])
        self.ch_combo.currentIndexChanged.connect(lambda idx: setattr(self.module, 'selected_channel', idx))
        ch_layout.addWidget(self.ch_combo)
        controls_layout.addLayout(ch_layout)
        
        # Speed
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Update Rate:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Fast (10Hz)", "Slow (2Hz)"])
        self.speed_combo.currentIndexChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_combo)
        controls_layout.addLayout(speed_layout)
        
        # Start/Stop
        self.run_btn = QPushButton("Start")
        self.run_btn.setCheckable(True)
        self.run_btn.clicked.connect(self.on_run_toggle)
        controls_layout.addWidget(self.run_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # --- History Plot ---
        self.plot_widget = pg.PlotWidget(title="Frequency Drift")
        self.plot_widget.setLabel('left', 'Frequency', units='Hz')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True)
        self.curve = self.plot_widget.plot(pen='y')
        layout.addWidget(self.plot_widget)
        
        self.setLayout(layout)

    def on_speed_changed(self, idx):
        if idx == 0: # Fast
            self.module.update_interval_ms = 100
        else: # Slow
            self.module.update_interval_ms = 500
        self.timer.setInterval(self.module.update_interval_ms)

    def on_run_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.run_btn.setText("Stop")
            self.module.start_time = time.time()
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.run_btn.setText("Start")

    def update_display(self):
        freq = self.module.process()
        
        # Update Amp
        self.amp_label.setText(f"{self.module.current_amp_db:.1f} dBFS")
        
        if freq is not None:
            # Update Label
            self.freq_label.setText(f"{freq:.5f} Hz")
            
            # Update History
            t = time.time() - self.module.start_time
            self.module.freq_history.append(freq)
            self.module.time_history.append(t)
            
            self.curve.setData(list(self.module.time_history), list(self.module.freq_history))
        else:
            self.freq_label.setText("---.----- Hz")
