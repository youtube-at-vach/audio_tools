import argparse
import numpy as np
import pyqtgraph as pg
from scipy.signal import hilbert
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QProgressBar)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class LockInAmplifier(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 4096 # Adjust for integration time
        self.input_data = np.zeros((self.buffer_size, 2))
        
        # Settings
        self.gen_frequency = 1000.0
        self.gen_amplitude = 0.5 # Linear 0-1
        self.output_channel = 0 # 0: Left, 1: Right
        
        self.signal_channel = 0 # 0: Left, 1: Right
        self.ref_channel = 1    # 0: Left, 1: Right
        
        # Results
        self.current_magnitude = 0.0
        self.current_phase = 0.0
        self.ref_freq = 0.0
        self.ref_level = 0.0
        
    @property
    def name(self) -> str:
        return "Lock-in Amplifier"

    @property
    def description(self) -> str:
        return "Dual-phase lock-in detection."

    def run(self, args: argparse.Namespace):
        print("Lock-in Amplifier running from CLI (not fully implemented)")

    def get_widget(self):
        return LockInAmplifierWidget(self)

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        self.input_data = np.zeros((self.buffer_size, 2))
        
        # Generator State
        self._phase = 0
        sample_rate = self.audio_engine.sample_rate
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
            # --- Input Capture ---
            if indata.shape[1] >= 2:
                new_data = indata[:, :2]
            else:
                new_data = np.column_stack((indata[:, 0], indata[:, 0]))
            
            # Roll buffer
            if len(new_data) > self.buffer_size:
                self.input_data[:] = new_data[-self.buffer_size:]
            else:
                self.input_data = np.roll(self.input_data, -len(new_data), axis=0)
                self.input_data[-len(new_data):] = new_data
            
            # --- Output Generation ---
            # Generate Sine Wave
            t = (np.arange(frames) + self._phase) / sample_rate
            self._phase += frames
            
            signal = self.gen_amplitude * np.sin(2 * np.pi * self.gen_frequency * t)
            
            # Fill Output Buffer
            outdata.fill(0)
            if outdata.shape[1] > self.output_channel:
                outdata[:, self.output_channel] = signal

        self.audio_engine.start_stream(callback, channels=2)

    def stop_analysis(self):
        if self.is_running:
            self.audio_engine.stop_stream()
            self.is_running = False

    def process_data(self):
        """
        Perform Lock-in calculation on the current buffer.
        """
        data = self.input_data
        sig = data[:, self.signal_channel]
        ref = data[:, self.ref_channel]
        
        # 1. Analyze Reference
        # Check if reference is present
        ref_rms = np.sqrt(np.mean(ref**2))
        self.ref_level = 20 * np.log10(ref_rms + 1e-12)
        
        if ref_rms < 0.001: # -60dB threshold
            self.current_magnitude = 0.0
            self.current_phase = 0.0
            self.ref_freq = 0.0
            return

        # Estimate Ref Frequency (Zero crossings or FFT)
        # Simple zero crossing for display
        crossings = np.where(np.diff(np.signbit(ref)))[0]
        if len(crossings) > 1:
            avg_period = (crossings[-1] - crossings[0]) / (len(crossings) - 1) * 2 # *2 because diff signbit counts both edges? No, signbit changes every zero crossing.
            # Wait, signbit changes + to - and - to +. Distance between changes is half period.
            # So period = 2 * avg_distance
            fs = self.audio_engine.sample_rate
            self.ref_freq = fs / (avg_period * 2) if avg_period > 0 else 0
        
        # 2. Lock-in Detection
        # Hilbert Transform to get analytic signal of Reference
        # This gives us a complex phasor rotating at the reference frequency
        ref_analytic = hilbert(ref)
        
        # Normalize to unit magnitude to extract just the phase information
        # Avoid divide by zero
        ref_phasor = ref_analytic / (np.abs(ref_analytic) + 1e-12)
        
        # Demodulate: Multiply Signal by Conjugate of Reference Phasor
        # Product = Sig * exp(-j*theta_ref)
        # If Sig = A * cos(w*t + phi) = (A/2)*(exp(j(wt+phi)) + exp(-j(wt+phi)))
        # Ref = exp(j*wt)
        # Product = (A/2) * (exp(j*phi) + exp(-j(2wt+phi)))
        # Lowpass filtering removes the 2wt term, leaving (A/2)*exp(j*phi)
        product = sig * np.conj(ref_phasor)
        
        # Low-pass filter (Mean over buffer)
        # Ensure buffer is integer number of cycles for best accuracy, or use windowing?
        # For lock-in, long integration time (large buffer) acts as narrow filter.
        # With 4096 samples at 48k, that's ~85ms. 1kHz has 85 cycles. Error is small.
        result = np.mean(product)
        
        # Magnitude is 2 * abs(result) because we discarded half power in negative freq?
        # Let's check: Sig=1*cos(wt), Ref=cos(wt)+j*sin(wt)=exp(jwt)
        # Product = cos(wt)*exp(-jwt) = cos(wt)*(cos(wt)-j*sin(wt)) = cos^2(wt) - j*sin(wt)cos(wt)
        # Mean(cos^2) = 0.5. Mean(sin*cos) = 0.
        # So Mean = 0.5. Magnitude should be 1. So we multiply by 2.
        self.current_magnitude = 2 * np.abs(result)
        
        # Phase
        self.current_phase = np.degrees(np.angle(result))


class LockInAmplifierWidget(QWidget):
    def __init__(self, module: LockInAmplifier):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.setInterval(100) # 10Hz update

    def init_ui(self):
        layout = QHBoxLayout()
        
        # --- Left Panel: Settings ---
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()
        
        # Output Controls
        self.toggle_btn = QPushButton("Start Output & Measure")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; font-weight: bold; padding: 10px; } QPushButton:checked { background-color: #ffcccc; }")
        settings_layout.addRow(self.toggle_btn)
        
        settings_layout.addRow(QLabel("<b>Output Generator</b>"))
        
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000)
        self.freq_spin.setValue(1000)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(self.on_freq_changed)
        settings_layout.addRow("Frequency:", self.freq_spin)
        
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(-120, 0)
        self.amp_spin.setValue(-6)
        self.amp_spin.setSuffix(" dBFS")
        self.amp_spin.valueChanged.connect(self.on_amp_changed)
        settings_layout.addRow("Amplitude:", self.amp_spin)
        
        self.out_ch_combo = QComboBox()
        self.out_ch_combo.addItems(["Left (Ch 1)", "Right (Ch 2)"])
        self.out_ch_combo.currentIndexChanged.connect(self.on_out_ch_changed)
        settings_layout.addRow("Output Ch:", self.out_ch_combo)
        
        settings_layout.addRow(QLabel("<b>Input Routing</b>"))
        
        self.sig_ch_combo = QComboBox()
        self.sig_ch_combo.addItems(["Left (Ch 1)", "Right (Ch 2)"])
        self.sig_ch_combo.setCurrentIndex(0) # Default Signal L
        self.sig_ch_combo.currentIndexChanged.connect(self.on_sig_ch_changed)
        settings_layout.addRow("Signal Input:", self.sig_ch_combo)
        
        self.ref_ch_combo = QComboBox()
        self.ref_ch_combo.addItems(["Left (Ch 1)", "Right (Ch 2)"])
        self.ref_ch_combo.setCurrentIndex(1) # Default Ref R
        self.ref_ch_combo.currentIndexChanged.connect(self.on_ref_ch_changed)
        settings_layout.addRow("Reference Input:", self.ref_ch_combo)
        
        # Integration Time (Buffer Size)
        self.time_combo = QComboBox()
        self.time_combo.addItems(["Fast (2048 samples)", "Medium (4096 samples)", "Slow (16384 samples)"])
        self.time_combo.setCurrentIndex(1)
        self.time_combo.currentIndexChanged.connect(self.on_time_changed)
        settings_layout.addRow("Integration:", self.time_combo)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group, stretch=1)
        
        # --- Right Panel: Meters ---
        meters_group = QGroupBox("Measurements")
        meters_layout = QVBoxLayout()
        
        # Magnitude
        meters_layout.addWidget(QLabel("Magnitude"))
        self.mag_label = QLabel("0.000 V")
        self.mag_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #00ff00;")
        self.mag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meters_layout.addWidget(self.mag_label)
        
        self.mag_db_label = QLabel("-inf dBFS")
        self.mag_db_label.setStyleSheet("font-size: 24px; color: #88ff88;")
        self.mag_db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meters_layout.addWidget(self.mag_db_label)
        
        meters_layout.addSpacing(20)
        
        # Phase
        meters_layout.addWidget(QLabel("Phase"))
        self.phase_label = QLabel("0.00 deg")
        self.phase_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #00ffff;")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meters_layout.addWidget(self.phase_label)
        
        meters_layout.addSpacing(20)
        
        # Reference Status
        ref_status_layout = QHBoxLayout()
        ref_status_layout.addWidget(QLabel("Reference Status:"))
        self.ref_status_label = QLabel("No Signal")
        self.ref_status_label.setStyleSheet("font-weight: bold; color: #ff0000;")
        ref_status_layout.addWidget(self.ref_status_label)
        meters_layout.addLayout(ref_status_layout)
        
        meters_layout.addStretch()
        meters_group.setLayout(meters_layout)
        layout.addWidget(meters_group, stretch=2)
        
        self.setLayout(layout)

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText("Stop")
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText("Start Output & Measure")

    def on_freq_changed(self, val):
        self.module.gen_frequency = val
        # Phase continuity is handled in callback by using self.module.gen_frequency

    def on_amp_changed(self, val):
        amp_linear = 10**(val/20)
        self.module.gen_amplitude = amp_linear

    def on_out_ch_changed(self, idx):
        self.module.output_channel = idx
        if self.module.is_running:
            # Restart to apply channel change
            self.module.stop_analysis()
            self.module.start_analysis()

    def on_sig_ch_changed(self, idx):
        self.module.signal_channel = idx

    def on_ref_ch_changed(self, idx):
        self.module.ref_channel = idx

    def on_time_changed(self, idx):
        if idx == 0: self.module.buffer_size = 2048
        elif idx == 1: self.module.buffer_size = 4096
        elif idx == 2: self.module.buffer_size = 16384
        
        # Re-allocate buffer
        self.module.input_data = np.zeros((self.module.buffer_size, 2))

    def update_ui(self):
        if not self.module.is_running:
            return
            
        self.module.process_data()
        
        # Update Meters
        mag = self.module.current_magnitude
        phase = self.module.current_phase
        
        self.mag_label.setText(f"{mag:.6f} V") # Assuming 0dBFS = 1.0V for now, or just linear units
        
        if mag > 0:
            db = 20 * np.log10(mag + 1e-12)
            self.mag_db_label.setText(f"{db:.2f} dBFS")
        else:
            self.mag_db_label.setText("-inf dBFS")
            
        self.phase_label.setText(f"{phase:.2f} deg")
        
        # Update Ref Status
        ref_level = self.module.ref_level
        ref_freq = self.module.ref_freq
        
        if ref_level > -60:
            self.ref_status_label.setText(f"Locked ({ref_freq:.1f} Hz, {ref_level:.1f} dB)")
            self.ref_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        else:
            self.ref_status_label.setText("No Signal / Low Level")
            self.ref_status_label.setStyleSheet("font-weight: bold; color: #ff0000;")
