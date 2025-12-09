
import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QSpinBox, QTabWidget, QProgressBar)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class LockInTHDAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 8192 # Good compromise for latency vs precision
        self.input_data = np.zeros(self.buffer_size)
        self.residual_data = np.zeros(self.buffer_size)
        
        # Generator Settings
        self.gen_frequency = 1000.0
        self.gen_amplitude = 0.5 
        self.output_channel = 0
        self.input_channel = 0
        self.output_enabled = True
        
        # Analysis State
        self.target_freq = 1000.0 # Typically same as gen_freq for internal mode
        self.harmonic_count = 10
        self.bw_low = 20.0
        self.bw_high = 20000.0
        
        # Results
        self.measured_freq = 0.0
        self.fund_amp = 0.0
        self.fund_phase = 0.0
        self.thdn_value = 0.0
        self.thdn_db = -140.0
        self.residual_rms = 0.0
        
        # DSP State
        self._phase_gen = 0.0
        self.callback_id = None
        
    @property
    def name(self) -> str:
        return "Lock-in THD+N"

    @property
    def description(self) -> str:
        return "High-precision THD+N measurement using lock-in fundamental removal."

    def run(self, args):
        print("Lock-in THD+N Analyzer running from CLI (not implemented)")

    def get_widget(self):
        return LockInTHDWidget(self)

    def start_analysis(self):
        print("DEBUG: start_analysis called. is_running:", self.is_running)
        if self.is_running: return
        self.is_running = True
        
        sample_rate = self.audio_engine.sample_rate
        self.input_data = np.zeros(self.buffer_size)
        self.residual_data = np.zeros(self.buffer_size)
        self._phase_gen = 0.0
        
        # Pre-calculate filters if needed
        # We'll do filtering in the callback or process loop
        
        def callback(indata, outdata, frames, time, status):
            if status: print(status)
            
            # Guard against zombie callback
            if not self.is_running:
                outdata.fill(0)
                return

            # 1. Generator
            outdata.fill(0)
            if self.output_enabled:
                t = (np.arange(frames) + self._phase_gen) / sample_rate
                self._phase_gen += frames
                sig = self.gen_amplitude * np.sin(2 * np.pi * self.gen_frequency * t)
                
                if self.output_channel == 2: # Stereo
                    if outdata.shape[1] >= 2: outdata[:, :2] = sig[:, np.newaxis]
                elif outdata.shape[1] > self.output_channel:
                    outdata[:, self.output_channel] = sig

            # 2. Input Capture
            if indata.shape[1] > self.input_channel:
                in_sig = indata[:, self.input_channel]
            else:
                in_sig = indata[:, 0]
                
            # Update Ring Buffer
            if len(in_sig) > self.buffer_size:
                self.input_data[:] = in_sig[-self.buffer_size:]
            else:
                self.input_data = np.roll(self.input_data, -len(in_sig))
                self.input_data[-len(in_sig):] = in_sig

        self.callback_id = self.audio_engine.register_callback(callback)
        print("DEBUG: Measured started. callback_id:", self.callback_id)
        
    def stop_analysis(self):
        print("DEBUG: stop_analysis called. is_running:", self.is_running, "callback_id:", self.callback_id)
        if self.is_running:
            self.is_running = False # Flag first
            if self.callback_id is not None:
                print("DEBUG: Unregistering callback:", self.callback_id)
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            else:
                print("DEBUG: callback_id is None, skipping unregister")

    def process(self):
        if not self.is_running: return
        
        # Snapshot of buffer
        data = self.input_data.copy()
        fs = self.audio_engine.sample_rate
        N = len(data)
        t = np.arange(N) / fs
        
        # Lock-in Detection (Post-processing on block)
        # We assume frequency is known (Internal mode)
        # If we need potential tuning, we could do a coarse FFT peak find first.
        f0 = self.gen_frequency
        
        # Create IQ Reference
        # Note: Phase here is relative to the start of *this block*. 
        # Since we reconstruct for *this block*, relative phase is all we need.
        ref_cos = np.cos(2 * np.pi * f0 * t)
        ref_sin = np.sin(2 * np.pi * f0 * t)
        
        # Demodulate
        # I = Signal * Cos, Q = Signal * Sin
        i_comp = np.mean(data * ref_cos)
        q_comp = np.mean(data * ref_sin)
        
        # Calculate Magnitude and Phase
        # Signal ~ A * cos(wt + phi) = A * (cos(wt)cos(phi) - sin(wt)sin(phi))
        # Mean(Sig * cos) = A/2 * cos(phi)
        # Mean(Sig * sin) = -A/2 * sin(phi)
        
        # So:
        # A = 2 * sqrt(I^2 + Q^2)
        # phi = atan2(-Q, I)  <-- Careful with signs definition
        
        amp = 2 * np.sqrt(i_comp**2 + q_comp**2)
        phase = np.arctan2(-q_comp, i_comp) 
        
        self.fund_amp = amp
        self.fund_phase = phase
        self.measured_freq = f0 # For display
        
        # Reconstruction
        # s_est = amp * cos(2*pi*f0*t + phase)
        s_est = amp * np.cos(2 * np.pi * f0 * t + phase)
        
        # Residual
        residual = data - s_est
        
        # Filtering (Bandwidth Limit)
        # 20 Hz - 20 kHz (or user defined)
        # Use SOS filters
        nyquist = 0.5 * fs
        low = max(0.1, self.bw_low)
        high = min(nyquist - 1, self.bw_high)
        
        if low > 0 and low < high:
            sos_hp = butter(4, low, 'hp', fs=fs, output='sos')
            residual = sosfiltfilt(sos_hp, residual)
            
        if high < nyquist:
            sos_lp = butter(4, high, 'lp', fs=fs, output='sos')
            residual = sosfiltfilt(sos_lp, residual)
        
        self.residual_data = residual
        
        # Calculate RMS
        # Remove edges to avoid filter artifacts ?
        trim = 100
        if len(residual) > 2*trim:
            res_valid = residual[trim:-trim]
        else:
            res_valid = residual
            
        self.residual_rms = np.sqrt(np.mean(res_valid**2))
        
        # THD+N
        if self.fund_amp > 1e-9:
            ratio = self.residual_rms / self.fund_amp
            self.thdn_db = 20 * np.log10(ratio)
            self.thdn_value = ratio * 100
        else:
            self.thdn_db = -140.0
            self.thdn_value = 0.0


class LockInTHDWidget(QWidget):
    def __init__(self, module: LockInTHDAnalyzer):
        super().__init__()
        self.module = module
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.setInterval(100) # 10 Hz

    def init_ui(self):
        layout = QHBoxLayout()
        
        # LEFT: Controls
        left_panel = QVBoxLayout()
        settings_group = QGroupBox(tr("Settings"))
        form = QFormLayout()
        
        self.btn_toggle = QPushButton(tr("Start Measurement"))
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.clicked.connect(self.on_toggle)
        self.btn_toggle.setStyleSheet("QPushButton:checked { background-color: #ccffcc; }")
        form.addRow(self.btn_toggle)
        
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000)
        self.freq_spin.setValue(1000)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(self.on_freq_changed)
        form.addRow(tr("Frequency:"), self.freq_spin)
        
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0, 1.0)
        self.amp_spin.setSingleStep(0.1)
        self.amp_spin.setValue(0.5)
        self.amp_spin.valueChanged.connect(lambda v: setattr(self.module, 'gen_amplitude', v))
        form.addRow(tr("Amplitude (0-1):"), self.amp_spin)
        
        self.bw_low_spin = QDoubleSpinBox()
        self.bw_low_spin.setRange(0, 1000)
        self.bw_low_spin.setValue(20)
        self.bw_low_spin.valueChanged.connect(lambda v: setattr(self.module, 'bw_low', v))
        form.addRow(tr("HPF (Hz):"), self.bw_low_spin)
        
        self.bw_high_spin = QDoubleSpinBox()
        self.bw_high_spin.setRange(1000, 48000)
        self.bw_high_spin.setValue(20000)
        self.bw_high_spin.valueChanged.connect(lambda v: setattr(self.module, 'bw_high', v))
        form.addRow(tr("LPF (Hz):"), self.bw_high_spin)
        
        settings_group.setLayout(form)
        left_panel.addWidget(settings_group)
        
        # Meters
        meters_group = QGroupBox(tr("Results"))
        meters_layout = QVBoxLayout()
        
        self.lbl_thdn = QLabel("--")
        self.lbl_thdn.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff5555;")
        self.lbl_thdn_db = QLabel("-- dB")
        meters_layout.addWidget(QLabel("THD+N:"))
        meters_layout.addWidget(self.lbl_thdn)
        meters_layout.addWidget(self.lbl_thdn_db)
        
        meters_layout.addSpacing(10)
        
        self.lbl_fund = QLabel("-- V")
        self.lbl_fund.setStyleSheet("font-size: 18px; color: #55ff55;")
        meters_layout.addWidget(QLabel("Fundamental (Lock-in):"))
        meters_layout.addWidget(self.lbl_fund)
        
        self.lbl_res = QLabel("-- V")
        meters_layout.addWidget(QLabel("Residual RMS:"))
        meters_layout.addWidget(self.lbl_res)
        
        meters_group.setLayout(meters_layout)
        left_panel.addWidget(meters_group)
        
        left_panel.addStretch()
        layout.addLayout(left_panel, 1)
        
        # RIGHT: Plots
        right_panel = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # Plot 1: Time Domain (Input vs Residual)
        self.plot_time = pg.PlotWidget(title="Time Domain")
        self.plot_time.addLegend()
        self.curve_input = self.plot_time.plot(pen='c', name="Input")
        self.curve_resid = self.plot_time.plot(pen='r', name="Residual (x10)")
        self.tabs.addTab(self.plot_time, "Waveform")
        
        # Plot 2: Spectrum (Residual)
        self.plot_spec = pg.PlotWidget(title="Residual Spectrum")
        self.plot_spec.setLogMode(x=True, y=False)
        self.plot_spec.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_spec.setLabel('left', 'Magnitude', units='dB')
        self.plot_spec.showGrid(x=True, y=True)
        self.curve_spec = self.plot_spec.plot(pen='y')
        self.tabs.addTab(self.plot_spec, "Spectrum")
        
        right_panel.addWidget(self.tabs)
        layout.addLayout(right_panel, 3)
        
        self.setLayout(layout)
        
    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.btn_toggle.setText(tr("Stop"))
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.btn_toggle.setText(tr("Start Measurement"))
            
    def on_freq_changed(self, val):
        self.module.gen_frequency = val
        self.module.target_freq = val
        
    def update_ui(self):
        if not self.module.is_running: return
        
        # Trigger processing in main thread to avoid threading issues with large data copies?
        # Ideally processing happens in separate thread, but for simplicity we call it here 
        # or have a worker. For now, calling process() here is okay if buffer isn't huge.
        # Better: process() buffers light data, filtering might be heavy.
        self.module.process()
        
        # Update Labels
        self.lbl_thdn.setText(f"{self.module.thdn_value:.4f} %")
        self.lbl_thdn_db.setText(f"{self.module.thdn_db:.2f} dB")
        self.lbl_fund.setText(f"{self.module.fund_amp:.5f} V")
        self.lbl_res.setText(f"{self.module.residual_rms:.2e} V")
        
        # Update Plots
        # Decimate for performance
        data = self.module.input_data
        res = self.module.residual_data
        
        step = max(1, len(data) // 1000)
        self.curve_input.setData(data[::step])
        # Scale residual for visibility
        self.curve_resid.setData(res[::step] * 10) 
        
        # Spectrum
        # Calculate FFT of residual
        if len(res) > 0:
            window = np.hanning(len(res))
            fft_res = np.fft.rfft(res * window)
            mag = 20 * np.log10(np.abs(fft_res) / len(res) * 2 + 1e-12)
            freqs = np.fft.rfftfreq(len(res), 1/self.module.audio_engine.sample_rate)
            
            # Skip DC
            self.curve_spec.setData(freqs[1:], mag[1:])
