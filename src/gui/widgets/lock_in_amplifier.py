import argparse
import numpy as np
import pyqtgraph as pg
from scipy.signal import hilbert
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QProgressBar, QSpinBox)
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QProgressBar, QSpinBox, QTabWidget, QApplication)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import time
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
        self.gen_amplitude = 0.5 # Linear 0-1
        self.output_channel = 0 # 0: Left, 1: Right
        self.external_mode = False
        
        self.signal_channel = 0 # 0: Left, 1: Right
        self.ref_channel = 1    # 0: Left, 1: Right
        
        # Results
        self.current_magnitude = 0.0
        self.current_phase = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.ref_freq = 0.0
        self.ref_level = 0.0
        
        # Averaging
        self.averaging_count = 1
        self.history = deque(maxlen=100)
        
        self.callback_id = None
        
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
            
            signal = self.gen_amplitude * np.cos(2 * np.pi * self.gen_frequency * t)
            
            # Fill Output Buffer
            outdata.fill(0)
            
            if not self.external_mode:
                if self.output_channel == 2: # Stereo
                    if outdata.shape[1] >= 1: outdata[:, 0] = signal
                    if outdata.shape[1] >= 2: outdata[:, 1] = signal
                elif outdata.shape[1] > self.output_channel:
                    outdata[:, self.output_channel] = signal

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
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
            self.current_x = 0.0
            self.current_y = 0.0
            self.ref_freq = 0.0
            return

        # Estimate Ref Frequency (Zero crossings or FFT)
        # Simple zero crossing for display
        crossings = np.where(np.diff(np.signbit(ref)))[0]
        if len(crossings) > 1:
            avg_period = (crossings[-1] - crossings[0]) / (len(crossings) - 1) * 2 
            fs = self.audio_engine.sample_rate
            self.ref_freq = fs / avg_period if avg_period > 0 else 0
        
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
        result = np.mean(product)
        
        # Averaging
        self.history.append(result)
        while len(self.history) > self.averaging_count:
            self.history.popleft()
            
        avg_result = np.mean(self.history)
        
        # Magnitude is 2 * abs(result)
        self.current_magnitude = 2 * np.abs(avg_result)
        
        # Phase
        self.current_phase = np.degrees(np.angle(avg_result))
        
        # X and Y (In-phase and Quadrature)
        # X = 2 * Real, Y = 2 * Imag
        self.current_x = 2 * np.real(avg_result)
        self.current_y = 2 * np.imag(avg_result)



class FRASweepWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(float, float, float) # freq, mag, phase
    finished_sweep = pyqtSignal()
    
    def __init__(self, module: LockInAmplifier, start_f, end_f, steps, log_sweep, settle_time):
        super().__init__()
        self.module = module
        self.start_f = start_f
        self.end_f = end_f
        self.steps = steps
        self.log_sweep = log_sweep
        self.settle_time = settle_time
        self.is_cancelled = False
        
    def run(self):
        if self.log_sweep:
            freqs = np.logspace(np.log10(self.start_f), np.log10(self.end_f), self.steps)
        else:
            freqs = np.linspace(self.start_f, self.end_f, self.steps)
            
        # Ensure module is running
        if not self.module.is_running:
            self.module.start_analysis()
            time.sleep(0.5) # Wait for start
            
        for i, f in enumerate(freqs):
            if self.is_cancelled: break
            
            self.module.gen_frequency = f
            
            # Wait for settling
            time.sleep(self.settle_time)
            
            # Measurement Loop
            # We need to capture 'averaging_count' buffers
            self.module.history.clear()
            
            # Calculate buffer duration
            sample_rate = self.module.audio_engine.sample_rate
            buffer_duration = self.module.buffer_size / sample_rate
            
            # Ensure we wait at least a bit to avoid CPU spin if buffer is tiny, 
            # though buffer_size is usually > 2048 (approx 40ms at 48k)
            wait_time = max(0.05, buffer_duration)
            
            # We need to fill the history with new data
            # The process_data() call processes the *current* buffer.
            # We need to wait for the audio callback to update the buffer.
            
            # First, wait for one full buffer fill to ensure we are past the settling time completely
            time.sleep(wait_time)
            
            for _ in range(self.module.averaging_count):
                if self.is_cancelled: break
                
                # Wait for next buffer update
                # Since we don't have precise synchronization with callback here, 
                # we sleep for the buffer duration.
                time.sleep(wait_time)
                
                # Process the current buffer state
                self.module.process_data()
            
            # Read measurement (which is now the average of the history)
            mag = self.module.current_magnitude
            phase = self.module.current_phase
            
            self.result.emit(f, mag, phase)
            self.progress.emit(int((i+1)/self.steps * 100))
            
        self.finished_sweep.emit()
        
    def cancel(self):
        self.is_cancelled = True

class LockInAmplifierWidget(QWidget):
    def __init__(self, module: LockInAmplifier):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.setInterval(100) # 10Hz update

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # Tabs for Modes
        self.tabs = QTabWidget()
        
        # --- Tab 1: Manual Control (Existing) ---
        manual_widget = QWidget()
        manual_layout = QHBoxLayout(manual_widget)
        
        # --- Left Panel: Settings ---
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()
        
        # Output Controls
        self.toggle_btn = QPushButton("Start Output & Measure")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; font-weight: bold; padding: 10px; color: black; } QPushButton:checked { background-color: #ffcccc; }")
            
        settings_layout.addRow(self.toggle_btn)
        
        settings_layout.addRow(self.toggle_btn)
        
        # External Mode
        self.ext_mode_check = QCheckBox("External Mode (No Output)")
        self.ext_mode_check.toggled.connect(self.on_ext_mode_toggled)
        settings_layout.addRow(self.ext_mode_check)
        
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
        self.amp_spin.valueChanged.connect(self.on_amp_spin_changed)
        
        self.gen_unit_combo = QComboBox()
        self.gen_unit_combo.addItems(['Linear (0-1)', 'dBFS', 'dBV', 'dBu', 'Vrms', 'Vpeak'])
        self.gen_unit_combo.setCurrentText('dBFS')
        self.gen_unit_combo.currentTextChanged.connect(self.on_gen_unit_changed)
        
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(self.amp_spin)
        amp_layout.addWidget(self.gen_unit_combo)
        settings_layout.addRow("Amplitude:", amp_layout)
        
        self.out_ch_combo = QComboBox()
        self.out_ch_combo.addItems(["Left (Ch 1)", "Right (Ch 2)", "Stereo (Both)"])
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
        
        self.avg_spin = QSpinBox()
        self.avg_spin.setRange(1, 100)
        self.avg_spin.setValue(1)
        self.avg_spin.valueChanged.connect(lambda v: setattr(self.module, 'averaging_count', v))
        settings_layout.addRow("Averaging:", self.avg_spin)
        
        settings_group.setLayout(settings_layout)
        manual_layout.addWidget(settings_group, stretch=1)
        
        # --- Right Panel: Meters ---
        meters_group = QGroupBox("Measurements")
        meters_layout = QVBoxLayout()
        
        # Magnitude
        meters_layout.addWidget(QLabel("Magnitude"))
        
        # Unit Selection
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["dBFS", "dBV", "dBu", "V", "mV"])
        self.unit_combo.setCurrentText("dBFS")
        self.unit_combo.currentIndexChanged.connect(self.update_ui) # Update immediately
        meters_layout.addWidget(self.unit_combo)
        
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
        
        # X / Y
        xy_layout = QHBoxLayout()
        
        x_group = QVBoxLayout()
        x_group.addWidget(QLabel("X (In-phase)"))
        self.x_label = QLabel("0.000 V")
        self.x_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffff00;")
        self.x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        x_group.addWidget(self.x_label)
        xy_layout.addLayout(x_group)
        
        y_group = QVBoxLayout()
        y_group.addWidget(QLabel("Y (Quadrature)"))
        self.y_label = QLabel("0.000 V")
        self.y_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff00ff;")
        self.y_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        y_group.addWidget(self.y_label)
        xy_layout.addLayout(y_group)
        
        meters_layout.addLayout(xy_layout)
        
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
        manual_layout.addWidget(meters_group, stretch=2)
        
        self.tabs.addTab(manual_widget, "Manual Control")
        
        # --- Tab 2: Frequency Response Analyzer (FRA) ---
        fra_widget = QWidget()
        fra_layout = QHBoxLayout(fra_widget)
        
        # FRA Settings
        fra_settings_group = QGroupBox("Sweep Settings")
        fra_form = QFormLayout()
        
        self.fra_start_spin = QDoubleSpinBox()
        self.fra_start_spin.setRange(20, 20000); self.fra_start_spin.setValue(20); self.fra_start_spin.setSuffix(" Hz")
        fra_form.addRow("Start Freq:", self.fra_start_spin)
        
        self.fra_end_spin = QDoubleSpinBox()
        self.fra_end_spin.setRange(20, 20000); self.fra_end_spin.setValue(20000); self.fra_end_spin.setSuffix(" Hz")
        fra_form.addRow("End Freq:", self.fra_end_spin)
        
        self.fra_steps_spin = QSpinBox()
        self.fra_steps_spin.setRange(10, 1000); self.fra_steps_spin.setValue(50)
        fra_form.addRow("Steps:", self.fra_steps_spin)
        
        self.fra_log_check = QCheckBox("Log Sweep"); self.fra_log_check.setChecked(True)
        fra_form.addRow(self.fra_log_check)
        
        self.fra_settle_spin = QDoubleSpinBox()
        self.fra_settle_spin.setRange(0.1, 5.0); self.fra_settle_spin.setValue(0.5); self.fra_settle_spin.setSuffix(" s")
        fra_form.addRow("Settling Time:", self.fra_settle_spin)
        
        # Plot Unit Selector
        self.fra_plot_unit_combo = QComboBox()
        self.fra_plot_unit_combo.addItems(['dBFS', 'dBV', 'dBu', 'Vrms', 'Vpeak'])
        self.fra_plot_unit_combo.setCurrentText('dBFS')
        fra_form.addRow("Plot Unit:", self.fra_plot_unit_combo)
        
        self.fra_start_btn = QPushButton("Start Sweep")
        self.fra_start_btn.clicked.connect(self.on_fra_start)
        fra_form.addRow(self.fra_start_btn)
        
        self.fra_progress = QProgressBar()
        fra_form.addRow(self.fra_progress)
        
        fra_settings_group.setLayout(fra_form)
        fra_layout.addWidget(fra_settings_group, stretch=1)
        
        # FRA Plot
        self.fra_plot = pg.PlotWidget(title="Bode Plot")
        self.fra_plot.setLabel('bottom', "Frequency", units='Hz')
        self.fra_plot.setLabel('left', "Magnitude", units='dB')
        self.fra_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fra_plot.addLegend()
        
        # Custom Axis for Log Frequency
        axis = self.fra_plot.getPlotItem().getAxis('bottom')
        axis.setLogMode(False) # We will handle log data manually
        
        # Dual Axis for Phase
        self.fra_plot_p = pg.ViewBox()
        self.fra_plot.scene().addItem(self.fra_plot_p)
        self.fra_plot.getPlotItem().showAxis('right')
        self.fra_plot.getPlotItem().scene().addItem(self.fra_plot_p)
        self.fra_plot.getPlotItem().getAxis('right').linkToView(self.fra_plot_p)
        self.fra_plot_p.setXLink(self.fra_plot.getPlotItem())
        self.fra_plot.getPlotItem().getAxis('right').setLabel('Phase', units='deg')
        
        # Handle resizing
        def update_views():
            self.fra_plot_p.setGeometry(self.fra_plot.getPlotItem().vb.sceneBoundingRect())
            self.fra_plot_p.linkedViewChanged(self.fra_plot.getPlotItem().vb, self.fra_plot_p.XAxis)
        self.fra_plot.getPlotItem().vb.sigResized.connect(update_views)
        
        self.fra_curve_mag = self.fra_plot.plot(pen='g', name='Magnitude (dB)')
        self.fra_curve_phase = pg.PlotCurveItem(pen='c', name='Phase (deg)')
        self.fra_plot_p.addItem(self.fra_curve_phase)
        
        fra_layout.addWidget(self.fra_plot, stretch=3)
        
        self.tabs.addTab(fra_widget, "Frequency Response")
        
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
        
        # Data storage for FRA
        self.fra_freqs = []
        self.fra_log_freqs = []
        self.fra_mags = []
        self.fra_phases = []
        self.fra_worker = None

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

    def calculate_linear_amplitude(self, val, unit):
        gain = self.module.audio_engine.calibration.output_gain
        amp_linear = 0.0
        
        if unit == 'Linear (0-1)':
            amp_linear = val
        elif unit == 'dBFS':
            amp_linear = 10**(val/20)
        elif unit == 'dBV':
            # val = 20 * log10(Vrms)
            v_rms = 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_linear = v_peak / gain
        elif unit == 'dBu':
            # val = 20 * log10(Vrms / 0.7746)
            v_rms = 0.7746 * 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_linear = v_peak / gain
        elif unit == 'Vrms':
            v_peak = val * np.sqrt(2)
            amp_linear = v_peak / gain
        elif unit == 'Vpeak':
            amp_linear = val / gain
            
        # Clamp
        if amp_linear > 1.0: amp_linear = 1.0
        elif amp_linear < 0.0: amp_linear = 0.0
        
        return amp_linear

    def on_amp_spin_changed(self, val):
        unit = self.gen_unit_combo.currentText()
        self.module.gen_amplitude = self.calculate_linear_amplitude(val, unit)

    def on_gen_unit_changed(self, unit):
        self.update_amp_display()

    def update_amp_display(self):
        unit = self.gen_unit_combo.currentText()
        amp_0_1 = self.module.gen_amplitude
        gain = self.module.audio_engine.calibration.output_gain
        
        self.amp_spin.blockSignals(True)
        
        if unit == 'Linear (0-1)':
            self.amp_spin.setRange(0, 1.0)
            self.amp_spin.setSingleStep(0.1)
            self.amp_spin.setValue(amp_0_1)
        elif unit == 'dBFS':
            self.amp_spin.setRange(-120, 0)
            self.amp_spin.setSingleStep(1.0)
            val = 20 * np.log10(amp_0_1 + 1e-12)
            self.amp_spin.setValue(val)
        elif unit == 'dBV':
            v_peak = amp_0_1 * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10(v_rms + 1e-12)
            self.amp_spin.setRange(-120, 20)
            self.amp_spin.setSingleStep(1.0)
            self.amp_spin.setValue(val)
        elif unit == 'dBu':
            v_peak = amp_0_1 * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10((v_rms + 1e-12) / 0.7746)
            self.amp_spin.setRange(-120, 20)
            self.amp_spin.setSingleStep(1.0)
            self.amp_spin.setValue(val)
        elif unit == 'Vrms':
            v_peak = amp_0_1 * gain
            v_rms = v_peak / np.sqrt(2)
            self.amp_spin.setRange(0, 100)
            self.amp_spin.setSingleStep(0.1)
            self.amp_spin.setValue(v_rms)
        elif unit == 'Vpeak':
            v_peak = amp_0_1 * gain
            self.amp_spin.setRange(0, 100)
            self.amp_spin.setSingleStep(0.1)
            self.amp_spin.setValue(v_peak)
            
        self.amp_spin.blockSignals(False)

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

    def on_ext_mode_toggled(self, checked):
        self.module.external_mode = checked
        
        # Disable/Enable Generator Controls
        self.freq_spin.setEnabled(not checked)
        self.amp_spin.setEnabled(not checked)
        self.gen_unit_combo.setEnabled(not checked)
        self.out_ch_combo.setEnabled(not checked)

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
        mag_fs = self.module.current_magnitude
        phase = self.module.current_phase
        
        # Calculate Voltage
        sensitivity = self.module.audio_engine.calibration.input_sensitivity # Vpeak at 0dBFS
        v_peak = mag_fs * sensitivity
        v_rms = v_peak / np.sqrt(2)
        
        unit = self.unit_combo.currentText()
        
        if unit == "dBFS":
            if mag_fs > 0:
                val = 20 * np.log10(mag_fs + 1e-12)
                self.mag_label.setText(f"{val:.2f} dBFS")
            else:
                self.mag_label.setText("-inf dBFS")
            self.mag_db_label.setText("") # Clear secondary
            
        elif unit == "dBV":
            if v_rms > 0:
                val = 20 * np.log10(v_rms + 1e-12)
                self.mag_label.setText(f"{val:.2f} dBV")
            else:
                self.mag_label.setText("-inf dBV")
            self.mag_db_label.setText("")
            
        elif unit == "dBu":
            if v_rms > 0:
                val = 20 * np.log10((v_rms + 1e-12) / 0.7746)
                self.mag_label.setText(f"{val:.2f} dBu")
            else:
                self.mag_label.setText("-inf dBu")
            self.mag_db_label.setText("")
            
        elif unit == "V":
            self.mag_label.setText(f"{v_rms:.6f} V")
            # Show dBFS as secondary
            if mag_fs > 0:
                db = 20 * np.log10(mag_fs + 1e-12)
                self.mag_db_label.setText(f"{db:.2f} dBFS")
            else:
                self.mag_db_label.setText("-inf dBFS")
                
        elif unit == "mV":
            self.mag_label.setText(f"{v_rms * 1000:.3f} mV")
            # Show dBFS as secondary
            if mag_fs > 0:
                db = 20 * np.log10(mag_fs + 1e-12)
                self.mag_db_label.setText(f"{db:.2f} dBFS")
            else:
                self.mag_db_label.setText("-inf dBFS")
            
        self.phase_label.setText(f"{phase:.2f} deg")
        
        # Update X/Y (Always in Voltage for now, or follow unit?)
        # Let's follow unit logic for X/Y roughly, but X/Y are signed.
        # dB is not good for signed X/Y.
        # So we stick to V or mV if V/mV/dBV/dBu is selected, and FS if dBFS.
        
        x_fs = self.module.current_x
        y_fs = self.module.current_y
        
        x_v = x_fs * sensitivity / np.sqrt(2) # RMS
        y_v = y_fs * sensitivity / np.sqrt(2) # RMS
        
        if unit == "dBFS":
            self.x_label.setText(f"{x_fs:.6f} FS")
            self.y_label.setText(f"{y_fs:.6f} FS")
        elif unit == "mV":
            self.x_label.setText(f"{x_v * 1000:.3f} mV")
            self.y_label.setText(f"{y_v * 1000:.3f} mV")
        else: # V, dBV, dBu -> Show V
            self.x_label.setText(f"{x_v:.6f} V")
            self.y_label.setText(f"{y_v:.6f} V")
        
        # Update Ref Status
        ref_level = self.module.ref_level
        ref_freq = self.module.ref_freq
        
        if ref_level > -60:
            self.ref_status_label.setText(f"Locked ({ref_freq:.1f} Hz, {ref_level:.1f} dB)")
            self.ref_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        else:
            self.ref_status_label.setText("No Signal / Low Level")
            self.ref_status_label.setStyleSheet("font-weight: bold; color: #ff0000;")

    def on_fra_start(self):
        if self.fra_worker is not None and self.fra_worker.isRunning():
            self.fra_worker.cancel()
            self.fra_start_btn.setText("Stopping...")
            self.fra_start_btn.setEnabled(False)
            # Do not wait() here, let the finished signal handle cleanup
            return
            
        # Clear Data
        self.fra_freqs = []
        self.fra_log_freqs = []
        self.fra_mags = []
        self.fra_phases = []
        self.fra_curve_mag.setData([], [])
        self.fra_curve_phase.setData([], [])
        
        # Reset View (Force AutoRange)
        self.fra_plot.getPlotItem().enableAutoRange()
        self.fra_plot_p.enableAutoRange()
        
        # Start Worker
        start = self.fra_start_spin.value()
        end = self.fra_end_spin.value()
        steps = self.fra_steps_spin.value()
        log = self.fra_log_check.isChecked()
        settle = self.fra_settle_spin.value()
        
        # Force Apply Settings (Channel Routing)
        # This ensures settings are applied even if Manual mode wasn't run
        self.module.output_channel = self.out_ch_combo.currentIndex()
        self.module.signal_channel = self.sig_ch_combo.currentIndex()
        self.module.ref_channel = self.ref_ch_combo.currentIndex()
        
        # Set Amplitude (Use Manual Settings)
        amp_val = self.amp_spin.value()
        amp_unit = self.gen_unit_combo.currentText()
        self.module.gen_amplitude = self.calculate_linear_amplitude(amp_val, amp_unit)
        
        # Setup Axis Ticks
        if log:
            axis = self.fra_plot.getPlotItem().getAxis('bottom')
            # Generate ticks
            ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            ticks_log = [(np.log10(t), str(t) if t < 1000 else f"{t/1000:.0f}k") for t in ticks]
            axis.setTicks([ticks_log])
        else:
            self.fra_plot.getPlotItem().getAxis('bottom').setTicks(None) # Auto
        
        self.fra_worker = FRASweepWorker(self.module, start, end, steps, log, settle)
        self.fra_worker.progress.connect(self.fra_progress.setValue)
        self.fra_worker.result.connect(self.on_fra_result)
        self.fra_worker.finished_sweep.connect(self.on_fra_finished)
        
        self.fra_worker.start()
        self.fra_start_btn.setText("Stop Sweep")
        
    def on_fra_result(self, f, mag, phase):
        self.fra_freqs.append(f)
        
        # Log X
        if self.fra_log_check.isChecked():
            x_val = np.log10(f)
        else:
            x_val = f
        self.fra_log_freqs.append(x_val)
        
        # Convert Mag to Selected Unit
        unit = self.fra_plot_unit_combo.currentText()
        sensitivity = self.module.audio_engine.calibration.input_sensitivity
        
        # mag is Linear (0-1 relative to Full Scale)
        
        y_val = 0.0
        if unit == 'dBFS':
            y_val = 20 * np.log10(mag + 1e-12)
            self.fra_plot.setLabel('left', "Magnitude", units='dBFS')
        elif unit == 'dBV':
            v_peak = mag * sensitivity
            v_rms = v_peak / np.sqrt(2)
            y_val = 20 * np.log10(v_rms + 1e-12)
            self.fra_plot.setLabel('left', "Magnitude", units='dBV')
        elif unit == 'dBu':
            v_peak = mag * sensitivity
            v_rms = v_peak / np.sqrt(2)
            y_val = 20 * np.log10((v_rms + 1e-12) / 0.7746)
            self.fra_plot.setLabel('left', "Magnitude", units='dBu')
        elif unit == 'Vrms':
            v_peak = mag * sensitivity
            y_val = v_peak / np.sqrt(2)
            self.fra_plot.setLabel('left', "Magnitude", units='V')
        elif unit == 'Vpeak':
            y_val = mag * sensitivity
            self.fra_plot.setLabel('left', "Magnitude", units='V')
            
        self.fra_mags.append(y_val)
        self.fra_phases.append(phase)
        
        self.fra_curve_mag.setData(self.fra_log_freqs, self.fra_mags)
        self.fra_curve_phase.setData(self.fra_log_freqs, self.fra_phases)
        
        # Auto-scale Phase View
        self.fra_plot_p.autoRange()
        
    def on_fra_finished(self):
        self.fra_start_btn.setText("Start Sweep")
        self.fra_start_btn.setEnabled(True)
        self.fra_progress.setValue(100)
        self.fra_plot.getPlotItem().autoRange()
        self.fra_plot_p.autoRange()

    def apply_theme(self, theme_name):
        if theme_name == 'system' and hasattr(self.app, 'theme_manager'):
            theme_name = self.app.theme_manager.get_effective_theme()
            
        if theme_name == 'dark':
            # Dark Theme
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; border: 1px solid #555; border-radius: 4px; padding: 10px; font-weight: bold; }"
                "QPushButton:checked { background-color: #c62828; color: white; border: 1px solid #555; border-radius: 4px; padding: 10px; }"
                "QPushButton:hover { background-color: #388e3c; }"
                "QPushButton:checked:hover { background-color: #d32f2f; }"
            )
            self.mag_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #00ff00;")
            self.mag_db_label.setStyleSheet("font-size: 24px; color: #88ff88;")
            self.phase_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #00ffff;")
            self.x_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffff00;")
            self.y_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff00ff;")
            
            # Ref status depends on state, but we set base style here or update in update_ui
            # update_ui sets color explicitly, so we might need to update that logic too.
            # Ideally apply_theme just sets the current state's color if we track it, 
            # or we let update_ui handle it and just use theme-aware colors there.
            # But update_ui runs on timer.
            pass

        else:
            # Light Theme
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #ccffcc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 10px; font-weight: bold; }"
                "QPushButton:checked { background-color: #ffcccc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 10px; }"
                "QPushButton:hover { background-color: #bbfebb; }"
                "QPushButton:checked:hover { background-color: #ffbbbb; }"
            )
            self.mag_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #008800;")
            self.mag_db_label.setStyleSheet("font-size: 24px; color: #006600;")
            self.phase_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #008888;")
            self.x_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #888800;")
            self.y_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #880088;")

        # Theme handling
        self.app = QApplication.instance()
        if hasattr(self.app, 'theme_manager'):
            self.app.theme_manager.theme_changed.connect(self.apply_theme)
            self.apply_theme(self.app.theme_manager.get_current_theme())
