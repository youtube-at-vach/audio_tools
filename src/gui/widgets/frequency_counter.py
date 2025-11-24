import numpy as np
import time
import pyqtgraph as pg
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
                             QComboBox, QGroupBox, QFormLayout, QFrame, QPushButton, QTabWidget,
                             QDialog, QMessageBox, QLineEdit)
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
        self.history_len = 2000 # Increased for Allan Plot
        self.freq_history = deque(maxlen=self.history_len)
        self.time_history = deque(maxlen=self.history_len)
        self.start_time = 0
        self.current_freq = 0.0
        self.current_amp_db = -100.0
        self.std_dev = 0.0
        self.allan_deviation = 0.0
        self.allan_taus = []
        self.allan_devs = []
        
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

    def set_update_interval(self, interval_ms):
        was_running = self.is_running
        if was_running:
            self.stop_analysis()
            
        self.update_interval_ms = interval_ms
        
        # Adjust buffer size to capture enough samples for the interval
        # Minimum buffer size for good FFT resolution is also a factor, 
        # but for sine fit we just need enough cycles.
        # Let's aim for exactly the interval length, or slightly more.
        # We need to know sample rate. If not running, guess 48000 or use current engine rate.
        sr = self.audio_engine.sample_rate
        if sr < 1000: sr = 48000 # Fallback
        
        # Calculate needed samples
        needed_samples = int(sr * interval_ms / 1000)
        
        # Ensure a minimum size (e.g. 8192 for fast updates)
        self.buffer_size = max(8192, needed_samples)
        
        if was_running:
            self.start_analysis()

    def process(self):
        if not self.is_running:
            return None
            
        # Ensure buffer is full enough for the requested interval?
        # With ring buffer, it's always "full" with something (zeros initially).
        
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
                
                # Apply Calibration
                cal_factor = self.audio_engine.calibration.frequency_calibration
                precise_freq *= cal_factor
                
                self.current_freq = precise_freq
                return precise_freq
            except:
                return coarse_freq
        else:
            return coarse_freq

    def calculate_stats(self):
        if len(self.freq_history) < 2:
            self.std_dev = 0.0
            self.allan_deviation = 0.0
            return

        data = np.array(self.freq_history)
        
        # Standard Deviation (Jitter)
        self.std_dev = np.std(data, ddof=1)
        
        # Allan Deviation (Tau = 1 sample)
        diffs = np.diff(data)
        self.allan_deviation = np.sqrt(0.5 * np.mean(diffs**2))

    def calculate_allan_plot_data(self):
        """
        Calculates Allan Deviation for multiple Tau values.
        Tau is in units of update_interval.
        """
        if len(self.freq_history) < 10:
            return [], []

        data = np.array(self.freq_history)
        n = len(data)
        
        taus = []
        devs = []
        
        # Calculate for Tau = 1, 2, 4, 8, ... up to N/2
        # m is the averaging factor (Tau = m * dt)
        
        max_m = n // 2
        m = 1
        while m <= max_m:
            # Create averaged data
            # We need non-overlapping averages of length m
            # But standard Allan Variance definition uses adjacent averages
            # Formula: sigma_y^2(tau) = 0.5 * < (y_{i+1} - y_i)^2 >
            # where y_i are averages over tau
            
            # Efficient implementation:
            # Reshape data to (N//m, m) and take mean along axis 1
            # This gives us the sequence of averages y_k
            
            num_samples = (n // m) * m
            if num_samples < 2 * m:
                break
                
            y = data[:num_samples].reshape(-1, m).mean(axis=1)
            
            if len(y) < 2:
                break
                
            diffs = np.diff(y)
            sigma = np.sqrt(0.5 * np.mean(diffs**2))
            
            tau_seconds = m * (self.update_interval_ms / 1000.0)
            taus.append(tau_seconds)
            devs.append(sigma)
            
            m *= 2
            
        self.allan_taus = taus
        self.allan_devs = devs
        self.allan_taus = taus
        self.allan_devs = devs
        return taus, devs

class FrequencyCalibrationDialog(QDialog):
    def __init__(self, module: FrequencyCounter, parent=None):
        super().__init__(parent)
        self.module = module
        self.setWindowTitle("Frequency Calibration")
        self.resize(400, 250)
        self.init_ui()
        
        # Measurement state
        self.measurements = []
        self.is_measuring = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_measure_tick)
        self.target_samples = 10 # Average over 10 samples
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("<b>Step 1:</b> Connect a known reference signal."))
        
        # Reference Input
        form = QFormLayout()
        self.ref_spin = QDoubleSpinBox()
        self.ref_spin.setRange(0, 100000000)
        self.ref_spin.setDecimals(6)
        self.ref_spin.setValue(1000.0)
        form.addRow("Reference Frequency (Hz):", self.ref_spin)
        layout.addLayout(form)
        
        layout.addWidget(QLabel("<b>Step 2:</b> Measure current frequency."))
        self.status_label = QLabel("Status: Idle")
        layout.addWidget(self.status_label)
        
        self.measure_btn = QPushButton("Measure & Calibrate")
        self.measure_btn.clicked.connect(self.start_measurement)
        layout.addWidget(self.measure_btn)
        
        # Current Factor
        curr_factor = self.module.audio_engine.calibration.frequency_calibration
        layout.addWidget(QLabel(f"Current Calibration Factor: {curr_factor:.8f}"))
        
        self.setLayout(layout)
        
    def start_measurement(self):
        self.measurements = []
        self.is_measuring = True
        self.measure_btn.setEnabled(False)
        self.status_label.setText("Status: Measuring... (0/10)")
        self.timer.start(int(self.module.update_interval_ms))
        
    def on_measure_tick(self):
        if not self.is_measuring:
            return
            
        # Get raw frequency (without calibration applied yet, or reverse it?)
        # The module.process() returns calibrated frequency if we changed the code.
        # But we want the RAW frequency to calculate the NEW factor.
        # So we should get the current_freq and divide by the OLD factor.
        
        # Wait, if we use process(), it updates current_freq.
        # Let's just use the latest value from module.
        
        calibrated_freq = self.module.current_freq
        current_factor = self.module.audio_engine.calibration.frequency_calibration
        
        if calibrated_freq <= 0:
            return # Wait for valid signal
            
        raw_freq = calibrated_freq / current_factor
        self.measurements.append(raw_freq)
        
        self.status_label.setText(f"Status: Measuring... ({len(self.measurements)}/{self.target_samples})")
        
        if len(self.measurements) >= self.target_samples:
            self.finish_calibration()
            
    def finish_calibration(self):
        self.is_measuring = False
        self.timer.stop()
        self.measure_btn.setEnabled(True)
        
        avg_raw = np.mean(self.measurements)
        target = self.ref_spin.value()
        
        if avg_raw < 1e-6:
            QMessageBox.warning(self, "Error", "Measured frequency is too low.")
            return
            
        new_factor = target / avg_raw
        
        ret = QMessageBox.question(self, "Confirm Calibration", 
                                   f"Average Raw Freq: {avg_raw:.6f} Hz\n"
                                   f"Target Freq: {target:.6f} Hz\n"
                                   f"New Factor: {new_factor:.8f}\n\n"
                                   "Apply this calibration?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                                   
        if ret == QMessageBox.StandardButton.Yes:
            self.module.audio_engine.calibration.set_frequency_calibration(new_factor)
            QMessageBox.information(self, "Success", "Calibration applied.")
            self.accept()
        else:
            self.status_label.setText("Status: Cancelled")

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

        # --- Stats Display ---
        stats_layout = QHBoxLayout()
        
        self.std_label = QLabel("Std Dev: -- Hz")
        self.std_label.setStyleSheet("color: #aaa; font-size: 14px;")
        stats_layout.addWidget(self.std_label)
        
        self.allan_label = QLabel("Allan Dev: -- Hz")
        self.allan_label.setStyleSheet("color: #aaa; font-size: 14px;")
        stats_layout.addWidget(self.allan_label)
        
        display_layout.addLayout(stats_layout)
        
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
        self.speed_combo.addItem("Fast (10Hz)", 100)
        self.speed_combo.addItem("Slow (2Hz)", 500)
        self.speed_combo.addItem("1 Sec (1Hz)", 1000)
        self.speed_combo.addItem("2 Sec (0.5Hz)", 2000)
        self.speed_combo.addItem("5 Sec (0.2Hz)", 5000)
        self.speed_combo.addItem("10 Sec (0.1Hz)", 10000)
        self.speed_combo.currentIndexChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_combo)
        controls_layout.addLayout(speed_layout)
        
        # Start/Stop
        self.run_btn = QPushButton("Start")
        self.run_btn.setCheckable(True)
        self.run_btn.clicked.connect(self.on_run_toggle)
        controls_layout.addWidget(self.run_btn)
        
        # Calibration
        self.cal_btn = QPushButton("Calibrate")
        self.cal_btn.clicked.connect(self.open_calibration)
        controls_layout.addWidget(self.cal_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # --- Tabs ---
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Tab 1: Frequency Drift
        self.plot_widget = pg.PlotWidget(title="Frequency Drift")
        self.plot_widget.setLabel('left', 'Frequency', units='Hz')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True)
        self.curve = self.plot_widget.plot(pen='y')
        self.tab_widget.addTab(self.plot_widget, "Frequency Drift")
        
        # Tab 2: Allan Deviation
        self.allan_plot = pg.PlotWidget(title="Allan Deviation")
        self.allan_plot.setLabel('left', 'Sigma_y(tau)')
        self.allan_plot.setLabel('bottom', 'Tau', units='s')
        self.allan_plot.showGrid(x=True, y=True)
        self.allan_plot.setLogMode(x=True, y=True)
        self.allan_curve = self.allan_plot.plot(pen='c', symbol='o', symbolSize=5)
        self.tab_widget.addTab(self.allan_plot, "Allan Deviation")
        
        self.setLayout(layout)

    def open_calibration(self):
        if not self.module.is_running:
            QMessageBox.warning(self, "Warning", "Please start the counter first.")
            return
            
        dlg = FrequencyCalibrationDialog(self.module, self)
        dlg.exec()

    def on_speed_changed(self, idx):
        interval_ms = self.speed_combo.currentData()
        if interval_ms is None:
            return
            
        self.module.set_update_interval(interval_ms)
        self.timer.setInterval(interval_ms)

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
            
            # Update Stats
            self.module.calculate_stats()
            self.std_label.setText(f"Std Dev: {self.module.std_dev:.5f} Hz")
            self.allan_label.setText(f"Allan Dev: {self.module.allan_deviation:.5f} Hz")
            
            # Update Plots based on visibility
            current_tab = self.tab_widget.currentIndex()
            
            if current_tab == 0: # Frequency Drift
                self.curve.setData(list(self.module.time_history), list(self.module.freq_history))
            
            elif current_tab == 1: # Allan Deviation
                # Update Allan Plot
                # For fast updates, limit to approx 2Hz (every 500ms) to save CPU
                # For slow updates (>= 1000ms), update every time
                should_update = self.module.update_interval_ms >= 1000 or (int(time.time() * 10) % 5 == 0)
                
                if len(self.module.freq_history) > 10 and should_update:
                    taus, devs = self.module.calculate_allan_plot_data()
                    if len(taus) > 0:
                        self.allan_curve.setData(taus, devs)
        else:
            self.freq_label.setText("---.----- Hz")
            self.std_label.setText("Std Dev: -- Hz")
            self.allan_label.setText("Allan Dev: -- Hz")
