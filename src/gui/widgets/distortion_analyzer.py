import argparse
import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt, get_window, iirnotch, filtfilt
from scipy.optimize import minimize_scalar
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QFormLayout, 
                             QSpinBox, QDoubleSpinBox, QTabWidget, QStackedWidget, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from src.measurement_modules.base import MeasurementModule
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.analysis import AudioCalc



class DistortionAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 16384 # Larger buffer for better frequency resolution
        self.input_data = np.zeros(self.buffer_size)
        
        # Generator Settings
        self.gen_frequency = 1000.0
        self.gen_amplitude = 0.5 # Linear 0-1
        self.output_channel = 0 # 0: Left, 1: Right
        self.output_enabled = True
        
        # Analysis Settings
        self.window_type = 'blackmanharris' # Good for distortion
        self.averaging = 0.0
        
        # IMD Settings
        self.imd_standard = 'smpte' # 'smpte' or 'ccif'
        self.imd_f1 = 60.0
        self.imd_f2 = 7000.0
        self.imd_ratio = 4.0 # 4:1 for SMPTE
        
        # Playback state
        self._phase_f1 = 0.0
        self._phase_f2 = 0.0
        
        # State
        self.current_result = None
        self._avg_thdn = None
        
        # Capture State
        self.capture_requested = False
        self.capture_ready = False
        self.captured_buffer = None
        
        # Sweep State
        self.sweep_mode = False
        self.sweep_running = False
        self.sweep_results = []
        
        self.callback_id = None

    @property
    def name(self) -> str:
        return "Distortion Analyzer"

    @property
    def description(self) -> str:
        return "THD, THD+N, and SINAD measurements."

    def run(self, args: argparse.Namespace):
        print("Distortion Analyzer running from CLI (not implemented)")

    def get_widget(self):
        return DistortionAnalyzerWidget(self)

    def start_analysis(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.input_data = np.zeros(self.buffer_size)
        self.current_result = None
        
        sample_rate = self.audio_engine.sample_rate
        phase = 0
        
        def callback(indata, outdata, frames, time, status):
            nonlocal phase
            if status:
                print(status)
                
            # Generate Signal
            outdata.fill(0)
            if self.output_enabled:
                # Check mode via a flag or infer from params? 
                # Ideally we should have a mode flag in module, but currently it's in Widget.
                # Let's add a mode flag to module.
                if hasattr(self, 'mode') and self.mode == 'IMD Measurement':
                    sine_wave = self._generate_dual_tone(frames, sample_rate)
                else:
                    t = (np.arange(frames) + phase) / sample_rate
                    phase += frames
                    sine_wave = self.gen_amplitude * np.sin(2 * np.pi * self.gen_frequency * t)
                
                if self.output_channel == 0:
                    outdata[:, 0] = sine_wave
                elif self.output_channel == 1:
                    if outdata.shape[1] > 1:
                        outdata[:, 1] = sine_wave
            else:
                pass
            
            # Capture Input
            capture_ch = self.output_channel if self.output_enabled else 0
            
            if indata.shape[1] > capture_ch:
                new_data = indata[:, capture_ch]
            else:
                new_data = indata[:, 0]
                
            # Ring buffer update
            if len(new_data) > self.buffer_size:
                self.input_data[:] = new_data[-self.buffer_size:]
            else:
                self.input_data = np.roll(self.input_data, -len(new_data))
                self.input_data[-len(new_data):] = new_data
                
            # Handle Capture Request (Thread-safe copy)
            if self.capture_requested:
                self.captured_buffer = self.input_data.copy()
                self.capture_requested = False
                self.capture_ready = True

        self.callback_id = self.audio_engine.register_callback(callback)

    def _generate_dual_tone(self, frames, sample_rate):
        # Calculate amplitudes based on ratio
        # Total amplitude should not exceed self.gen_amplitude
        
        if self.imd_standard == 'smpte':
            # ratio = amp_f1 / amp_f2
            # amp_f2 * (ratio + 1) = self.gen_amplitude
            amp_f2 = self.gen_amplitude / (self.imd_ratio + 1)
            amp_f1 = amp_f2 * self.imd_ratio
        else: # CCIF
            # 1:1 ratio usually
            amp_f1 = self.gen_amplitude / 2
            amp_f2 = self.gen_amplitude / 2
            
        # Generate phases
        phase_inc_f1 = 2 * np.pi * self.imd_f1 / sample_rate
        phase_inc_f2 = 2 * np.pi * self.imd_f2 / sample_rate
        
        t = np.arange(frames)
        phases_f1 = self._phase_f1 + t * phase_inc_f1
        phases_f2 = self._phase_f2 + t * phase_inc_f2
        
        # Update state
        self._phase_f1 = (self._phase_f1 + frames * phase_inc_f1) % (2 * np.pi)
        self._phase_f2 = (self._phase_f2 + frames * phase_inc_f2) % (2 * np.pi)
        
        signal = amp_f1 * np.sin(phases_f1) + amp_f2 * np.sin(phases_f2)
        return signal

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False
            
    def request_capture(self):
        """Request a thread-safe capture of the current input buffer."""
        self.capture_ready = False
        self.capture_requested = True

class SweepWorker(QThread):
    result_ready = pyqtSignal(dict)
    finished = pyqtSignal()
    progress = pyqtSignal(int, int)

    def __init__(self, module, sweep_type, start, end, steps, duration_ms=1000):
        super().__init__()
        self.module = module
        self.sweep_type = sweep_type # 'frequency' or 'amplitude'
        self.start_val = start
        self.end_val = end
        self.steps = steps
        self.duration_ms = duration_ms
        self.is_running = True

    def run(self):
        # Generate steps
        if self.sweep_type == 'frequency':
            # Logarithmic sweep for frequency
            values = np.logspace(np.log10(self.start_val), np.log10(self.end_val), self.steps)
        else:
            # Linear sweep for amplitude (dB)
            values = np.linspace(self.start_val, self.end_val, self.steps)

        for i, val in enumerate(values):
            if not self.is_running:
                break
            
            # Set Generator
            if self.sweep_type == 'frequency':
                self.module.gen_frequency = val
            else:
                # val is dBFS, convert to linear
                self.module.gen_amplitude = 10**(val/20)
            
            # Wait for settling (Generator update + Audio Buffer Latency)
            # Ensure at least 300ms wait
            wait_time = max(300, self.duration_ms)
            self.msleep(wait_time)
            
            # Use safe capture
            self.module.request_capture()
            # Wait for capture
            timeout = 0
            while not self.module.capture_ready and timeout < 50: # 500ms timeout
                self.msleep(10)
                timeout += 1
            
            if self.module.capture_ready:
                data = self.module.captured_buffer
            else:
                data = self.module.input_data.copy() # Fallback
            
            sample_rate = self.module.audio_engine.sample_rate
            
            results = AudioCalc.analyze_harmonics(
                data, 
                self.module.gen_frequency, 
                self.module.window_type, 
                sample_rate
            )
            
            # Add sweep parameter to results
            results['sweep_param'] = val
            self.result_ready.emit(results)
            self.progress.emit(i + 1, self.steps)
            
        self.finished.emit()

    def stop(self):
        self.is_running = False


class DistortionAnalyzerWidget(QWidget):
    def __init__(self, module: DistortionAnalyzer):
        super().__init__()
        self.module = module
        self.sweep_worker = None
        self.module = module
        self.sweep_worker = None
        self.init_ui()
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_realtime_analysis)
        self.timer.setInterval(100) # 10Hz update

    def init_ui(self):
        layout = QHBoxLayout()
        
        # --- Left Panel: Controls & Meters ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        # 1. Mode Selection
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Real-time", "Frequency Sweep", "Amplitude Sweep", "IMD Measurement"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        left_panel.addWidget(mode_group)
        
        # 2. Generator/Sweep Controls (Stacked)
        self.controls_stack = QStackedWidget()
        
        # Page 1: Real-time Controls
        rt_widget = QWidget()
        rt_layout = QFormLayout()
        
        # Output Mode
        self.out_mode_combo = QComboBox()
        self.out_mode_combo.addItems(["Internal Sine", "Off (External Source)"])
        self.out_mode_combo.currentIndexChanged.connect(self.on_out_mode_changed)
        rt_layout.addRow("Output Mode:", self.out_mode_combo)
        
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000)
        self.freq_spin.setValue(1000)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(self.on_freq_changed)
        rt_layout.addRow("Frequency:", self.freq_spin)
        
        # Amplitude with Units
        amp_layout = QHBoxLayout()
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(-120, 20) # Allow positive for dBV/dBu
        self.amp_spin.setValue(-6)
        self.amp_spin.valueChanged.connect(self.on_amp_changed)
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['dBFS', 'dBV', 'dBu', 'Vrms'])
        self.unit_combo.currentTextChanged.connect(self.on_unit_changed)
        
        amp_layout.addWidget(self.amp_spin)
        amp_layout.addWidget(self.unit_combo)
        rt_layout.addRow("Amplitude:", amp_layout)
        
        rt_widget.setLayout(rt_layout)
        self.controls_stack.addWidget(rt_widget)
        
        # Page 2: Sweep Controls
        sweep_widget = QWidget()
        sweep_layout = QFormLayout()
        
        self.sweep_start_spin = QDoubleSpinBox()
        self.sweep_start_spin.setRange(-120, 20000)
        self.sweep_start_spin.setValue(20)
        sweep_layout.addRow("Start:", self.sweep_start_spin)
        
        self.sweep_end_spin = QDoubleSpinBox()
        self.sweep_end_spin.setRange(-120, 20000)
        self.sweep_end_spin.setValue(20000)
        sweep_layout.addRow("End:", self.sweep_end_spin)
        
        self.sweep_steps_spin = QSpinBox()
        self.sweep_steps_spin.setRange(2, 1000)
        self.sweep_steps_spin.setValue(30)
        sweep_layout.addRow("Steps:", self.sweep_steps_spin)
        
        sweep_widget.setLayout(sweep_layout)
        self.controls_stack.addWidget(sweep_widget)
        
        # Page 3: IMD Controls
        imd_widget = QWidget()
        imd_layout = QFormLayout()
        
        self.imd_std_combo = QComboBox()
        self.imd_std_combo.addItems(['smpte', 'ccif'])
        self.imd_std_combo.currentTextChanged.connect(self.on_imd_std_changed)
        imd_layout.addRow("Standard:", self.imd_std_combo)
        
        self.imd_f1_spin = QDoubleSpinBox()
        self.imd_f1_spin.setRange(10, 20000)
        self.imd_f1_spin.setValue(self.module.imd_f1)
        self.imd_f1_spin.valueChanged.connect(lambda v: setattr(self.module, 'imd_f1', v))
        imd_layout.addRow("Freq 1 (Hz):", self.imd_f1_spin)
        
        self.imd_f2_spin = QDoubleSpinBox()
        self.imd_f2_spin.setRange(10, 24000)
        self.imd_f2_spin.setValue(self.module.imd_f2)
        self.imd_f2_spin.valueChanged.connect(lambda v: setattr(self.module, 'imd_f2', v))
        imd_layout.addRow("Freq 2 (Hz):", self.imd_f2_spin)
        
        self.imd_ratio_spin = QDoubleSpinBox()
        self.imd_ratio_spin.setRange(1, 10)
        self.imd_ratio_spin.setValue(self.module.imd_ratio)
        self.imd_ratio_spin.valueChanged.connect(lambda v: setattr(self.module, 'imd_ratio', v))
        imd_layout.addRow("Ratio (F1:F2):", self.imd_ratio_spin)
        
        # Reuse Amplitude controls for IMD? Yes, but maybe duplicate for clarity or just switch back to RT page?
        # Better to have dedicated amplitude here or shared. 
        # For simplicity, let's add amplitude here too or rely on the fact that we can switch back.
        # But user wants to adjust while in IMD mode.
        # Let's add amplitude control here too.
        
        imd_amp_layout = QHBoxLayout()
        self.imd_amp_spin = QDoubleSpinBox()
        self.imd_amp_spin.setRange(-120, 0)
        self.imd_amp_spin.setValue(-6)
        self.imd_amp_spin.valueChanged.connect(self.on_imd_amp_changed)
        imd_amp_layout.addWidget(self.imd_amp_spin)
        imd_amp_layout.addWidget(QLabel("dBFS")) # Simplified for IMD
        imd_layout.addRow("Amplitude:", imd_amp_layout)
        
        imd_widget.setLayout(imd_layout)
        self.controls_stack.addWidget(imd_widget)
        
        left_panel.addWidget(self.controls_stack)
        
        # Common Controls
        common_group = QGroupBox("Settings")
        common_layout = QFormLayout()
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Left (Ch 1)", "Right (Ch 2)"])
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        common_layout.addRow("Output Ch:", self.channel_combo)
        common_group.setLayout(common_layout)
        left_panel.addWidget(common_group)
        
        # Action Buttons
        btn_layout = QVBoxLayout()
        self.action_btn = QPushButton("Start Measurement")
        self.action_btn.setCheckable(True)
        self.action_btn.clicked.connect(self.on_action)
        self.action_btn.setStyleSheet("QPushButton:checked { background-color: #ccffcc; }")
        btn_layout.addWidget(self.action_btn)
        
        self.action_btn.setStyleSheet("QPushButton:checked { background-color: #ccffcc; }")
        btn_layout.addWidget(self.action_btn)
        
        left_panel.addLayout(btn_layout)
        
        # 3. Meters (Real-time only)
        self.meters_group = QGroupBox("Measurements")
        meters_layout = QVBoxLayout()
        
        self.thdn_label = QLabel("-- %")
        self.thdn_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff5555;")
        self.thdn_db_label = QLabel("-- dB")
        meters_layout.addWidget(QLabel("THD+N:"))
        meters_layout.addWidget(self.thdn_label)
        meters_layout.addWidget(self.thdn_db_label)
        
        self.thd_label = QLabel("-- %")
        self.thd_label.setStyleSheet("font-size: 18px; color: #ffaa55;")
        meters_layout.addWidget(QLabel("THD:"))
        meters_layout.addWidget(self.thd_label)
        
        self.sinad_label = QLabel("-- dB")
        self.sinad_label.setStyleSheet("font-size: 18px; color: #55ffff;")
        meters_layout.addWidget(QLabel("SINAD:"))
        meters_layout.addWidget(self.sinad_label)
        
        # IMD Meter (Hidden by default)
        self.imd_label = QLabel("-- %")
        self.imd_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff55ff;")
        self.imd_db_label = QLabel("-- dB")
        self.imd_meter_widget = QWidget()
        imd_meter_layout = QVBoxLayout(self.imd_meter_widget)
        imd_meter_layout.setContentsMargins(0,0,0,0)
        imd_meter_layout.addWidget(QLabel("IMD:"))
        imd_meter_layout.addWidget(self.imd_label)
        imd_meter_layout.addWidget(self.imd_db_label)
        meters_layout.addWidget(self.imd_meter_widget)
        self.imd_meter_widget.setVisible(False)
        
        self.meters_group.setLayout(meters_layout)
        left_panel.addWidget(self.meters_group)
        
        left_panel.addStretch()
        layout.addLayout(left_panel, 1)
        
        # --- Right Panel: Plots & Tables ---
        right_panel = QVBoxLayout()
        
        self.tabs = QTabWidget()
        
        # Tab 1: Spectrum
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel('left', 'Amplitude', units='dBFS')
        self.spectrum_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.spectrum_plot.setLogMode(x=True, y=False)
        self.spectrum_plot.setYRange(-140, 0)
        self.spectrum_plot.setYRange(-140, 0)
        self.spectrum_plot.showGrid(x=True, y=True)
        
        # Custom Axis Ticks for Spectrum
        axis_spec = self.spectrum_plot.getPlotItem().getAxis('bottom')
        ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        ticks_log = [(np.log10(t), str(t) if t < 1000 else f"{t/1000:.0f}k") for t in ticks]
        axis_spec.setTicks([ticks_log])
        
        # Set Range (log domain)
        self.spectrum_plot.setXRange(np.log10(20), np.log10(20000))
        
        self.spectrum_curve = self.spectrum_plot.plot(pen='y')
        self.tabs.addTab(self.spectrum_plot, "Spectrum")
        
        # Tab 2: Harmonics Table
        self.harmonics_table = QTableWidget()
        self.harmonics_table.setColumnCount(4)
        self.harmonics_table.setHorizontalHeaderLabels(["Order", "Freq (Hz)", "Level (dBr)", "Level (Linear)"])
        self.harmonics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tabs.addTab(self.harmonics_table, "Harmonics")
        
        # Tab 3: Sweep Results
        self.sweep_plot = pg.PlotWidget()
        self.sweep_plot.setLabel('left', 'THD+N', units='dB')
        self.sweep_plot.setLabel('bottom', 'Frequency', units='Hz') # Dynamic label
        self.sweep_plot.setLogMode(x=True, y=False)
        self.sweep_plot.setLogMode(x=True, y=False)
        self.sweep_plot.showGrid(x=True, y=True)
        
        # Custom Axis Ticks for Sweep (Frequency Mode)
        # Note: If mode changes to Amplitude Sweep, we might need to reset this?
        # The user only requested "like Spectrum Analyzer", which implies Frequency domain.
        # We'll set it here, and handle mode changes if necessary.
        self.sweep_axis = self.sweep_plot.getPlotItem().getAxis('bottom')
        self.sweep_axis.setTicks([ticks_log])
        
        # Set Range (log domain) for Frequency Sweep default
        self.sweep_plot.setXRange(np.log10(20), np.log10(20000))
        
        self.sweep_curve = self.sweep_plot.plot(pen='c', symbol='o')
        self.tabs.addTab(self.sweep_plot, "Sweep Results")
        
        right_panel.addWidget(self.tabs)
        layout.addLayout(right_panel, 3)
        
        self.setLayout(layout)
        
        # Initial update
        self.on_unit_changed(self.unit_combo.currentText())

    def on_mode_changed(self, idx):
        mode = self.mode_combo.currentText()
        self.module.mode = mode # Update mode in module
        
        if mode == "Real-time":
            self.controls_stack.setCurrentIndex(0)
            self.meters_group.setVisible(True)
            self.set_meters_mode('thd')
            self.tabs.setCurrentIndex(0)
        elif mode == "IMD Measurement":
            self.controls_stack.setCurrentIndex(2)
            self.meters_group.setVisible(True)
            self.set_meters_mode('imd')
            self.tabs.setCurrentIndex(0)
            self.on_imd_std_changed(self.imd_std_combo.currentText()) # Init controls
        else:
            self.controls_stack.setCurrentIndex(1)
            self.meters_group.setVisible(False)
            self.tabs.setCurrentIndex(2)
            
            if mode == "Frequency Sweep":
                self.sweep_start_spin.setSuffix(" Hz")
                self.sweep_end_spin.setSuffix(" Hz")
                self.sweep_start_spin.setValue(20)
                self.sweep_end_spin.setValue(20000)
                self.sweep_plot.setLabel('bottom', 'Frequency', units='Hz')
                self.sweep_plot.setLogMode(x=True, y=False)
                # Restore custom ticks for frequency
                ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
                ticks_log = [(np.log10(t), str(t) if t < 1000 else f"{t/1000:.0f}k") for t in ticks]
                self.sweep_axis.setTicks([ticks_log])
                self.sweep_plot.setXRange(np.log10(20), np.log10(20000))
            else: # Amplitude Sweep
                self.sweep_start_spin.setSuffix(" dBFS")
                self.sweep_end_spin.setSuffix(" dBFS")
                self.sweep_start_spin.setValue(-60)
                self.sweep_end_spin.setValue(0)
                self.sweep_plot.setLabel('bottom', 'Amplitude', units='dBFS')
                self.sweep_plot.setLogMode(x=False, y=False)
                # Reset ticks to auto for amplitude
                self.sweep_axis.setTicks(None)

    def on_out_mode_changed(self, idx):
        if idx == 0: # Internal Sine
            self.module.output_enabled = True
            self.freq_spin.setEnabled(True)
            self.amp_spin.setEnabled(True)
            self.unit_combo.setEnabled(True)
        else: # Off
            self.module.output_enabled = False
            self.freq_spin.setEnabled(True) # Still need freq for analysis reference
            self.amp_spin.setEnabled(False)
            self.unit_combo.setEnabled(False)

    def on_unit_changed(self, unit):
        # Update spin box range/value based on current amplitude
        # Current amplitude is stored in module as Linear (0-1)
        # But we need to convert it.
        # Actually, let's just update the display value.
        
        amp_linear = self.module.gen_amplitude
        gain = self.module.audio_engine.calibration.output_gain
        
        self.amp_spin.blockSignals(True)
        
        if unit == 'dBFS':
            val = 20 * np.log10(amp_linear + 1e-12)
        elif unit == 'dBV':
            v_peak = amp_linear * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10(v_rms + 1e-12)
        elif unit == 'dBu':
            v_peak = amp_linear * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10((v_rms + 1e-12) / 0.7746)
        elif unit == 'Vrms':
            v_peak = amp_linear * gain
            val = v_peak / np.sqrt(2)
            
        self.amp_spin.setValue(val)
        self.amp_spin.blockSignals(False)

    def on_amp_changed(self, val):
        unit = self.unit_combo.currentText()
        gain = self.module.audio_engine.calibration.output_gain
        amp_linear = 0.0
        
        if unit == 'dBFS':
            amp_linear = 10**(val/20)
        elif unit == 'dBV':
            v_rms = 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_linear = v_peak / gain
        elif unit == 'dBu':
            v_rms = 0.7746 * 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_linear = v_peak / gain
        elif unit == 'Vrms':
            v_peak = val * np.sqrt(2)
            amp_linear = v_peak / gain
            
        # Clamp
        if amp_linear > 1.0:
            amp_linear = 1.0
        elif amp_linear < 0.0:
            amp_linear = 0.0
            
        self.module.gen_amplitude = amp_linear

    def on_action(self, checked):
        mode = self.mode_combo.currentText()
        if mode == "Real-time":
            self.on_toggle_realtime(checked)
        else:
            if checked:
                self.start_sweep(mode)
            else:
                self.stop_sweep()

    def on_toggle_realtime(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.action_btn.setText("Stop Measurement")
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.action_btn.setText("Start Measurement")

    def set_meters_mode(self, mode):
        if mode == 'thd':
            self.thdn_label.setVisible(True)
            self.thdn_db_label.setVisible(True)
            self.thd_label.setVisible(True)
            self.sinad_label.setVisible(True)
            self.imd_meter_widget.setVisible(False)
        else:
            self.thdn_label.setVisible(False)
            self.thdn_db_label.setVisible(False)
            self.thd_label.setVisible(False)
            self.sinad_label.setVisible(False)
            self.imd_meter_widget.setVisible(True)

    def on_imd_std_changed(self, text):
        self.module.imd_standard = text
        if text == 'smpte':
            self.module.imd_f1 = 60.0
            self.module.imd_f2 = 7000.0
            self.imd_ratio_spin.setEnabled(True)
        else: # ccif
            self.module.imd_f1 = 19000.0
            self.module.imd_f2 = 20000.0
            self.imd_ratio_spin.setEnabled(False)
            
        self.imd_f1_spin.setValue(self.module.imd_f1)
        self.imd_f2_spin.setValue(self.module.imd_f2)

    def on_imd_amp_changed(self, val):
        self.module.gen_amplitude = 10**(val/20)

    def start_sweep(self, mode):
        self.module.start_analysis() # Ensure audio is running
        self.action_btn.setText("Stop Sweep")
        self.module.sweep_results = []
        self.sweep_curve.setData([], [])
        
        sweep_type = 'frequency' if mode == "Frequency Sweep" else 'amplitude'
        start = self.sweep_start_spin.value()
        end = self.sweep_end_spin.value()
        steps = self.sweep_steps_spin.value()
        
        self.sweep_worker = SweepWorker(self.module, sweep_type, start, end, steps)
        self.sweep_worker.result_ready.connect(self.on_sweep_result)
        self.sweep_worker.finished.connect(self.on_sweep_finished)
        self.sweep_worker.start()

    def stop_sweep(self):
        if self.sweep_worker:
            self.sweep_worker.stop()
            self.sweep_worker.wait()
        self.module.stop_analysis()
        self.action_btn.setText("Start Measurement")
        self.action_btn.setChecked(False)

    def on_sweep_result(self, result):
        self.module.sweep_results.append(result)
        
        # Update Plot
        x_data = [r['sweep_param'] for r in self.module.sweep_results]
        y_data = [r['thdn_db'] for r in self.module.sweep_results]
        
        mode = self.mode_combo.currentText()
        if mode == "Frequency Sweep":
            x_plot = np.array(x_data)
        else:
            x_plot = np.array(x_data)
            
        self.sweep_curve.setData(x_plot, y_data)

    def on_sweep_finished(self):
        self.stop_sweep()

    def on_freq_changed(self, val):
        self.module.gen_frequency = val

    def on_channel_changed(self, idx):
        self.module.output_channel = idx

    def update_realtime_analysis(self):
        if not self.module.is_running:
            return
            
        data = self.module.input_data
        sample_rate = self.module.audio_engine.sample_rate
        
        # Perform Analysis
        # Perform Analysis
        if self.module.mode == "IMD Measurement":
            window = get_window(self.module.window_type, len(data))
            fft_data = np.fft.rfft(data * window)
            mag = np.abs(fft_data) * (2 / np.sum(window)) # Linear magnitude for IMD calc
            freqs = np.fft.rfftfreq(len(data), 1/sample_rate)
            
            if self.module.imd_standard == 'smpte':
                res = AudioCalc.calculate_imd_smpte(mag, freqs, self.module.imd_f1, self.module.imd_f2)
            else:
                res = AudioCalc.calculate_imd_ccif(mag, freqs, self.module.imd_f1, self.module.imd_f2)
                
            self.imd_label.setText(f"{res['imd']:.4f} %")
            self.imd_db_label.setText(f"{res['imd_db']:.2f} dB")
            
            # For plotting, convert to dBFS
            mag_db = 20 * np.log10(mag + 1e-12)
            self.spectrum_curve.setData(freqs[1:], mag_db[1:])
            
        else:
            results = AudioCalc.analyze_harmonics(
                data, 
                self.module.gen_frequency, 
                self.module.window_type, 
                sample_rate
            )
            self.module.current_result = results
            
            # Update Meters
            self.thdn_label.setText(f"{results['thdn_percent']:.4f} %")
            self.thdn_db_label.setText(f"{results['thdn_db']:.2f} dB")
            self.thd_label.setText(f"{results['thd_percent']:.4f} %")
            self.sinad_label.setText(f"{results['sinad_db']:.2f} dB")
            
            # Update Harmonics Table
            self.harmonics_table.setRowCount(len(results['harmonics']))
            for i, h in enumerate(results['harmonics']):
                self.harmonics_table.setItem(i, 0, QTableWidgetItem(str(h['order'])))
                self.harmonics_table.setItem(i, 1, QTableWidgetItem(f"{h['frequency']:.1f}"))
                self.harmonics_table.setItem(i, 2, QTableWidgetItem(f"{h['amplitude_dbr']:.2f}"))
                self.harmonics_table.setItem(i, 3, QTableWidgetItem(f"{h['amplitude_linear']:.6f}"))
            
            # Update Spectrum Plot
            window = get_window(self.module.window_type, len(data))
            fft_data = np.fft.rfft(data * window)
            mag = 20 * np.log10(np.abs(fft_data) / len(data) * 2 + 1e-12)
            freqs = np.fft.rfftfreq(len(data), 1/sample_rate)
            
            self.spectrum_curve.setData(freqs[1:], mag[1:])
