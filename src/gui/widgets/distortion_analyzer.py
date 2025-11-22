import argparse
import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt, get_window
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QFormLayout, 
                             QSpinBox, QDoubleSpinBox, QTabWidget, QStackedWidget, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class AudioCalc:
    """
    Ported from legacy/audio_analyzer/audiocalc.py
    """
    @staticmethod
    def bandpass_filter(signal, sampling_rate, lowcut=20.0, highcut=20000.0):
        nyquist = 0.5 * sampling_rate
        # Ensure valid bounds
        lowcut = max(0.1, lowcut)
        highcut = min(nyquist - 1, highcut)
        if lowcut >= highcut:
            return signal
        sos = butter(8, [lowcut / nyquist, highcut / nyquist], btype='bandpass', output='sos')
        return sosfiltfilt(sos, signal)
    
    @staticmethod
    def notch_filter(signal, sampling_rate, target_frequency, quality_factor=30):
        nyquist = 0.5 * sampling_rate
        w0 = target_frequency / nyquist
        bandwidth = w0 / quality_factor
        sos = butter(2, [w0 - bandwidth/2, w0 + bandwidth/2], btype='bandstop', output='sos')
        return sosfiltfilt(sos, signal)
    
    @staticmethod
    def calculate_thdn_components(signal, sampling_rate, target_frequency, window_name='hanning'):
        """
        Returns (fundamental_rms, residue_rms) for THD+N calculation.
        """
        if np.max(np.abs(signal)) == 0:
            return 0.0, 0.0
        
        # Apply window
        window = get_window(window_name, len(signal))
        windowed_signal = signal * window
        
        # Notch out fundamental
        filtered_signal = AudioCalc.notch_filter(windowed_signal, sampling_rate, target_frequency)
        
        # Calculate RMS
        fundamental_rms = np.sqrt(np.mean(windowed_signal**2))
        residue_rms = np.sqrt(np.mean(filtered_signal**2))
        
        return fundamental_rms, residue_rms
    
    @staticmethod
    def calculate_thdn(signal, sampling_rate, target_frequency, window_name='hanning', min_db=-140.0):
        fund_rms, res_rms = AudioCalc.calculate_thdn_components(signal, sampling_rate, target_frequency, window_name)
        
        if fund_rms == 0:
            return min_db
            
        thdn_value = 20 * np.log10(res_rms / fund_rms + 1e-12)
        return thdn_value
    
    @staticmethod
    def analyze_harmonics(audio_data, fundamental_freq, window_name, sampling_rate, min_db=-140.0):
        window = get_window(window_name, len(audio_data))
        windowed_data = audio_data * window
        fft_result = np.fft.rfft(windowed_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1/sampling_rate)

        # Coherent gain correction
        coherent_gain = np.sum(window) / len(window)
        
        # Amplitude spectrum (Peak)
        # rfft returns N/2+1 bins. Magnitude is |X|/N * 2 (except DC and Nyquist)
        amplitude_spectrum = (np.abs(fft_result) / len(audio_data)) * 2 / coherent_gain
        
        # Find Fundamental Peak
        # Search near expected frequency
        search_window = 0.1 * fundamental_freq # +/- 10%
        idx_min = np.searchsorted(freqs, fundamental_freq - search_window)
        idx_max = np.searchsorted(freqs, fundamental_freq + search_window)
        if idx_max <= idx_min:
            idx_max = idx_min + 1
            
        # Find max in range
        if idx_max < len(amplitude_spectrum):
            subset = amplitude_spectrum[idx_min:idx_max]
            if len(subset) > 0:
                local_max_idx = np.argmax(subset)
                peak_idx = idx_min + local_max_idx
            else:
                peak_idx = np.argmin(np.abs(freqs - fundamental_freq))
        else:
            peak_idx = np.argmin(np.abs(freqs - fundamental_freq))
            
        max_freq = freqs[peak_idx]
        max_amplitude = amplitude_spectrum[peak_idx]
        amp_dbfs = 20 * np.log10(max_amplitude + 1e-12)
        
        result = {
            'frequency': max_freq,
            'amplitude_dbfs': amp_dbfs,
            'max_amplitude': max_amplitude
        }
        
        # Harmonics
        harmonic_results = []
        harmonic_amplitudes_linear = []
        
        # Up to 10th harmonic
        for i in range(2, 11): 
            harmonic_freq = max_freq * i
            if harmonic_freq >= sampling_rate / 2:
                break
                
            # Search near harmonic
            h_idx_min = np.searchsorted(freqs, harmonic_freq - search_window)
            h_idx_max = np.searchsorted(freqs, harmonic_freq + search_window)
            
            if h_idx_max < len(amplitude_spectrum) and h_idx_max > h_idx_min:
                subset = amplitude_spectrum[h_idx_min:h_idx_max]
                local_max_h = np.argmax(subset)
                h_peak_idx = h_idx_min + local_max_h
                
                h_amp = amplitude_spectrum[h_peak_idx]
                h_freq = freqs[h_peak_idx]
                
                relative_amp = h_amp / max_amplitude if max_amplitude > 0 else 0
                amp_db = 20 * np.log10(relative_amp + 1e-12)
                
                harmonic_results.append({
                    'order': i,
                    'frequency': h_freq,
                    'amplitude_dbr': amp_db,
                    'amplitude_linear': h_amp
                })
                harmonic_amplitudes_linear.append(h_amp)
            else:
                 harmonic_results.append({
                    'order': i,
                    'frequency': harmonic_freq,
                    'amplitude_dbr': min_db,
                    'amplitude_linear': 0
                })

        # THD Calculation
        # THD = sqrt(sum(harmonics^2)) / fundamental
        if max_amplitude > 0:
            thd_linear = np.sqrt(sum(a**2 for a in harmonic_amplitudes_linear)) / max_amplitude
            thd_percent = thd_linear * 100
            thd_db = 20 * np.log10(thd_linear + 1e-12)
        else:
            thd_percent = 0
            thd_db = min_db
            
        # THD+N Calculation (Time domain notch)
        fund_rms, res_rms = AudioCalc.calculate_thdn_components(audio_data, sampling_rate, max_freq, window_name)
        if fund_rms > 0:
            thdn_linear = res_rms / fund_rms
            thdn_db = 20 * np.log10(thdn_linear + 1e-12)
        else:
            thdn_linear = 0
            thdn_db = min_db
            
        thdn_percent = thdn_linear * 100
        sinad_db = -thdn_db
        
        return {
            'basic_wave': result,
            'harmonics': harmonic_results,
            'thd_percent': thd_percent,
            'thd_db': thd_db,
            'thdn_percent': thdn_percent,
            'thdn_db': thdn_db,
            'sinad_db': sinad_db,
            # Raw components for averaging
            'raw_fund_rms': fund_rms,
            'raw_res_rms': res_rms,
            'raw_harmonics': harmonic_amplitudes_linear,
            'raw_fund_amp': max_amplitude
        }

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
                
            # Generate Sine
            outdata.fill(0)
            if self.output_enabled:
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

        self.audio_engine.start_stream(callback, channels=2)

    def stop_analysis(self):
        if self.is_running:
            self.audio_engine.stop_stream()
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
            
            self.msleep(self.duration_ms)
            
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

class HighPrecisionWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)

    def __init__(self, module, duration_sec=3.0):
        super().__init__()
        self.module = module
        self.duration_sec = duration_sec
        self.is_running = True

    def run(self):
        sample_rate = self.module.audio_engine.sample_rate
        num_snapshots = int(self.duration_sec * 5) # 5 snapshots per second (every 200ms)
        
        # Accumulators for Power Averaging
        fund_power_acc = 0.0
        res_power_acc = 0.0
        harmonics_power_acc = []
        fund_amp_acc = 0.0 # For THD denominator
        
        count = 0
        
        for i in range(num_snapshots):
            if not self.is_running:
                break
                
            # Request Capture
            self.module.request_capture()
            
            # Wait for capture (poll)
            timeout = 0
            while not self.module.capture_ready and timeout < 50: # 500ms timeout
                self.msleep(10)
                timeout += 1
                
            if not self.module.capture_ready:
                continue # Skip if failed
                
            data = self.module.captured_buffer
            
            results = AudioCalc.analyze_harmonics(
                data, 
                self.module.gen_frequency, 
                self.module.window_type, 
                sample_rate
            )
            
            # Accumulate Powers (Mean Square)
            fund_power_acc += results['raw_fund_rms']**2
            res_power_acc += results['raw_res_rms']**2
            fund_amp_acc += results['raw_fund_amp'] # Peak amplitude linear
            
            # Harmonics
            raw_harmonics = results['raw_harmonics']
            if len(harmonics_power_acc) == 0:
                harmonics_power_acc = [h**2 for h in raw_harmonics]
            else:
                for j, h in enumerate(raw_harmonics):
                    if j < len(harmonics_power_acc):
                        harmonics_power_acc[j] += h**2
            
            count += 1
            self.progress.emit(int((i + 1) / num_snapshots * 100))
            
            # Wait a bit before next capture to get fresh data
            self.msleep(150) 

        if count > 0:
            # Calculate Averaged RMS
            avg_fund_rms = np.sqrt(fund_power_acc / count)
            avg_res_rms = np.sqrt(res_power_acc / count)
            
            # THD+N
            if avg_fund_rms > 0:
                thdn_linear = avg_res_rms / avg_fund_rms
                thdn_db = 20 * np.log10(thdn_linear + 1e-12)
            else:
                thdn_linear = 0
                thdn_db = -140.0
                
            thdn_percent = thdn_linear * 100
            sinad_db = -thdn_db
            
            # THD
            avg_fund_peak = fund_amp_acc / count
            
            avg_harmonics_power_sum = sum([p/count for p in harmonics_power_acc])
            avg_harmonics_rms = np.sqrt(avg_harmonics_power_sum) 
            
            if avg_fund_peak > 0:
                thd_linear = avg_harmonics_rms / avg_fund_peak
                thd_db = 20 * np.log10(thd_linear + 1e-12)
            else:
                thd_linear = 0
                thd_db = -140.0
                
            thd_percent = thd_linear * 100
            
            final_results = {
                'thd_percent': thd_percent,
                'thd_db': thd_db,
                'thdn_percent': thdn_percent,
                'thdn_db': thdn_db,
                'sinad_db': sinad_db
            }
            
            self.finished.emit(final_results)

    def stop(self):
        self.is_running = False

class DistortionAnalyzerWidget(QWidget):
    def __init__(self, module: DistortionAnalyzer):
        super().__init__()
        self.module = module
        self.sweep_worker = None
        self.hp_worker = None
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
        self.mode_combo.addItems(["Real-time", "Frequency Sweep", "Amplitude Sweep"])
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
        
        self.hp_btn = QPushButton("High Precision Measurement")
        self.hp_btn.clicked.connect(self.on_high_precision)
        btn_layout.addWidget(self.hp_btn)
        
        self.hp_progress = QProgressBar()
        self.hp_progress.setVisible(False)
        btn_layout.addWidget(self.hp_progress)
        
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
        self.spectrum_plot.showGrid(x=True, y=True)
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
        self.sweep_plot.showGrid(x=True, y=True)
        self.sweep_curve = self.sweep_plot.plot(pen='c', symbol='o')
        self.tabs.addTab(self.sweep_plot, "Sweep Results")
        
        right_panel.addWidget(self.tabs)
        layout.addLayout(right_panel, 3)
        
        self.setLayout(layout)
        
        # Initial update
        self.on_unit_changed(self.unit_combo.currentText())

    def on_mode_changed(self, idx):
        mode = self.mode_combo.currentText()
        if mode == "Real-time":
            self.controls_stack.setCurrentIndex(0)
            self.meters_group.setVisible(True)
            self.hp_btn.setVisible(True)
            self.tabs.setCurrentIndex(0)
        else:
            self.controls_stack.setCurrentIndex(1)
            self.meters_group.setVisible(False)
            self.hp_btn.setVisible(False)
            self.tabs.setCurrentIndex(2)
            
            if mode == "Frequency Sweep":
                self.sweep_start_spin.setSuffix(" Hz")
                self.sweep_end_spin.setSuffix(" Hz")
                self.sweep_start_spin.setValue(20)
                self.sweep_end_spin.setValue(20000)
                self.sweep_plot.setLabel('bottom', 'Frequency', units='Hz')
                self.sweep_plot.setLogMode(x=True, y=False)
            else: # Amplitude Sweep
                self.sweep_start_spin.setSuffix(" dBFS")
                self.sweep_end_spin.setSuffix(" dBFS")
                self.sweep_start_spin.setValue(-60)
                self.sweep_end_spin.setValue(0)
                self.sweep_plot.setLabel('bottom', 'Amplitude', units='dBFS')
                self.sweep_plot.setLogMode(x=False, y=False)

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

    def on_high_precision(self):
        self.module.start_analysis() # Ensure running
        self.hp_btn.setEnabled(False)
        self.action_btn.setEnabled(False)
        self.hp_progress.setVisible(True)
        self.hp_progress.setValue(0)
        
        self.hp_worker = HighPrecisionWorker(self.module)
        self.hp_worker.progress.connect(self.hp_progress.setValue)
        self.hp_worker.finished.connect(self.on_hp_finished)
        self.hp_worker.start()

    def on_hp_finished(self, results):
        self.hp_btn.setEnabled(True)
        self.action_btn.setEnabled(True)
        self.hp_progress.setVisible(False)
        
        # Update Meters with high precision results
        self.thdn_label.setText(f"{results['thdn_percent']:.4f} %")
        self.thdn_db_label.setText(f"{results['thdn_db']:.2f} dB")
        self.thd_label.setText(f"{results['thd_percent']:.4f} %")
        self.sinad_label.setText(f"{results['sinad_db']:.2f} dB")
        
        # Stop if it was only for HP
        if not self.action_btn.isChecked():
            self.module.stop_analysis()

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
            x_plot = np.log10(np.array(x_data) + 1e-12)
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
        
        self.spectrum_curve.setData(np.log10(freqs[1:]+1e-12), mag[1:])
