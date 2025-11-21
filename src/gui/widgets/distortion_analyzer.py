import argparse
import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt, get_window
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QTabWidget, QStackedWidget)
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
    def calculate_thdn(signal, sampling_rate, target_frequency, window_name='hanning', min_db=-140.0):
        if np.max(np.abs(signal)) == 0:
            return min_db
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        # Apply window
        window = get_window(window_name, len(signal))
        windowed_signal = signal * window
        
        # Notch out fundamental
        filtered_signal = AudioCalc.notch_filter(windowed_signal, sampling_rate, target_frequency)
        
        # Calculate RMS
        # Note: Windowing affects RMS, but we apply it to both, so ratio should be roughly preserved?
        # Actually, for THD+N, standard practice is often:
        # 1. Measure RMS of total signal (or fundamental)
        # 2. Notch fundamental
        # 3. Measure RMS of residue
        # Windowing before notch might smear the notch if not careful, but usually fine for steady state.
        # Legacy code applied window then notch.
        
        fundamental_rms = np.sqrt(np.mean(windowed_signal**2))
        filtered_rms = np.sqrt(np.mean(filtered_signal**2))
        
        if fundamental_rms == 0:
            return min_db
            
        thdn_value = 20 * np.log10(filtered_rms / fundamental_rms)
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
        # Handle DC and Nyquist if needed, but usually negligible for THD
        
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
        
        for i in range(2, 10): # 2nd to 9th harmonic
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
        thdn_db = AudioCalc.calculate_thdn(audio_data, sampling_rate, max_freq, window_name, min_db)
        thdn_percent = 10**(thdn_db/20) * 100
        
        sinad_db = -thdn_db
        
        return {
            'basic_wave': result,
            'harmonics': harmonic_results,
            'thd_percent': thd_percent,
            'thd_db': thd_db,
            'thdn_percent': thdn_percent,
            'thdn_db': thdn_db,
            'sinad_db': sinad_db
        }

class DistortionAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 16384 # Larger buffer for better frequency resolution
        self.input_data = np.zeros(self.buffer_size)
        
        # Generator Settings
        self.gen_frequency = 1000.0
        self.gen_amplitude_dbfs = -6.0
        self.output_channel = 0 # 0: Left, 1: Right
        
        # Analysis Settings
        self.window_type = 'blackmanharris' # Good for distortion
        self.averaging = 0.0
        
        # State
        self.current_result = None
        self._avg_thdn = None
        
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
        
        # Start Generator
        # We need to generate a sine wave continuously
        # AudioEngine doesn't have a built-in generator, we must provide it in the callback
        
        sample_rate = self.audio_engine.sample_rate
        phase = 0
        
        def callback(indata, outdata, frames, time, status):
            nonlocal phase
            if status:
                print(status)
                
            # Generate Sine
            t = (np.arange(frames) + phase) / sample_rate
            phase += frames
            
            amp_linear = 10**(self.gen_amplitude_dbfs/20)
            sine_wave = amp_linear * np.sin(2 * np.pi * self.gen_frequency * t)
            
            outdata.fill(0)
            if self.output_channel == 0:
                outdata[:, 0] = sine_wave
            elif self.output_channel == 1:
                if outdata.shape[1] > 1:
                    outdata[:, 1] = sine_wave
            
            # Capture Input
            # We analyze the input channel corresponding to the output (loopback) 
            # or let user select? For now assume loopback on same channel or user routed it.
            # Let's capture both and mix or select. 
            # Ideally we should have an input channel selector.
            # For now, let's use the same channel index as input if possible, or just mix.
            # But distortion is usually single channel.
            # Let's default to capturing the same channel index.
            
            if indata.shape[1] > self.output_channel:
                new_data = indata[:, self.output_channel]
            else:
                new_data = indata[:, 0]
                
            # Ring buffer
            if len(new_data) > self.buffer_size:
                self.input_data[:] = new_data[-self.buffer_size:]
            else:
                self.input_data = np.roll(self.input_data, -len(new_data))
                self.input_data[-len(new_data):] = new_data

        self.audio_engine.start_stream(callback, channels=2)

    def stop_analysis(self):
        if self.is_running:
            self.audio_engine.stop_stream()
            self.is_running = False

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
                self.module.gen_amplitude_dbfs = val
            
            # Wait for settling and measurement
            # We need to wait at least buffer fill time + some extra
            # Buffer is 16384 @ 48k ~= 340ms
            # duration_ms should be > 340ms
            self.msleep(self.duration_ms)
            
            # Perform Analysis on current buffer
            # Note: This is a bit racy if buffer isn't fully fresh, but with sufficient wait it's okay.
            # Ideally we'd sync with callback, but for now sleep is robust enough.
            
            data = self.module.input_data.copy()
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
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000)
        self.freq_spin.setValue(1000)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(self.on_freq_changed)
        rt_layout.addRow("Frequency:", self.freq_spin)
        
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(-120, 0)
        self.amp_spin.setValue(-6)
        self.amp_spin.setSuffix(" dBFS")
        self.amp_spin.valueChanged.connect(self.on_amp_changed)
        rt_layout.addRow("Amplitude:", self.amp_spin)
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
        
        # Action Button
        self.action_btn = QPushButton("Start Measurement")
        self.action_btn.setCheckable(True)
        self.action_btn.clicked.connect(self.on_action)
        self.action_btn.setStyleSheet("QPushButton:checked { background-color: #ccffcc; }")
        left_panel.addWidget(self.action_btn)
        
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
        
        # --- Right Panel: Plots ---
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
        
        # Tab 2: Sweep Results
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

    def on_mode_changed(self, idx):
        mode = self.mode_combo.currentText()
        if mode == "Real-time":
            self.controls_stack.setCurrentIndex(0)
            self.meters_group.setVisible(True)
            self.tabs.setCurrentIndex(0)
        else:
            self.controls_stack.setCurrentIndex(1)
            self.meters_group.setVisible(False)
            self.tabs.setCurrentIndex(1)
            
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
        
        # If Frequency Sweep, X is log. If Amplitude, X is linear.
        # PlotWidget LogMode handles the display, but we pass linear values usually?
        # Wait, if setLogMode(x=True), we pass log10(x) or linear x?
        # In SpectrumAnalyzer we did log10(x).
        # Let's check. If setLogMode(x=True), pyqtgraph expects log values?
        # "If True, the axis will be logarithmic. The data plotted must be logarithmic."
        # So yes, we must transform.
        
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

    def on_amp_changed(self, val):
        self.module.gen_amplitude_dbfs = val

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
        
        # Update Spectrum Plot
        window = get_window(self.module.window_type, len(data))
        fft_data = np.fft.rfft(data * window)
        mag = 20 * np.log10(np.abs(fft_data) / len(data) * 2 + 1e-12)
        freqs = np.fft.rfftfreq(len(data), 1/sample_rate)
        
        self.spectrum_curve.setData(np.log10(freqs[1:]+1e-12), mag[1:])
