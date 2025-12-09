import argparse
import numpy as np
import scipy.signal
import time
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
                             QPushButton, QComboBox, QGroupBox, QFormLayout, QProgressBar, QCheckBox, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
import pyqtgraph as pg
import threading

from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class NetworkAnalyzerSignals(QObject):
    update_plot = pyqtSignal(float, float, float) # freq, mag_db, phase_deg
    sweep_finished = pyqtSignal()
    progress = pyqtSignal(int)
    latency_result = pyqtSignal(float)
    error = pyqtSignal(str)

class PlayRecSession:
    def __init__(self, audio_engine, output_data, input_channels=1):
        self.audio_engine = audio_engine
        self.output_data = output_data
        self.total_frames = len(output_data)
        self.input_channels = input_channels
        self.input_data = np.zeros((self.total_frames, input_channels), dtype=np.float32)
        self.current_frame = 0
        self.is_complete = False
        self.callback_id = None
        self.lock = threading.Lock()
        self.completion_event = threading.Event()
        self.error = None

    def start(self):
        self.callback_id = self.audio_engine.register_callback(self._callback)

    def stop(self):
        if self.callback_id is not None:
            self.audio_engine.unregister_callback(self.callback_id)
            self.callback_id = None

    def wait(self, timeout=None):
        return self.completion_event.wait(timeout)

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Stream status: {status}")

        with self.lock:
            if self.is_complete:
                outdata.fill(0)
                return

            remaining = self.total_frames - self.current_frame
            chunk = min(frames, remaining)
            
            # Output
            outdata[:chunk, :] = self.output_data[self.current_frame:self.current_frame+chunk, :]
            if chunk < frames:
                outdata[chunk:, :] = 0
                
            # Input
            if indata.shape[1] > 0:
                # Capture requested number of channels
                ch_to_copy = min(self.input_channels, indata.shape[1])
                self.input_data[self.current_frame:self.current_frame+chunk, :ch_to_copy] = indata[:chunk, :ch_to_copy]
            
            self.current_frame += chunk
            
            if self.current_frame >= self.total_frames:
                self.is_complete = True
                self.completion_event.set()

class SweepWorker(QThread):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.is_running = True

    def run(self):
        try:
            self.analyzer._execute_sweep(self)
        except Exception as e:
            self.analyzer.signals.error.emit(str(e))
        finally:
            self.analyzer.signals.sweep_finished.emit()

    def stop(self):
        self.is_running = False

class CalibrationWorker(QThread):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer

    def run(self):
        self.analyzer.calibrate_latency()

class FastSweepWorker(QThread):
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.is_running = True

    def run(self):
        try:
            self.analyzer._execute_fast_sweep(self)
        except Exception as e:
            self.analyzer.signals.error.emit(str(e))
        finally:
            self.analyzer.signals.sweep_finished.emit()

    def stop(self):
        self.is_running = False

class NetworkAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.signals = NetworkAnalyzerSignals()
        
        # Parameters
        self.start_freq = 20.0
        self.end_freq = 20000.0
        self.steps_per_octave = 12
        self.steps_per_octave = 12
        self.amplitude = 0.5
        self.gen_unit = 'Amplitude' # 'Amplitude', 'dBFS', 'dBV', 'dBu', 'Vrms', 'Vpeak'
        self.duration_per_step = 0.5
        self.latency_sec = 0.0
        self.num_averages = 1
        
        # Routing
        self.output_channel = 'STEREO' # 'L', 'R', 'STEREO'
        self.input_mode = 'L' # 'L', 'R', 'XFER'
        
        # Fast Sweep Parameters
        self.sweep_mode = "Stepped Sine" # or "Fast Chirp"
        self.chirp_duration = 1.0
        
        self.worker = None
        self.calibration_worker = None
        
        self.reference_trace = None

    @property
    def name(self) -> str:
        return "Network Analyzer"

    @property
    def description(self) -> str:
        return "Bode Plot (Gain & Phase) with XFER support"

    def run(self, args: argparse.Namespace):
        print("CLI not implemented")

    def get_widget(self):
        return NetworkAnalyzerWidget(self)

    def run_play_rec(self, output_data, input_channels=1):
        """
        Helper to run a play/record session.
        output_data: (N, 2) numpy array
        Returns: (N, input_channels) numpy array
        """
        session = PlayRecSession(self.audio_engine, output_data, input_channels)
        session.start()
        session.wait()
        session.stop()
        return session.input_data

    def get_output_amplitude(self):
        """Returns the linear amplitude (0-1) for signal generation."""
        # self.amplitude is already stored as linear amplitude (0-1) by the widget
        return max(0.0, min(1.0, self.amplitude))

    def calibrate_latency(self):
        """Measures loopback latency using a chirp signal."""
        sample_rate = self.audio_engine.sample_rate
        duration = 0.5
        
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        chirp = scipy.signal.chirp(t, f0=20, t1=duration, f1=10000, method='logarithmic')
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        chirp = scipy.signal.chirp(t, f0=20, t1=duration, f1=10000, method='logarithmic')
        chirp *= self.get_output_amplitude()
        
        try:
            out_data = np.zeros((len(chirp), 2), dtype=np.float32)
            out_data[:, 0] = chirp
            out_data[:, 1] = chirp
            
            print("Playing chirp for latency calibration...")
            
            # Always capture 1 channel for latency cal (assume Ch 0 loopback)
            rec_data = self.run_play_rec(out_data, input_channels=1)
            recorded = rec_data[:, 0]
            
            correlation = scipy.signal.correlate(recorded, chirp, mode='full')
            lags = scipy.signal.correlation_lags(len(recorded), len(chirp), mode='full')
            lag = lags[np.argmax(correlation)]
            
            latency_samples = lag
            self.latency_sec = latency_samples / sample_rate
            
            if self.latency_sec < 0:
                self.latency_sec = 0
                
            self.signals.latency_result.emit(self.latency_sec)
            print(f"Measured Latency: {self.latency_sec*1000:.2f} ms")
            
        except Exception as e:
            self.signals.error.emit(f"Calibration failed: {e}")

    def start_sweep(self):
        if self.worker and self.worker.isRunning():
            return
        if self.sweep_mode == "Fast Chirp":
            self.worker = FastSweepWorker(self)
        else:
            self.worker = SweepWorker(self)
        self.worker.start()

    def start_calibration(self):
        if self.calibration_worker and self.calibration_worker.isRunning():
            return
        self.calibration_worker = CalibrationWorker(self)
        self.calibration_worker.start()

    def stop_sweep(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()

    def _prepare_output_buffer(self, signal):
        """Prepares stereo output buffer based on routing."""
        out_data = np.zeros((len(signal), 2), dtype=np.float32)
        if self.output_channel in ['L', 'STEREO']:
            out_data[:, 0] = signal
        if self.output_channel in ['R', 'STEREO']:
            out_data[:, 1] = signal
        return out_data

    def _execute_fast_sweep(self, worker):
        sample_rate = self.audio_engine.sample_rate
        
        # 1. Generate Log Chirp
        num_samples = int(sample_rate * self.chirp_duration)
        t = np.linspace(0, self.chirp_duration, num_samples, endpoint=False)
        
        w1 = 2 * np.pi * self.start_freq
        w2 = 2 * np.pi * self.end_freq
        T = self.chirp_duration
        L = np.log(self.end_freq / self.start_freq)
        
        
        phase = (w1 * T / L) * (np.exp(t * L / T) - 1)
        chirp = self.get_output_amplitude() * np.sin(phase)
        
        window = scipy.signal.windows.tukey(num_samples, alpha=0.05)
        chirp *= window
        
        # 2. Generate Inverse Filter
        inv_envelope = np.exp(t * L / T)
        inv_filter = inv_envelope * np.sin(phase)
        inv_filter *= window
        inv_filter = np.flip(inv_filter)
        
        test_conv = scipy.signal.fftconvolve(chirp, inv_filter, mode='full')
        norm_factor = np.max(np.abs(test_conv))
        if norm_factor > 1e-9:
            inv_filter /= norm_factor
        
        # 3. Play and Record
        padding_sec = 1.0
        padding_samples = int(padding_sec * sample_rate)
        
        # Prepare output
        out_signal = np.concatenate([chirp, np.zeros(padding_samples)])
        out_data = self._prepare_output_buffer(out_signal)
        
        self.signals.progress.emit(10)
        if not worker.is_running: return
        
        # Determine input channels
        input_ch_count = 2 # Always capture stereo to avoid channel mapping issues
        
        rec_data = self.run_play_rec(out_data, input_channels=input_ch_count)
        
        self.signals.progress.emit(50)
        if not worker.is_running: return

        # 4. Process
        def get_ir(signal):
            return scipy.signal.fftconvolve(signal, inv_filter, mode='full')
            
        if self.input_mode == 'XFER':
            # XFER Mode: Ref = Ch0, Meas = Ch1
            # We assume Ch0 is Reference (e.g. Source Loopback) and Ch1 is Measurement (DUT)
            # Or user can physically patch it.
            # Standard XFER: H = Meas / Ref
            
            ref_sig = rec_data[:, 0]
            meas_sig = rec_data[:, 1]
            
            ir_ref = get_ir(ref_sig)
            ir_meas = get_ir(meas_sig)
            
            # Find peak in Ref to align
            peak_idx = np.argmax(np.abs(ir_ref))
            
            # Window both
            pre = int(0.01 * sample_rate)
            post = int(0.5 * sample_rate)
            start = max(0, peak_idx - pre)
            end = min(len(ir_ref), peak_idx + post)
            
            # Ensure same length
            len_win = end - start
            
            win_ref = ir_ref[start:end]
            win_meas = ir_meas[start:end] # Use same window for Meas
            
            H_ref = np.fft.rfft(win_ref)
            H_meas = np.fft.rfft(win_meas)
            freqs = np.fft.rfftfreq(len_win, d=1/sample_rate)
            
            # Transfer Function
            # Avoid div by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                H_xfer = H_meas / H_ref
                H_xfer = np.nan_to_num(H_xfer)
                
            # Mask
            mask = (freqs >= self.start_freq) & (freqs <= self.end_freq)
            valid_freqs = freqs[mask]
            valid_H = H_xfer[mask]
            
            mag_db = 20 * np.log10(np.abs(valid_H) + 1e-12)
            phase_rad = np.angle(valid_H)
            phase_rad = np.unwrap(phase_rad)
            phase_deg = np.degrees(phase_rad)
            phase_deg = (phase_deg + 180) % 360 - 180
            
        else:
            # Single Channel Mode
            ch_idx = 1 if self.input_mode == 'R' else 0
            # If we recorded 1 channel, it's at index 0
            if rec_data.shape[1] == 1:
                sig = rec_data[:, 0]
            else:
                sig = rec_data[:, ch_idx]
                
            ir = get_ir(sig)
            peak_idx = np.argmax(np.abs(ir))
            
            pre = int(0.01 * sample_rate)
            post = int(0.5 * sample_rate)
            start = max(0, peak_idx - pre)
            end = min(len(ir), peak_idx + post)
            
            ir_win = ir[start:end]
            H = np.fft.rfft(ir_win)
            freqs = np.fft.rfftfreq(len(ir_win), d=1/sample_rate)
            
            mask = (freqs >= self.start_freq) & (freqs <= self.end_freq)
            valid_freqs = freqs[mask]
            valid_H = H[mask]
            
            mag_db = 20 * np.log10(np.abs(valid_H) + 1e-12)
            
            # If not XFER, adjust for output amplitude to show Absolute Level
            if self.input_mode != 'XFER':
                mag_db += 20 * np.log10(self.get_output_amplitude() + 1e-12)
            phase_rad = np.angle(valid_H)
            phase_rad = np.unwrap(phase_rad)
            
            # Latency Comp
            delay_samples = peak_idx - start
            phase_rad += 2 * np.pi * valid_freqs * (delay_samples / sample_rate)
            
            phase_deg = np.degrees(phase_rad)
            phase_deg = (phase_deg + 180) % 360 - 180

        # Emit
        step = max(1, len(valid_freqs) // 500)
        for i in range(0, len(valid_freqs), step):
            if not worker.is_running: break
            self.signals.update_plot.emit(valid_freqs[i], mag_db[i], phase_deg[i])
            
        self.signals.progress.emit(100)

    def _execute_sweep(self, worker):
        sample_rate = self.audio_engine.sample_rate
        freqs = self._generate_log_freqs(self.start_freq, self.end_freq, self.steps_per_octave)
        total_steps = len(freqs)
        
        input_ch_count = 2 # Always capture stereo
        
        for i, freq in enumerate(freqs):
            if not worker.is_running: break
            
            avg_mag = 0.0
            avg_phase = 0.0 + 0.0j
            
            for _ in range(self.num_averages):
                if not worker.is_running: break

                num_samples = int(sample_rate * self.duration_per_step)
                num_samples = int(sample_rate * self.duration_per_step)
                t = np.arange(num_samples) / sample_rate
                tone = self.get_output_amplitude() * np.cos(2 * np.pi * freq * t)
                
                padding = int((self.latency_sec + 0.1) * sample_rate)
                out_signal = np.concatenate([tone, np.zeros(padding)])
                out_data = self._prepare_output_buffer(out_signal)
                
                rec_data = self.run_play_rec(out_data, input_channels=input_ch_count)
                
                start_idx = int(self.latency_sec * sample_rate)
                end_idx = start_idx + len(tone)
                if end_idx > len(rec_data): end_idx = len(rec_data)
                
                if self.input_mode == 'XFER':
                    # XFER Analysis
                    ref_seg = rec_data[start_idx:end_idx, 0]
                    meas_seg = rec_data[start_idx:end_idx, 1]
                    tone_seg = tone[:len(ref_seg)]
                    
                    mag_ref, phase_ref = self._analyze_tone(ref_seg, tone_seg, freq, sample_rate, comp_latency=False)
                    mag_meas, phase_meas = self._analyze_tone(meas_seg, tone_seg, freq, sample_rate, comp_latency=False)
                    
                    # H = Meas / Ref
                    mag_ratio = mag_meas / (mag_ref + 1e-12)
                    phase_diff = phase_meas - phase_ref
                    
                    avg_mag += mag_ratio
                    avg_phase += np.exp(1j * np.radians(phase_diff))
                    
                else:
                    # Single Channel
                    ch_idx = 1 if self.input_mode == 'R' else 0
                    if rec_data.shape[1] == 1: ch_idx = 0
                    
                    seg = rec_data[start_idx:end_idx, ch_idx]
                    tone_seg = tone[:len(seg)]
                    
                    mag, phase = self._analyze_tone(seg, tone_seg, freq, sample_rate, comp_latency=True)
                    
                    avg_mag += mag
                    avg_phase += np.exp(1j * np.radians(phase))
            
            avg_mag /= self.num_averages
            avg_phase /= self.num_averages
            
            final_mag_db = 20 * np.log10(avg_mag + 1e-12)
            final_phase_deg = np.degrees(np.angle(avg_phase))
            
            self.signals.update_plot.emit(freq, final_mag_db, final_phase_deg)
            self.signals.progress.emit(int((i + 1) / total_steps * 100))

    def _generate_log_freqs(self, start, end, steps_per_oct):
        if start >= end:
            return [start]
        freqs = []
        curr = start
        while curr < end:
            freqs.append(curr)
            curr *= 2 ** (1 / steps_per_oct)
        if not freqs or freqs[-1] < end:
            freqs.append(end)
        return freqs

    def _analyze_tone(self, recorded, reference, freq, sample_rate, comp_latency=True):
        window = scipy.signal.windows.hann(len(recorded))
        rec_windowed = recorded * window
        
        fft_rec = np.fft.rfft(rec_windowed)
        idx = np.argmax(np.abs(fft_rec))
        
        mag_linear = np.abs(fft_rec[idx]) * 2 / np.sum(window)
        phase_rec = np.angle(fft_rec[idx])
        
        if comp_latency:
            latency_samples = self.latency_sec * sample_rate
            fractional_sec = (latency_samples - int(latency_samples)) / sample_rate
            phase_delay_comp = 2 * np.pi * freq * fractional_sec
            phase_sys_rad = phase_rec + phase_delay_comp
        else:
            phase_sys_rad = phase_rec
        
        phase_sys_rad = (phase_sys_rad + np.pi) % (2 * np.pi) - np.pi
        return mag_linear, np.degrees(phase_sys_rad)

class NetworkAnalyzerWidget(QWidget):
    def __init__(self, module: NetworkAnalyzer):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.module.signals.update_plot.connect(self.update_plot)
        self.module.signals.sweep_finished.connect(self.on_sweep_finished)
        self.module.signals.progress.connect(self.progress_bar.setValue)
        self.module.signals.latency_result.connect(self.on_latency_result)
        self.module.signals.error.connect(self.on_error)
        
        self.freqs = []
        self.mags = []
        self.phases = []

    def init_ui(self):
        layout = QHBoxLayout()
        
        # Left Panel Container
        left_panel = QWidget()
        left_panel.setFixedWidth(360)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create Tab Widget
        tabs = QTabWidget()
        # tabs.setFixedWidth(340) # Removed fixed width from tabs
        
        # --- Tab 1: Settings ---
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        
        # Controls Group
        controls_group = QGroupBox(tr("Sweep Settings"))
        form = QFormLayout()
        
        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("Stepped Sine"), "Stepped Sine")
        self.mode_combo.addItem(tr("Fast Chirp"), "Fast Chirp")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        form.addRow(tr("Sweep Mode:"), self.mode_combo)
        
        # Routing
        self.out_combo = QComboBox()
        self.out_combo.addItem(tr("Left"), "L")
        self.out_combo.addItem(tr("Right"), "R")
        self.out_combo.addItem(tr("Stereo"), "STEREO")
        self.out_combo.setCurrentIndex(2)
        self.out_combo.currentIndexChanged.connect(self.on_routing_changed)
        form.addRow(tr("Output Ch:"), self.out_combo)
        
        self.in_combo = QComboBox()
        self.in_combo.addItem(tr("Left (Ch1)"), "L")
        self.in_combo.addItem(tr("Right (Ch2)"), "R")
        self.in_combo.addItem(tr("XFER (Ref=L, Meas=R)"), "XFER")
        self.in_combo.setCurrentIndex(0)
        self.in_combo.currentIndexChanged.connect(self.on_routing_changed)
        form.addRow(tr("Input Mode:"), self.in_combo)
        
        # Freqs
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(10, 20000); self.start_spin.setValue(20)
        self.start_spin.valueChanged.connect(lambda v: setattr(self.module, 'start_freq', v))
        form.addRow(tr("Start Freq:"), self.start_spin)
        
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(10, 24000); self.end_spin.setValue(20000)
        self.end_spin.valueChanged.connect(lambda v: setattr(self.module, 'end_freq', v))
        form.addRow(tr("End Freq:"), self.end_spin)
        
        self.steps_spin = QDoubleSpinBox() 
        self.steps_spin.setDecimals(0)
        self.steps_spin.setRange(1, 48); self.steps_spin.setValue(12)
        self.steps_spin.valueChanged.connect(lambda v: setattr(self.module, 'steps_per_octave', int(v)))
        self.steps_label = QLabel(tr("Steps/Octave:"))
        form.addRow(self.steps_label, self.steps_spin)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 60.0); self.duration_spin.setValue(1.0)
        self.duration_spin.valueChanged.connect(lambda v: setattr(self.module, 'chirp_duration', v))
        self.duration_label = QLabel(tr("Duration (s):"))
        form.addRow(self.duration_label, self.duration_spin)
        self.duration_label.hide(); self.duration_spin.hide()
        
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0, 1); self.amp_spin.setValue(0.5); self.amp_spin.setSingleStep(0.1)
        self.amp_spin.valueChanged.connect(self.on_amp_spin_changed)
        
        self.gen_unit_combo = QComboBox()
        self.gen_unit_combo.addItems(['Amplitude', 'dBFS', 'dBV', 'dBu', 'Vrms', 'Vpeak'])
        self.gen_unit_combo.currentTextChanged.connect(self.on_gen_unit_changed)
        
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(self.amp_spin)
        amp_layout.addWidget(self.gen_unit_combo)
        form.addRow(tr("Amplitude:"), amp_layout)
        
        self.avg_spin = QDoubleSpinBox()
        self.avg_spin.setDecimals(0)
        self.avg_spin.setRange(1, 10); self.avg_spin.setValue(1)
        self.avg_spin.valueChanged.connect(lambda v: setattr(self.module, 'num_averages', int(v)))
        form.addRow(tr("Averages:"), self.avg_spin)
        
        controls_group.setLayout(form)
        settings_layout.addWidget(controls_group)
        settings_layout.addStretch()
        settings_tab.setLayout(settings_layout)
        tabs.addTab(settings_tab, tr("Settings"))
        
        # --- Tab 2: Display ---
        display_tab = QWidget()
        display_layout = QVBoxLayout()
        
        display_group = QGroupBox(tr("Display Settings"))
        display_form = QFormLayout()
        
        # Limit Plot Freq (Max)
        self.limit_check = QCheckBox(tr("Limit Max"))
        self.limit_check.toggled.connect(self.refresh_plots)
        self.limit_spin = QDoubleSpinBox()
        self.limit_spin.setRange(10, 24000); self.limit_spin.setValue(20000)
        self.limit_spin.valueChanged.connect(self.refresh_plots)
        
        limit_layout = QHBoxLayout()
        limit_layout.addWidget(self.limit_check)
        limit_layout.addWidget(self.limit_spin)
        display_form.addRow(tr("Max Freq:"), limit_layout)

        # Limit Plot Freq (Min)
        self.min_limit_check = QCheckBox(tr("Limit Min"))
        self.min_limit_check.toggled.connect(self.refresh_plots)
        self.min_limit_spin = QDoubleSpinBox()
        self.min_limit_spin.setRange(10, 24000); self.min_limit_spin.setValue(20)
        self.min_limit_spin.valueChanged.connect(self.refresh_plots)
        
        min_limit_layout = QHBoxLayout()
        min_limit_layout.addWidget(self.min_limit_check)
        min_limit_layout.addWidget(self.min_limit_spin)
        display_form.addRow(tr("Min Freq:"), min_limit_layout)
        
        self.smooth_combo = QComboBox()
        self.smooth_combo.addItems([tr("Off"), tr("Light"), tr("Medium"), tr("Heavy")])
        self.smooth_combo.currentTextChanged.connect(self.refresh_plots)
        display_form.addRow(tr("Smoothing:"), self.smooth_combo)
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["dBFS", "dBV", "dBu", "Vrms", "Vpeak"])
        self.unit_combo.currentTextChanged.connect(self.refresh_plots)
        display_form.addRow(tr("Unit:"), self.unit_combo)
        
        self.gd_check = QCheckBox(tr("Show Group Delay"))
        self.gd_check.toggled.connect(self.refresh_plots)
        display_form.addRow(self.gd_check)
        
        display_group.setLayout(display_form)
        display_layout.addWidget(display_group)
        display_layout.addStretch()
        display_tab.setLayout(display_layout)
        tabs.addTab(display_tab, tr("Display"))
        
        # --- Tab 3: Calibration ---
        cal_tab = QWidget()
        cal_tab_layout = QVBoxLayout()
        
        # Latency
        lat_group = QGroupBox(tr("Latency"))
        lat_form = QFormLayout()
        self.lat_btn = QPushButton(tr("Calibrate Latency"))
        self.lat_btn.clicked.connect(self.calibrate)
        lat_form.addRow(self.lat_btn)
        self.lat_label = QLabel(tr("Latency: 0.00 ms"))
        lat_form.addRow(self.lat_label)
        lat_group.setLayout(lat_form)
        cal_tab_layout.addWidget(lat_group)
        
        # Reference
        cal_group = QGroupBox(tr("Reference Trace"))
        cal_layout = QFormLayout()
        self.store_ref_btn = QPushButton(tr("Store Reference"))
        self.store_ref_btn.clicked.connect(self.on_store_reference)
        cal_layout.addRow(self.store_ref_btn)
        self.clear_ref_btn = QPushButton(tr("Clear Reference"))
        self.clear_ref_btn.clicked.connect(self.on_clear_reference)
        cal_layout.addRow(self.clear_ref_btn)
        self.apply_ref_check = QCheckBox(tr("Apply Reference"))
        self.apply_ref_check.toggled.connect(self.on_apply_reference_changed)
        cal_layout.addRow(self.apply_ref_check)
        cal_group.setLayout(cal_layout)
        cal_tab_layout.addWidget(cal_group)
        
        cal_tab_layout.addStretch()
        cal_tab.setLayout(cal_tab_layout)
        tabs.addTab(cal_tab, tr("Calibration"))
        
        # Add tabs to left layout
        left_layout.addWidget(tabs)
        
        # Buttons
        self.start_btn = QPushButton(tr("Start Sweep"))
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.on_start_stop)
        self.start_btn.setFixedHeight(40)
        left_layout.addWidget(self.start_btn)
        
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        layout.addWidget(left_panel)
        
        # Plots
        plot_layout = QVBoxLayout()
        self.mag_plot = pg.PlotWidget(title=tr("Magnitude Response"))
        self.mag_plot.setLabel('left', tr('Magnitude'), units='dB')
        self.mag_plot.setLabel('bottom', tr('Frequency'), units='Hz')
        self.mag_plot.setLogMode(x=True, y=False)
        self.mag_plot.showGrid(x=True, y=True)
        self.mag_curve = self.mag_plot.plot(pen='g')
        plot_layout.addWidget(self.mag_plot)
        
        self.phase_plot = pg.PlotWidget(title=tr("Phase Response"))
        self.phase_plot.setLabel('left', tr('Phase'), units='deg')
        self.phase_plot.setLabel('bottom', tr('Frequency'), units='Hz')
        self.phase_plot.setLogMode(x=True, y=False)
        self.phase_plot.showGrid(x=True, y=True)
        self.phase_curve = self.phase_plot.plot(pen='y')
        
        # Group Delay Axis (Right)
        self.gd_axis = pg.AxisItem('right')
        self.gd_axis.setLabel(tr('Group Delay'), units='ms')
        self.phase_plot.plotItem.layout.addItem(self.gd_axis, 2, 3)
        
        self.gd_view = pg.ViewBox()
        self.gd_axis.linkToView(self.gd_view)
        self.phase_plot.plotItem.scene().addItem(self.gd_view)
        self.gd_view.setXLink(self.phase_plot.plotItem.vb)
        
        # Disable log mode for the overlay view (we will manually log the data)
        self.gd_view.setLogMode(False, False)
        
        self.gd_curve = pg.PlotCurveItem(pen='r')
        self.gd_view.addItem(self.gd_curve)
        
        # 同期処理
        self.phase_plot.plotItem.vb.sigResized.connect(self.update_gd_views)
        
        plot_layout.addWidget(self.phase_plot)
        
        layout.addLayout(plot_layout)
        self.setLayout(layout)

    def on_mode_changed(self, index):
        mode = self.mode_combo.itemData(index)
        self.module.sweep_mode = mode
        if mode == "Stepped Sine":
            self.steps_label.show(); self.steps_spin.show()
            self.duration_label.hide(); self.duration_spin.hide()
        else:
            self.steps_label.hide(); self.steps_spin.hide()
            self.duration_label.show(); self.duration_spin.show()

    def on_routing_changed(self, index):
        self.module.output_channel = self.out_combo.currentData()
        self.module.input_mode = self.in_combo.currentData()
        
        # Update UI hints
        if self.module.input_mode == 'XFER':
            self.mag_plot.setTitle(tr("Transfer Function (Meas / Ref)"))
            self.unit_combo.setEnabled(False) # XFER is always relative dB
        else:
            self.mag_plot.setTitle(tr("Magnitude Response"))
            self.unit_combo.setEnabled(True)

    def on_gen_unit_changed(self, unit):
        self.module.gen_unit = unit
        # Update display to show current amplitude in new unit
        self.update_amp_display_value(self.module.amplitude)

    def update_amp_display_value(self, amp_0_1):
        unit = self.gen_unit_combo.currentText()
        try:
            gain = self.module.audio_engine.calibration.output_gain
        except:
            gain = 1.0
        
        self.amp_spin.blockSignals(True)
        
        if unit == 'Amplitude':
            self.amp_spin.setRange(0, 1.0)
            self.amp_spin.setSingleStep(0.1)
            self.amp_spin.setSuffix("")
            self.amp_spin.setValue(amp_0_1)
        elif unit == 'dBFS':
            self.amp_spin.setRange(-120, 0)
            self.amp_spin.setSingleStep(1.0)
            self.amp_spin.setSuffix(" dB")
            val = 20 * np.log10(amp_0_1 + 1e-12)
            self.amp_spin.setValue(val)
        elif unit == 'dBV':
            v_peak = amp_0_1 * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10(v_rms + 1e-12)
            self.amp_spin.setRange(-120, 20)
            self.amp_spin.setSingleStep(1.0)
            self.amp_spin.setSuffix(" dB")
            self.amp_spin.setValue(val)
        elif unit == 'dBu':
            v_peak = amp_0_1 * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10((v_rms + 1e-12) / 0.7746)
            self.amp_spin.setRange(-120, 20)
            self.amp_spin.setSingleStep(1.0)
            self.amp_spin.setSuffix(" dB")
            self.amp_spin.setValue(val)
        elif unit == 'Vrms':
            v_peak = amp_0_1 * gain
            v_rms = v_peak / np.sqrt(2)
            self.amp_spin.setRange(0, 100)
            self.amp_spin.setSingleStep(0.1)
            self.amp_spin.setSuffix(" V")
            self.amp_spin.setValue(v_rms)
        elif unit == 'Vpeak':
            v_peak = amp_0_1 * gain
            self.amp_spin.setRange(0, 100)
            self.amp_spin.setSingleStep(0.1)
            self.amp_spin.setSuffix(" V")
            self.amp_spin.setValue(v_peak)
            
        self.amp_spin.blockSignals(False)

    def on_amp_spin_changed(self, val):
        unit = self.gen_unit_combo.currentText()
        try:
            gain = self.module.audio_engine.calibration.output_gain
        except:
            gain = 1.0
            
        amp_0_1 = 0.0
        
        if unit == 'Amplitude':
            amp_0_1 = val
        elif unit == 'dBFS':
            amp_0_1 = 10**(val/20)
        elif unit == 'dBV':
            v_rms = 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_0_1 = v_peak / gain
        elif unit == 'dBu':
            v_rms = 0.7746 * 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_0_1 = v_peak / gain
        elif unit == 'Vrms':
            v_peak = val * np.sqrt(2)
            amp_0_1 = v_peak / gain
        elif unit == 'Vpeak':
            amp_0_1 = val / gain
            
        if amp_0_1 > 1.0: amp_0_1 = 1.0
        elif amp_0_1 < 0.0: amp_0_1 = 0.0
            
        self.module.amplitude = amp_0_1

    def calibrate(self):
        self.lat_btn.setEnabled(False)
        self.lat_label.setText(tr("Calibrating..."))
        self.module.start_calibration()

    def on_latency_result(self, lat):
        self.lat_label.setText(tr("Latency: {0:.2f} ms").format(lat*1000))
        self.lat_btn.setEnabled(True)
    
    def on_error(self, msg):
        print(f"Error: {msg}")
        self.start_btn.setChecked(False)
        self.start_btn.setText(tr("Start Sweep"))

    def on_store_reference(self):
        if not self.freqs: return
        self.module.reference_trace = {
            'freqs': np.array(self.freqs),
            'mags': np.array(self.mags),
            'phases': np.array(self.phases)
        }
        print("Reference trace stored.")

    def on_clear_reference(self):
        self.module.reference_trace = None
        self.refresh_plots()

    def on_apply_reference_changed(self, checked):
        self.refresh_plots()

    def on_start_stop(self, checked):
        if checked:
            self.freqs = []
            self.mags = []
            self.phases = []
            self.mag_curve.setData([], [])
            self.phase_curve.setData([], [])
            self.gd_curve.setData([], [])
            self.start_btn.setText(tr("Stop Sweep"))
            self.module.start_sweep()
        else:
            self.module.stop_sweep()
            self.start_btn.setText(tr("Start Sweep"))

    def on_sweep_finished(self):
        self.start_btn.setChecked(False)
        self.start_btn.setText(tr("Start Sweep"))

    def update_gd_views(self):
        # Keep the GD view aligned with the main view
        self.gd_view.setGeometry(self.phase_plot.plotItem.vb.sceneBoundingRect())

    def update_plot(self, freq, mag, phase):
        self.freqs.append(freq)
        self.mags.append(mag)
        self.phases.append(phase)
        self.refresh_plots()

    def refresh_plots(self):
        if not self.freqs: return
        
        smooth_mode = self.smooth_combo.currentText()
        unit = self.unit_combo.currentText()
        
        # Filter data if limit is enabled
        freqs_arr = np.array(self.freqs)
        mags_arr = np.array(self.mags)
        phases_arr = np.array(self.phases)
        
        # Create mask for filtering
        mask = np.ones(len(freqs_arr), dtype=bool)
        
        if self.limit_check.isChecked():
            limit = self.limit_spin.value()
            mask &= (freqs_arr <= limit)
            
        if self.min_limit_check.isChecked():
            min_limit = self.min_limit_spin.value()
            mask &= (freqs_arr >= min_limit)
            
        freqs_to_plot = freqs_arr[mask]
        mags_to_plot = mags_arr[mask]
        phases_to_plot = phases_arr[mask]
            
        if len(freqs_to_plot) == 0: return
        
        if self.module.input_mode == 'XFER':
            # XFER is already in dB relative
            y_values = mags_to_plot
            self.mag_plot.setLabel('left', tr('Gain'), units='dB')
        else:
            # Standard conversion logic (same as before)
            mags_linear = 10 ** (mags_to_plot / 20)
            try:
                input_sensitivity = self.module.audio_engine.calibration.input_sensitivity
            except:
                input_sensitivity = 1.0
                
            if unit == "dBFS":
                y_values = mags_to_plot
                self.mag_plot.setLabel('left', tr('Magnitude'), units='dBFS')
            elif unit == "dBV":
                v_peak = mags_linear * input_sensitivity
                v_rms = v_peak / np.sqrt(2)
                y_values = 20 * np.log10(v_rms + 1e-12)
                self.mag_plot.setLabel('left', tr('Magnitude'), units='dBV')
            elif unit == "dBu":
                v_peak = mags_linear * input_sensitivity
                v_rms = v_peak / np.sqrt(2)
                y_values = 20 * np.log10((v_rms + 1e-12) / 0.7746)
                self.mag_plot.setLabel('left', tr('Magnitude'), units='dBu')
            elif unit == "Vrms":
                v_peak = mags_linear * input_sensitivity
                y_values = v_peak / np.sqrt(2)
                self.mag_plot.setLabel('left', tr('Magnitude'), units='V')
            elif unit == "Vpeak":
                y_values = mags_linear * input_sensitivity
                self.mag_plot.setLabel('left', tr('Magnitude'), units='V')
            else:
                y_values = mags_to_plot
        
        # Apply Reference
        if self.apply_ref_check.isChecked() and self.module.reference_trace is not None:
            ref = self.module.reference_trace
            if len(ref['freqs']) > 1:
                interp_mags = np.interp(freqs_to_plot, ref['freqs'], ref['mags'])
                
                # If XFER, just subtract dB
                if self.module.input_mode == 'XFER':
                    y_values -= interp_mags
                else:
                    # If not XFER, we need to handle units carefully
                    # But usually reference is stored in dBFS (base unit)
                    # So if we are displaying dB, we subtract.
                    # If linear, we divide.
                    if "dB" in unit:
                        # We need to convert ref to target unit first?
                        # Actually, if we store ref in dBFS, and current is in dBV,
                        # Ref in dBV would be Ref_dBFS + Offset.
                        # Current in dBV is Curr_dBFS + Offset.
                        # Diff is Curr_dBFS - Ref_dBFS.
                        # So simple subtraction works for dB units.
                        y_values -= interp_mags
                    else:
                        # Linear
                        ref_linear = 10 ** (interp_mags / 20)
                        # Scale ref to unit?
                        # Ratio = Curr_Linear / Ref_Linear
                        # This is unitless.
                        # So we display Ratio? Or normalized V?
                        # Standard practice: Normalized Magnitude (Unitless or %)
                        y_values /= (ref_linear + 1e-12)
                        
            # Phase Subtraction
            if len(ref['phases']) > 1:
                interp_phases = np.interp(freqs_to_plot, ref['freqs'], ref['phases'])
                phases_to_plot -= interp_phases
                # Wrap to [-180, 180]
                phases_to_plot = (phases_to_plot + 180) % 360 - 180

        self.mag_curve.setData(freqs_to_plot, y_values)
        self.phase_curve.setData(freqs_to_plot, phases_to_plot)
        
        # Group Delay Calculation
        if self.gd_check.isChecked() and len(freqs_to_plot) > 1:
            self.gd_axis.show()
            
            # Unwrap phase (in degrees) -> radians
            # Note: phases_to_plot might be wrapped to [-180, 180] or relative
            # We should use the raw accumulated phase for GD if possible, 
            # but self.phases stores wrapped phase usually?
            # self.phases stores what update_plot sends.
            # In _analyze_tone, it returns degrees in [-180, 180].
            # So we need to unwrap here.
            
            # Use the raw phases from self.phases, not the potentially modified phases_to_plot
            # But we need to filter them too if we are filtering
            # Actually phases_to_plot IS filtered above.
            # But wait, the original code used self.phases (raw) here.
            # If we want raw phases but filtered, we should use the filtered raw phases.
            # Let's re-extract raw filtered phases if needed, or just use phases_to_plot if that's what we want.
            # The comment says "Use the raw phases from self.phases".
            # So we should use the filtered version of self.phases.
            # We already have phases_to_plot which is filtered self.phases (before ref subtraction).
            # Wait, phases_to_plot is modified by ref subtraction in lines 925.
            # So we need a clean filtered raw phase.
            
            # If reference is applied, we should probably calculate GD of the *corrected* phase?
            # Yes, Group Delay of the system as displayed.
            # So use phases_to_plot (which includes ref subtraction).
            
            # Unwrap requires radians usually, or we can just unwrap degrees with period 360
            phases_rad = np.radians(phases_to_plot)
            phases_unwrapped = np.unwrap(phases_rad)
            
            # dPhi / dOmega
            # dOmega = 2 * pi * dF
            # GD = - dPhi / dOmega
            
            # Calculate derivative
            d_phi = np.diff(phases_unwrapped)
            d_freq = np.diff(freqs_to_plot)
            
            # Avoid div by zero
            d_freq[d_freq == 0] = 1e-12
            
            group_delay_sec = - d_phi / (2 * np.pi * d_freq)
            group_delay_ms = group_delay_sec * 1000.0
            
            # Plot against mid-points of freqs
            freq_mids = (freqs_to_plot[:-1] + freqs_to_plot[1:]) / 2
            
            # Manually log X for the overlay view
            log_freq_mids = np.log10(freq_mids)
            
            self.gd_curve.setData(log_freq_mids, group_delay_ms)
            self.gd_view.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            self.update_gd_views()
            
        else:
            self.gd_axis.hide()
            self.gd_curve.setData([], [])
