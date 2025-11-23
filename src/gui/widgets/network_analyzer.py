import argparse
import numpy as np
import scipy.signal
import time
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
                             QPushButton, QComboBox, QGroupBox, QFormLayout, QProgressBar, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
import pyqtgraph as pg
import threading

from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class NetworkAnalyzerSignals(QObject):
    update_plot = pyqtSignal(float, float, float) # freq, mag_db, phase_deg
    sweep_finished = pyqtSignal()
    progress = pyqtSignal(int)
    latency_result = pyqtSignal(float)
    latency_result = pyqtSignal(float)
    error = pyqtSignal(str)

class PlayRecSession:
    def __init__(self, audio_engine, output_data, input_channels=1):
        self.audio_engine = audio_engine
        self.output_data = output_data
        self.total_frames = len(output_data)
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
            # Handle stereo output mapping
            # output_data is expected to be (N, 2)
            outdata[:chunk, :] = self.output_data[self.current_frame:self.current_frame+chunk, :]
            if chunk < frames:
                outdata[chunk:, :] = 0
                
            # Input
            # Capture specified input channels (default 1, usually ch 0)
            # indata is (frames, channels)
            # We want to store into input_data (total_frames, input_channels)
            
            # Determine which input channel to capture. 
            # For now, let's assume we capture channel 0 (Left) as per original code.
            # If input_channels > 1, we might need more logic.
            # Original code: recorded = rec_data[:, 0]
            
            if indata.shape[1] > 0:
                self.input_data[self.current_frame:self.current_frame+chunk, 0] = indata[:chunk, 0]
            
            self.current_frame += chunk
            
            if self.current_frame >= self.total_frames:
                self.is_complete = True
                self.completion_event.set()
                # We can't unregister here easily without deadlock or complex logic, 
                # so we rely on the main thread to call stop() after wait() returns.

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
        self.amplitude = 0.5
        self.duration_per_step = 0.5
        self.latency_sec = 0.0
        self.num_averages = 1
        
        # Fast Sweep Parameters
        self.sweep_mode = "Stepped Sine" # or "Fast Chirp"
        self.chirp_duration = 1.0
        
        self.worker = None
        self.calibration_worker = None
        
        # Calibration (Reference Trace)
        # Format: {'freqs': np.array, 'mags': np.array, 'phases': np.array}
        self.reference_trace = None

    @property
    def name(self) -> str:
        return "Network Analyzer"

    @property
    def description(self) -> str:
        return "Bode Plot (Gain & Phase) with Latency Compensation"

    def run(self, args: argparse.Namespace):
        print("CLI not implemented")

    def get_widget(self):
        return NetworkAnalyzerWidget(self)

    def run_play_rec(self, output_data):
        """
        Helper to run a play/record session using AudioEngine.
        output_data: (N, 2) numpy array
        Returns: (N, 1) numpy array (recorded data)
        """
        session = PlayRecSession(self.audio_engine, output_data)
        session.start()
        
        # Wait for completion
        # We need to handle potential aborts from workers
        # But here we are blocking the worker thread, which is fine.
        # However, we should verify if we need to check for worker cancellation?
        # The session.wait() blocks.
        
        session.wait()
        session.stop()
        
        return session.input_data

    def calibrate_latency(self):
        """
        Measures loopback latency using a chirp signal.
        """
        sample_rate = self.audio_engine.sample_rate
        duration = 0.5
        
        # Generate Chirp
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        chirp = scipy.signal.chirp(t, f0=20, t1=duration, f1=10000, method='logarithmic')
        chirp *= self.amplitude
        
        try:
            # Prepare stereo output (duplicate mono chirp)
            out_data = np.zeros((len(chirp), 2), dtype=np.float32)
            out_data[:, 0] = chirp
            out_data[:, 1] = chirp
            
            print("Playing chirp for latency calibration...")
            
            # Use AudioEngine
            rec_data = self.run_play_rec(out_data)
            
            # Analyze
            # Cross-correlate reference (chirp) with recorded signal (channel 0)
            recorded = rec_data[:, 0]
            
            # Simple peak finding in cross-correlation
            correlation = scipy.signal.correlate(recorded, chirp, mode='full')
            lags = scipy.signal.correlation_lags(len(recorded), len(chirp), mode='full')
            lag = lags[np.argmax(correlation)]
            
            latency_samples = lag
            self.latency_sec = latency_samples / sample_rate
            
            # Sanity check: latency shouldn't be negative or huge
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
        chirp = self.amplitude * np.sin(phase)
        
        # Apply Tapering (Tukey Window) to reduce ringing
        # Fade in/out over 5% of duration
        window = scipy.signal.windows.tukey(num_samples, alpha=0.05)
        chirp *= window
        
        # 2. Generate Inverse Filter
        # The inverse filter must compensate for the pink spectrum of the log chirp (-3dB/oct).
        # It needs to have a blue spectrum (+3dB/oct).
        # Since we time-reverse the chirp (High->Low freq), the envelope must start High and end Low.
        # This means the pre-reversed envelope (Low->High freq) must GROW with frequency.
        # Envelope m(t) = exp(t * L / T)
        
        inv_envelope = np.exp(t * L / T)
        inv_filter = inv_envelope * np.sin(phase)
        
        # Apply window to inverse filter as well to avoid edge artifacts
        inv_filter *= window
        
        inv_filter = np.flip(inv_filter) # Time reverse
        
        # Normalize Inverse Filter
        # We want the convolution of chirp and inv_filter to have a peak of 1.0 (0dB) for a unity gain system.
        # Peak of autocorrelation of chirp * inv_filter
        # Let's calculate the scaling factor.
        # Ideally: scale = 1 / max(fftconvolve(chirp, inv_filter))
        # But we can just normalize the inverse filter by its energy or similar.
        # Let's do a quick pre-calculation of the normalization factor.
        
        test_conv = scipy.signal.fftconvolve(chirp, inv_filter, mode='full')
        norm_factor = np.max(np.abs(test_conv))
        if norm_factor > 1e-9:
            inv_filter /= norm_factor
        
        # 3. Play and Record
        
        # Padding for latency and decay
        padding_sec = 1.0 # Generous padding
        padding_samples = int(padding_sec * sample_rate)
        
        out_data = np.zeros((len(chirp) + padding_samples, 2), dtype=np.float32)
        out_data[:len(chirp), 0] = chirp
        out_data[:len(chirp), 1] = chirp
        
        self.signals.progress.emit(10)
        
        if not worker.is_running: return
        
        # Use AudioEngine
        rec_data = self.run_play_rec(out_data)
        recorded = rec_data[:, 0]
        
        self.signals.progress.emit(50)
        
        if not worker.is_running: return

        # 4. Deconvolution (Convolution with Inverse Filter)
        # Use FFT convolution for speed
        ir = scipy.signal.fftconvolve(recorded, inv_filter, mode='full')
        
        # 5. Find Impulse Response Peak and Window it
        peak_idx = np.argmax(np.abs(ir))
        
        # Window the IR
        # We want to capture the main impulse.
        # Center around peak.
        
        pre_peak_samples = int(0.01 * sample_rate) # 10ms before
        post_peak_samples = int(0.5 * sample_rate) # 500ms after
        
        start_win = max(0, peak_idx - pre_peak_samples)
        end_win = min(len(ir), peak_idx + post_peak_samples)
        
        ir_windowed = ir[start_win:end_win]
        
        # 6. FFT to get Frequency Response
        H = np.fft.rfft(ir_windowed)
        freqs = np.fft.rfftfreq(len(ir_windowed), d=1/sample_rate)
        
        # 7. Extract Magnitude and Phase
        mask = (freqs >= self.start_freq) & (freqs <= self.end_freq)
        valid_freqs = freqs[mask]
        valid_H = H[mask]
        
        # Magnitude in dB
        # Since we normalized the inverse filter, a unity gain system should give ~1.0 magnitude (0dB).
        mag_db = 20 * np.log10(np.abs(valid_H) + 1e-12)
        
        phase_rad = np.angle(valid_H)
        phase_rad = np.unwrap(phase_rad)
        
        # Compensate for the window shift
        # The peak was at 'peak_idx'. We started window at 'start_win'.
        # The effective delay in the windowed buffer is 'peak_idx - start_win'.
        # We want to remove this delay to get phase relative to the impulse peak.
        
        delay_samples = peak_idx - start_win
        phase_rad += 2 * np.pi * valid_freqs * (delay_samples / sample_rate)
        
        # Wrap to -180, 180
        phase_deg = np.degrees(phase_rad)
        phase_deg = (phase_deg + 180) % 360 - 180
        
        # Emit points
        step = max(1, len(valid_freqs) // 500)
        
        for i in range(0, len(valid_freqs), step):
            if not worker.is_running: break
            self.signals.update_plot.emit(valid_freqs[i], mag_db[i], phase_deg[i])
            
        self.signals.progress.emit(100)

    def _execute_sweep(self, worker):
        sample_rate = self.audio_engine.sample_rate
        
        # Generate Frequencies
        freqs = self._generate_log_freqs(self.start_freq, self.end_freq, self.steps_per_octave)
        total_steps = len(freqs)
        
        import sounddevice as sd
        
        for i, freq in enumerate(freqs):
            if not worker.is_running:
                break
            
            avg_mag_linear = 0.0
            avg_phase_vector = 0.0 + 0.0j
            
            # Averaging Loop
            for _ in range(self.num_averages):
                if not worker.is_running:
                    break

                # Generate Tone
                num_samples = int(sample_rate * self.duration_per_step)
                t = np.arange(num_samples) / sample_rate
                tone = self.amplitude * np.cos(2 * np.pi * freq * t)
                
                # Padding for latency
                # We need to ensure we capture the full tone after it travels through the loopback
                # Latency can be up to self.latency_sec. Add a safety buffer.
                safety_buffer_sec = 0.1
                padding_samples = int((self.latency_sec + safety_buffer_sec) * sample_rate)
                
                # Output: Tone + Silence
                out_data = np.zeros((len(tone) + padding_samples, 2), dtype=np.float32)
                out_data[:len(tone), 0] = tone
                out_data[:len(tone), 1] = tone
                
                # Play/Record
                rec_data = self.run_play_rec(out_data)
                recorded = rec_data[:, 0]
                
                # Extract the relevant segment based on latency
                # We expect the signal to start at latency_sec
                start_idx = int(self.latency_sec * sample_rate)
                end_idx = start_idx + len(tone)
                
                if end_idx > len(recorded):
                    # Should not happen with sufficient padding, but clamp just in case
                    end_idx = len(recorded)
                    
                recorded_segment = recorded[start_idx:end_idx]
                
                # If segment is shorter than tone (due to clamping), truncate tone too
                if len(recorded_segment) < len(tone):
                    tone_segment = tone[:len(recorded_segment)]
                else:
                    tone_segment = tone
                
                # Analyze
                mag_linear, phase_deg = self._analyze_tone(recorded_segment, tone_segment, freq, sample_rate)
                
                avg_mag_linear += mag_linear
                # Convert phase to vector for averaging
                avg_phase_vector += np.exp(1j * np.radians(phase_deg))
            
            # Finalize Average
            avg_mag_linear /= self.num_averages
            avg_phase_vector /= self.num_averages
            
            final_mag_db = 20 * np.log10(avg_mag_linear) if avg_mag_linear > 1e-9 else -100
            final_phase_deg = np.degrees(np.angle(avg_phase_vector))
            
            self.signals.update_plot.emit(freq, final_mag_db, final_phase_deg)
            self.signals.progress.emit(int((i + 1) / total_steps * 100))

    def _generate_log_freqs(self, start, end, steps_per_oct):
        freqs = []
        curr = start
        while curr < end:
            freqs.append(curr)
            curr *= 2 ** (1 / steps_per_oct)
        if freqs[-1] < end:
            freqs.append(end)
        return freqs

    def _analyze_tone(self, recorded, reference, freq, sample_rate):
        # Apply Window
        window = scipy.signal.windows.hann(len(recorded))
        rec_windowed = recorded * window
        
        # FFT
        fft_rec = np.fft.rfft(rec_windowed)
        freqs = np.fft.rfftfreq(len(recorded), d=1/sample_rate)
        
        # Find Peak
        idx = np.argmax(np.abs(fft_rec))
        peak_freq = freqs[idx]
        
        # Magnitude
        # Scale for window and RMS
        mag_linear = np.abs(fft_rec[idx]) * 2 / np.sum(window)
        
        # Phase
        # Phase of recorded signal at peak
        phase_rec = np.angle(fft_rec[idx])
        
        # Since we time-shifted the recorded signal by 'latency_sec' (by extracting the segment),
        # we have already compensated for the integer sample delay.
        # However, there might be a fractional sample delay left.
        # fractional_delay = latency_sec * sample_rate - int(latency_sec * sample_rate)
        # phase_correction = 2 * pi * freq * (fractional_delay / sample_rate)
        
        latency_samples = self.latency_sec * sample_rate
        fractional_samples = latency_samples - int(latency_samples)
        fractional_sec = fractional_samples / sample_rate
        
        phase_delay_comp = 2 * np.pi * freq * fractional_sec
        phase_sys_rad = phase_rec + phase_delay_comp
        
        # Wrap to -pi, pi
        phase_sys_rad = (phase_sys_rad + np.pi) % (2 * np.pi) - np.pi
        phase_sys_deg = np.degrees(phase_sys_rad)
        
        return mag_linear, phase_sys_deg

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
        
        # Controls
        controls_group = QGroupBox("Settings")
        controls_group.setFixedWidth(300)
        form = QFormLayout()
        
        # Mode Selector
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Stepped Sine", "Fast Chirp"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        form.addRow("Mode:", self.mode_combo)
        
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(10, 20000); self.start_spin.setValue(20)
        self.start_spin.valueChanged.connect(lambda v: setattr(self.module, 'start_freq', v))
        form.addRow("Start Freq:", self.start_spin)
        
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(10, 24000); self.end_spin.setValue(20000)
        self.end_spin.valueChanged.connect(lambda v: setattr(self.module, 'end_freq', v))
        form.addRow("End Freq:", self.end_spin)
        
        self.steps_spin = QDoubleSpinBox() 
        self.steps_spin.setDecimals(0)
        self.steps_spin.setRange(1, 48); self.steps_spin.setValue(12)
        self.steps_spin.valueChanged.connect(lambda v: setattr(self.module, 'steps_per_octave', int(v)))
        self.steps_label = QLabel("Steps/Octave:")
        form.addRow(self.steps_label, self.steps_spin)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 60.0); self.duration_spin.setValue(1.0)
        self.duration_spin.valueChanged.connect(lambda v: setattr(self.module, 'chirp_duration', v))
        self.duration_label = QLabel("Duration (s):")
        form.addRow(self.duration_label, self.duration_spin)
        self.duration_label.hide()
        self.duration_spin.hide()
        
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0, 1); self.amp_spin.setValue(0.5); self.amp_spin.setSingleStep(0.1)
        self.amp_spin.valueChanged.connect(lambda v: setattr(self.module, 'amplitude', v))
        form.addRow("Amplitude:", self.amp_spin)
        
        self.avg_spin = QDoubleSpinBox()
        self.avg_spin.setDecimals(0)
        self.avg_spin.setRange(1, 10); self.avg_spin.setValue(1)
        self.avg_spin.valueChanged.connect(lambda v: setattr(self.module, 'num_averages', int(v)))
        form.addRow("Averages:", self.avg_spin)
        
        self.smooth_combo = QComboBox()
        self.smooth_combo.addItems(["Off", "Light", "Medium", "Heavy"])
        self.smooth_combo.currentTextChanged.connect(self.refresh_plots)
        form.addRow("Smoothing:", self.smooth_combo)
        
        # Unit Selector
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["dBFS", "dBV", "dBu", "Vrms", "Vpeak"])
        self.unit_combo.currentTextChanged.connect(self.refresh_plots)
        form.addRow("Unit:", self.unit_combo)
        
        self.lat_btn = QPushButton("Calibrate Latency")
        self.lat_btn.clicked.connect(self.calibrate)
        form.addRow(self.lat_btn)
        
        self.lat_label = QLabel("Latency: 0.00 ms")
        form.addRow(self.lat_label)
        
        controls_group.setLayout(form)
        
        # Calibration Group
        cal_group = QGroupBox("Calibration")
        cal_layout = QFormLayout()
        
        self.store_ref_btn = QPushButton("Store Reference")
        self.store_ref_btn.clicked.connect(self.on_store_reference)
        cal_layout.addRow(self.store_ref_btn)
        
        self.clear_ref_btn = QPushButton("Clear Reference")
        self.clear_ref_btn.clicked.connect(self.on_clear_reference)
        cal_layout.addRow(self.clear_ref_btn)
        
        self.apply_ref_check = QCheckBox("Apply Reference")
        self.apply_ref_check.toggled.connect(self.on_apply_reference_changed)
        cal_layout.addRow(self.apply_ref_check)
        
        cal_group.setLayout(cal_layout)
        
        # Buttons
        btn_layout = QVBoxLayout()
        self.start_btn = QPushButton("Start Sweep")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.on_start_stop)
        btn_layout.addWidget(self.start_btn)
        
        self.progress_bar = QProgressBar()
        btn_layout.addWidget(self.progress_bar)
        
        btn_layout.addStretch()
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(controls_group)
        left_layout.addWidget(cal_group)
        left_layout.addLayout(btn_layout)
        layout.addLayout(left_layout)
        
        # Plots
        plot_layout = QVBoxLayout()
        
        self.mag_plot = pg.PlotWidget(title="Magnitude Response")
        self.mag_plot.setLabel('left', 'Magnitude', units='dB')
        self.mag_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.mag_plot.setLogMode(x=True, y=False)
        self.mag_plot.showGrid(x=True, y=True)
        self.mag_curve = self.mag_plot.plot(pen='g')
        plot_layout.addWidget(self.mag_plot)
        
        self.phase_plot = pg.PlotWidget(title="Phase Response")
        self.phase_plot.setLabel('left', 'Phase', units='deg')
        self.phase_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.phase_plot.setLogMode(x=True, y=False)
        self.phase_plot.showGrid(x=True, y=True)
        self.phase_curve = self.phase_plot.plot(pen='y')
        plot_layout.addWidget(self.phase_plot)
        
        layout.addLayout(plot_layout)
        self.setLayout(layout)

    def on_mode_changed(self, mode):
        self.module.sweep_mode = mode
        if mode == "Stepped Sine":
            self.steps_label.show(); self.steps_spin.show()
            self.duration_label.hide(); self.duration_spin.hide()
        else:
            self.steps_label.hide(); self.steps_spin.hide()
            self.duration_label.show(); self.duration_spin.show()

    def calibrate(self):
        self.lat_btn.setEnabled(False)
        self.lat_label.setText("Calibrating...")
        self.module.start_calibration()

    def on_latency_result(self, lat):
        self.lat_label.setText(f"Latency: {lat*1000:.2f} ms")
        self.lat_btn.setEnabled(True)

    def on_store_reference(self):
        if not self.freqs:
            return
        
        self.module.reference_trace = {
            'freqs': np.array(self.freqs),
            'mags': np.array(self.mags),
            'phases': np.array(self.phases)
        }
        print("Reference trace stored.")

    def on_clear_reference(self):
        self.module.reference_trace = None
        self.refresh_plots()
        print("Reference trace cleared.")

    def on_apply_reference_changed(self, checked):
        self.refresh_plots()

    def on_start_stop(self, checked):
        if checked:
            self.freqs = []
            self.mags = []
            self.phases = []
            self.mag_curve.setData([], [])
            self.phase_curve.setData([], [])
            self.start_btn.setText("Stop Sweep")
            self.module.start_sweep()
        else:
            self.module.stop_sweep()
            self.start_btn.setText("Start Sweep")

    def update_plot(self, freq, mag, phase):
        self.freqs.append(freq)
        self.mags.append(mag)
        self.phases.append(phase)
        self.refresh_plots()

    def refresh_plots(self):
        if not self.freqs:
            return
            
        smooth_mode = self.smooth_combo.currentText()
        unit = self.unit_combo.currentText()
        
        mags_to_plot = np.array(self.mags)
        phases_to_plot = np.array(self.phases)
        
        # Unit Conversion
        # self.mags is in dBFS
        # Convert to Linear (Full Scale ratio)
        mags_linear = 10 ** (mags_to_plot / 20)
        
        # Get Input Sensitivity (Volts for 0dBFS)
        # Default to 1.0 if not available
        try:
            input_sensitivity = self.module.audio_engine.calibration.input_sensitivity
        except:
            input_sensitivity = 1.0
            
        if unit == "dBFS":
            y_values = mags_to_plot
            self.mag_plot.setLabel('left', 'Magnitude', units='dBFS')
        elif unit == "dBV":
            # dBV = 20 * log10(Vrms)
            # Vpeak = linear * input_sensitivity
            # Vrms = Vpeak / sqrt(2)
            v_peak = mags_linear * input_sensitivity
            v_rms = v_peak / np.sqrt(2)
            y_values = 20 * np.log10(v_rms + 1e-12)
            self.mag_plot.setLabel('left', 'Magnitude', units='dBV')
        elif unit == "dBu":
            # dBu = 20 * log10(Vrms / 0.7746)
            v_peak = mags_linear * input_sensitivity
            v_rms = v_peak / np.sqrt(2)
            y_values = 20 * np.log10((v_rms + 1e-12) / 0.7746)
            self.mag_plot.setLabel('left', 'Magnitude', units='dBu')
        elif unit == "Vrms":
            v_peak = mags_linear * input_sensitivity
            y_values = v_peak / np.sqrt(2)
            self.mag_plot.setLabel('left', 'Magnitude', units='V')
        elif unit == "Vpeak":
            y_values = mags_linear * input_sensitivity
            self.mag_plot.setLabel('left', 'Magnitude', units='V')
        else:
            y_values = mags_to_plot
            
        # Apply Reference if enabled
        if self.apply_ref_check.isChecked() and self.module.reference_trace is not None:
            ref = self.module.reference_trace
            ref_freqs = ref['freqs']
            ref_mags = ref['mags'] # dBFS
            ref_phases = ref['phases']
            
            # Interpolate reference to current frequencies
            if len(ref_freqs) > 1:
                interp_mags_dbfs = np.interp(self.freqs, ref_freqs, ref_mags)
                interp_phases = np.interp(self.freqs, ref_freqs, ref_phases)
                
                # Calculate Reference in Target Unit
                ref_mags_linear = 10 ** (interp_mags_dbfs / 20)
                
                if unit == "dBFS":
                    ref_y = interp_mags_dbfs
                elif unit == "dBV":
                    v_peak = ref_mags_linear * input_sensitivity
                    v_rms = v_peak / np.sqrt(2)
                    ref_y = 20 * np.log10(v_rms + 1e-12)
                elif unit == "dBu":
                    v_peak = ref_mags_linear * input_sensitivity
                    v_rms = v_peak / np.sqrt(2)
                    ref_y = 20 * np.log10((v_rms + 1e-12) / 0.7746)
                elif unit == "Vrms":
                    v_peak = ref_mags_linear * input_sensitivity
                    ref_y = v_peak / np.sqrt(2)
                elif unit == "Vpeak":
                    ref_y = ref_mags_linear * input_sensitivity
                
                # Normalize
                if "dB" in unit:
                    y_values = y_values - ref_y
                else:
                    # Linear division
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_values = y_values / ref_y
                        y_values = np.nan_to_num(y_values)
                
                # Phase normalization
                current_phases = np.array(phases_to_plot)
                phases_to_plot = current_phases - interp_phases
                phases_to_plot = (phases_to_plot + 180) % 360 - 180

        if smooth_mode != "Off" and len(y_values) > 3:
            window_size = 3
            if smooth_mode == "Medium": window_size = 5
            if smooth_mode == "Heavy": window_size = 9
            
            # Simple moving average
            y_values = scipy.signal.savgol_filter(y_values, min(len(y_values), window_size), 1) if len(y_values) >= window_size else y_values
            phases_to_plot = scipy.signal.savgol_filter(phases_to_plot, min(len(phases_to_plot), window_size), 1) if len(phases_to_plot) >= window_size else phases_to_plot

        self.mag_curve.setData(self.freqs, y_values)
        self.phase_curve.setData(self.freqs, phases_to_plot)

    def on_sweep_finished(self):
        self.start_btn.setChecked(False)
        self.start_btn.setText("Start Sweep")
        self.progress_bar.setValue(100)

    def on_error(self, msg):
        print(f"Error: {msg}")
        self.lat_btn.setEnabled(True)
