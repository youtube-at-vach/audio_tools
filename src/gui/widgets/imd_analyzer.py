import argparse
import numpy as np
import scipy.signal
import queue
import threading
import time
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
                             QPushButton, QComboBox, QGroupBox, QFormLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
import pyqtgraph as pg

from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class IMDAnalyzerSignals(QObject):
    update_results = pyqtSignal(dict)
    update_plot = pyqtSignal(object, object) # freqs, magnitude

class AnalysisWorker(threading.Thread):
    def __init__(self, input_queue, signals, module_params):
        super().__init__()
        self.input_queue = input_queue
        self.signals = signals
        self.params = module_params # Reference to module to read params safely
        self.running = True
        self.daemon = True

    def run(self):
        while self.running:
            try:
                # Get data with timeout to allow checking self.running
                data, sample_rate = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._analyze_chunk(data, sample_rate)
            except Exception as e:
                print(f"Error in analysis worker: {e}")
            finally:
                self.input_queue.task_done()

    def stop(self):
        self.running = False

    def _analyze_chunk(self, recorded, sample_rate):
        # Perform FFT
        N = len(recorded)
        # Read params safely (assuming atomic reads or acceptable race for UI params)
        window_name = self.params.window_name
        standard = self.params.standard
        f1 = self.params.f1
        f2 = self.params.f2
        num_sidebands = self.params.num_sidebands

        window = scipy.signal.get_window(window_name, N)
        windowed = recorded * window
        fft_res = np.fft.rfft(windowed)
        mag = np.abs(fft_res) * (2 / np.sum(window))
        freqs = np.fft.rfftfreq(N, d=1/sample_rate)
        
        # Emit for plotting
        self.signals.update_plot.emit(freqs, mag)
        
        # Calculate IMD
        if standard == 'smpte':
            results = self._calc_imd_smpte(mag, freqs, f1, f2, num_sidebands)
        else:
            results = self._calc_imd_ccif(mag, freqs, f1, f2)
            
        self.signals.update_results.emit(results)

    def _find_peak(self, mag, freqs, target_freq, width=20.0):
        mask = (freqs >= target_freq - width) & (freqs <= target_freq + width)
        if not np.any(mask):
            return 0.0
        return np.max(mag[mask])

    def _calc_imd_smpte(self, mag, freqs, f1, f2, num_sidebands):
        # SMPTE: f1 (low), f2 (high). IMD products at f2 +/- n*f1
        amp_f2 = self._find_peak(mag, freqs, f2, width=max(50.0, f1*0.1))
        
        if amp_f2 < 1e-6:
            return {'imd': 0.0, 'imd_db': -100.0}
            
        sum_sq_sidebands = 0.0
        for n in range(1, num_sidebands + 1):
            sb_upper = f2 + n * f1
            sb_lower = f2 - n * f1
            
            amp_upper = self._find_peak(mag, freqs, sb_upper)
            amp_lower = self._find_peak(mag, freqs, sb_lower)
            
            sum_sq_sidebands += amp_upper**2 + amp_lower**2
            
        imd = np.sqrt(sum_sq_sidebands) / amp_f2
        return {
            'imd': imd * 100,
            'imd_db': 20 * np.log10(imd) if imd > 1e-9 else -100.0
        }

    def _calc_imd_ccif(self, mag, freqs, f1, f2):
        # CCIF: f1, f2 close (e.g. 19k, 20k). 
        # d2 = f2 - f1
        # d3 = 2f1 - f2, 2f2 - f1
        
        amp_f1 = self._find_peak(mag, freqs, f1)
        amp_f2 = self._find_peak(mag, freqs, f2)
        total_amp = amp_f1 + amp_f2
        
        if total_amp < 1e-6:
            return {'imd': 0.0, 'imd_db': -100.0}
            
        # d2
        d2_freq = abs(f2 - f1)
        amp_d2 = self._find_peak(mag, freqs, d2_freq)
        
        # d3
        d3_low = 2*f1 - f2
        d3_high = 2*f2 - f1
        amp_d3_low = self._find_peak(mag, freqs, d3_low) if d3_low > 0 else 0
        amp_d3_high = self._find_peak(mag, freqs, d3_high)
        
        distortion_sum_sq = amp_d2**2 + amp_d3_low**2 + amp_d3_high**2
        imd = np.sqrt(distortion_sum_sq) / total_amp
        
        return {
            'imd': imd * 100,
            'imd_db': 20 * np.log10(imd) if imd > 1e-9 else -100.0
        }

class IMDAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.signals = IMDAnalyzerSignals()
        
        # Parameters
        self.standard = 'smpte' # 'smpte' or 'ccif'
        self.f1 = 60.0
        self.f2 = 7000.0
        self.amplitude = 0.5 # Linear amplitude (approx -6 dBFS)
        self.ratio = 4.0 # f1/f2 ratio for SMPTE
        self.is_running = False
        
        # Analysis state
        self.window_name = 'blackmanharris'
        self.num_sidebands = 3 # For SMPTE
        
        # Playback state
        self._phase_f1 = 0.0
        self._phase_f2 = 0.0

        # Threading
        self.analysis_queue = queue.Queue(maxsize=2) # Small buffer to drop old frames if slow
        self.worker = None
        
        self.callback_id = None

    @property
    def name(self) -> str:
        return "IMD Analyzer"

    @property
    def description(self) -> str:
        return "Measures Intermodulation Distortion (SMPTE / CCIF)"

    def run(self, args: argparse.Namespace):
        print("IMD Analyzer CLI not implemented yet.")

    def get_widget(self):
        return IMDAnalyzerWidget(self)

    def start_analysis(self):
        if self.is_running:
            return
        
        self.is_running = True
        self._phase_f1 = 0.0
        self._phase_f2 = 0.0
        
        # Clear queue
        while not self.analysis_queue.empty():
            try:
                self.analysis_queue.get_nowait()
            except queue.Empty:
                break

        # Start worker
        self.worker = AnalysisWorker(self.analysis_queue, self.signals, self)
        self.worker.start()
        
        self.callback_id = self.audio_engine.register_callback(self._audio_callback)

    def stop_analysis(self):
        if not self.is_running:
            return
            
        if self.callback_id is not None:
            self.audio_engine.unregister_callback(self.callback_id)
            self.callback_id = None
            
        self.is_running = False
        
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=1.0)
            self.worker = None

    def _generate_dual_tone(self, num_frames, sample_rate):
        t = np.arange(num_frames) / sample_rate
        
        # Calculate amplitudes based on ratio
        # Total amplitude should not exceed self.amplitude
        # SMPTE: f1 is usually 4x stronger than f2 (4:1 ratio)
        # CCIF: usually 1:1 ratio
        
        if self.standard == 'smpte':
            # ratio = amp_f1 / amp_f2
            # amp_f1 + amp_f2 = self.amplitude (roughly, to avoid clipping)
            # amp_f2 * ratio + amp_f2 = self.amplitude
            # amp_f2 * (ratio + 1) = self.amplitude
            amp_f2 = self.amplitude / (self.ratio + 1)
            amp_f1 = amp_f2 * self.ratio
        else: # CCIF
            # 1:1 ratio usually
            amp_f1 = self.amplitude / 2
            amp_f2 = self.amplitude / 2
            
        # Generate phases
        phase_inc_f1 = 2 * np.pi * self.f1 / sample_rate
        phase_inc_f2 = 2 * np.pi * self.f2 / sample_rate
        
        phases_f1 = self._phase_f1 + np.arange(num_frames) * phase_inc_f1
        phases_f2 = self._phase_f2 + np.arange(num_frames) * phase_inc_f2
        
        # Update state
        self._phase_f1 = (self._phase_f1 + num_frames * phase_inc_f1) % (2 * np.pi)
        self._phase_f2 = (self._phase_f2 + num_frames * phase_inc_f2) % (2 * np.pi)
        
        signal = amp_f1 * np.sin(phases_f1) + amp_f2 * np.sin(phases_f2)
        return signal

    def _audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(f"Stream status: {status}")
            
        sample_rate = self.audio_engine.sample_rate
        
        # 1. Generate Signal
        signal = self._generate_dual_tone(frames, sample_rate)
        
        # Output to channels (Mono -> Stereo)
        if outdata.shape[1] >= 1:
            outdata[:, 0] = signal
        if outdata.shape[1] >= 2:
            outdata[:, 1] = signal
            
        # 2. Analyze Input (Push to queue)
        # Use the first input channel
        if indata.shape[1] > 0:
            recorded = indata[:, 0].copy() # Copy is essential for threading
            try:
                self.analysis_queue.put_nowait((recorded, sample_rate))
            except queue.Full:
                # Drop frame if worker is too slow
                pass

class IMDAnalyzerWidget(QWidget):
    def __init__(self, module: IMDAnalyzer):
        super().__init__()
        self.module = module
        self.init_ui()
        
        # Connect signals
        self.module.signals.update_results.connect(self.update_results)
        self.module.signals.update_plot.connect(self.update_plot)

    def init_ui(self):
        layout = QHBoxLayout()
        
        # --- Controls Panel ---
        controls_group = QGroupBox("Settings")
        controls_group.setFixedWidth(300)
        form_layout = QFormLayout()
        
        # Standard
        self.std_combo = QComboBox()
        self.std_combo.addItems(['smpte', 'ccif'])
        self.std_combo.currentTextChanged.connect(self.on_std_changed)
        form_layout.addRow("Standard:", self.std_combo)
        
        # F1
        self.f1_spin = QDoubleSpinBox()
        self.f1_spin.setRange(10, 20000)
        self.f1_spin.setValue(self.module.f1)
        self.f1_spin.valueChanged.connect(lambda v: setattr(self.module, 'f1', v))
        form_layout.addRow("Freq 1 (Hz):", self.f1_spin)
        
        # F2
        self.f2_spin = QDoubleSpinBox()
        self.f2_spin.setRange(10, 24000)
        self.f2_spin.setValue(self.module.f2)
        self.f2_spin.valueChanged.connect(lambda v: setattr(self.module, 'f2', v))
        form_layout.addRow("Freq 2 (Hz):", self.f2_spin)
        
        # Amplitude
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0, 1.0)
        self.amp_spin.setSingleStep(0.1)
        self.amp_spin.setValue(self.module.amplitude)
        self.amp_spin.valueChanged.connect(lambda v: setattr(self.module, 'amplitude', v))
        form_layout.addRow("Amplitude (0-1):", self.amp_spin)
        
        # Ratio (SMPTE only)
        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(1, 10)
        self.ratio_spin.setValue(self.module.ratio)
        self.ratio_spin.valueChanged.connect(lambda v: setattr(self.module, 'ratio', v))
        self.ratio_label = QLabel("Ratio (F1:F2):")
        form_layout.addRow(self.ratio_label, self.ratio_spin)
        
        controls_group.setLayout(form_layout)
        
        # Buttons & Results
        btn_layout = QVBoxLayout()
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.on_start_stop)
        btn_layout.addWidget(self.start_btn)
        
        self.result_label = QLabel("IMD: -- %")
        self.result_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #333;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(self.result_label)
        
        self.db_label = QLabel("-- dB")
        self.db_label.setStyleSheet("font-size: 14pt; color: #666;")
        self.db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(self.db_label)
        
        btn_layout.addStretch()
        
        # Combine controls
        left_layout = QVBoxLayout()
        left_layout.addWidget(controls_group)
        left_layout.addLayout(btn_layout)
        layout.addLayout(left_layout)
        
        # --- Plot ---
        self.plot_widget = pg.PlotWidget(title="Spectrum")
        self.plot_widget.setLabel('left', 'Magnitude')
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_widget.setLogMode(x=True, y=True)
        self.plot_widget.showGrid(x=True, y=True)
        
        # Custom Axis Ticks
        axis = self.plot_widget.getPlotItem().getAxis('bottom')
        ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        ticks_log = [(np.log10(t), str(t) if t < 1000 else f"{t/1000:.0f}k") for t in ticks]
        axis.setTicks([ticks_log])
        
        self.plot_curve = self.plot_widget.plot(pen='y')
        layout.addWidget(self.plot_widget)
        
        self.setLayout(layout)
        
        # Init state
        self.on_std_changed('smpte')

    def on_std_changed(self, text):
        self.module.standard = text
        if text == 'smpte':
            self.module.f1 = 60.0
            self.module.f2 = 7000.0
            self.ratio_spin.setEnabled(True)
        else: # ccif
            self.module.f1 = 19000.0
            self.module.f2 = 20000.0
            self.ratio_spin.setEnabled(False)
            
        self.f1_spin.setValue(self.module.f1)
        self.f2_spin.setValue(self.module.f2)

    def on_start_stop(self, checked):
        if checked:
            self.module.start_analysis()
            self.start_btn.setText("Stop Analysis")
            self.start_btn.setStyleSheet("background-color: #ffcccc")
        else:
            self.module.stop_analysis()
            self.start_btn.setText("Start Analysis")
            self.start_btn.setStyleSheet("")

    def update_results(self, results):
        self.result_label.setText(f"IMD: {results['imd']:.4f} %")
        self.db_label.setText(f"{results['imd_db']:.2f} dB")

    def update_plot(self, freqs, mag):
        # Avoid updating too frequently if needed, but for now direct update
        # Log mode requires positive values
        self.plot_curve.setData(freqs, mag)
