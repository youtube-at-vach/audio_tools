import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QGroupBox, QDoubleSpinBox, QSpinBox, QProgressBar, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
import scipy.signal

class SweepWorker(QThread):
    progress = pyqtSignal(int, str)
    result_point = pyqtSignal(float, float, float) # freq, amp_db, phase_deg
    finished_sweep = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, module, start_f, end_f, steps_per_oct, amp_db, duration):
        super().__init__()
        self.module = module
        self.audio_engine = module.audio_engine
        self.start_f = start_f
        self.end_f = end_f
        self.steps_per_oct = steps_per_oct
        self.amp_db = amp_db
        self.duration = duration
        self.is_running = True

    def run(self):
        try:
            self.audio_engine.stop_stream() # Ensure stream is stopped
            self.module.perform_sweep(
                self.start_f, self.end_f, self.steps_per_oct, self.amp_db, self.duration,
                progress_callback=self.report_progress,
                result_callback=self.report_result,
                check_stop=self.check_stop
            )
            self.finished_sweep.emit()
        except Exception as e:
            self.error.emit(str(e))

    def report_progress(self, val, msg):
        self.progress.emit(val, msg)

    def report_result(self, freq, amp, phase):
        self.result_point.emit(freq, amp, phase)

    def check_stop(self):
        return not self.is_running

    def stop(self):
        self.is_running = False

class FreqResponseAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.worker = None

    @property
    def name(self) -> str:
        return "Freq Response"

    @property
    def description(self) -> str:
        return "Measures frequency response using a stepped sine sweep."

    def perform_sweep(self, start_f, end_f, steps_per_oct, amp_db, duration, 
                      progress_callback=None, result_callback=None, check_stop=None):
        
        freqs = []
        curr = start_f
        while curr <= end_f:
            freqs.append(curr)
            curr *= 2**(1/steps_per_oct)
        
        total_steps = len(freqs)
        
        # Device IDs
        input_device = self.audio_engine.input_device
        output_device = self.audio_engine.output_device
        sample_rate = self.audio_engine.sample_rate
        
        # Channel Mapping
        out_ch_idx = 1 if self.audio_engine.output_channel_mode == 'right' else 0
        in_ch_idx = 1 if self.audio_engine.input_channel_mode == 'right' else 0
        
        try:
            out_info = sd.query_devices(output_device)
            in_info = sd.query_devices(input_device)
            max_out = out_info['max_output_channels']
            max_in = in_info['max_input_channels']
        except Exception as e:
            raise Exception(f"Device query failed: {e}")

        amp_linear = 10**(amp_db/20)
        results = []

        for i, f in enumerate(freqs):
            if check_stop and check_stop():
                break
            
            if progress_callback:
                progress_callback(int((i/total_steps)*100), f"Measuring {f:.1f} Hz...")
            
            # Generate Tone
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = amp_linear * np.sin(2 * np.pi * f * t)
            
            # Prepare Output
            out_data = np.zeros((len(tone), max_out), dtype=np.float32)
            if out_ch_idx < max_out:
                out_data[:, out_ch_idx] = tone
            
            # Play/Record
            try:
                rec_data = sd.playrec(out_data, samplerate=sample_rate, channels=max_in, 
                                    device=(input_device, output_device), blocking=True)
            except Exception as e:
                raise Exception(f"IO Error: {e}")
            
            # Analyze
            if in_ch_idx < max_in:
                signal_in = rec_data[:, in_ch_idx]
                
                # Apply window
                window = np.hanning(len(signal_in))
                fft_res = np.fft.rfft(signal_in * window)
                fft_freqs = np.fft.rfftfreq(len(signal_in), 1/sample_rate)
                
                target_idx = np.argmin(np.abs(fft_freqs - f))
                mag = np.abs(fft_res[target_idx]) * 2 / np.sum(window)
                
                measured_db = 20*np.log10(mag + 1e-12)
                
                # Phase
                phase_rad = np.angle(fft_res[target_idx])
                phase_deg = np.degrees(phase_rad)
                
                if result_callback:
                    result_callback(f, measured_db, phase_deg)
                
                results.append((f, measured_db, phase_deg))
        
        return results

    def run(self, args):
        print("Running Freq Response Analyzer...")
        # Defaults for CLI
        start_f = 20
        end_f = 20000
        steps = 3
        amp = -20
        dur = 0.5
        
        results = self.perform_sweep(start_f, end_f, steps, amp, dur,
                                     progress_callback=lambda p, m: print(f"{p}%: {m}"),
                                     result_callback=lambda f, a, p: print(f"{f:.1f}Hz: {a:.1f}dB, {p:.1f}deg"))
        print("Sweep Complete.")

    def get_widget(self):
        return FreqResponseWidget(self)

class FreqResponseWidget(QWidget):
    def __init__(self, module: FreqResponseAnalyzer):
        super().__init__()
        self.module = module
        self.x_data = []
        self.y_amp = []
        self.y_phase = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Settings
        settings_group = QGroupBox("Sweep Settings")
        settings_layout = QHBoxLayout()
        
        settings_layout.addWidget(QLabel("Start (Hz):"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(20, 20000)
        self.start_spin.setValue(20)
        settings_layout.addWidget(self.start_spin)
        
        settings_layout.addWidget(QLabel("End (Hz):"))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(20, 20000)
        self.end_spin.setValue(20000)
        settings_layout.addWidget(self.end_spin)
        
        settings_layout.addWidget(QLabel("Steps/Oct:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 48)
        self.steps_spin.setValue(3)
        settings_layout.addWidget(self.steps_spin)
        
        settings_layout.addWidget(QLabel("Level (dB):"))
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(-60, 0)
        self.amp_spin.setValue(-20)
        settings_layout.addWidget(self.amp_spin)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Actions
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Sweep")
        self.start_btn.clicked.connect(self.start_sweep)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_sweep)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_plot)
        btn_layout.addWidget(self.clear_btn)
        
        layout.addLayout(btn_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Plots
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude', units='dB')
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setYRange(-60, 10)
        
        self.amp_curve = self.plot_widget.plot(pen='y', name='Amplitude')
        
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def start_sweep(self):
        if self.module.audio_engine.stream and self.module.audio_engine.stream.active:
            self.module.audio_engine.stop_stream()
            
        self.x_data = []
        self.y_amp = []
        self.y_phase = []
        self.amp_curve.setData([], [])
        
        self.module.worker = SweepWorker(
            self.module,
            self.start_spin.value(),
            self.end_spin.value(),
            self.steps_spin.value(),
            self.amp_spin.value(),
            0.5 # Duration per step
        )
        
        self.module.worker.progress.connect(self.update_progress)
        self.module.worker.result_point.connect(self.add_point)
        self.module.worker.finished_sweep.connect(self.sweep_finished)
        self.module.worker.error.connect(self.show_error)
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.module.worker.start()

    def stop_sweep(self):
        if self.module.worker:
            self.module.worker.stop()
            self.module.worker.wait()
        self.sweep_finished()

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.status_label.setText(msg)

    def add_point(self, freq, amp, phase):
        self.x_data.append(np.log10(freq))
        self.y_amp.append(amp)
        self.y_phase.append(phase)
        self.amp_curve.setData(self.x_data, self.y_amp)

    def sweep_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Sweep Completed")

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.sweep_finished()

    def clear_plot(self):
        self.x_data = []
        self.y_amp = []
        self.y_phase = []
        self.amp_curve.setData([], [])
