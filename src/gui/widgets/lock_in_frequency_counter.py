import argparse
import numpy as np
import pyqtgraph as pg
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QProgressBar, QSpinBox, QTabWidget, QApplication, QSplitter)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class LockInFrequencyCounter(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 4096 
        self.input_data = np.zeros((self.buffer_size, 2))

        # Signal detect (match FrequencyCounter-style amplitude gate)
        self.gate_threshold_db = -60.0
        self.current_amp_db = -120.0
        self.signal_present = False
        
        # Settings
        self.gen_frequency = 1000.0 # NCO Frequency
        self.signal_channel = 0 # 0: Ch1 (L), 1: Ch2 (R)
        self.ref_channel = 1
        self.ref_mode = "internal" # internal, loopback
        # Display is ~1000 points @ 10 Hz = ~100 s window. A ~2 s EMA time constant
        # provides stable readout without feeling laggy.
        self.smoothing_tau = 2.0
        
        # Internal State
        self._nco_phase = 0.0
        self._last_unwrapped_phase = 0.0
        self._first_run = True

        # Startup transient handling
        self._samples_received = 0
        # Only enable estimate discarding during real-time streaming (set in start_analysis).
        # This keeps offline/unit-test calls to process_data() responsive.
        self._discard_initial_estimates = 0
        self._estimates_discarded = 0
        
        # Plot Data Buffers
        self.max_history = 1000 # points on plot
        self.time_axis = deque(maxlen=self.max_history)
        self.freq_dev_history = deque(maxlen=self.max_history)
        self.phase_history = deque(maxlen=self.max_history)
        self.iq_history_i = deque(maxlen=self.max_history) # For I-Q plot
        self.iq_history_q = deque(maxlen=self.max_history)
        
        self.start_time = 0
        
        # Current Value
        self.current_freq_dev = 0.0
        self.smoothed_freq_dev = 0.0
        self.current_phase_deg = 0.0
        self.phase_std = 0.0
        
        self.callback_id = None

    @property
    def name(self) -> str:
        return "Lock-in Frequency Counter"

    @property
    def description(self) -> str:
        return "Precision Frequency & Phase Drift Measurement using Lock-in Principle."

    def run(self, args: argparse.Namespace):
        print("CLI not implemented")

    def get_widget(self):
        return LockInFrequencyCounterWidget(self)

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        self.input_data = np.zeros((self.buffer_size, 2))
        
        # Reset State
        self._nco_phase = 0.0
        self._last_unwrapped_phase = 0.0
        self._first_run = True
        self.start_time = 0
        self.smoothed_freq_dev = 0.0
        self.current_phase_deg = 0.0

        self.current_amp_db = -120.0
        self.signal_present = False

        self._samples_received = 0
        self._discard_initial_estimates = 3
        self._estimates_discarded = 0
        
        self.time_axis.clear()
        self.freq_dev_history.clear()
        self.phase_history.clear()
        self.iq_history_i.clear()
        self.iq_history_q.clear()
        
        sample_rate = self.audio_engine.sample_rate
        
        def callback(indata, outdata, frames, time_info, status):
            # Input Capture
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

            self._samples_received += frames
                
            # Output Generation
            outdata.fill(0)
            if self.ref_mode == "loopback":
                t = (np.arange(frames) + self._nco_phase) / sample_rate
                sig = 0.5 * np.cos(2 * np.pi * self.gen_frequency * t)
                outdata[:, 0] = sig
                outdata[:, 1] = sig
                
            self._nco_phase += frames

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def process_data(self):
        if not self.is_running:
            return

        # Wait until we have a fully populated buffer, then discard a few initial
        # estimates to avoid startup transients causing a "first point jump".
        if self._samples_received < self.buffer_size:
            # In normal operation, samples arrive via the audio callback and the
            # buffer starts as zeros. In unit tests / offline use, input_data may
            # be populated directly without advancing _samples_received.
            if self._samples_received == 0 and np.any(self.input_data):
                pass
            else:
                return

        # Get Snapshot
        data = self.input_data
        sig = data[:, self.signal_channel]
        n_samples = len(sig)
        sr = self.audio_engine.sample_rate

        # 1) Signal detect (RMS gate)
        rms = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2)))
        self.current_amp_db = float(20.0 * np.log10(rms + 1e-12))
        if self.current_amp_db < float(getattr(self, 'gate_threshold_db', -60.0)):
            self.signal_present = False
            return
        
        t = np.arange(n_samples) / sr
        
        # NCO
        osc = np.exp(-1j * 2 * np.pi * self.gen_frequency * t)
        
        # Mixing
        z = sig * osc 
        
        # Split into segments to find slope
        n_segments = 4
        seg_len = n_samples // n_segments
        
        # Calculate Phase for each segment
        seg_phases = []
        seg_centers = []
        
        for i in range(n_segments):
            start = i * seg_len
            end = start + seg_len
            segment = z[start:end]
            win = np.hanning(len(segment))
            avg = np.mean(segment * win)
            
            if np.abs(avg) < 1e-9:
                self.signal_present = False
                return # Noise
                
            phi = np.angle(avg)
            seg_phases.append(phi)
            seg_centers.append(start + seg_len/2)
            
        # Unwrap phases across segments
        seg_phases_unwrapped = np.unwrap(seg_phases)
        
        # Helper time array
        t_centers = np.array(seg_centers) / sr
        
        if len(t_centers) > 1:
            slope, intercept = np.polyfit(t_centers, seg_phases_unwrapped, 1)
            delta_f = slope / (2*np.pi)

            if self._estimates_discarded < self._discard_initial_estimates:
                self._estimates_discarded += 1
                # Keep the UI stable at start: don't integrate or append history yet.
                self.current_freq_dev = 0.0
                self.smoothed_freq_dev = 0.0
                return
            
            # Mean Vector for IQ
            mean_vec = np.mean(z)
            self.iq_history_i.append(np.real(mean_vec))
            self.iq_history_q.append(np.imag(mean_vec))

            self.signal_present = True
            
            self.current_freq_dev = delta_f
            
            # Smoothing (EMA)
            dt = n_samples / sr # approx time per buffer
            tau = self.smoothing_tau
            if tau > 0:
                alpha = dt / (tau + dt)
                # Initialize smoothed value if first valid
                if self.start_time == 0: 
                     self.smoothed_freq_dev = delta_f
                else:
                     self.smoothed_freq_dev = self.smoothed_freq_dev + alpha * (delta_f - self.smoothed_freq_dev)
            else:
                self.smoothed_freq_dev = delta_f
            
            import time
            now = time.time()
            if self.start_time == 0:
                self.start_time = now
            
            # Integrate Phase using RAW Delta F for physical correctness
            self.current_phase_deg += (delta_f * 360.0 * 0.1) 
            
            self.freq_dev_history.append(delta_f)
            self.phase_history.append(self.current_phase_deg)
            self.time_axis.append(now - self.start_time)


class LockInFrequencyCounterWidget(QWidget):
    def __init__(self, module: LockInFrequencyCounter):
        super().__init__()
        self.module = module
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100) # 10Hz

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # -- Controls --
        controls_group = QGroupBox(tr("Settings"))
        controls_layout = QFormLayout()
        
        # NCO Freq
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000)
        self.freq_spin.setValue(1000.0)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.setDecimals(4)
        self.freq_spin.valueChanged.connect(self.on_freq_changed)
        controls_layout.addRow(tr("NCO Frequency:"), self.freq_spin)
        
        # Ref Mode
        self.ref_combo = QComboBox()
        self.ref_combo.addItems(["Internal (NCO)", "Loopback (Ref Out)"])
        self.ref_combo.currentIndexChanged.connect(self.on_ref_mode_changed)
        controls_layout.addRow(tr("Reference Mode:"), self.ref_combo)

        # Input Channel (L/R)
        self.input_ch_combo = QComboBox()
        self.input_ch_combo.addItems([tr("Ch 1"), tr("Ch 2")])
        self.input_ch_combo.setCurrentIndex(int(getattr(self.module, 'signal_channel', 0)))
        self.input_ch_combo.currentIndexChanged.connect(self.on_input_channel_changed)
        controls_layout.addRow(tr("Channel:"), self.input_ch_combo)
        
        # Start/Stop
        self.btn_run = QPushButton(tr("Start"))
        self.btn_run.setCheckable(True)
        self.btn_run.clicked.connect(self.on_run_clicked)
        controls_layout.addRow(self.btn_run)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # -- Plots --
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Frequency Deviation Data
        self.plot_freq = pg.PlotWidget(title=tr("Frequency Deviation Δf (Hz)"))
        self.plot_freq.showGrid(x=True, y=True)
        self.curve_freq = self.plot_freq.plot(pen='g')
        left_layout.addWidget(self.plot_freq)
        
        # Phase Data
        self.plot_phase = pg.PlotWidget(title=tr("Integrated Phase φ (deg)"))
        self.plot_phase.showGrid(x=True, y=True)
        self.curve_phase = self.plot_phase.plot(pen='c')

        left_layout.addWidget(self.plot_phase)
        
        splitter.addWidget(left_widget)
        
        # I-Q Plot
        self.plot_iq = pg.PlotWidget(title=tr("I-Q Phase Space"))
        self.plot_iq.setAspectLocked(True)
        self.plot_iq.showGrid(x=True, y=True)
        self.plot_iq.setXRange(-1, 1)
        self.plot_iq.setYRange(-1, 1)
        self.scatter_iq = pg.ScatterPlotItem(pen=None, brush='y', size=5)
        self.plot_iq.addItem(self.scatter_iq)
        splitter.addWidget(self.plot_iq)
        
        # -- Meters --
        meters_layout = QHBoxLayout()

        # Signal indicator (shown only when signal is missing)
        self.lbl_signal_status = QLabel(tr("No Signal"))
        self.lbl_signal_status.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.lbl_signal_status.setVisible(False)
        meters_layout.addWidget(self.lbl_signal_status)

        self.lbl_delta_f = QLabel("Δf: 0.000 Hz")
        self.lbl_delta_f.setStyleSheet("font-size: 16px; font-weight: bold;")
        meters_layout.addWidget(self.lbl_delta_f)
        
        self.lbl_phase = QLabel("φ: 0.00°")
        self.lbl_phase.setStyleSheet("font-size: 16px; font-weight: bold;")
        meters_layout.addWidget(self.lbl_phase)
        
        layout.addLayout(meters_layout)
        
        splitter.setSizes([600, 300])

    def on_freq_changed(self, val):
        self.module.gen_frequency = val

    def on_ref_mode_changed(self, idx):
        modes = ["internal", "loopback"]
        self.module.ref_mode = modes[idx]

    def on_input_channel_changed(self, idx):
        self.module.signal_channel = int(idx)

    def on_run_clicked(self, checked):
        if checked:
            self.module.start_analysis()
            self.btn_run.setText(tr("Stop"))
            self.btn_run.setStyleSheet("background-color: #ffcccc;")
        else:
            self.module.stop_analysis()
            self.btn_run.setText(tr("Start"))
            self.btn_run.setStyleSheet("")

    def update_ui(self):
        if self.module.is_running:
            self.module.process_data()

            # Signal present indicator
            has_signal = bool(getattr(self.module, 'signal_present', False))
            self.lbl_signal_status.setVisible(not has_signal)
            
            delta_f = self.module.current_freq_dev
            delta_f_smooth = self.module.smoothed_freq_dev
            
            t_data = list(self.module.time_axis)
            f_data = list(self.module.freq_dev_history)
            p_data = list(self.module.phase_history)
            
            if len(t_data) > 0:
                self.curve_freq.setData(t_data, f_data)
                self.curve_phase.setData(t_data, p_data)
                
                i_data = list(self.module.iq_history_i)
                q_data = list(self.module.iq_history_q)
                
                n_tail = 50
                if len(i_data) > n_tail:
                    self.scatter_iq.setData(i_data[-n_tail:], q_data[-n_tail:])
                else:
                    self.scatter_iq.setData(i_data, q_data)
            
            # Meters (Smoothed for consistency)
            self.lbl_delta_f.setText(f"Δf: {delta_f_smooth:.6f} Hz") 
            self.lbl_phase.setText(f"φ: {self.module.current_phase_deg:.2f}°")
