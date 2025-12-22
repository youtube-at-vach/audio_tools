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
        
        # Settings
        self.gen_frequency = 1000.0 # NCO Frequency
        self.signal_channel = 0 # 0: Left
        self.ref_channel = 1    # 1: Right (Used in Audio REF mode)
        self.ref_mode = "internal" # internal, loopback, audio
        
        # Internal State
        self._nco_phase = 0.0
        self._last_unwrapped_phase = 0.0
        self._first_run = True
        
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
                
            # Output Generation (NCO Loopback or Silence)
            outdata.fill(0)
            
            # Ref Mode Logic could go here (e.g. if Loopback, output sine)
            # But the 'Lock-in' implies we might output the REF signal for the device under test to lock to,
            # OR we are just passively measuring.
            # User request: Loopback (DAC -> ADC).
            
            if self.ref_mode == "loopback":
                # Generate Sine for Output
                t = (np.arange(frames) + self._nco_phase) / sample_rate
                # self._nco_phase updated below? No, NCO phase for DEMOD vs Output should be synced if "Internal"
                # But here 'Loopback' explicitly means we output the 'Reference'
                
                # Careful: The NCO used for demodulation is updated in process_data (or here?).
                # Ideally, we use the SAME phase counter for both if we want perfect "Internal" loopback.
                # However, process_data runs on the GUI thread / timer usually in this architecture (snapshot),
                # while this callback is audio thread.
                # To really be accurate, we should probably generate output here.
                
                # For simplicity in this architecture where process_data analyzes the buffer *snapshot*:
                # We'll just generate output here for loopback.
                # The Demodulation NCO in process_data will be independent (software NCO).
                # This mimics "Two independent oscillators" unless we are very careful.
                # But wait, user wants "Phase Integral Counter".
                # If we output 1000Hz and measure 1000Hz, we expect constant phase.
                # If we rely on process_data (snapshot) NCO, it's fine as long as we know 't'.
                
                sig = 0.5 * np.cos(2 * np.pi * self.gen_frequency * t)
                outdata[:, 0] = sig
                outdata[:, 1] = sig
                
            # Update phase for next block (only used for output generation if loopback)
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

        # Get Snapshot
        data = self.input_data
        sig = data[:, self.signal_channel]
        n_samples = len(sig)
        sr = self.audio_engine.sample_rate
        
        # Audio REF Mode: Update gen_frequency from CH2?
        # User said: "Audio REF will measure the frequency of CH2 once ... and set NCO"
        # Let's do a quick zero-crossing or dominant frequency check on Ref channel if requested.
        # For now, let's assume 'Audio REF' simply uses the *input* from CH2 as the Reference Vector?
        # "That reference near which NCO is placed".
        # Interpretation: The NCO frequency is set MANUALLY, or snapped to CH2.
        # Let's stick to: NCO is manual, but maybe we have a button "Snap to Ref".
        
        # Demodulation against NCO
        # We need to know the exact time of this buffer to generate coherent NCO?
        # This architecture (buffer snapshot) is tricky for "Continuous Phase" over long time 
        # because we might miss frames between snapshots or overlap.
        # ACTUALLY: The correct way for a "Counter" is to process EVERY frame in the audio callback.
        # But `process_data` is called by a Timer in `LockInAmplifier`.
        # If we just analyze the snapshot, we assume the signal is stationary.
        # BUT, for "Phase Integration", we need continuity. 
        # If we restart the NCO phase at 0 for every buffer, we measure "Phase offset vs t=0 of buffer", 
        # which is scrambling if the buffer start time is random.
        
        # FIX: We can't easily do continuous phase integration on snapshots without timestamps.
        # HOWEVER, we can measure the *Frequency Offset* in the buffer locally,
        # and Integrate that over time?
        # Or, we can assume the NCO starts at t=0 of the *buffer* and we measure the phase *relative to the start of the buffer*.
        # If the actual signal is drifting, the Phase within the buffer will have a slope.
        # Slope = Frequency Difference.
        # Phase (Average of buffer) = Instantaneous Phase difference.
        
        # Let's try "Slope Method":
        # 1. Demodulate with NCO (freq = f_nco) for the whole buffer.
        # 2. Extract Phase(t) within the buffer.
        # 3. Fit linear slope to Phase(t).
        #    Slope = d(Phi)/dt = 2*pi * delta_f.
        #    Intercept = Instantaneous Phase.
        
        t = np.arange(n_samples) / sr
        
        # NCO
        osc = np.exp(-1j * 2 * np.pi * self.gen_frequency * t)
        
        # Demodulate
        # analytic_sig = sig * osc (freq shift)
        # Low pass?
        # If we just do Multi:
        # I = sig * cos(...)
        # Q = sig * -sin(...)
        # signal ~ cos(w_s t + phi)
        # result has diff freq (w_s - w_nco) and sum freq.
        # We want the diff freq component.
        
        # Analytic signal approach (Hilbert) on the *Demodulated* result? 
        # Or simpler:
        dt = 1/sr
        
        # Baseband signal (Complex)
        # z(t) = sig(t) * exp(-i * w_nco * t)
        # If sig(t) = A * cos(w_sig * t + phi) = A/2 * (exp(i...) + exp(-i...))
        # z(t) = A/2 * exp(i * (w_sig - w_nco)t + phi) + (high freq term)
        
        # We filter out the high freq term. A simple mean over the buffer is a Sync Filter (Sinc).
        # For a "Counter", we want the time-evolution within the buffer.
        
        z = sig * osc # This mixes down.
        
        # Instantaneous Phase of z(t)
        # We need to remove the 2*w component first.
        # A simple moving average or low-pass filter?
        # Or just use the fact that 2*w is fast.
        
        # Let's use a simple windowed projection to get a SINGLE point (Average Phase, Average Freq Diff) for this buffer.
        # This is robust.
        # BUT, we want to plot Delta f(t) and Phi(t) continuously.
        # If we just get one point per buffer update (10Hz), that's fine for the graph.
        
        # 1. Average Frequency Difference in this buffer
        # dPhi/dt. 
        # Calculate unwrapped phase of z(t).
        # But z(t) is noisy and has 2w component.
        # Better: FFT of z(t)? No, too coarse.
        # Linear Regression on Phase of z(t)?
        # Let's try Linear Regression on Unwrapped Phase of z(t). 
        # To avoid 2w ripple, we should filter z(t) first?
        # or just fit? If buffer is long (4096 @ 48k ~ 85ms), 1000Hz -> many cycles. 2w ripple averages out?
        
        # Let's calc phase array
        # Unwrap?
        # But arc2(z) is messy if z crosses zero or has noise.
        
        # Robust approach:
        # Sum(z * window) -> Average Vector.
        # Angle(Average Vector) -> Average Phase Diff in this window.
        # But this loses the Frequency Diff info if it rotates > 2pi in the window (washed out).
        # If Delta f is small (< 1/T_buffer), Vector Average is fine.
        # If Delta f is large, Vector Average magnitude drops.
        
        # Assumption: We are "Locked in" or close. Delta f is small.
        # We calculate:
        # I_mean = mean(real(z))
        # Q_mean = mean(imag(z))
        # Phase = atan2(Q, I)
        
        # How to get Delta F?
        # We can compare Phase of *this* buffer vs *previous* buffer?
        # Problem: We don't know exact time gap between buffers (system timer jitter).
        # Unless we use audio indices... but we are in a separate thread/timer usually.
        # Wait, measurement_modules usually run `process_data` on a timer?
        # Yes, `QTimer` in the widget updates UI, but `process_data` is called... 
        # In LIA, `process_data` is called MANUALLY by the Worker (if in Sweep) or effectively by the UI timer?
        # In LIA widget: `process_data` is called... wait.
        # LIA code line 462: `self.timer.timeout.connect(self.update_ui)`
        # LIA update_ui calls process_data? No.
        # LIA `process_data` is NOT called by timer automatically in the provided code snippet!
        # Ah, look at `LockInAmplifier.start_analysis`: it registers a callback.
        # The callback FILLS `self.input_data`.
        # Who calls `process_data`?
        # In `LockInAmplifierWidget.update_ui` (not shown fully in snippet, but implied), it reads `self.current_magnitude`.
        # BUT `process_data` calculates `current_magnitude`.
        # I need to check if `LockInAmplifier` handles `process_data` internally or if the widget calls it.
        # Checking `LockInAmplifier` class... it has `process_data`.
        # Line 101: `get_widget` returns `LockInAmplifierWidget`.
        # The Widget timer calls `update_ui`.
        # I suspect `update_ui` calls `process_data`? Or `process_data` should be called on a timer?
        
        # Ah, looking at LIA logic again:
        # The `process_data` method reads `self.input_data` (buffer) and computes.
        # It is likely intended that the Widget's timer calls `module.process_data()` before updating labels.
        # I will assume that pattern.
        
        # Back to Frequency Calculation:
        # If we can't trust time between `process_data` calls (jitter), we can't differentiate Phase across calls accurately for fine Hz.
        # UNLESS we rely on the buffer's internal timeline.
        # We can split the buffer into 2 halves?
        # Phase1 = Phase of 1st half.
        # Phase2 = Phase of 2nd half.
        # dt = (N/2) / SR.
        # Delta f = (Phase2 - Phase1) / (2*pi * dt).
        # This is robust and self-contained in one snapshot!
        
        # Implementation:
        w = np.hanning(n_samples)
        z = sig * osc # Mixing
        
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
            # Windowing?
            # A simple mean is a boxcar. Hanning is better.
            win = np.hanning(len(segment))
            avg = np.mean(segment * win)
            
            # If magnitude is too low, phase is noise.
            if np.abs(avg) < 1e-9:
                return # Noise
                
            phi = np.angle(avg)
            seg_phases.append(phi)
            seg_centers.append(start + seg_len/2)
            
        # Unwrap phases across segments
        seg_phases_unwrapped = np.unwrap(seg_phases)
        
        # Helper time array
        t_centers = np.array(seg_centers) / sr
        
        # Linear Regression: Phi = m*t + c
        # m = dPhi/dt (rad/s)
        # Delta f = m / (2*pi)
        
        if len(t_centers) > 1:
            slope, intercept = np.polyfit(t_centers, seg_phases_unwrapped, 1)
            delta_f = slope / (2*np.pi)
            
            # Phase at center of buffer?
            # avg_phase = np.mean(seg_phases_unwrapped) # or evaluate line at center
            # Let's take the mean phase of the whole buffer for the "I-Q" plot
            # (Vector sum of whole buffer)
            
            # Whole buffer vector
            # But if delta_f is large, the vector sum cancels out!
            # If we want to visualize "Phase vs Time", we are visualizing the phase of the beat note.
            # If Delta_f is 1Hz, the phase rotates 360 deg per second.
            
            # For I-Q plot: we plot I and Q of the *averaged* vector?
            # If rotating, it will be near zero.
            # Maybe we plot the *Instantaneous* I/Q? 
            # "Phase Space (I-Q Plot) -> X: I, Y: Q"
            # If it's a counter, usually we are close to lock. 
            # Let's plot the average vector of the buffer. 
            # If it's spinning fast, the user sees a dot at 0, which is correct (not locked).
            # If it's spinning slow (1Hz), the dot moves in a circle on the plot update (10fps).
            
            # Mean Vector
            mean_vec = np.mean(z) # No window for raw mean?
            self.iq_history_i.append(np.real(mean_vec))
            self.iq_history_q.append(np.imag(mean_vec))
            
            # Phase Tracking
            # We want to accumulate phase over time for the "Phase vs Time" plot.
            # We have delta_f.
            # Current absolute phase (relative to NCO t=0 which is arbitrary per buffer)
            # The "Intercept" from polyfit gives us Phase at t=0 of THIS BUFFER.
            # BUT this buffer's t=0 is essentially "now" relative to the NCO restart.
            # Since we restart NCO at t=0 for every process_data call (in this logic),
            # Phase is always relative to a cosine starting *at the beginning of the acquisition*.
            # This is fine for checking stability.
            
            # Wait, if I assume NCO starts at 0 every time, but the Signal is continuous...
            # Then "Phase" will jump randomly because we don't know the phase of the Signal relative to the buffer start.
            # UNLESS the buffer capture is perfectly contiguous?
            # It IS contiguous in the audio engine, but `input_data` is a circular buffer or just latest chunk?
            # `LockInAmplifier` just copies `[-buffer_size:]`. 
            # If we call `process_data` faster than real time, we share data.
            # If slower, we skip.
            
            # CRITICAL: We cannot track absolute accumulated phase without precise timestamps or handling the NCO continuity.
            # The "Frequency Deviation" calculation (Slope method) is robust to phase jumps between buffers.
            # So Graph 1 (Delta F) is fine.
            # Graph 2 (Phase) will be jumpy if we don't integrate Delta F ourselves.
            
            # Solution:
            # Integrate the measured Delta F to get "Synthesized Phase".
            # Phi_accum += Delta_f * dt_update.
            # This shows the "Phase Drift".
            
            self.current_freq_dev = delta_f
            
            # For display, we accumulate phase
            dt_update = 0.1 # approx if timer is 100ms.
            # Better: use real time delta?
            import time
            now = time.time()
            if self.start_time == 0:
                self.start_time = now
            
            # Accumulate Phase
            # dPhi = 2 * pi * delta_f * dt
            # But wait, we can just display the "Phase Offset" relative to the NCO *frame*.
            # If we can't sync NCO frame-to-frame, we can't show "Absolute Phase Diff".
            
            # Let's stick to "Frequency Deviation".
            # And "Phase Noise" (Std Dev within buffer).
            # And for the Phase Plot... maybe just integral of Delta F?
            # User wants "Phase difference phi(t)".
            # If we assume the Widget Timer is reasonably steady, we can integrate.
            
            self.current_phase_deg += (delta_f * 360.0 * dt_update)
            # Normalize? No, users want to see drift (wrapping or multiple cycles).
            
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
        self.ref_combo.addItems(["Internal (NCO)", "Loopback (Ref Out)", "Audio REF (CH2)"])
        self.ref_combo.currentIndexChanged.connect(self.on_ref_mode_changed)
        controls_layout.addRow(tr("Reference Mode:"), self.ref_combo)
        
        # Start/Stop
        self.btn_run = QPushButton(tr("Start"))
        self.btn_run.setCheckable(True)
        self.btn_run.clicked.connect(self.on_run_clicked)
        controls_layout.addRow(self.btn_run)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # -- Plots --
        # Splitter: Left (Graphs), Right (IQ)
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
        
        # Phase Drift Indicator (Label above plot)
        self.lbl_drift = QLabel(tr("Phase Drift: --"))
        self.lbl_drift.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_drift.setStyleSheet("font-size: 14pt; font-weight: bold; color: #aaaaaa;")
        left_layout.addWidget(self.lbl_drift)
        
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
        self.lbl_delta_f = QLabel("Δf: 0.000 Hz")
        self.lbl_delta_f.setStyleSheet("font-size: 16px; font-weight: bold;")
        meters_layout.addWidget(self.lbl_delta_f)
        
        self.lbl_phase = QLabel("φ: 0.00°")
        self.lbl_phase.setStyleSheet("font-size: 16px; font-weight: bold;")
        meters_layout.addWidget(self.lbl_phase)
        
        layout.addLayout(meters_layout)
        
        # Set splitter sizes
        splitter.setSizes([600, 300])

    def on_freq_changed(self, val):
        self.module.gen_frequency = val

    def on_ref_mode_changed(self, idx):
        modes = ["internal", "loopback", "audio"]
        self.module.ref_mode = modes[idx]

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
            
            # Use raw deviation for plot
            delta_f = self.module.current_freq_dev
            
            # Update Plots
            t_data = list(self.module.time_axis)
            f_data = list(self.module.freq_dev_history)
            p_data = list(self.module.phase_history)
            
            if len(t_data) > 0:
                self.curve_freq.setData(t_data, f_data)
                self.curve_phase.setData(t_data, p_data)
                
                # Update Drift Indicator
                drift_uhz = delta_f * 1_000_000
                
                if drift_uhz > 0:
                    arrow = "↑"
                    color = "#00ffff" # Cyan
                else:
                    arrow = "↓"
                    color = "#00ffff" # Cyan
                
                # To distinguish zero?
                if abs(drift_uhz) < 0.001:
                    arrow = "-"
                    color = "#88ff88" # Green for "Locked"
                
                self.lbl_drift.setText(f"Phase Drift: {arrow} {drift_uhz:+.3f} µHz")
                self.lbl_drift.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {color};")
                
                # IQ
                i_data = list(self.module.iq_history_i)
                q_data = list(self.module.iq_history_q)
                
                # Just show last N points for IQ to avoid clutter
                n_tail = 50
                if len(i_data) > n_tail:
                    self.scatter_iq.setData(i_data[-n_tail:], q_data[-n_tail:])
                else:
                    self.scatter_iq.setData(i_data, q_data)
            
            # Meters
            self.lbl_delta_f.setText(f"Δf: {delta_f:.6f} Hz") # High precision on label too
            self.lbl_phase.setText(f"φ: {self.module.current_phase_deg:.2f}°")
