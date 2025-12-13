
import numpy as np
import scipy.signal
import time
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, 
                             QGridLayout, QGroupBox, QHBoxLayout, QDoubleSpinBox)
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg

from src.core.audio_engine import AudioEngine
from src.core.localization import tr
from src.measurement_modules.base import MeasurementModule

class SoundLevelMeter(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        super().__init__()
        self.audio_engine = audio_engine
        self.is_running = False
        
        # Measurement parameters
        self.freq_weighting = 'A'  # A, C, Z
        self.time_weighting = 'FAST'  # FAST, SLOW, IMPULSE, 10ms
        self.channel = 0 # 0 for Left, 1 for Right
        self.target_duration = None # None means continuous
        self.sampling_period = 0.1 # seconds
        self.start_time = None
        self.bandwidth_mode = '20Hz - 20kHz (Wide)' # Default
        
        # State variables
        self.leq_integrator = 0.0
        self.leq_samples = 0
        self.lmax = -np.inf
        self.lmin = np.inf
        self.lpeak = -np.inf
        self.le_integrator = 0.0
        
        self.callback_id = None

        # Filters
        self.sos_filter = None
        self.filter_state = None
        self.bw_filter = None
        self.bw_filter_state = None
        
        # Time weighting constants (tau)
        self.TIME_CONSTANTS = {
            'FAST': 0.125,
            'SLOW': 1.0,
            'IMPULSE': 0.035, # Rise time. Fall is different, see logic below
            '10ms': 0.010
        }
        self.IMPULSE_FALL_TAU = 1.5
        
        # Current instantaneous level (squared pressure)
        self.current_sq_val = 0.0
        
        # Results (thread-safe access needed ideally, but Python GIL helps basic reads)
        self.results = {
            'Lp': -np.inf,
            'Leq': -np.inf,
            'LE': -np.inf,
            'Lmax': -np.inf,
            'Lmin': -np.inf,
            'Lpeak': -np.inf
        }

    @property
    def name(self):
        return "Sound Level Meter"

    @property
    def description(self):
        return "Advanced sound pressure level meter with A/C/Z weighting and time constants."

    def run(self, args):
        pass

    def get_widget(self):
        return SoundLevelMeterWidget(self)

    def set_freq_weighting(self, weighting):
        self.freq_weighting = weighting
        self._update_filters()

    def set_time_weighting(self, weighting):
        self.time_weighting = weighting
        # Reset integration if needed or just continue? Usually better to keep current val but adapt tau.
        # But for impulse it's stateful.

    def set_channel(self, channel):
        self.channel = channel
        self.reset_measurements()

    def set_target_duration(self, duration_str):
        if duration_str == 'Continuous':
            self.target_duration = None
        else:
            # Parse "1s", "1min"
            val = int(duration_str[:-1]) if duration_str[-1] == 's' else int(duration_str[:-3]) * 60
            self.target_duration = float(val)
        # Don't reset immediately, applies to next start? Or reset if running?
        # Usually settings apply to next run.

    def set_sampling_period(self, period):
        self.sampling_period = period

    def set_bandwidth_mode(self, mode):
        # mode: String from combobox
        self.bandwidth_mode = mode
        self._update_filters()

    def reset_measurements(self):
        self.leq_integrator = 0.0
        self.leq_samples = 0
        self.lmax = -np.inf
        self.lmin = np.inf
        self.lpeak = -np.inf
        self.le_integrator = 0.0
        # Reset current value effectively? Maybe keep instant.
        
    def _update_filters(self):
        sr = self.audio_engine.sample_rate
        if not sr: 
            return

        # Bandwidth Filter Design
        # 20Hz is common lower bound.
        # Upper: 12.5k, 20k, 8k
        upper_freq = 20000
        if '12.5kHz' in self.bandwidth_mode:
            upper_freq = 12500
        elif '8kHz' in self.bandwidth_mode:
            upper_freq = 8000
        elif '20kHz' in self.bandwidth_mode:
            upper_freq = 20000
        
        # Ensure upper freq is below Nyquist
        nyquist = sr / 2.0
        if upper_freq >= nyquist * 0.95:
            # Just Highpass 20Hz
            self.bw_filter = scipy.signal.butter(4, 20, btype='highpass', fs=sr, output='sos')
        else:
            # Bandpass 20 - upper
            self.bw_filter = scipy.signal.butter(4, [20, upper_freq], btype='bandpass', fs=sr, output='sos')
            
        self.bw_filter_state = np.zeros((self.bw_filter.shape[0], 2))

        if self.freq_weighting == 'Z':
            self.sos_filter = None
            self.filter_state = None
        elif self.freq_weighting == 'A':
            # A-weighting design (approximate) via bilinear transform or standard library if available.
            # Using scipy.signal.bilinear_zpk or similar if we had analog poles/zeros.
            # Using a standard approximation function here.
            self.sos_filter = self._design_a_weighting(sr)
            self.filter_state = np.zeros((self.sos_filter.shape[0], 2))
        elif self.freq_weighting == 'C':
            self.sos_filter = self._design_c_weighting(sr)
            self.filter_state = np.zeros((self.sos_filter.shape[0], 2))

    def _design_a_weighting(self, fs):
        """Design A-weighting filter."""
        # Constants for A-weighting
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        A1000 = 1.9997
        
        numer = np.poly([0, 0, 0, 0])
        denom = np.convolve(np.poly([-2*np.pi*f4, -2*np.pi*f4]),
                             np.convolve(np.poly([-2*np.pi*f1, -2*np.pi*f1]),
                                         np.poly([-2*np.pi*f2, -2*np.pi*f3])))
        
        # Gain at 1 kHz should be 0 dB.
        # Analog generic formulation. Converting to digital SOS.
        # Simplify: Use scipy's generic filter design tools if possible, or predefined coefficients.
        # For robustness and "good enough" accuracy, let's use the 'ABC_weighting' library logic logic or standard implementation.
        # Here is a standard implementation using scipy.signal.zpk2sos and bilinear.
        
        # Analog poles and zeros
        # z: 0, 0, 0, 0 (s-plane zeros at 0)
        # p: -2*pi*f1, -2*pi*f1, -2*pi*f2, -2*pi*f3, -2*pi*f4, -2*pi*f4
        
        pi = np.pi
        z = [0, 0, 0, 0]
        p = [
            -2*pi*f1, -2*pi*f1,
            -2*pi*f2,
            -2*pi*f3,
            -2*pi*f4, -2*pi*f4
        ]
        k = 1.0 # normalize later
        
        # Normalize to 0dB at 1000Hz
        # Filter response H(s). |H(j*2*pi*1000)| = 1 (0dB) ideally? 
        # Actually standard definition constants usually result in unity gain at passband tip, but let's normalize strictly at 1k.
        
        # Convert to discrete
        zd, pd, kd = scipy.signal.bilinear_zpk(z, p, k, fs)
        sos = scipy.signal.zpk2sos(zd, pd, kd)
        
        # Frequency response check at 1kHz
        w, h = scipy.signal.sosfreqz(sos, worN=[1000], fs=fs)
        gain_1k = np.abs(h[0])
        sos[0, :3] /= gain_1k # Normalization
        
        return sos

    def _design_c_weighting(self, fs):
        """Design C-weighting filter."""
        f1 = 20.598997
        f4 = 12194.217
        
        pi = np.pi
        z = [0, 0]
        p = [
            -2*pi*f1, -2*pi*f1,
            -2*pi*f4, -2*pi*f4
        ]
        k = 1.0
        
        zd, pd, kd = scipy.signal.bilinear_zpk(z, p, k, fs)
        sos = scipy.signal.zpk2sos(zd, pd, kd)
        
        # Normalize at 1kHz
        w, h = scipy.signal.sosfreqz(sos, worN=[1000], fs=fs)
        gain_1k = np.abs(h[0])
        sos[0, :3] /= gain_1k
        
        return sos

    def start_analysis(self):
        if self.is_running:
            return
        
        self.is_running = True
        self._update_filters()
        self.reset_measurements()
        self.start_time = time.time()
        
        # Setup callback
        try:
            self.callback_id = self.audio_engine.register_callback(self.callback)
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            self.is_running = False

    def stop_analysis(self):
        if not self.is_running:
            return
        self.is_running = False
        if self.callback_id is not None:
             self.audio_engine.unregister_callback(self.callback_id)
             self.callback_id = None

    def callback(self, indata, outdata, frames, time_info, status):
        if not self.is_running:
            return

        # Check duration
        if self.target_duration is not None:
            if time.time() - self.start_time >= self.target_duration:
                self.is_running = False
                return

        # Mono processing for now (use channel 0 or mix?)
        # Let's use the first selected input channel or average if stereo.
        # Assuming indata is (frames, channels).
        
        if indata.shape[1] > self.channel:
            sig = indata[:, self.channel]
        else:
            # Fallback to channel 0 if requested channel doesn't exist
            sig = indata[:, 0]

        # If calibration is available, apply it to convert to Pascal? 
        # BUT wait, the calibration in settings is global input scale factor? 
        # Or usually 'dBFS to real unit'.
        # Let's assume sig is in FS (-1.0 to 1.0).
        # We need a calibration factor to convert FS to Pa.
        # For now, let's treat FS as the unit and user sees dBFS-based SPL unless calibrated.
        # NOTE: The requirement says "use settings calibration value".
        # We will apply that in the final dB calculation or here. 
        # Typically calibration is "X dB at 1.0 FS" or "Sensitivity X V/Pa".
        # Let's do raw calculations and offset dB at display time or use a sensitivity factor here.
        # For robustness, let's calculate in FS and add offset in display.
        # Ah, but integrators need linear values.
        # Let's assume 1.0 FS = 0 dB for internal math, then add global offset.
        
        # Apply Bandwidth Filter (HighSens/Wide/Normal)
        if self.bw_filter is not None and self.bw_filter_state is not None:
            sig, self.bw_filter_state = scipy.signal.sosfilt(self.bw_filter, sig, zi=self.bw_filter_state)

        # Apply Frequency Weighting
        if self.sos_filter is not None and self.filter_state is not None:
             sig, self.filter_state = scipy.signal.sosfilt(self.sos_filter, sig, zi=self.filter_state)
        
        # Square signal for power
        sq_sig = sig**2
        
        # Time Weighting
        # Digital implementation of RC low-pass on squared signal
        # y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        # alpha = 1 - exp(-1 / (fs * tau))
        
        sr = self.audio_engine.sample_rate or 48000
        
        tau = self.TIME_CONSTANTS.get(self.time_weighting, 0.125)
        
        # Impulse is special: Rise fast, fall very slow
        if self.time_weighting == 'IMPULSE':
            alpha_rise = 1.0 - np.exp(-1.0 / (sr * 0.035))
            alpha_fall = 1.0 - np.exp(-1.0 / (sr * 1.5))
            
            # This is sample-by-sample, which is slow in Python.
            # We can try a block approximation if frames is small, or numba if allowed.
            # Vectorized approximation:
            # It's hard to vectorize conditional IIR.
            # Fallback: strict simple IIR with one tau if we can't do dual easily, 
            # OR simple max envelope per chunk for peak? No, Impulse is defined dynamics.
            # Let's do a simple Cython-like optimization or just use slow Python loop for Impulse?
            # Or... just use the small tau for everything? No, that's wrong.
            # Let's approximate: 
            # Since audio callbacks are small chunks (e.g. 1024 frames), 
            # doing a loop of 1024 in Python is arguably ok-ish for a specialized meter?
            # Actually, standard Impulse metering is often just "Fast" with peak hold, but the standard says different.
            # Let's implement a simplified block-based approach or use scipy lfilter if possible.
            # Since true Impulse weighting requires conditional update, it's non-linear.
            # We will use a naive python loop for now. It might eat CPU.
            # Optimization: Use just FAST for now if Impulse is too heavy? 
            # Let's try to implement a slightly optimized numpy version if possible.
            # Actually, scipy.signal.lfilter is good for constant alpha.
            # For 'IMPULSE', we use a distinct detector.
            
            # Hack for performance: Use 'FAST' alpha for the whole block for now if time is tight,
            # but that's not 'IMPULSE'.
            # Let's stick to 'FAST', 'SLOW', '10ms' using lfilter for now.
            # 'IMPULSE' logic:
            current_val = self.current_sq_val
            
            # Simple Python loop - optimizing by minimizing lookups
            # This is risky for performance.
            # Let's default Impulse to Fast for the initial 'execute' phase to ensure stability 
            # unless we find a fast way.
            # User asked for Impulse. Let's do it properly but watch out.
            
            # Actually, let's implement the block filter with constant time constant for F/S/10ms
            # and only do special logic for Impulse.
            pass
        
        alpha = 1.0 - np.exp(-1.0 / (sr * tau))
        
        if self.time_weighting != 'IMPULSE':
            # Exponential moving average filter
            # y = lfilter([alpha], [1, -(1-alpha)], x)
            # We need to maintain state.
            
            # initial state
            zi = [self.current_sq_val * (1 - alpha)]
            
            filtered_sq, zf = scipy.signal.lfilter([alpha], [1, -(1-alpha)], sq_sig, zi=zi)
            
            self.current_sq_val = filtered_sq[-1]
            block_vals = filtered_sq
        else:
            # Impulse implementation
            # Need to iterate.
            # x[n] is input power.
            # if x[n] > y[n-1]: tau = 35ms
            # else: tau = 1500ms
            # This is a Peak detector with different attack/release.
            
            vals = []
            curr = self.current_sq_val
            alpha_rise = 1.0 - np.exp(-1.0 / (sr * 0.035))
            alpha_fall = 1.0 - np.exp(-1.0 / (sr * 1.5))
            
            # Very slow in pure python. 
            # Let's process in small chunks or use a numba JIT if available (likely not in this env).
            # We'll assume the user isn't running very high sample rates or accept some CPU load.
            # Vectorization trick:
            # It's an attack/release filter.
            # We can use a recursive approach.
            
            for s in sq_sig:
                if s > curr:
                    curr = alpha_rise * s + (1 - alpha_rise) * curr
                else:
                    curr = alpha_fall * s + (1 - alpha_fall) * curr
                vals.append(curr)
            
            self.current_sq_val = curr
            block_vals = np.array(vals)
            
        # Update Measurements
        
        # Lp (Instantaneous / Time-weighted level)
        # Taking the last value of the block is common for display update, 
        # but for max/min we should scan the block.
        lp_inst = block_vals[-1]
        
        # Lmax, Lmin
        # We assume Lmax/Lmin are derived from the Time-Weighted signal.
        blk_max = np.max(block_vals)
        blk_min = np.min(block_vals)
        
        if blk_max > 1e-12: # avoid log of zero issues implicitly later
            if blk_max > self.current_sq_val: pass # logic check
            
        # Update state Lmax (store in linear power to avoid log calls in callback, convert later)
        # Actually usually stored in dB, but linear is safer for aggregation.
        # Wait, Lmax is max of the weighted level? Yes, usually.
        self.lmax = max(self.lmax, blk_max)
        self.lmin = min(self.lmin, blk_min)
        
        # Leq (Equivalent Continuous Sound Level)
        # Average of squared pressure over time (unweighted by time constant, but freq weighted).
        # So we integrate the raw 'sq_sig' (which is freq weighted but not time weighted).
        self.leq_integrator += np.sum(sq_sig)
        self.leq_samples += len(sq_sig)
        
        # LE (Sound Exposure Level)
        # Energy normalized to 1 second.
        self.le_integrator += np.sum(sq_sig) / sr # Energy accumulator? 
        # LE = 10 log10( sum(p^2 * dt) / p0^2 / T0 ) where T0 = 1s.
        # dt = 1/sr. sum(p^2) / sr.
        
        # Lpeak (Peak Sound Level)
        # Max of the absolute raw signal (freq weighted, NO time weighting).
        # raw peak
        peak_curr = np.max(sq_sig)
        self.lpeak = max(self.lpeak, peak_curr)
        
        # Store for display (Atomic update preferred)
        self.results['Lp'] = 10 * np.log10(lp_inst + 1e-12)
        self.results['Leq'] = 10 * np.log10((self.leq_integrator / (self.leq_samples + 1e-12)) + 1e-12)
        # LE: 10 log10 ( Integral(p^2) dt ). dt = 1/sr. 
        # Leq = LE - 10 log10(T).
        # LE = 10 log10 ( sum(sq_sig) / sr ).
        total_energy = self.le_integrator # sum(sq_sig) // wait, leq_int is sum(sq_sig).
        # My le_integrator variable above was accumulating sum/sr?
        # Let's fix. le_integrator will track sum of p^2.
        # Actually, self.leq_integrator tracks sum(p^2).
        
        le_val = (self.leq_integrator / sr) + 1e-12
        self.results['LE'] = 10 * np.log10(le_val)
        
        self.results['Lmax'] = 10 * np.log10(self.lmax + 1e-12)
        self.results['Lmin'] = 10 * np.log10(self.lmin + 1e-12)
        self.results['Lpeak'] = 10 * np.log10(self.lpeak + 1e-12)


class SoundLevelMeterWidget(QWidget):
    def __init__(self, module: SoundLevelMeter):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(50) # 20Hz refresh

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Controls Group
        controls_group = QGroupBox(tr("Settings"))
        
        # Use a vertical layout to hold two rows of controls
        settings_layout = QVBoxLayout()
        row1_layout = QHBoxLayout()
        row2_layout = QHBoxLayout()
        
        # --- Row 1 ---
        
        # Start/Stop
        self.btn_start = QPushButton(tr("Start"))
        self.btn_start.setCheckable(True)
        self.btn_start.toggled.connect(self.on_start_toggle)
        row1_layout.addWidget(self.btn_start)
        
        # Reset
        self.btn_reset = QPushButton(tr("Reset"))
        self.btn_reset.clicked.connect(self.module.reset_measurements)
        row1_layout.addWidget(self.btn_reset)
        
        # Channel Selection
        row1_layout.addWidget(QLabel(tr("Channel:")))
        self.combo_channel = QComboBox()
        self.combo_channel.addItems(['L', 'R'])
        self.combo_channel.currentIndexChanged.connect(self.module.set_channel)
        row1_layout.addWidget(self.combo_channel)

        # Freq Weighting
        row1_layout.addWidget(QLabel(tr("Freq Weight:")))
        self.combo_freq = QComboBox()
        self.combo_freq.addItems(['A', 'C', 'Z'])
        self.combo_freq.currentTextChanged.connect(self.module.set_freq_weighting)
        row1_layout.addWidget(self.combo_freq)
        
        row1_layout.addStretch()

        # --- Row 2 ---

        # Bandwidth
        row2_layout.addWidget(QLabel(tr("Bandwidth:")))
        self.combo_bw = QComboBox()
        self.combo_bw.addItems(['20Hz - 20kHz (Wide)', '20Hz - 12.5kHz', '20Hz - 8kHz (Normal)'])
        self.combo_bw.currentTextChanged.connect(self.module.set_bandwidth_mode)
        row2_layout.addWidget(self.combo_bw)
        
        # Time Weighting
        row2_layout.addWidget(QLabel(tr("Time Weight:")))
        self.combo_time = QComboBox()
        self.combo_time.addItems(['FAST', 'SLOW', 'IMPULSE', '10ms'])
        self.combo_time.currentTextChanged.connect(self.module.set_time_weighting)
        row2_layout.addWidget(self.combo_time)

        # Measurement Time
        row2_layout.addWidget(QLabel(tr("Duration:")))
        self.combo_duration = QComboBox()
        self.combo_duration.addItems(['Continuous', '1s', '3s', '5s', '10s', '20s', '30s', '1min'])
        self.combo_duration.currentTextChanged.connect(self.module.set_target_duration)
        row2_layout.addWidget(self.combo_duration)

        # Sampling Period
        row2_layout.addWidget(QLabel(tr("Lp Interval:")))
        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(0.01, 10.0)
        self.spin_interval.setSingleStep(0.1)
        self.spin_interval.setValue(0.1)
        self.spin_interval.setSuffix(" s")
        self.spin_interval.valueChanged.connect(self.module.set_sampling_period)
        row2_layout.addWidget(self.spin_interval)
        
        row2_layout.addStretch()
        
        settings_layout.addLayout(row1_layout)
        settings_layout.addLayout(row2_layout)
        controls_group.setLayout(settings_layout)
        layout.addWidget(controls_group)
        
        # Display Grid
        display_group = QGroupBox(tr("Measurements"))
        grid_layout = QGridLayout()
        
        # Create labels
        self.labels = {}
        metrics = [
            ('Lp', tr("Sound Pressure Level"), 0, 0),
            ('Leq', tr("Equivalent Continuous Level"), 0, 1),
            ('Lmax', tr("Maximum Level"), 1, 0),
            ('Lmin', tr("Minimum Level"), 1, 1),
            ('Lpeak', tr("Peak Level"), 2, 0),
            ('LE', tr("Sound Exposure Level"), 2, 1)
        ]
        
        font_style = "font-size: 24px; font-weight: bold; color: #00ff00;"
        
        for key, desc, r, c in metrics:
            container = QWidget()
            v_box = QVBoxLayout()
            
            title = QLabel(key)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("font-weight: bold; font-size: 14pt;")
            
            desc_lbl = QLabel(desc)
            desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc_lbl.setStyleSheet("font-size: 10pt; color: #aaa;")
            
            value_lbl = QLabel("--.-")
            value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_lbl.setStyleSheet(font_style)
            
            v_box.addWidget(title)
            v_box.addWidget(desc_lbl)
            v_box.addWidget(value_lbl)
            container.setLayout(v_box)
            
            # Add frame or background?
            container.setStyleSheet("background-color: #111; border-radius: 5px; margin: 5px;")
            
            grid_layout.addWidget(container, r, c)
            self.labels[key] = value_lbl
            
        display_group.setLayout(grid_layout)
        layout.addWidget(display_group)
        
        layout.addStretch()
        self.setLayout(layout)

    def on_start_toggle(self, checked):
        if checked:
            self.btn_start.setText(tr("Stop"))
            self.module.start_analysis()
        else:
            self.btn_start.setText(tr("Start"))
            self.module.stop_analysis()

    def update_display(self):
        # Check if stopped automatically
        if not self.module.is_running and self.btn_start.isChecked():
            self.btn_start.setChecked(False)
            self.btn_start.setText(tr("Start"))
            self.module.stop_analysis()

        if not self.module.is_running:
            return
            
        # Get calibration offset from AudioEngine
        cal_db = 0.0
        if hasattr(self.module.audio_engine, 'calibration'):
             cal_ptr = self.module.audio_engine.calibration
             if hasattr(cal_ptr, 'get_spl_offset_db'):
                 cal_db = cal_ptr.get_spl_offset_db() or 0.0
        
        vals = self.module.results
        
        for key, lbl in self.labels.items():
            val = vals.get(key, -np.inf)
            if np.isinf(val):
                lbl.setText("--.-")
            else:
                # Apply calibration
                display_val = val + cal_db
                lbl.setText(f"{display_val:.1f} dB")

