
import time

import numpy as np
import pyqtgraph as pg
import scipy.signal
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

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

        # LN Statistics
        self.ln_history = []
        self.last_sample_time = 0.0
        self.LN_SAMPLING_PERIOD = 0.1 # Fixed 0.1s for statistics as requested

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
            'Lpeak': -np.inf,
            'LN': {} # L5, L10, L50, L90, L95, Lhigh, Llow, Lave
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

        # Reset LN Statistics
        self.ln_history = []
        self.last_sample_time = time.time()

        # Reset Results
        self.results = {
            'Lp': -np.inf,
            'Leq': -np.inf,
            'LE': -np.inf,
            'Lmax': -np.inf,
            'Lmin': -np.inf,
            'Lpeak': -np.inf,
            'LN': {}
        }

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

        np.poly([0, 0, 0, 0])
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
        self.last_sample_time = self.start_time

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
            self.current_sq_val

            # Simple Python loop - optimizing by minimizing lookups
            # This is risky for performance.
            # Let's default Impulse to Fast for the initial 'execute' phase to ensure stability
            # unless we find a fast way.
            # User asked for Impulse. Let's do it properly but watch out.

            # Actually, let's implement the block filter with constant time constant for F/S/10ms
            # and only do special logic for Impulse.

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


        # LN Data Collection (0.1s interval)
        current_time = time.time()
        if current_time - self.last_sample_time >= self.LN_SAMPLING_PERIOD:
            # Add current instantaneous level (Lp) to history
            # Lp is already calculated as linear power 'lp_inst'
            # Convert to dB for storage? Better to store linear for averaging (Lave),
            # but usually Ln percentiles are on dB values.
            # Lave is "Energy Average" usually -> Leq.
            # But "Lave" might effectively be Leq.
            # Let's store LINEAR power to support accurate Leq/Lave calculation of the subset,
            # and convert to dB for sorting/percentiles.
            self.ln_history.append(lp_inst)
            self.last_sample_time = current_time

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

        # Calculate/Update Statistics periodically (or on demand)
        # Since this is the audio callback, we should NOT sort a potentially large array here.
        # It's better to defer this to the GUI timer or a separate method called by GUI.
        # We just collected the data.

    def calculate_ln_statistics(self):
        """Calculate LN statistics from history. Called by GUI."""
        if not self.ln_history:
            return {}

        data_linear = np.array(self.ln_history)
        data_db = 10 * np.log10(data_linear + 1e-12)

        # Percentiles (Ln is level EXCEEDED n% of time)
        # So L5 is the 95th percentile of the distribution (high level).
        # L95 is the 5th percentile (low level).
        # numpy.percentile calculates "p-th percentile below which p% of observations fall".
        # So L5 = np.percentile(data, 95)
        # L95 = np.percentile(data, 5)

        p_vals = np.percentile(data_db, [95, 90, 50, 10, 5])
        l5, l10, l50, l90, l95 = p_vals

        lhigh = np.max(data_db)
        llow = np.min(data_db)

        # Lave (Arithmetic average of levels or Energy average?)
        # Standard in acoustics: Lave usually implies Leq (Energy Average).
        # But if it means "Average of the sampled dB values", that's different.
        # "Lave" (Label) usually acts as Leq short term or specified period.
        # Since we have the linear history, we can calc Leq of the history.
        lave = 10 * np.log10(np.mean(data_linear) + 1e-12)

        return {
            'L5': l5,
            'L10': l10,
            'L50': l50,
            'L90': l90,
            'L95': l95,
            'Lhigh': lhigh,
            'Llow': llow,
            'Lave': lave
        }

    def get_ln_histogram(self, bin_size=0.5):
        """
        Calculate histogram of LN history.
        Args:
            bin_size (float): Bin size in dB.
        Returns:
            tuple: (bins, probabilities)
                   bins: Center frequencies of bins
                   probabilities: Normalized count (pdf) or absolute count? 
                                  Let's return normalized probability (sum=100% or 1.0)
        """
        if not self.ln_history:
            return np.array([]), np.array([])

        data_linear = np.array(self.ln_history)
        data_db = 10 * np.log10(data_linear + 1e-12)

        # Determine range
        min_val = np.min(data_db)
        max_val = np.max(data_db)

        # Align bins to bin_size
        start = np.floor(min_val / bin_size) * bin_size
        end = np.ceil(max_val / bin_size) * bin_size

        # Create bins edges
        # If flat, make sure we have at least one bin
        if start == end:
            end += bin_size

        bins = np.arange(start, end + bin_size, bin_size)

        hist, bin_edges = np.histogram(data_db, bins=bins, density=False)

        # Calculate centers
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize to percent?
        total = np.sum(hist)
        if total > 0:
            probs = (hist / total) * 100.0
        else:
            probs = hist

        return centers, probs


class SoundLevelMeterWidget(QWidget):
    def __init__(self, module: SoundLevelMeter):
        super().__init__()
        self.module = module
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(50) # 20Hz refresh

    def init_ui(self):
        # Main Layout: Sidebar (Left) + Content (Right)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Sidebar ---
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 10, 10, 10)

        # Controls Group
        controls_group = QGroupBox(tr("Controls"))
        controls_layout = QVBoxLayout()

        self.btn_start = QPushButton(tr("Start"))
        self.btn_start.setCheckable(True)
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.btn_start.toggled.connect(self.on_start_toggle)
        controls_layout.addWidget(self.btn_start)

        self.btn_reset = QPushButton(tr("Reset"))
        self.btn_reset.clicked.connect(self.module.reset_measurements)
        controls_layout.addWidget(self.btn_reset)

        controls_group.setLayout(controls_layout)
        sidebar_layout.addWidget(controls_group)

        # Settings Group
        settings_group = QGroupBox(tr("Settings"))
        settings_layout = QVBoxLayout()

        # Channel
        settings_layout.addWidget(QLabel(tr("Channel:")))
        self.combo_channel = QComboBox()
        self.combo_channel.addItems(['L', 'R'])
        self.combo_channel.currentIndexChanged.connect(self.module.set_channel)
        settings_layout.addWidget(self.combo_channel)

        # Freq Weight
        settings_layout.addWidget(QLabel(tr("Freq Weight:")))
        self.combo_freq = QComboBox()
        self.combo_freq.addItems(['A', 'C', 'Z'])
        self.combo_freq.currentTextChanged.connect(self.module.set_freq_weighting)
        settings_layout.addWidget(self.combo_freq)

        # Time Weight
        settings_layout.addWidget(QLabel(tr("Time Weight:")))
        self.combo_time = QComboBox()
        self.combo_time.addItems(['FAST', 'SLOW', 'IMPULSE', '10ms'])
        self.combo_time.currentTextChanged.connect(self.module.set_time_weighting)
        settings_layout.addWidget(self.combo_time)

        # Bandwidth
        settings_layout.addWidget(QLabel(tr("Bandwidth:")))
        self.combo_bw = QComboBox()
        self.combo_bw.addItems(['20Hz - 20kHz', '20Hz - 12.5kHz', '20Hz - 8kHz'])
        self.combo_bw.currentTextChanged.connect(self.module.set_bandwidth_mode)
        settings_layout.addWidget(self.combo_bw)

        # Duration
        settings_layout.addWidget(QLabel(tr("Duration:")))
        self.combo_duration = QComboBox()
        self.combo_duration.addItems(['Continuous', '1s', '3s', '5s', '10s', '20s', '30s', '1min', '2min', '5min', '10min', '15min', '30min'])
        self.combo_duration.currentTextChanged.connect(self.module.set_target_duration)
        settings_layout.addWidget(self.combo_duration)

        # Sampling Period
        settings_layout.addWidget(QLabel(tr("Lp Interval:")))
        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(0.01, 10.0)
        self.spin_interval.setSingleStep(0.1)
        self.spin_interval.setValue(0.1)
        self.spin_interval.setSuffix(" s")
        self.spin_interval.valueChanged.connect(self.module.set_sampling_period)
        settings_layout.addWidget(self.spin_interval)

        settings_group.setLayout(settings_layout)
        sidebar_layout.addWidget(settings_group)

        sidebar_layout.addStretch()
        self.sidebar.setLayout(sidebar_layout)

        # --- Main Content Area ---
        content_area = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 10, 10, 10)

        # 1. Main Display (Big Numbers)
        display_frame = QWidget()
        display_frame.setStyleSheet("background-color: #000; border-radius: 8px; margin-bottom: 10px;")
        display_layout = QHBoxLayout()

        # Lp Display
        self.disp_lp = self._create_big_display(tr("Instantaneous (Lp)"), "#00ff00")
        display_layout.addWidget(self.disp_lp['container'])

        # Leq Display
        self.disp_leq = self._create_big_display(tr("Equivalent (Leq)"), "#00ccff")
        display_layout.addWidget(self.disp_leq['container'])

        display_frame.setLayout(display_layout)
        content_layout.addWidget(display_frame)

        # 2. Tabs for Graphs and Stats
        self.tabs = QTabWidget()

        # Tab 1: Histogram (Graph)
        self.tab_hist = QWidget()
        hist_layout = QVBoxLayout()

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#111')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('bottom', "Level", units='dB')
        self.plot_widget.setLabel('left', "Probability", units='%')

        self.hist_item = pg.BarGraphItem(x=[0], height=[0], width=0.4, brush='g')
        self.plot_widget.addItem(self.hist_item)

        hist_layout.addWidget(self.plot_widget)
        self.tab_hist.setLayout(hist_layout)
        self.tabs.addTab(self.tab_hist, tr("Histogram (LN)"))

        # Tab 2: Statistics (LN Table)
        self.tab_stats = QWidget()
        stats_layout = QVBoxLayout()
        ln_grid = QGridLayout()
        self.ln_labels = {}
        ln_metrics = [
            ('L5', 'L5', 0, 0), ('L10', 'L10', 0, 1), ('L50', 'L50', 0, 2), ('L90', 'L90', 0, 3),
            ('L95', 'L95', 1, 0), ('Lhigh', 'Lhigh', 1, 1), ('Llow', 'Llow', 1, 2), ('Lave', 'Lave', 1, 3)
        ]
        ln_font_style = "font-size: 24px; font-weight: bold; color: #00ffff;"

        for key, title_text, r, c in ln_metrics:
            container = QWidget()
            v_box = QVBoxLayout()
            title = QLabel(title_text)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("font-weight: bold; font-size: 14pt; color: #eee;")
            val_lbl = QLabel("--.-")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl.setStyleSheet(ln_font_style)
            v_box.addWidget(title)
            v_box.addWidget(val_lbl)
            container.setLayout(v_box)
            container.setStyleSheet("background-color: #222; border-radius: 6px; margin: 4px;")
            ln_grid.addWidget(container, r, c)
            self.ln_labels[key] = val_lbl

        stats_layout.addLayout(ln_grid)
        stats_layout.addStretch()
        self.tab_stats.setLayout(stats_layout)
        self.tabs.addTab(self.tab_stats, tr("Statistics"))

        # Tab 3: Detailed Metrics
        self.tab_metrics = QWidget()
        metrics_layout = QVBoxLayout()
        m_grid = QGridLayout()
        self.metric_labels = {}
        metrics_list = [
            ('Lmax', tr("Maximum Level"), 0, 0),
            ('Lmin', tr("Minimum Level"), 0, 1),
            ('Lpeak', tr("Peak Level"), 1, 0),
            ('LE', tr("Sound Exposure Level"), 1, 1)
        ]

        for key, desc, r, c in metrics_list:
            container = QWidget()
            v_box = QVBoxLayout()
            lbl_title = QLabel(key)
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_title.setStyleSheet("font-weight: bold; font-size: 16pt;")

            lbl_desc = QLabel(desc)
            lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_desc.setStyleSheet("font-size: 11pt; color: #aaa;")

            lbl_val = QLabel("--.-")
            lbl_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_val.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffaa00;")

            v_box.addWidget(lbl_title)
            v_box.addWidget(lbl_desc)
            v_box.addWidget(lbl_val)
            container.setLayout(v_box)
            container.setStyleSheet("background-color: #1a1a1a; border-radius: 6px; margin: 5px;")
            m_grid.addWidget(container, r, c)
            self.metric_labels[key] = lbl_val

        metrics_layout.addLayout(m_grid)
        metrics_layout.addStretch()
        self.tab_metrics.setLayout(metrics_layout)
        self.tabs.addTab(self.tab_metrics, tr("Details"))

        content_layout.addWidget(self.tabs)
        content_area.setLayout(content_layout)

        # Assemble
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(content_area)
        self.setLayout(main_layout)

    def _create_big_display(self, title, color):
        container = QWidget()
        layout = QVBoxLayout()

        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("color: #aaa; font-size: 14pt; margin-top: 10px;")

        lbl_val = QLabel("--.-")
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_val.setStyleSheet(f"color: {color}; font-size: 64px; font-weight: bold;")

        lbl_unit = QLabel("dB")
        lbl_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_unit.setStyleSheet(f"color: {color}; font-size: 18pt; margin-bottom: 15px;")

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        layout.addWidget(lbl_unit)
        container.setLayout(layout)
        return {'container': container, 'label': lbl_val}

    def on_start_toggle(self, checked):
        if checked:
            self.btn_start.setText(tr("Stop"))
            self.btn_start.setStyleSheet("background-color: #aa3333; font-weight: bold; font-size: 14px;")
            self.module.start_analysis()
        else:
            self.btn_start.setText(tr("Start"))
            self.btn_start.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.module.stop_analysis()

    def update_display(self):
        # Check if stopped automatically
        if not self.module.is_running and self.btn_start.isChecked():
            self.btn_start.setChecked(False)
            self.on_start_toggle(False)

        if not self.module.is_running:
            return

        # Get calibration offset
        cal_db = 0.0
        if hasattr(self.module.audio_engine, 'calibration'):
             cal_ptr = self.module.audio_engine.calibration
             if hasattr(cal_ptr, 'get_spl_offset_db'):
                 cal_db = cal_ptr.get_spl_offset_db() or 0.0

        vals = self.module.results

        # Helper formatter
        def fmt(v):
            return f"{v + cal_db:.1f}" if not np.isinf(v) and not np.isnan(v) else "--.-"

        # Update Big Displays
        self.disp_lp['label'].setText(fmt(vals.get('Lp', -np.inf)))
        self.disp_leq['label'].setText(fmt(vals.get('Leq', -np.inf)))

        # Update Details
        for key, lbl in self.metric_labels.items():
            lbl.setText(fmt(vals.get(key, -np.inf)) + " dB")

        # Update Stats (LN)
        # Always calculate? It might be heavy if history is huge.
        # But user wants to see it 'live' usually.
        # Only calculate if tab is visible?
        # Optimization: Only calculate if current tab is Histogram or Stats
        current_idx = self.tabs.currentIndex()
        is_hist_tab = (self.tabs.widget(current_idx) == self.tab_hist)
        is_stats_tab = (self.tabs.widget(current_idx) == self.tab_stats)

        if (is_hist_tab or is_stats_tab) and self.module.ln_history:
            # We can optimize by not calculating every GUI frame (50ms), maybe every 250ms?
            # But let's try direct first.

            # For Stats tab
            if is_stats_tab:
                ln_stats = self.module.calculate_ln_statistics()
                for key, lbl in self.ln_labels.items():
                    val = ln_stats.get(key, -np.inf)
                    lbl.setText(fmt(val) + " dB")

            # For Histogram tab
            if is_hist_tab:
                centers, probs = self.module.get_ln_histogram(bin_size=0.5)
                if len(centers) > 0:
                    # Update bar graph
                    # BarGraphItem needs x, height, width
                    self.hist_item.setOpts(x=centers, height=probs, width=0.4)

                    # Auto range y?
                    # self.plot_widget.setYRange(0, np.max(probs)*1.1)

