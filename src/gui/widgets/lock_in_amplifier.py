import argparse
import numpy as np
import pyqtgraph as pg
from scipy.signal import hilbert
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QProgressBar, QSpinBox, QTabWidget, QApplication, QFileDialog, QMessageBox)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import time
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class LockInAmplifier(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 4096 # Adjust for integration time
        self.input_data = np.zeros((self.buffer_size, 2))
        
        # Settings
        self.gen_frequency = 1000.0
        self.gen_amplitude = 0.5 # Linear 0-1
        self.gen_amplitude = 0.5 # Linear 0-1
        self.output_channel = 0 # 0: Left, 1: Right
        self.external_mode = False

        self.harmonic_order = 1
        self.apply_calibration = False
        
        self.signal_channel = 0 # 0: Left, 1: Right
        self.ref_channel = 1    # 0: Left, 1: Right
        
        # Results
        self.current_magnitude = 0.0
        self.current_phase = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.ref_freq = 0.0
        self.ref_freq = 0.0
        self.ref_level = 0.0
        self.ref_coherence = 0.0
        
        # Statistics
        self.current_magnitude_std = 0.0
        self.current_phase_std = 0.0
        
        # Averaging
        self.averaging_count = 1
        self.history = deque(maxlen=300)
        
        self.callback_id = None
        
    @property
    def name(self) -> str:
        return "Lock-in Amplifier"

    @property
    def description(self) -> str:
        return "Dual-phase lock-in detection."

    def run(self, args: argparse.Namespace):
        print("Lock-in Amplifier running from CLI (not fully implemented)")

    def get_widget(self):
        return LockInAmplifierWidget(self)

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        self.input_data = np.zeros((self.buffer_size, 2))
        
        # Generator State
        self._phase = 0
        sample_rate = self.audio_engine.sample_rate
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
            # --- Input Capture ---
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
            
            # --- Output Generation ---
            # Generate Sine Wave
            t = (np.arange(frames) + self._phase) / sample_rate
            self._phase += frames
            
            signal = self.gen_amplitude * np.cos(2 * np.pi * self.gen_frequency * t)
            
            # Fill Output Buffer
            outdata.fill(0)
            
            if not self.external_mode:
                if self.output_channel == 2: # Stereo
                    if outdata.shape[1] >= 1: outdata[:, 0] = signal
                    if outdata.shape[1] >= 2: outdata[:, 1] = signal
                elif outdata.shape[1] > self.output_channel:
                    outdata[:, self.output_channel] = signal

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def process_data(self):
        """
        Perform Lock-in calculation on the current buffer.
        """
        data = self.input_data
        sig = data[:, self.signal_channel]
        ref = data[:, self.ref_channel]
        
        # 1. Analyze Reference
        # Check if reference is present
        ref_rms = np.sqrt(np.mean(ref**2))
        self.ref_level = 20 * np.log10(ref_rms + 1e-12)
        
        if ref_rms < 0.001: # -60dB threshold
            self.current_magnitude = 0.0
            self.current_phase = 0.0
            self.current_x = 0.0
            self.current_y = 0.0
            self.ref_freq = 0.0
            return

        # Estimate Ref Frequency and Coherence
        # Use Hilbert Transform to get analytic signal
        ref_analytic = hilbert(ref)
        
        # Trim edges to remove Hilbert artifacts (e.g., 5% from each side)
        trim_percent = 0.05
        trim_len = int(len(ref) * trim_percent)
        if trim_len < 10: trim_len = 0 # Don't trim if too short (shouldn't happen with 4096)
        
        if trim_len > 0:
            ref_analytic_trimmed = ref_analytic[trim_len:-trim_len]
            ref_trimmed = ref[trim_len:-trim_len]
        else:
            ref_analytic_trimmed = ref_analytic
            ref_trimmed = ref
            
        if len(ref_analytic_trimmed) > 10:
            # Linear Regression on Phase
            ref_inst_phase = np.unwrap(np.angle(ref_analytic_trimmed))
            t_trimmed = np.arange(len(ref_inst_phase)) / self.audio_engine.sample_rate
            
            # Polyfit degree 1 (Linear) -> slope is angular frequency (rad/s)
            slope, intercept = np.polyfit(t_trimmed, ref_inst_phase, 1)
            self.ref_freq = slope / (2.0 * np.pi)
        else:
            self.ref_freq = 0.0
            
        # Calculate Coherence (Spectral Purity)
        # Coherence = magnitude of component at ref_freq / total peak level
        # Use the same coherent projection used by the lock-in detector.
        t_coh = np.arange(len(ref_trimmed)) / self.audio_engine.sample_rate
        osc_coh = np.exp(-1j * 2 * np.pi * self.ref_freq * t_coh)
        ref_component = np.abs(2 * np.mean(ref_trimmed * osc_coh))
        ref_rms_val = np.sqrt(np.mean(ref_trimmed**2))
        ref_peak_val = ref_rms_val * np.sqrt(2)
        
        if ref_peak_val > 1e-9:
            self.ref_coherence = ref_component / ref_peak_val
            if self.ref_coherence > 1.0: self.ref_coherence = 1.0
        else:
            self.ref_coherence = 0.0
            
        # Force Internal Mode stats
        if not self.external_mode:
            self.ref_freq = self.gen_frequency
            self.ref_coherence = 1.0
        
        # 2. Lock-in Detection
        # Coherent demodulation (single-bin DFT) of both channels, then remove reference phase.
        # This avoids Hilbert edge artifacts and stabilizes low-frequency phase when only a
        # few cycles fit in the buffer.
        ref_freq = self.ref_freq
        if not np.isfinite(ref_freq) or ref_freq <= 0:
            self.current_magnitude = 0.0
            self.current_phase = 0.0
            self.current_x = 0.0
            self.current_y = 0.0
            return

        demod_freq = ref_freq * max(int(self.harmonic_order), 1)
        t = np.arange(len(sig)) / self.audio_engine.sample_rate
        osc = np.exp(-1j * 2 * np.pi * demod_freq * t)

        # Complex amplitudes (peak) at demod_freq
        sig_c = 2 * np.mean(sig * osc)
        ref_c = 2 * np.mean(ref * osc)

        # Remove reference phase while keeping signal amplitude.
        # Then correct scalloping loss using the reference channel's time-domain peak estimate.
        # For a near-sinusoidal reference, ref_rms*sqrt(2) ≈ A_ref (peak), while |ref_c| = A_ref*|H|.
        # correction ≈ 1/|H| and applies equally to the signal projection at the same frequency.
        ref_amp_est = ref_rms * np.sqrt(2)
        ref_proj_mag = np.abs(ref_c)
        correction = ref_amp_est / (ref_proj_mag + 1e-12)

        # rel = A_sig * exp(j*(phi_sig - phi_ref))
        result = (sig_c * np.conj(ref_c) / (ref_proj_mag + 1e-12)) * correction
        
        # Averaging
        self.history.append(result)
        while len(self.history) > self.averaging_count:
            self.history.popleft()
            
        avg_result = np.mean(self.history)
        
        # Magnitude is abs(result) (peak)
        self.current_magnitude = np.abs(avg_result)
        
        # Phase
        self.current_phase = np.degrees(np.angle(avg_result))
        
        # Statistics
        if len(self.history) > 1:
            # Magnitude Std
            self.current_magnitude_std = np.std(np.abs(self.history))
            
            # Phase Std (Account for wrapping)
            phases = np.angle(self.history)
            # Use circular standard deviation for phase? 
            # Or just unwrap if spread is small. 
            # Since lock-in phase is usually stable, standard std on unwrapped or complex phasor is fine.
            # Using scipy.stats.circstd is better but we might not want to import it if not needed.
            # Simple approximation: std of angle of unit vectors
            # R = |sum(exp(i*theta))| / N
            # std = sqrt(-2*ln(R))
            # But let's stick to simple std on unwrapped phases for now, it's robust enough for "display precision"
            # We assume the phase doesn't wrap around 180/-180 wildly within the averaging window if it's a stable signal.
            # But to be safe, we can use the angular spread.
            
            # Just use numpy std on the complex result's angle? 
            # No, that wraps.
            # Let's use the std of the complex values relative to the mean, projected onto the perpendicular?
            # Or just:
            # phase_variance = E[phi^2] - E[phi]^2? No.
            
            # Let's simply compute std on unwrapped phases.
            unwrapped_phases = np.unwrap(phases)
            self.current_phase_std = np.degrees(np.std(unwrapped_phases))
        else:
            self.current_magnitude_std = 0.0
            self.current_phase_std = 0.0
        
        # Apply Calibration
        if self.apply_calibration:
            mag_corr, phase_corr = self.audio_engine.calibration.get_frequency_correction(self.gen_frequency)
            gain_offset = self.audio_engine.calibration.lockin_gain_offset
            
            # Total Correction
            # Corrected_dB = Measured_dB - (Map_Correction + Gain_Offset)
            total_mag_corr_db = mag_corr + gain_offset
            
            corr_factor = 10 ** (-total_mag_corr_db / 20)
            self.current_magnitude *= corr_factor
            self.current_phase -= phase_corr
        
        # X and Y (In-phase and Quadrature)
        # Recalculate X/Y based on corrected Mag/Phase
        rad_phase = np.radians(self.current_phase)
        self.current_x = self.current_magnitude * np.cos(rad_phase)
        self.current_y = self.current_magnitude * np.sin(rad_phase)



class FRASweepWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(float, float, float) # freq, mag, phase
    finished_sweep = pyqtSignal()
    
    def __init__(self, module: LockInAmplifier, start_f, end_f, steps, log_sweep, settle_time):
        super().__init__()
        self.module = module
        self.start_f = start_f
        self.end_f = end_f
        self.steps = steps
        self.log_sweep = log_sweep
        self.settle_time = settle_time
        self.is_cancelled = False
        
    def run(self):
        if self.log_sweep:
            freqs = np.logspace(np.log10(self.start_f), np.log10(self.end_f), self.steps)
        else:
            freqs = np.linspace(self.start_f, self.end_f, self.steps)
            
        # Ensure module is running
        if not self.module.is_running:
            self.module.start_analysis()
            time.sleep(0.5) # Wait for start
            
        for i, f in enumerate(freqs):
            if self.is_cancelled: break
            
            self.module.gen_frequency = f
            
            # Wait for settling
            time.sleep(self.settle_time)
            
            # Measurement Loop
            # We need to capture 'averaging_count' buffers
            self.module.history.clear()
            
            # Calculate buffer duration
            sample_rate = self.module.audio_engine.sample_rate
            buffer_duration = self.module.buffer_size / sample_rate
            
            # Ensure we wait at least a bit to avoid CPU spin if buffer is tiny, 
            # though buffer_size is usually > 2048 (approx 40ms at 48k)
            wait_time = max(0.05, buffer_duration)
            
            # We need to fill the history with new data
            # The process_data() call processes the *current* buffer.
            # We need to wait for the audio callback to update the buffer.
            
            # First, wait for one full buffer fill to ensure we are past the settling time completely
            time.sleep(wait_time)
            
            for _ in range(self.module.averaging_count):
                if self.is_cancelled: break
                
                # Wait for next buffer update
                # Since we don't have precise synchronization with callback here, 
                # we sleep for the buffer duration.
                time.sleep(wait_time)
                
                # Process the current buffer state
                self.module.process_data()
            
            # Read measurement (which is now the average of the history)
            mag = self.module.current_magnitude
            phase = self.module.current_phase
            
            self.result.emit(f, mag, phase)
            self.progress.emit(int((i+1)/self.steps * 100))
            
        self.finished_sweep.emit()
        
    def cancel(self):
        self.is_cancelled = True

class LockInAmplifierWidget(QWidget):
    def __init__(self, module: LockInAmplifier):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.setInterval(100) # 10Hz update

    def get_decimal_places(self, val_std, val_abs=None, is_db=False, default=3, max_places=6):
        if val_std <= 0: return default
        try:
            if is_db:
                if val_abs is None or val_abs <= 1e-12: return default
                # 8.686 * (std / abs)
                std_to_use = 8.686 * (val_std / val_abs)
            else:
                std_to_use = val_std
            
            if std_to_use <= 1e-9: return max_places
            places = -int(np.floor(np.log10(std_to_use)))
            
            # Adjust strategy: 
            # If std is 0.001, places=3 -> shows noise digit.
            # If we want to hide noise, we might subtract 1?
            # User wants "Display optimal". Usually means showing the stable digits + 1 noise digit.
            # So places is correct.
            
            if places < 0: places = 0
            if places > max_places: places = max_places
            return places
        except:
            return default

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # Tabs for Modes
        self.tabs = QTabWidget()
        
        # --- Tab 1: Manual Control (Existing) ---
        manual_widget = QWidget()
        manual_layout = QHBoxLayout(manual_widget)
        
        # --- Left Panel: Settings ---
        settings_group = QGroupBox(tr("Settings"))
        settings_layout = QFormLayout()
        
        # Output Controls
        self.toggle_btn = QPushButton(tr("Start Output & Measure"))
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; font-weight: bold; padding: 10px; color: black; } QPushButton:checked { background-color: #ffcccc; }")
            
        settings_layout.addRow(self.toggle_btn)
        
        settings_layout.addRow(self.toggle_btn)
        
        # External Mode
        self.ext_mode_check = QCheckBox(tr("External Mode (No Output)"))
        self.ext_mode_check.toggled.connect(self.on_ext_mode_toggled)
        settings_layout.addRow(self.ext_mode_check)
        
        settings_layout.addRow(QLabel(tr("<b>Output Generator</b>")))
        
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000)
        self.freq_spin.setValue(1000)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(self.on_freq_changed)
        settings_layout.addRow(tr("Frequency:"), self.freq_spin)
        
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(-120, 0)
        self.amp_spin.setValue(-6)
        self.amp_spin.valueChanged.connect(self.on_amp_spin_changed)
        
        self.gen_unit_combo = QComboBox()
        self.gen_unit_combo.addItems(['Linear (0-1)', 'dBFS', 'dBV', 'dBu', 'Vrms', 'Vpeak'])
        self.gen_unit_combo.setCurrentText('dBFS')
        self.gen_unit_combo.currentTextChanged.connect(self.on_gen_unit_changed)
        
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(self.amp_spin)
        amp_layout.addWidget(self.gen_unit_combo)
        settings_layout.addRow(tr("Amplitude:"), amp_layout)
        
        self.out_ch_combo = QComboBox()
        self.out_ch_combo.addItems([tr("Left (Ch 1)"), tr("Right (Ch 2)"), tr("Stereo (Both)")])
        self.out_ch_combo.currentIndexChanged.connect(self.on_out_ch_changed)
        settings_layout.addRow(tr("Output Ch:"), self.out_ch_combo)
        
        settings_layout.addRow(QLabel(tr("<b>Input Routing</b>")))
        
        self.sig_ch_combo = QComboBox()
        self.sig_ch_combo.addItems([tr("Left (Ch 1)"), tr("Right (Ch 2)")])
        self.sig_ch_combo.setCurrentIndex(0) # Default Signal L
        self.sig_ch_combo.currentIndexChanged.connect(self.on_sig_ch_changed)
        settings_layout.addRow(tr("Signal Input:"), self.sig_ch_combo)
        
        self.ref_ch_combo = QComboBox()
        self.ref_ch_combo.addItems([tr("Left (Ch 1)"), tr("Right (Ch 2)")])
        self.ref_ch_combo.setCurrentIndex(1) # Default Ref R
        self.ref_ch_combo.currentIndexChanged.connect(self.on_ref_ch_changed)
        settings_layout.addRow(tr("Reference Input:"), self.ref_ch_combo)
        
        # Integration Time (Buffer Size)
        self.time_combo = QComboBox()
        self.time_combo.addItems([tr("Fast (2048 samples)"), tr("Medium (4096 samples)"), tr("Slow (16384 samples)")])
        self.time_combo.setCurrentIndex(1)
        self.time_combo.currentIndexChanged.connect(self.on_time_changed)
        settings_layout.addRow(tr("Integration:"), self.time_combo)
        
        self.avg_spin = QSpinBox()
        self.avg_spin.setRange(1, 300)
        self.avg_spin.setValue(1)
        self.avg_spin.valueChanged.connect(lambda v: setattr(self.module, 'averaging_count', v))
        settings_layout.addRow(tr("Averaging:"), self.avg_spin)
        
        self.harmonic_spin = QSpinBox()
        self.harmonic_spin.setRange(1, 10)
        self.harmonic_spin.setValue(1)
        self.harmonic_spin.valueChanged.connect(lambda v: setattr(self.module, 'harmonic_order', v))
        settings_layout.addRow(tr("Harmonic:"), self.harmonic_spin)
        
        settings_group.setLayout(settings_layout)
        manual_layout.addWidget(settings_group, stretch=1)
        
        # --- Right Panel: Meters ---
        meters_group = QGroupBox(tr("Measurements"))
        meters_layout = QVBoxLayout()
        
        # Magnitude
        meters_layout.addWidget(QLabel(tr("Magnitude")))
        
        # Unit Selection
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["dBFS", "dBV", "dBu", "V", "mV"])
        self.unit_combo.setCurrentText("dBFS")
        self.unit_combo.currentIndexChanged.connect(self.update_ui) # Update immediately
        meters_layout.addWidget(self.unit_combo)
        
        self.mag_label = QLabel(tr("0.000 V"))
        self.mag_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #00ff00;")
        self.mag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meters_layout.addWidget(self.mag_label)
        
        self.mag_db_label = QLabel(tr("-inf dBFS"))
        self.mag_db_label.setStyleSheet("font-size: 24px; color: #88ff88;")
        self.mag_db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meters_layout.addWidget(self.mag_db_label)
        
        meters_layout.addSpacing(20)
        
        # Phase
        meters_layout.addWidget(QLabel(tr("Phase")))
        self.phase_label = QLabel(tr("0.000 deg"))
        self.phase_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #00ffff;")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meters_layout.addWidget(self.phase_label)
        
        meters_layout.addSpacing(20)
        
        # X / Y
        xy_layout = QHBoxLayout()
        
        x_group = QVBoxLayout()
        x_group.addWidget(QLabel(tr("X (In-phase)")))
        self.x_label = QLabel(tr("0.000 V"))
        self.x_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffff00;")
        self.x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        x_group.addWidget(self.x_label)
        xy_layout.addLayout(x_group)
        
        y_group = QVBoxLayout()
        y_group.addWidget(QLabel(tr("Y (Quadrature)")))
        self.y_label = QLabel(tr("0.000 V"))
        self.y_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff00ff;")
        self.y_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        y_group.addWidget(self.y_label)
        xy_layout.addLayout(y_group)
        
        meters_layout.addLayout(xy_layout)
        
        meters_layout.addSpacing(20)
        
        # Reference Status
        ref_status_layout = QHBoxLayout()
        ref_status_layout.addWidget(QLabel(tr("Reference Status:")))
        self.ref_status_label = QLabel(tr("No Signal"))
        self.ref_status_label.setStyleSheet("font-weight: bold; color: #ff0000;")
        ref_status_layout.addWidget(self.ref_status_label)
        meters_layout.addLayout(ref_status_layout)
        
        meters_layout.addStretch()
        meters_group.setLayout(meters_layout)
        manual_layout.addWidget(meters_group, stretch=2)
        
        self.tabs.addTab(manual_widget, tr("Manual Control"))
        
        # --- Tab 2: Frequency Response Analyzer (FRA) ---
        fra_widget = QWidget()
        fra_layout = QHBoxLayout(fra_widget)
        
        # FRA Settings
        fra_settings_group = QGroupBox(tr("Sweep Settings"))
        fra_form = QFormLayout()
        
        self.fra_start_spin = QDoubleSpinBox()
        self.fra_start_spin.setRange(20, 20000); self.fra_start_spin.setValue(20); self.fra_start_spin.setSuffix(" Hz")
        fra_form.addRow(tr("Start Freq:"), self.fra_start_spin)
        
        self.fra_end_spin = QDoubleSpinBox()
        self.fra_end_spin.setRange(20, 20000); self.fra_end_spin.setValue(20000); self.fra_end_spin.setSuffix(" Hz")
        fra_form.addRow(tr("End Freq:"), self.fra_end_spin)
        
        self.fra_steps_spin = QSpinBox()
        self.fra_steps_spin.setRange(10, 1000); self.fra_steps_spin.setValue(50)
        fra_form.addRow(tr("Steps:"), self.fra_steps_spin)
        
        self.fra_log_check = QCheckBox(tr("Log Sweep")); self.fra_log_check.setChecked(True)
        fra_form.addRow(self.fra_log_check)
        
        self.fra_settle_spin = QDoubleSpinBox()
        self.fra_settle_spin.setRange(0.1, 5.0); self.fra_settle_spin.setValue(0.5); self.fra_settle_spin.setSuffix(" s")
        fra_form.addRow(tr("Settling Time:"), self.fra_settle_spin)
        
        # Plot Unit Selector
        self.fra_plot_unit_combo = QComboBox()
        self.fra_plot_unit_combo.addItems(['dBFS', 'dBV', 'dBu', 'Vrms', 'Vpeak'])
        self.fra_plot_unit_combo.setCurrentText('dBFS')
        self.fra_plot_unit_combo.currentTextChanged.connect(self.update_fra_plot)
        fra_form.addRow(tr("Plot Unit:"), self.fra_plot_unit_combo)
        
        self.fra_start_btn = QPushButton(tr("Start Sweep"))
        self.fra_start_btn.clicked.connect(self.on_fra_start)
        fra_form.addRow(self.fra_start_btn)
        
        self.fra_progress = QProgressBar()
        fra_form.addRow(self.fra_progress)
        
        fra_settings_group.setLayout(fra_form)
        fra_layout.addWidget(fra_settings_group, stretch=1)
        
        # FRA Plot
        self.fra_plot = pg.PlotWidget(title=tr("Bode Plot"))
        self.fra_plot.setLabel('bottom', tr("Frequency"), units='Hz')
        self.fra_plot.setLabel('left', tr("Magnitude"), units='dB')
        self.fra_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fra_plot.addLegend()
        
        # Custom Axis for Log Frequency
        axis = self.fra_plot.getPlotItem().getAxis('bottom')
        axis.setLogMode(False) # We will handle log data manually
        
        # Dual Axis for Phase
        self.fra_plot_p = pg.ViewBox()
        self.fra_plot.scene().addItem(self.fra_plot_p)
        self.fra_plot.getPlotItem().showAxis('right')
        self.fra_plot.getPlotItem().getAxis('right').linkToView(self.fra_plot_p)
        self.fra_plot_p.setXLink(self.fra_plot.getPlotItem())
        self.fra_plot.getPlotItem().getAxis('right').setLabel(tr('Phase'), units='deg')
        
        # Handle resizing
        def update_views():
            self.fra_plot_p.setGeometry(self.fra_plot.getPlotItem().vb.sceneBoundingRect())
            self.fra_plot_p.linkedViewChanged(self.fra_plot.getPlotItem().vb, self.fra_plot_p.XAxis)
        self.fra_plot.getPlotItem().vb.sigResized.connect(update_views)
        
        self.fra_curve_mag = self.fra_plot.plot(pen='g', name=tr('Magnitude (dB)'))
        self.fra_curve_phase = pg.PlotCurveItem(pen='c', name=tr('Phase (deg)'))
        self.fra_plot_p.addItem(self.fra_curve_phase)
        
        fra_layout.addWidget(self.fra_plot, stretch=3)
        
        self.tabs.addTab(fra_widget, tr("Frequency Response"))
        
        # --- Tab 3: Calibration ---
        cal_widget = QWidget()
        cal_layout = QHBoxLayout(cal_widget)
        
        # Settings
        cal_settings = QGroupBox(tr("Calibration Settings"))
        cal_form = QFormLayout()
        
        self.cal_start_spin = QDoubleSpinBox()
        self.cal_start_spin.setRange(20, 20000); self.cal_start_spin.setValue(20); self.cal_start_spin.setSuffix(" Hz")
        cal_form.addRow(tr("Start Freq:"), self.cal_start_spin)
        
        self.cal_end_spin = QDoubleSpinBox()
        self.cal_end_spin.setRange(20, 20000); self.cal_end_spin.setValue(20000); self.cal_end_spin.setSuffix(" Hz")
        cal_form.addRow(tr("End Freq:"), self.cal_end_spin)
        
        self.cal_steps_spin = QSpinBox()
        self.cal_steps_spin.setRange(10, 5000); self.cal_steps_spin.setValue(100)
        cal_form.addRow(tr("Steps:"), self.cal_steps_spin)
        
        self.cal_settle_spin = QDoubleSpinBox()
        self.cal_settle_spin.setRange(0.1, 5.0); self.cal_settle_spin.setValue(0.5); self.cal_settle_spin.setSuffix(" s")
        cal_form.addRow(tr("Settling Time:"), self.cal_settle_spin)
        
        self.cal_start_btn = QPushButton(tr("Run Relative Map Sweep"))
        self.cal_start_btn.clicked.connect(self.on_cal_start)
        cal_form.addRow(self.cal_start_btn)
        
        self.cal_save_btn = QPushButton(tr("Save Map"))
        self.cal_save_btn.clicked.connect(self.on_cal_save)
        self.cal_save_btn.setEnabled(False)
        cal_form.addRow(self.cal_save_btn)
        
        self.cal_load_btn = QPushButton(tr("Load Map"))
        self.cal_load_btn.clicked.connect(self.on_cal_load)
        cal_form.addRow(self.cal_load_btn)
        
        self.cal_apply_check = QCheckBox(tr("Apply Calibration"))
        self.cal_apply_check.toggled.connect(self.on_cal_apply_toggled)
        cal_form.addRow(self.cal_apply_check)
        
        # Absolute Gain Calibration
        cal_form.addRow(QLabel(tr("<b>Absolute Gain Calibration</b>")))
        
        self.abs_cal_target_spin = QDoubleSpinBox()
        self.abs_cal_target_spin.setRange(-200, 200)
        self.abs_cal_target_spin.setValue(1.0) # Default 1.0 V
        self.abs_cal_target_spin.setDecimals(6)
        
        self.abs_cal_unit_combo = QComboBox()
        self.abs_cal_unit_combo.addItems(['Vrms', 'dBV', 'dBu', 'dBFS'])
        self.abs_cal_unit_combo.setCurrentText('Vrms')
        
        abs_target_layout = QHBoxLayout()
        abs_target_layout.addWidget(self.abs_cal_target_spin)
        abs_target_layout.addWidget(self.abs_cal_unit_combo)
        cal_form.addRow(tr("Target Value:"), abs_target_layout)
        
        self.abs_cal_btn = QPushButton(tr("Calibrate Absolute Gain (1-Point)"))
        self.abs_cal_btn.clicked.connect(self.on_abs_cal_click)
        cal_form.addRow(self.abs_cal_btn)
        
        self.abs_cal_status = QLabel(tr("Current Offset: {0:.3f} dB").format(self.module.audio_engine.calibration.lockin_gain_offset))
        cal_form.addRow(self.abs_cal_status)
        
        self.cal_progress = QProgressBar()
        cal_form.addRow(self.cal_progress)
        
        cal_settings.setLayout(cal_form)
        cal_layout.addWidget(cal_settings, stretch=1)
        
        # Plot
        self.cal_plot = pg.PlotWidget(title=tr("Calibration Map"))
        self.cal_plot.setLabel('bottom', tr("Frequency"), units='Hz')
        self.cal_plot.setLabel('left', tr("Correction"), units='dB')
        self.cal_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cal_plot.addLegend()
        self.cal_plot.getPlotItem().getAxis('bottom').setLogMode(True)
        
        self.cal_curve_mag = self.cal_plot.plot(pen='y', name=tr('Mag Correction (dB)'))
        self.cal_curve_phase = pg.PlotCurveItem(pen='c', name=tr('Phase Correction (deg)'))
        
        # Dual Axis for Phase
        self.cal_plot_p = pg.ViewBox()
        self.cal_plot.scene().addItem(self.cal_plot_p)
        self.cal_plot.getPlotItem().showAxis('right')
        self.cal_plot.getPlotItem().getAxis('right').linkToView(self.cal_plot_p)
        self.cal_plot_p.setXLink(self.cal_plot.getPlotItem())
        self.cal_plot.getPlotItem().getAxis('right').setLabel(tr('Phase'), units='deg')
        
        self.cal_plot_p.addItem(self.cal_curve_phase)
        
        # Handle resizing for cal plot
        def update_cal_views():
            self.cal_plot_p.setGeometry(self.cal_plot.getPlotItem().vb.sceneBoundingRect())
            self.cal_plot_p.linkedViewChanged(self.cal_plot.getPlotItem().vb, self.cal_plot_p.XAxis)
        self.cal_plot.getPlotItem().vb.sigResized.connect(update_cal_views)
        
        cal_layout.addWidget(self.cal_plot, stretch=3)
        
        self.tabs.addTab(cal_widget, tr("Calibration"))
        
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
        
        # Data storage for FRA
        self.fra_freqs = []
        self.fra_log_freqs = []
        self.fra_raw_mags = [] # Linear (0-1)
        self.fra_phases = []
        self.fra_worker = None
        
        # Data storage for Calibration
        self.cal_data = [] # List of [freq, mag_db, phase_deg]
        self.cal_worker = None

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText(tr("Stop"))
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText(tr("Start Output & Measure"))

    def on_freq_changed(self, val):
        self.module.gen_frequency = val
        # Phase continuity is handled in callback by using self.module.gen_frequency

    def calculate_linear_amplitude(self, val, unit):
        gain = self.module.audio_engine.calibration.output_gain
        amp_linear = 0.0
        
        if unit == 'Linear (0-1)':
            amp_linear = val
        elif unit == 'dBFS':
            amp_linear = 10**(val/20)
        elif unit == 'dBV':
            # val = 20 * log10(Vrms)
            v_rms = 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_linear = v_peak / gain
        elif unit == 'dBu':
            # val = 20 * log10(Vrms / 0.7746)
            v_rms = 0.7746 * 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_linear = v_peak / gain
        elif unit == 'Vrms':
            v_peak = val * np.sqrt(2)
            amp_linear = v_peak / gain
        elif unit == 'Vpeak':
            amp_linear = val / gain
            
        # Clamp
        if amp_linear > 1.0: amp_linear = 1.0
        elif amp_linear < 0.0: amp_linear = 0.0
        
        return amp_linear

    def on_amp_spin_changed(self, val):
        unit = self.gen_unit_combo.currentText()
        self.module.gen_amplitude = self.calculate_linear_amplitude(val, unit)

    def on_gen_unit_changed(self, unit):
        self.update_amp_display()

    def update_amp_display(self):
        unit = self.gen_unit_combo.currentText()
        amp_0_1 = self.module.gen_amplitude
        gain = self.module.audio_engine.calibration.output_gain
        
        self.amp_spin.blockSignals(True)
        
        if unit == 'Linear (0-1)':
            self.amp_spin.setRange(0, 1.0)
            self.amp_spin.setSingleStep(0.1)
            self.amp_spin.setValue(amp_0_1)
        elif unit == 'dBFS':
            self.amp_spin.setRange(-120, 0)
            self.amp_spin.setSingleStep(1.0)
            val = 20 * np.log10(amp_0_1 + 1e-12)
            self.amp_spin.setValue(val)
        elif unit == 'dBV':
            v_peak = amp_0_1 * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10(v_rms + 1e-12)
            self.amp_spin.setRange(-120, 20)
            self.amp_spin.setSingleStep(1.0)
            self.amp_spin.setValue(val)
        elif unit == 'dBu':
            v_peak = amp_0_1 * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10((v_rms + 1e-12) / 0.7746)
            self.amp_spin.setRange(-120, 20)
            self.amp_spin.setSingleStep(1.0)
            self.amp_spin.setValue(val)
        elif unit == 'Vrms':
            v_peak = amp_0_1 * gain
            v_rms = v_peak / np.sqrt(2)
            self.amp_spin.setRange(0, 100)
            self.amp_spin.setSingleStep(0.1)
            self.amp_spin.setValue(v_rms)
        elif unit == 'Vpeak':
            v_peak = amp_0_1 * gain
            self.amp_spin.setRange(0, 100)
            self.amp_spin.setSingleStep(0.1)
            self.amp_spin.setValue(v_peak)
            
        self.amp_spin.blockSignals(False)

    def on_out_ch_changed(self, idx):
        self.module.output_channel = idx
        if self.module.is_running:
            # Restart to apply channel change
            self.module.stop_analysis()
            self.module.start_analysis()

    def on_sig_ch_changed(self, idx):
        self.module.signal_channel = idx

    def on_ref_ch_changed(self, idx):
        self.module.ref_channel = idx

    def on_ext_mode_toggled(self, checked):
        self.module.external_mode = checked
        
        # Disable/Enable Generator Controls
        self.freq_spin.setEnabled(not checked)
        self.amp_spin.setEnabled(not checked)
        self.gen_unit_combo.setEnabled(not checked)
        self.out_ch_combo.setEnabled(not checked)

    def on_time_changed(self, idx):
        if idx == 0: self.module.buffer_size = 2048
        elif idx == 1: self.module.buffer_size = 4096
        elif idx == 2: self.module.buffer_size = 16384
        
        # Re-allocate buffer
        self.module.input_data = np.zeros((self.module.buffer_size, 2))

    def update_ui(self):
        if not self.module.is_running:
            return
            
        self.module.process_data()
        
        # Calculate Precision
        prec_mag = 3
        prec_db = 3
        prec_phase = 3
        prec_xy = 6
        
        if self.module.averaging_count > 1 and len(self.module.history) > 1:
            mag_std = self.module.current_magnitude_std
            mag_abs = self.module.current_magnitude
            phase_std = self.module.current_phase_std
            
            prec_mag = self.get_decimal_places(mag_std, default=3)
            prec_db = self.get_decimal_places(mag_std, val_abs=mag_abs, is_db=True, default=3)
            # Phase is usually noisier, default 3 might be high if noisy
            prec_phase = self.get_decimal_places(phase_std, default=3)
            
            # For XY, use linear mag precision (approx)
            prec_xy = prec_mag + 1
            if prec_xy > 6: prec_xy = 6
        
        # Update Meters
        mag_fs = self.module.current_magnitude
        phase = self.module.current_phase
        
        # Calculate Voltage
        sensitivity = self.module.audio_engine.calibration.input_sensitivity # Vpeak at 0dBFS
        v_peak = mag_fs * sensitivity
        v_rms = v_peak / np.sqrt(2)
        
        unit = self.unit_combo.currentText()
        
        if unit == "dBFS":
            if mag_fs > 0:
                val = 20 * np.log10(mag_fs + 1e-12)
                fmt = "{0:." + str(prec_db) + "f} dBFS"
                self.mag_label.setText(tr(fmt).format(val))
            else:
                self.mag_label.setText(tr("-inf dBFS"))
            self.mag_db_label.setText("") # Clear secondary
            
        elif unit == "dBV":
            if v_rms > 0:
                val = 20 * np.log10(v_rms + 1e-12)
                fmt = "{0:." + str(prec_db) + "f} dBV"
                self.mag_label.setText(tr(fmt).format(val))
            else:
                self.mag_label.setText(tr("-inf dBV"))
            self.mag_db_label.setText("")
            
        elif unit == "dBu":
            if v_rms > 0:
                val = 20 * np.log10((v_rms + 1e-12) / 0.7746)
                fmt = "{0:." + str(prec_db) + "f} dBu"
                self.mag_label.setText(tr(fmt).format(val))
            else:
                self.mag_label.setText(tr("-inf dBu"))
            self.mag_db_label.setText("")
            
        elif unit == "V":
            fmt = "{0:." + str(prec_mag + 3) + "f} V" # V usually needs more decimal places than raw linear if < 1
            if v_rms < 1.0: fmt = "{0:." + str(prec_mag + 4) + "f} V" # e.g. 0.00123
            
            # If prec_mag is calculated from FS (0-1), 
            # if std(FS) = 1e-6 (6 places), then std(V) ~ 1e-6 * sensitivity.
            # So places should be roughly same or +sensitivity factor.
            # Let's use simple logic: recalculate prec for V? 
            # or just use prec_mag + 2 safe buffer?
            # Re-calculating proper clean precision for V:
            # std_v = mag_std * sensitivity / sqrt(2)
            # prec_v = get_decimal_places(std_v)
            if self.module.averaging_count > 1:
                std_v = self.module.current_magnitude_std * sensitivity / np.sqrt(2)
                prec_v_val = self.get_decimal_places(std_v, default=6)
                fmt = "{0:." + str(prec_v_val) + "f} V"
            else:
                fmt = "{0:.6f} V"

            self.mag_label.setText(tr(fmt).format(v_rms))
            # Show dBFS as secondary
            if mag_fs > 0:
                db = 20 * np.log10(mag_fs + 1e-12)
                fmt_db = "{0:." + str(prec_db) + "f} dBFS"
                self.mag_db_label.setText(tr(fmt_db).format(db))
            else:
                self.mag_db_label.setText(tr("-inf dBFS"))
                
        elif unit == "mV":
            # std_mv = std_v * 1000
            if self.module.averaging_count > 1:
                std_mv = self.module.current_magnitude_std * sensitivity / np.sqrt(2) * 1000
                prec_mv_val = self.get_decimal_places(std_mv, default=3)
                fmt = "{0:." + str(prec_mv_val) + "f} mV"
            else:
                fmt = "{0:.3f} mV"
                
            self.mag_label.setText(tr(fmt).format(v_rms * 1000))
            # Show dBFS as secondary
            if mag_fs > 0:
                db = 20 * np.log10(mag_fs + 1e-12)
                fmt_db = "{0:." + str(prec_db) + "f} dBFS"
                self.mag_db_label.setText(tr(fmt_db).format(db))
            else:
                self.mag_db_label.setText(tr("-inf dBFS"))
            
        fmt_phase = "{0:." + str(prec_phase) + "f} deg"
        self.phase_label.setText(tr(fmt_phase).format(phase))
        
        # Update X/Y
        x_fs = self.module.current_x
        y_fs = self.module.current_y
        
        x_v = x_fs * sensitivity / np.sqrt(2) # RMS
        y_v = y_fs * sensitivity / np.sqrt(2) # RMS
        
        if unit == "dBFS":
            # For X/Y in FS, use prec_xy
            fmt_xy = "{0:." + str(prec_xy) + "f} FS"
            self.x_label.setText(tr(fmt_xy).format(x_fs))
            self.y_label.setText(tr(fmt_xy).format(y_fs))
        elif unit == "mV":
             # Use prec_mv from above logic or similar
            if self.module.averaging_count > 1:
                std_mv = self.module.current_magnitude_std * sensitivity / np.sqrt(2) * 1000
                prec_mv_val = self.get_decimal_places(std_mv, default=3)
                fmt_xy = "{0:." + str(prec_mv_val) + "f} mV"
            else:
                fmt_xy = "{0:.3f} mV"
            self.x_label.setText(tr(fmt_xy).format(x_v * 1000))
            self.y_label.setText(tr(fmt_xy).format(y_v * 1000))
        else: # V, dBV, dBu -> Show V
            # Use prec_v
            if self.module.averaging_count > 1:
                std_v = self.module.current_magnitude_std * sensitivity / np.sqrt(2)
                prec_v_val = self.get_decimal_places(std_v, default=6)
                fmt_xy = "{0:." + str(prec_v_val) + "f} V"
            else:
                fmt_xy = "{0:.6f} V"
            self.x_label.setText(tr(fmt_xy).format(x_v))
            self.y_label.setText(tr(fmt_xy).format(y_v))
        
        # Update Ref Status
        self.module.ref_level
        ref_freq = self.module.ref_freq
        coherence = self.module.ref_coherence
        
        if coherence >= 0.95:
            self.ref_status_label.setText(tr("Locked ({0:.1f} Hz, Coh: {1:.2f})").format(ref_freq, coherence))
            self.ref_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        elif coherence >= 0.8:
            self.ref_status_label.setText(tr("Unstable ({0:.1f} Hz, Coh: {1:.2f})").format(ref_freq, coherence))
            self.ref_status_label.setStyleSheet("font-weight: bold; color: #ffff00;")
        else:
            self.ref_status_label.setText(tr("Unlocked ({0:.1f} Hz, Coh: {1:.2f})").format(ref_freq, coherence))
            self.ref_status_label.setStyleSheet("font-weight: bold; color: #ff0000;")

    def on_fra_start(self):
        if self.fra_worker is not None and self.fra_worker.isRunning():
            self.fra_worker.cancel()
            self.fra_start_btn.setText(tr("Stopping..."))
            self.fra_start_btn.setEnabled(False)
            # Do not wait() here, let the finished signal handle cleanup
            return
            
        # Clear Data
        self.fra_freqs = []
        self.fra_log_freqs = []
        self.fra_raw_mags = []
        self.fra_phases = []
        self.fra_curve_mag.setData([], [])
        self.fra_curve_phase.setData([], [])
        
        # Reset View (Force AutoRange)
        self.fra_plot.getPlotItem().enableAutoRange()
        self.fra_plot_p.enableAutoRange()
        
        # Start Worker
        start = self.fra_start_spin.value()
        end = self.fra_end_spin.value()
        steps = self.fra_steps_spin.value()
        log = self.fra_log_check.isChecked()
        settle = self.fra_settle_spin.value()
        
        # Force Apply Settings (Channel Routing)
        # This ensures settings are applied even if Manual mode wasn't run
        self.module.output_channel = self.out_ch_combo.currentIndex()
        self.module.signal_channel = self.sig_ch_combo.currentIndex()
        self.module.ref_channel = self.ref_ch_combo.currentIndex()
        
        # Set Amplitude (Use Manual Settings)
        amp_val = self.amp_spin.value()
        amp_unit = self.gen_unit_combo.currentText()
        self.module.gen_amplitude = self.calculate_linear_amplitude(amp_val, amp_unit)
        
        # Setup Axis Ticks
        if log:
            axis = self.fra_plot.getPlotItem().getAxis('bottom')
            # Generate ticks
            ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            ticks_log = [(np.log10(t), str(t) if t < 1000 else f"{t/1000:.0f}k") for t in ticks]
            axis.setTicks([ticks_log])
        else:
            self.fra_plot.getPlotItem().getAxis('bottom').setTicks(None) # Auto
        
        self.fra_worker = FRASweepWorker(self.module, start, end, steps, log, settle)
        self.fra_worker.progress.connect(self.fra_progress.setValue)
        self.fra_worker.result.connect(self.on_fra_result)
        self.fra_worker.finished_sweep.connect(self.on_fra_finished)
        self.fra_worker.start()
        
        self.fra_start_btn.setText(tr("Stop Sweep"))
        
    def on_fra_result(self, f, mag, phase):
        self.fra_freqs.append(f)
        
        # Log X
        if self.fra_log_check.isChecked():
            x_val = np.log10(f) if f > 0 else 0
        else:
            x_val = f
        self.fra_log_freqs.append(x_val)
        
        self.fra_raw_mags.append(mag)
        self.fra_phases.append(phase)
        
        self.update_fra_plot()
        
        # Auto-scale Phase View
        self.fra_plot_p.autoRange()

    def update_fra_plot(self):
        if not self.fra_raw_mags:
            return
            
        unit = self.fra_plot_unit_combo.currentText()
        sensitivity = self.module.audio_engine.calibration.input_sensitivity
        
        plot_mags = []
        
        for mag in self.fra_raw_mags:
            y_val = 0.0
            if unit == 'dBFS':
                y_val = 20 * np.log10(mag + 1e-12)
            elif unit == 'dBV':
                v_peak = mag * sensitivity
                v_rms = v_peak / np.sqrt(2)
                y_val = 20 * np.log10(v_rms + 1e-12)
            elif unit == 'dBu':
                v_peak = mag * sensitivity
                v_rms = v_peak / np.sqrt(2)
                y_val = 20 * np.log10((v_rms + 1e-12) / 0.7746)
            elif unit == 'Vrms':
                v_peak = mag * sensitivity
                y_val = v_peak / np.sqrt(2)
            elif unit == 'Vpeak':
                y_val = mag * sensitivity
            plot_mags.append(y_val)
            
        # Update Axis Label
        if unit == 'dBFS': self.fra_plot.setLabel('left', tr("Magnitude"), units='dBFS')
        elif unit == 'dBV': self.fra_plot.setLabel('left', tr("Magnitude"), units='dBV')
        elif unit == 'dBu': self.fra_plot.setLabel('left', tr("Magnitude"), units='dBu')
        elif unit == 'Vrms': self.fra_plot.setLabel('left', tr("Magnitude"), units='V')
        elif unit == 'Vpeak': self.fra_plot.setLabel('left', tr("Magnitude"), units='V')

        self.fra_curve_mag.setData(self.fra_log_freqs, plot_mags)
        self.fra_curve_phase.setData(self.fra_log_freqs, self.fra_phases)

    def on_fra_finished(self):
        self.fra_start_btn.setText(tr("Start Sweep"))
        self.fra_start_btn.setEnabled(True)
        self.module.stop_analysis() # Stop generator
        
    # --- Calibration Methods ---
    
    def on_cal_start(self):
        if self.cal_worker is not None and self.cal_worker.isRunning():
            self.cal_worker.cancel()
            self.cal_start_btn.setText(tr("Stopping..."))
            self.cal_start_btn.setEnabled(False)
            return
            
        # Clear Data
        self.cal_data = []
        self.cal_curve_mag.setData([], [])
        self.cal_curve_phase.setData([], [])
        
        # Start Worker (Reuse FRASweepWorker logic)
        start = self.cal_start_spin.value()
        end = self.cal_end_spin.value()
        steps = self.cal_steps_spin.value()
        settle = self.cal_settle_spin.value()
        
        # Force Apply Settings
        self.module.output_channel = self.out_ch_combo.currentIndex()
        self.module.signal_channel = self.sig_ch_combo.currentIndex()
        self.module.ref_channel = self.ref_ch_combo.currentIndex()
        
        # Set Amplitude
        amp_val = self.amp_spin.value()
        amp_unit = self.gen_unit_combo.currentText()
        self.module.gen_amplitude = self.calculate_linear_amplitude(amp_val, amp_unit)
        
        # Disable Calibration Application during calibration sweep!
        self.cal_apply_check.setChecked(False)
        
        self.cal_worker = FRASweepWorker(self.module, start, end, steps, True, settle) # Always log sweep for cal?
        self.cal_worker.progress.connect(self.cal_progress.setValue)
        self.cal_worker.result.connect(self.on_cal_result)
        self.cal_worker.finished_sweep.connect(self.on_cal_finished)
        self.cal_worker.start()
        
        self.cal_start_btn.setText(tr("Stop Calibration"))
        self.cal_save_btn.setEnabled(False)
        
    def on_cal_result(self, f, mag, phase):
        # Store raw data
        if mag > 0:
            mag_db = 20 * np.log10(mag + 1e-12)
        else:
            mag_db = -120
            
        self.cal_data.append([f, mag_db, phase])
        
        # Update Plot
        freqs = [x[0] for x in self.cal_data]
        log_freqs = [np.log10(x) for x in freqs]
        mags = [x[1] for x in self.cal_data]
        phases = [x[2] for x in self.cal_data]
        
        self.cal_curve_mag.setData(log_freqs, mags)
        self.cal_curve_phase.setData(log_freqs, phases)
        
    def on_cal_finished(self):
        self.cal_start_btn.setText(tr("Run Relative Map Sweep"))
        self.cal_start_btn.setEnabled(True)
        self.cal_save_btn.setEnabled(True)
        self.module.stop_analysis()
        
        # Normalize Map to 1kHz (or nearest)
        # Find 1kHz index
        if not self.cal_data: return
        
        freqs = [x[0] for x in self.cal_data]
        mags = [x[1] for x in self.cal_data]
        phases = [x[2] for x in self.cal_data]
        
        # Find nearest to 1000Hz
        idx = (np.abs(np.array(freqs) - 1000)).argmin()
        ref_mag = mags[idx]
        ref_phase = phases[idx] # Optional: Normalize phase too? Usually yes for relative map.
        
        # Normalize
        norm_data = []
        for f, m, p in self.cal_data:
            norm_data.append([f, m - ref_mag, p - ref_phase]) # Relative to 1kHz
            
        self.cal_data = norm_data
        
        # Update Plot with Normalized Data
        log_freqs = [np.log10(x[0]) for x in self.cal_data]
        norm_mags = [x[1] for x in self.cal_data]
        norm_phases = [x[2] for x in self.cal_data]
        
        self.cal_curve_mag.setData(log_freqs, norm_mags)
        self.cal_curve_phase.setData(log_freqs, norm_phases)
        
        QMessageBox.information(self, tr("Calibration Complete"), 
                              tr("Sweep completed.\nMap normalized to 1kHz (Ref: {0:.2f} dB, {1:.2f} deg).\n"
                              "This map captures RELATIVE frequency response.\n"
                              "Use 'Absolute Gain Calibration' to fix the absolute level.").format(ref_mag, ref_phase))

    def on_abs_cal_click(self):
        if not self.module.is_running:
            QMessageBox.warning(self, tr("Error"), tr("Please start the measurement first (Manual Control -> Start)."))
            return
            
        # Get Current Measurement (Uncorrected or Corrected? Should be Uncorrected for calculating offset?)
        # Actually, if we are calibrating absolute gain, we want the result AFTER Map correction (if applied) 
        # to match the Target.
        # But wait, if Map is relative (0dB at 1k), then at 1k Map correction is 0.
        # So at 1k, Corrected = Measured - Offset.
        # We want Corrected = Target.
        # So Target = Measured - Offset => Offset = Measured - Target.
        
        # If we are at 10k, and Map says +1dB relative to 1k.
        # Measured = X.
        # Corrected = X - (+1dB) - Offset.
        # We want Corrected = Target.
        # Offset = X - 1dB - Target.
        
        # So we need the measurement *with Map applied but without Offset applied*.
        # Or simpler: Just take the current displayed value (which has both applied), 
        # and adjust the offset by the difference.
        
        # Current Displayed Magnitude (dBFS)
        # self.module.current_magnitude is already corrected in process_data if apply_calibration is True.
        # But wait, process_data uses the *current* offset.
        # So Current_Displayed = Raw - Map - Old_Offset.
        # We want New_Displayed = Target.
        # Target = Raw - Map - New_Offset.
        # New_Offset = Raw - Map - Target.
        # New_Offset = (Current_Displayed + Old_Offset) - Target.
        # New_Offset = Old_Offset + (Current_Displayed - Target).
        
        # This works regardless of whether Map is applied or not, as long as we are consistent.
        # If Map NOT applied: Current = Raw. New_Offset = Raw - Target. (Assuming Map=0)
        
        # Let's use the current magnitude from module (which is linear 0-1 FS).
        current_mag_linear = self.module.current_magnitude
        if current_mag_linear <= 0:
            QMessageBox.warning(self, tr("Error"), tr("Signal too low for calibration."))
            return
            
        current_mag_dbfs = 20 * np.log10(current_mag_linear + 1e-12)
        
        # Calculate Target in dBFS
        target_val = self.abs_cal_target_spin.value()
        target_unit = self.abs_cal_unit_combo.currentText()
        sensitivity = self.module.audio_engine.calibration.input_sensitivity
        
        target_dbfs = 0.0
        if target_unit == 'dBFS':
            target_dbfs = target_val
        elif target_unit == 'dBV':
            # val = 20log(Vrms)
            # Vrms = 10^(val/20)
            # Vpeak = Vrms * sqrt(2)
            # dBFS = 20log(Vpeak / sensitivity)
            v_rms = 10**(target_val/20)
            v_peak = v_rms * np.sqrt(2)
            target_dbfs = 20 * np.log10(v_peak / sensitivity)
        elif target_unit == 'dBu':
            v_rms = 0.7746 * 10**(target_val/20)
            v_peak = v_rms * np.sqrt(2)
            target_dbfs = 20 * np.log10(v_peak / sensitivity)
        elif target_unit == 'Vrms':
            v_peak = target_val * np.sqrt(2)
            target_dbfs = 20 * np.log10(v_peak / sensitivity)
            
        # Calculate Error
        # Current (Displayed) = Target + Error
        # We want to subtract Error from the reading.
        # Since reading = Raw - Offset, we need to INCREASE Offset by Error.
        # Error = Current - Target
        # New_Offset = Old_Offset + Error
        
        # Wait, let's verify sign.
        # Correction logic: Corrected = Measured - Offset.
        # If Measured = -10dB, Target = -12dB.
        # We are reading 2dB too high.
        # We need to subtract 2dB more.
        # So Offset should increase by 2dB.
        # Error = -10 - (-12) = +2dB.
        # New_Offset = Old_Offset + 2dB. Correct.
        
        # But wait, self.module.current_magnitude ALREADY includes the current offset if applied!
        # If apply_calibration is False, it does NOT include offset.
        
        old_offset = 0.0
        if self.module.apply_calibration:
            old_offset = self.module.audio_engine.calibration.lockin_gain_offset
            
        # Reconstruct "Raw - Map" (The value before offset correction)
        # If applied: Current = (Raw - Map) - Old_Offset
        # So (Raw - Map) = Current + Old_Offset
        
        # If NOT applied: Current = Raw
        # And we assume Map is 0 (or ignored).
        # But if we calibrate Absolute Gain, we usually want it to work WITH the map.
        # If the user hasn't enabled calibration, we should probably warn them or enable it?
        # Or just calculate the offset assuming they WILL enable it.
        
        # Case 1: Calibration Enabled.
        # We adjust offset to make display match target.
        # New_Offset = Old_Offset + (Current_dBFS - Target_dBFS)
        
        # Case 2: Calibration Disabled.
        # We calculate offset such that IF enabled, it matches.
        # Raw = Current.
        # Target = Raw - Map - New_Offset.
        # New_Offset = Raw - Map - Target.
        # We need the Map value at current frequency.
        
        if not self.module.apply_calibration:
            ret = QMessageBox.question(self, tr("Enable Calibration?"), 
                                     tr("Calibration is currently disabled. To calibrate absolute gain correctly with the frequency map, we should enable calibration first.\n\nEnable and proceed?"),
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret == QMessageBox.StandardButton.Yes:
                self.cal_apply_check.setChecked(True)
                # Wait for one cycle? No, just proceed with logic, 
                # but we need to be careful about what 'current_magnitude' represents.
                # It represents the LAST processed buffer.
                # If we just enabled it, the next process_data hasn't run yet.
                # So current_magnitude is still Raw.
                # So we use Case 2 logic but set Old_Offset to what is in calibration (which is not applied yet).
                
                # Actually, simpler: Just force enable, wait a bit (or manually trigger process), then read.
                self.module.process_data() # Force process with new setting
                current_mag_linear = self.module.current_magnitude
                current_mag_dbfs = 20 * np.log10(current_mag_linear + 1e-12)
                old_offset = self.module.audio_engine.calibration.lockin_gain_offset
            else:
                return

        # Recalculate diff
        diff = current_mag_dbfs - target_dbfs
        new_offset = old_offset + diff
        
        self.module.audio_engine.calibration.set_lockin_gain_offset(new_offset)
        self.abs_cal_status.setText(tr("Current Offset: {0:.3f} dB").format(new_offset))
        
        QMessageBox.information(self, tr("Success"), tr("Absolute Gain Calibrated.\nOffset adjusted by {0:+.3f} dB.\nNew Offset: {1:.3f} dB").format(diff, new_offset))
        
    def on_cal_save(self):
        if not self.cal_data:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, tr("Save Calibration Map"), "", tr("JSON Files (*.json)"))
        if path:
            if self.module.audio_engine.calibration.save_frequency_map(path, self.cal_data):
                QMessageBox.information(self, tr("Success"), tr("Calibration map saved successfully."))
            else:
                QMessageBox.critical(self, tr("Error"), tr("Failed to save calibration map."))
                
    def on_cal_load(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Load Calibration Map"), "", tr("JSON Files (*.json)"))
        if path:
            if self.module.audio_engine.calibration.load_frequency_map(path):
                QMessageBox.information(self, tr("Success"), tr("Calibration map loaded successfully."))
                # Update plot with loaded data
                self.cal_data = self.module.audio_engine.calibration.frequency_map
                freqs = [x[0] for x in self.cal_data]
                log_freqs = [np.log10(x) for x in freqs]
                mags = [x[1] for x in self.cal_data]
                phases = [x[2] for x in self.cal_data]
                self.cal_curve_mag.setData(log_freqs, mags)
                self.cal_curve_phase.setData(log_freqs, phases)
            else:
                QMessageBox.critical(self, tr("Error"), tr("Failed to load calibration map."))

    def on_cal_apply_toggled(self, checked):
        self.module.apply_calibration = checked


    def apply_theme(self, theme_name):
        if theme_name == 'system' and hasattr(self.app, 'theme_manager'):
            theme_name = self.app.theme_manager.get_effective_theme()
            
        if theme_name == 'dark':
            # Dark Theme
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; border: 1px solid #555; border-radius: 4px; padding: 10px; font-weight: bold; }"
                "QPushButton:checked { background-color: #c62828; color: white; border: 1px solid #555; border-radius: 4px; padding: 10px; }"
                "QPushButton:hover { background-color: #388e3c; }"
                "QPushButton:checked:hover { background-color: #d32f2f; }"
            )
            self.mag_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #00ff00;")
            self.mag_db_label.setStyleSheet("font-size: 24px; color: #88ff88;")
            self.phase_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #00ffff;")
            self.x_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffff00;")
            self.y_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff00ff;")
            
            # Ref status depends on state, but we set base style here or update in update_ui
            # update_ui sets color explicitly, so we might need to update that logic too.
            # Ideally apply_theme just sets the current state's color if we track it, 
            # or we let update_ui handle it and just use theme-aware colors there.
            # But update_ui runs on timer.

        else:
            # Light Theme
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #ccffcc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 10px; font-weight: bold; }"
                "QPushButton:checked { background-color: #ffcccc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 10px; }"
                "QPushButton:hover { background-color: #bbfebb; }"
                "QPushButton:checked:hover { background-color: #ffbbbb; }"
            )
            self.mag_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #008800;")
            self.mag_db_label.setStyleSheet("font-size: 24px; color: #006600;")
            self.phase_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #008888;")
            self.x_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #888800;")
            self.y_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #880088;")

        # Theme handling
        self.app = QApplication.instance()
        if hasattr(self.app, 'theme_manager'):
            self.app.theme_manager.theme_changed.connect(self.apply_theme)
            self.apply_theme(self.app.theme_manager.get_current_theme())
