
import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, sosfiltfilt
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QSpinBox, QTabWidget)
from PyQt6.QtCore import QTimer
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class LockInTHDAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 8192 # Good compromise for latency vs precision
        self.input_data = np.zeros(self.buffer_size)
        self.residual_data = np.zeros(self.buffer_size)
        
        # Generator Settings
        self.gen_frequency = 1000.0
        self.gen_amplitude = 0.5 
        self.output_channel = 0
        self.input_channel = 0
        self.output_enabled = True
        
        # Analysis State
        self.target_freq = 1000.0 # Typically same as gen_freq for internal mode
        self.harmonic_count = 10
        self.bw_low = 20.0
        self.bw_high = 20000.0
        
        # Averaging
        self.average_count = 1
        self.amp_history = deque(maxlen=1)
        self.res_history = deque(maxlen=1)
        
        # Results
        self.measured_freq = 0.0
        self.fund_amp = 0.0
        self.fund_phase = 0.0
        self.thdn_value = 0.0
        self.thdn_db = -140.0
        self.residual_rms = 0.0
        
        # DSP State
        self._phase_gen = 0.0
        self.callback_id = None
        self.history_len = self.buffer_size * 10
        self.residual_history = deque(maxlen=self.history_len)
        
    @property
    def name(self) -> str:
        return "Lock-in THD+N"

    @property
    def description(self) -> str:
        return "High-precision THD+N measurement using lock-in fundamental removal."

    def run(self, args):
        # CLI entry not implemented
        pass

    def get_widget(self):
        return LockInTHDWidget(self)

    def start_analysis(self):
        if self.is_running: return
        self.is_running = True
        
        sample_rate = self.audio_engine.sample_rate
        self.input_data = np.zeros(self.buffer_size)
        self.residual_data = np.zeros(self.buffer_size)
        self._phase_gen = 0.0
        
        # Pre-calculate filters if needed
        # We'll do filtering in the callback or process loop
        
        def callback(indata, outdata, frames, time, status):
            if status: print(status)
            
            # Guard against zombie callback
            if not self.is_running:
                outdata.fill(0)
                return

            # 1. Generator
            outdata.fill(0)
            if self.output_enabled:
                t = (np.arange(frames) + self._phase_gen) / sample_rate
                self._phase_gen += frames
                sig = self.gen_amplitude * np.sin(2 * np.pi * self.gen_frequency * t)
                
                if self.output_channel == 2: # Stereo
                    if outdata.shape[1] >= 2: outdata[:, :2] = sig[:, np.newaxis]
                elif outdata.shape[1] > self.output_channel:
                    outdata[:, self.output_channel] = sig

            # 2. Input Capture
            if indata.shape[1] > self.input_channel:
                in_sig = indata[:, self.input_channel]
            else:
                in_sig = indata[:, 0]
                
            # Update Ring Buffer
            if len(in_sig) > self.buffer_size:
                self.input_data[:] = in_sig[-self.buffer_size:]
            else:
                self.input_data = np.roll(self.input_data, -len(in_sig))
                self.input_data[-len(in_sig):] = in_sig

        self.callback_id = self.audio_engine.register_callback(callback)
        
    def stop_analysis(self):
        if self.is_running:
            self.is_running = False # Flag first
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None

    def process(self):
        if not self.is_running: return
        
        # Snapshot of buffer
        data_full = self.input_data.copy()
        fs = self.audio_engine.sample_rate
        
        # 1. Determine Analysis Window (Integer Cycles)
        # This is crucial for Lock-in detection to avoid spectral leakage
        # when the buffer size is not an integer multiple of the signal period.
        f0 = self.gen_frequency
        if f0 <= 0: return

        samples_per_cycle = fs / f0
        n_cycles = int(len(data_full) / samples_per_cycle)
        
        if n_cycles < 1:
            # Frequency too low for buffer, use full buffer but expect leakage
            n_samples = len(data_full)
        else:
            # Use integer number of cycles to minimize leakage
            n_samples = int(n_cycles * samples_per_cycle)
            
        # Slice data
        data = data_full[:n_samples]
        N = len(data)
        t = np.arange(N) / fs
        
        # Lock-in Detection (Post-processing on block)
        # We assume frequency is known (Internal mode)
        # If we need potential tuning, we could do a coarse FFT peak find first.
        
        # Create IQ Reference
        # Note: Phase here is relative to the start of *this block*. 
        # Since we reconstruct for *this block*, relative phase is all we need.
        ref_cos = np.cos(2 * np.pi * f0 * t)
        ref_sin = np.sin(2 * np.pi * f0 * t)
        
        # Demodulate
        # I = Signal * Cos, Q = Signal * Sin
        i_comp = np.mean(data * ref_cos)
        q_comp = np.mean(data * ref_sin)
        
        # Calculate Magnitude and Phase
        # Signal ~ A * cos(wt + phi) = A * (cos(wt)cos(phi) - sin(wt)sin(phi))
        # Mean(Sig * cos) = A/2 * cos(phi)
        # Mean(Sig * sin) = -A/2 * sin(phi)
        
        # So:
        # A = 2 * sqrt(I^2 + Q^2)
        # phi = atan2(-Q, I)  <-- Careful with signs definition
        
        amp = 2 * np.sqrt(i_comp**2 + q_comp**2)
        phase = np.arctan2(-q_comp, i_comp) 
        
        self.fund_amp = amp
        self.fund_phase = phase
        self.measured_freq = f0 # For display
        
        # Reconstruction
        # s_est = amp * cos(2*pi*f0*t + phase)
        s_est = amp * np.cos(2 * np.pi * f0 * t + phase)
        
        # Residual
        residual = data - s_est
        
        # Filtering (Bandwidth Limit)
        # 20 Hz - 20 kHz (or user defined)
        # Use SOS filters
        nyquist = 0.5 * fs
        low = max(0.1, self.bw_low)
        high = min(nyquist - 1, self.bw_high)
        
        if low > 0 and low < high:
            sos_hp = butter(4, low, 'hp', fs=fs, output='sos')
            residual = sosfiltfilt(sos_hp, residual)
            
        if high < nyquist:
            sos_lp = butter(4, high, 'lp', fs=fs, output='sos')
            residual = sosfiltfilt(sos_lp, residual)
        
        self.residual_data = residual
        if self.residual_history.maxlen != self.history_len:
            self.residual_history = deque(maxlen=self.history_len)
        self.residual_history.extend(residual.tolist())
        
        # Calculate RMS
        # Remove edges to avoid filter artifacts ?
        trim = 100
        if len(residual) > 2*trim:
            res_valid = residual[trim:-trim]
        else:
            res_valid = residual
            
        self.residual_rms = np.sqrt(np.mean(res_valid**2))
        
        # Averaging
        if self.amp_history.maxlen != self.average_count:
            self.amp_history = deque(maxlen=self.average_count)
            self.res_history = deque(maxlen=self.average_count)
            
        self.amp_history.append(amp)
        self.res_history.append(self.residual_rms)
        
        avg_amp = np.mean(self.amp_history)
        avg_res = np.mean(self.res_history)
        
        self.fund_amp = avg_amp
        self.residual_rms = avg_res
        
        # THD+N
        if self.fund_amp > 1e-9:
            # Calculate using RMS of fundamental
            fund_rms = self.fund_amp / np.sqrt(2)
            ratio = self.residual_rms / fund_rms
            self.thdn_db = 20 * np.log10(ratio)
            self.thdn_value = ratio * 100
        else:
            self.thdn_db = -140.0
            self.thdn_value = 0.0


class LockInTHDWidget(QWidget):
    def __init__(self, module: LockInTHDAnalyzer):
        super().__init__()
        self.module = module
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.setInterval(100) # 10 Hz

    def init_ui(self):
        layout = QHBoxLayout()
        
        # LEFT: Controls
        left_panel = QVBoxLayout()
        settings_group = QGroupBox(tr("Settings"))
        form = QFormLayout()
        
        self.btn_toggle = QPushButton(tr("Start Measurement"))
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.clicked.connect(self.on_toggle)
        self.btn_toggle.setStyleSheet("QPushButton:checked { background-color: #ccffcc; }")
        form.addRow(self.btn_toggle)
        
        # Input Channel
        self.combo_input_ch = QComboBox()
        self.combo_input_ch.addItems([tr("Left"), tr("Right")])
        # Set initial index based on module state (0=Left, 1=Right)
        initial_idx = 0 if self.module.input_channel == 0 else 1
        self.combo_input_ch.setCurrentIndex(initial_idx)
        self.combo_input_ch.currentIndexChanged.connect(lambda v: setattr(self.module, 'input_channel', v))
        form.addRow(tr("Input Channel:"), self.combo_input_ch)

        # Output Channel
        self.combo_output_ch = QComboBox()
        self.combo_output_ch.addItems([tr("Left (Ch 1)"), tr("Right (Ch 2)"), tr("Stereo (Both)")])
        out_idx = 2 if self.module.output_channel == 2 else (1 if self.module.output_channel == 1 else 0)
        self.combo_output_ch.setCurrentIndex(out_idx)
        self.combo_output_ch.currentIndexChanged.connect(self.on_output_ch_changed)
        form.addRow(tr("Output Ch:"), self.combo_output_ch)
        
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000)
        self.freq_spin.setValue(1000)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(self.on_freq_changed)
        form.addRow(tr("Frequency:"), self.freq_spin)
        
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(-120, 20)
        self.amp_spin.setSingleStep(1.0)
        self.amp_spin.setValue(-6)
        self.amp_spin.valueChanged.connect(self.on_amp_changed)

        self.amp_unit_combo = QComboBox()
        self.amp_unit_combo.addItems(['dBFS', 'dBV', 'dBu', 'Vrms'])
        self.amp_unit_combo.currentTextChanged.connect(self.on_amp_unit_changed)

        amp_layout = QHBoxLayout()
        amp_layout.addWidget(self.amp_spin)
        amp_layout.addWidget(self.amp_unit_combo)
        form.addRow(tr("Amplitude:"), amp_layout)
        
        self.bw_low_spin = QDoubleSpinBox()
        self.bw_low_spin.setRange(0, 1000)
        self.bw_low_spin.setValue(20)
        self.bw_low_spin.valueChanged.connect(lambda v: setattr(self.module, 'bw_low', v))
        form.addRow(tr("HPF (Hz):"), self.bw_low_spin)
        
        self.bw_high_spin = QDoubleSpinBox()
        self.bw_high_spin.setRange(1000, 48000)
        self.bw_high_spin.setValue(20000)
        self.bw_high_spin.valueChanged.connect(lambda v: setattr(self.module, 'bw_high', v))
        form.addRow(tr("LPF (Hz):"), self.bw_high_spin)
        
        # Averaging
        self.spin_avg = QSpinBox()
        self.spin_avg.setRange(1, 100)
        self.spin_avg.setValue(1)
        self.spin_avg.valueChanged.connect(lambda v: setattr(self.module, 'average_count', v))
        form.addRow(tr("Averages:"), self.spin_avg)
        
        settings_group.setLayout(form)
        left_panel.addWidget(settings_group)
        
        # Meters
        meters_group = QGroupBox(tr("Results"))
        meters_layout = QVBoxLayout()
        
        # Unit Selection
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel(tr("Unit:")))
        self.combo_unit = QComboBox()
        self.combo_unit.addItems(["dBV", "dBFS"])
        self.combo_unit.setCurrentText("dBV")
        unit_layout.addWidget(self.combo_unit)
        meters_layout.addLayout(unit_layout)
        
        self.lbl_thdn = QLabel("--")
        self.lbl_thdn.setStyleSheet("font-size: 24px; font-weight: bold; color: #ff5555;")
        self.lbl_thdn_db = QLabel("-- dB")
        meters_layout.addWidget(QLabel("THD+N:"))
        meters_layout.addWidget(self.lbl_thdn)
        meters_layout.addWidget(self.lbl_thdn_db)
        
        meters_layout.addSpacing(10)
        
        self.lbl_fund = QLabel("-- V")
        self.lbl_fund.setStyleSheet("font-size: 18px; color: #55ff55;")
        meters_layout.addWidget(QLabel("Fundamental (Lock-in):"))
        meters_layout.addWidget(self.lbl_fund)
        
        self.lbl_res = QLabel("-- V")
        meters_layout.addWidget(QLabel("Residual RMS:"))
        meters_layout.addWidget(self.lbl_res)
        
        meters_group.setLayout(meters_layout)
        left_panel.addWidget(meters_group)
        
        left_panel.addStretch()
        layout.addLayout(left_panel, 1)
        
        # RIGHT: Plots
        right_panel = QVBoxLayout()
        self.tabs = QTabWidget()

        # Plot 1: Time Domain (Input vs Residual)
        self.plot_time = pg.PlotWidget(title="Time Domain")
        self.plot_time.addLegend()

        # Plot Controls
        plot_ctrl_layout = QHBoxLayout()
        self.chk_show_input = QCheckBox(tr("Show Input Trace"))
        self.chk_show_input.setChecked(True)
        self.chk_show_input.toggled.connect(self.update_plot_visibility)
        plot_ctrl_layout.addWidget(self.chk_show_input)
        plot_ctrl_layout.addStretch()

        plot_widget_container = QWidget()
        pwc_layout = QVBoxLayout()
        pwc_layout.addLayout(plot_ctrl_layout)
        pwc_layout.addWidget(self.plot_time)
        plot_widget_container.setLayout(pwc_layout)

        self.curve_input = self.plot_time.plot(pen='c', name="Input")
        self.curve_resid = self.plot_time.plot(pen='r', name="Residual (x10)")
        self.tabs.addTab(plot_widget_container, "Waveform")

        # Plot 2: Residual-only Time Domain
        self.plot_res_time = pg.PlotWidget(title="Residual Only")
        self.plot_res_time.addLegend()
        self.curve_res_time = self.plot_res_time.plot(pen='m', name="Residual")
        self.curve_res_time_avg = self.plot_res_time.plot(pen='y', name="Moving Avg")
        self.tabs.addTab(self.plot_res_time, "Residual")

        # Plot 3: Spectrum (Residual)
        self.plot_spec = pg.PlotWidget(title="Residual Spectrum")
        self.plot_spec.setLogMode(x=True, y=False)
        self.plot_spec.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_spec.setLabel('left', 'Magnitude', units='dB')
        self.plot_spec.showGrid(x=True, y=True)
        self.curve_spec = self.plot_spec.plot(pen='y')
        self.tabs.addTab(self.plot_spec, "Spectrum")
        
        right_panel.addWidget(self.tabs)
        layout.addLayout(right_panel, 3)
        
        self.setLayout(layout)

        # Initialize amplitude display to current module state
        self.on_amp_unit_changed(self.amp_unit_combo.currentText())

    def _format_si(self, value, unit):
        if value == 0:
            return f"0 {unit}"
        exponent = int(np.floor(np.log10(abs(value)) / 3) * 3)
        exponent = max(min(exponent, 9), -12)
        prefixes = {
            -12: 'p',
            -9: 'n',
            -6: 'u',
            -3: 'm',
            0: '',
            3: 'k',
            6: 'M',
            9: 'G'
        }
        scaled = value / (10 ** exponent)
        prefix = prefixes.get(exponent, '')
        return f"{scaled:.3g} {prefix}{unit}"
        
    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.btn_toggle.setText(tr("Stop"))
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.btn_toggle.setText(tr("Start Measurement"))

    def on_amp_unit_changed(self, unit):
        # Convert current linear amplitude (0-1 FS) into selected unit using calibration gain
        amp_linear = self.module.gen_amplitude
        gain = self.module.audio_engine.calibration.output_gain

        self.amp_spin.blockSignals(True)
        if unit == 'dBFS':
            val = 20 * np.log10(amp_linear + 1e-12)
            self.amp_spin.setRange(-120, 6)
            self.amp_spin.setSingleStep(1.0)
        elif unit == 'dBV':
            v_peak = amp_linear * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10(v_rms + 1e-12)
            self.amp_spin.setRange(-120, 20)
            self.amp_spin.setSingleStep(0.5)
        elif unit == 'dBu':
            v_peak = amp_linear * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10((v_rms + 1e-12) / 0.7746)
            self.amp_spin.setRange(-120, 20)
            self.amp_spin.setSingleStep(0.5)
        else: # Vrms
            v_peak = amp_linear * gain
            val = v_peak / np.sqrt(2)
            self.amp_spin.setRange(0, 100)
            self.amp_spin.setSingleStep(0.01)

        self.amp_spin.setValue(val)
        self.amp_spin.blockSignals(False)

    def on_amp_changed(self, val):
        unit = self.amp_unit_combo.currentText()
        gain = self.module.audio_engine.calibration.output_gain

        if unit == 'dBFS':
            amp_linear = 10**(val/20)
        elif unit == 'dBV':
            v_rms = 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_linear = v_peak / gain
        elif unit == 'dBu':
            v_rms = 0.7746 * 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_linear = v_peak / gain
        else: # Vrms
            v_peak = val * np.sqrt(2)
            amp_linear = v_peak / gain

        amp_linear = max(0.0, min(1.0, amp_linear))
        self.module.gen_amplitude = amp_linear
            
    def on_freq_changed(self, val):
        self.module.gen_frequency = val
        self.module.target_freq = val

    def on_output_ch_changed(self, idx):
        self.module.output_channel = idx
        
    def update_plot_visibility(self, checked):
        if checked:
            self.curve_input.setVisible(True)
        else:
            self.curve_input.setVisible(False)

    def update_ui(self):
        if not self.module.is_running: return
        
        # Trigger processing
        self.module.process()
        
        # Update Labels
        self.lbl_thdn.setText(f"{self.module.thdn_value:.4f} %")
        self.lbl_thdn_db.setText(f"{self.module.thdn_db:.2f} dB")
        
        # Unit Conversion
        unit = self.combo_unit.currentText()
        fund_peak_fs = self.module.fund_amp
        res_rms_fs = self.module.residual_rms
        
        calibration = self.module.audio_engine.calibration
        offset_db = calibration.get_input_offset_db()
        sensitivity = calibration.input_sensitivity
        
        if unit == "dBV":
            # RMS conversions
            fund_rms_fs = fund_peak_fs / np.sqrt(2)
            fund_dbv = 20 * np.log10(fund_rms_fs + 1e-12) + offset_db
            res_dbv = 20 * np.log10(res_rms_fs + 1e-12) + offset_db

            fund_rms_v = fund_rms_fs * sensitivity
            res_rms_v = res_rms_fs * sensitivity

            fund_str = f"{fund_dbv:.2f} dBV ( {self._format_si(fund_rms_v, 'V')} rms )"
            res_str = f"{res_dbv:.2f} dBV ( {self._format_si(res_rms_v, 'V')} rms )"
            
        else: # dBFS
            fund_dbfs = 20 * np.log10(fund_peak_fs + 1e-12)
            res_dbfs = 20 * np.log10(res_rms_fs + 1e-12)
            fund_str = f"{fund_dbfs:.2f} dBFS"
            res_str = f"{res_dbfs:.2f} dBFS"
            
        self.lbl_fund.setText(fund_str)
        self.lbl_res.setText(res_str)
        
        # Update Plots
        # Decimate for performance
        data = self.module.input_data
        res = self.module.residual_data
        res_hist = np.array(self.module.residual_history)
        
        step = max(1, len(data) // 1000)
        
        if self.chk_show_input.isChecked():
            self.curve_input.setData(data[::step])
        
        # Scale residual for visibility
        self.curve_resid.setData(res[::step] * 10) 
        # Long residual history with optional smoothing
        if len(res_hist) > 0:
            fs = self.module.audio_engine.sample_rate
            x_hist = np.arange(len(res_hist)) / fs
            self.curve_res_time.setData(x_hist, res_hist)
            # Moving average to show slow drift / integration
            win = max(10, min(len(res_hist)//20, 2000))
            if win < len(res_hist):
                kernel = np.ones(win) / win
                ma = np.convolve(res_hist, kernel, mode='valid')
                x_ma = x_hist[win-1:]
                self.curve_res_time_avg.setData(x_ma, ma)
            else:
                self.curve_res_time_avg.setData([], [])
        else:
            self.curve_res_time.setData([])
            self.curve_res_time_avg.setData([])
        
        # Spectrum
        # Calculate FFT of residual
        if len(res) > 0:
            window = np.hanning(len(res))
            fft_res = np.fft.rfft(res * window)
            mag = 20 * np.log10(np.abs(fft_res) / len(res) * 2 + 1e-12)
            freqs = np.fft.rfftfreq(len(res), 1/self.module.audio_engine.sample_rate)
            
            # Skip DC
            self.curve_spec.setData(freqs[1:], mag[1:])
