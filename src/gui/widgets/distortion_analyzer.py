import argparse

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import get_window

from src.core.analysis import AudioCalc
from src.core.audio_engine import AudioEngine
from src.core.localization import tr
from src.measurement_modules.base import MeasurementModule


class DistortionAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 16384 # Larger buffer for better frequency resolution
        self.input_data = np.zeros(self.buffer_size)

        # Generator Settings
        self.gen_frequency = 1000.0
        self.gen_frequency = 1000.0
        self._gen_amplitude = 0.5 # Linear 0-1
        self.output_channel = 0 # 0: Left, 1: Right
        self.input_channel = 0 # 0: Left, 1: Right
        self.output_enabled = True

        # Analysis Settings
        self.window_type = 'blackmanharris' # Good for distortion
        self.averaging = 0.0

        # IMD Settings
        self.imd_standard = 'smpte' # 'smpte' or 'ccif'
        self.imd_f1 = 60.0
        self.imd_f2 = 7000.0
        self.imd_ratio = 4.0 # 4:1 for SMPTE

        # Playback state
        self._phase_f1 = 0.0
        self._phase_f1 = 0.0
        self._phase_f2 = 0.0

        # Mode
        self.mode = 'Real-time'
        self.signal_type = 'sine' # 'sine', 'smpte', 'ccif'

        # State
        self.current_result = None
        self._avg_thdn = None
        self._avg_state = None
        self._avg_spectrum = None
        self._avg_imd_ratio = None

        # Capture State
        self.capture_requested = False
        self.capture_ready = False
        self.captured_buffer = None

        # Sweep State
        self.sweep_mode = False
        self.sweep_running = False
        self.sweep_results = []

        self.callback_id = None

    def reset_averaging_state(self):
        """Clear cached averaging state when settings change."""
        self._avg_thdn = None
        self._avg_state = None
        self._avg_spectrum = None
        self._avg_imd_ratio = None

    def _apply_result_averaging(self, results: dict) -> dict:
        """Apply exponential averaging to harmonic metrics using raw components."""
        alpha = self.averaging
        if alpha <= 0:
            self.reset_averaging_state()
            return results

        raw_fund_rms = float(results.get('raw_fund_rms', 0.0))
        raw_res_rms = float(results.get('raw_res_rms', 0.0))
        raw_fund_amp = float(results.get('raw_fund_amp', 0.0))
        raw_freq = float(results.get('basic_wave', {}).get('frequency', self.gen_frequency))
        raw_amp_dbfs = float(results.get('basic_wave', {}).get('amplitude_dbfs', -140.0))
        raw_harmonics = np.array(results.get('raw_harmonics', []), dtype=float)

        # Initialize or update state
        if self._avg_state is None or self._avg_state['harmonics'].shape != raw_harmonics.shape:
            self._avg_state = {
                'fund_rms': raw_fund_rms,
                'res_rms': raw_res_rms,
                'fund_amp': raw_fund_amp,
                'frequency': raw_freq,
                'amplitude_dbfs': raw_amp_dbfs,
                'harmonics': raw_harmonics,
            }
        else:
            self._avg_state['fund_rms'] = alpha * self._avg_state['fund_rms'] + (1 - alpha) * raw_fund_rms
            self._avg_state['res_rms'] = alpha * self._avg_state['res_rms'] + (1 - alpha) * raw_res_rms
            self._avg_state['fund_amp'] = alpha * self._avg_state['fund_amp'] + (1 - alpha) * raw_fund_amp
            self._avg_state['frequency'] = alpha * self._avg_state['frequency'] + (1 - alpha) * raw_freq
            self._avg_state['amplitude_dbfs'] = alpha * self._avg_state['amplitude_dbfs'] + (1 - alpha) * raw_amp_dbfs
            self._avg_state['harmonics'] = alpha * self._avg_state['harmonics'] + (1 - alpha) * raw_harmonics

        state = self._avg_state

        fund_amp = max(state['fund_amp'], 1e-12)
        fund_rms = max(state['fund_rms'], 1e-12)
        res_rms = max(state['res_rms'], 0.0)

        thd_linear = 0.0
        if fund_amp > 0 and state['harmonics'].size:
            thd_linear = np.sqrt(np.sum(state['harmonics']**2)) / fund_amp

        thd_percent = thd_linear * 100
        thd_db = 20 * np.log10(thd_linear + 1e-12)

        thdn_linear = res_rms / fund_rms if fund_rms > 0 else 0.0
        thdn_percent = thdn_linear * 100
        thdn_db = 20 * np.log10(thdn_linear + 1e-12)
        sinad_db = -thdn_db

        # Rebuild harmonics list using averaged fundamentals for relative levels
        harmonics = []
        base_freq = state['frequency']
        for idx, amp in enumerate(state['harmonics']):
            order = idx + 2
            rel_amp = amp / fund_amp if fund_amp > 0 else 0.0
            harmonics.append({
                'order': order,
                'frequency': base_freq * order,
                'amplitude_dbr': 20 * np.log10(rel_amp + 1e-12),
                'amplitude_linear': float(amp)
            })

        averaged = {
            'basic_wave': {
                'frequency': state['frequency'],
                'amplitude_dbfs': state['amplitude_dbfs'],
                'max_amplitude': state['fund_amp']
            },
            'harmonics': harmonics,
            'thd_percent': thd_percent,
            'thd_db': thd_db,
            'thdn_percent': thdn_percent,
            'thdn_db': thdn_db,
            'sinad_db': sinad_db,
            # Preserve averaged raw components for downstream use/inspection
            'raw_fund_rms': state['fund_rms'],
            'raw_res_rms': state['res_rms'],
            'raw_harmonics': state['harmonics'],
            'raw_fund_amp': state['fund_amp']
        }

        return averaged

    def _apply_imd_averaging(self, imd_result: dict) -> dict:
        """Apply exponential averaging to IMD results in the linear domain."""
        alpha = self.averaging
        if alpha <= 0:
            self._avg_imd_ratio = None
            return imd_result

        raw_ratio = max(float(imd_result.get('imd', 0.0)) / 100.0, 0.0)
        if self._avg_imd_ratio is None:
            self._avg_imd_ratio = raw_ratio
        else:
            self._avg_imd_ratio = alpha * self._avg_imd_ratio + (1 - alpha) * raw_ratio

        ratio = self._avg_imd_ratio
        imd_percent = ratio * 100.0
        imd_db = 20 * np.log10(ratio) if ratio > 1e-12 else -100.0

        return {
            'imd': imd_percent,
            'imd_db': imd_db,
            'raw_imd_ratio': ratio
        }

    def apply_spectrum_averaging(self, mag_linear: np.ndarray) -> np.ndarray:
        """Smooth spectrum magnitude with exponential averaging (linear domain)."""
        alpha = self.averaging
        if alpha <= 0:
            self._avg_spectrum = None
            return mag_linear

        if self._avg_spectrum is None or self._avg_spectrum.shape != mag_linear.shape:
            self._avg_spectrum = mag_linear
        else:
            self._avg_spectrum = alpha * self._avg_spectrum + (1 - alpha) * mag_linear

        return self._avg_spectrum

    @property
    def gen_amplitude(self):
        return self._gen_amplitude

    @gen_amplitude.setter
    def gen_amplitude(self, value):
        # Clamp to prevent overflow (e.g. if user enters Hz in dB field)
        # Max 10.0 (20dB headroom above 0dBFS) is plenty.
        if value > 10.0:
            value = 10.0
        elif value < 0.0:
            value = 0.0
        self._gen_amplitude = value

    @property
    def name(self) -> str:
        return "Distortion Analyzer"

    @property
    def description(self) -> str:
        return "THD, THD+N, and SINAD measurements."

    def run(self, args: argparse.Namespace):
        print("Distortion Analyzer running from CLI (not implemented)")

    def get_widget(self):
        return DistortionAnalyzerWidget(self)

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        self.reset_averaging_state()
        self.input_data = np.zeros(self.buffer_size)
        self.current_result = None

        sample_rate = self.audio_engine.sample_rate
        phase = 0

        def callback(indata, outdata, frames, time, status):
            nonlocal phase
            if status:
                print(status)

            # Generate Signal
            outdata.fill(0)
            if self.output_enabled:
                # Check signal type
                if self.signal_type == 'smpte' or self.signal_type == 'ccif':
                    sine_wave = self._generate_dual_tone(frames, sample_rate)
                else:
                    t = (np.arange(frames) + phase) / sample_rate
                    phase += frames
                    sine_wave = self.gen_amplitude * np.sin(2 * np.pi * self.gen_frequency * t)

                if self.output_channel == 0:
                    outdata[:, 0] = sine_wave
                elif self.output_channel == 1:
                    if outdata.shape[1] > 1:
                        outdata[:, 1] = sine_wave
            else:
                pass

            # Capture Input
            capture_ch = self.input_channel

            if indata.shape[1] > capture_ch:
                new_data = indata[:, capture_ch]
            else:
                new_data = indata[:, 0]

            # Ring buffer update
            if len(new_data) > self.buffer_size:
                self.input_data[:] = new_data[-self.buffer_size:]
            else:
                self.input_data = np.roll(self.input_data, -len(new_data))
                self.input_data[-len(new_data):] = new_data

            # Handle Capture Request (Thread-safe copy)
            if self.capture_requested:
                self.captured_buffer = self.input_data.copy()
                self.capture_requested = False
                self.capture_ready = True

        self.callback_id = self.audio_engine.register_callback(callback)

    def _generate_dual_tone(self, frames, sample_rate):
        # Calculate amplitudes based on ratio
        # Total amplitude should not exceed self.gen_amplitude

        if self.imd_standard == 'smpte':
            # ratio = amp_f1 / amp_f2
            # amp_f2 * (ratio + 1) = self.gen_amplitude
            amp_f2 = self.gen_amplitude / (self.imd_ratio + 1)
            amp_f1 = amp_f2 * self.imd_ratio
        else: # CCIF
            # 1:1 ratio usually
            amp_f1 = self.gen_amplitude / 2
            amp_f2 = self.gen_amplitude / 2

        # Generate phases
        phase_inc_f1 = 2 * np.pi * self.imd_f1 / sample_rate
        phase_inc_f2 = 2 * np.pi * self.imd_f2 / sample_rate

        t = np.arange(frames)
        phases_f1 = self._phase_f1 + t * phase_inc_f1
        phases_f2 = self._phase_f2 + t * phase_inc_f2

        # Update state
        self._phase_f1 = (self._phase_f1 + frames * phase_inc_f1) % (2 * np.pi)
        self._phase_f2 = (self._phase_f2 + frames * phase_inc_f2) % (2 * np.pi)

        signal = amp_f1 * np.sin(phases_f1) + amp_f2 * np.sin(phases_f2)
        return signal

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def request_capture(self):
        """Request a thread-safe capture of the current input buffer."""
        self.capture_ready = False
        self.capture_requested = True

class SweepWorker(QThread):
    result_ready = pyqtSignal(dict)
    finished = pyqtSignal()
    progress = pyqtSignal(int, int)

    def __init__(self, module, sweep_type, start, end, steps, duration_ms=1000):
        super().__init__()
        self.module = module
        self.sweep_type = sweep_type # 'frequency' or 'amplitude'
        self.start_val = start
        self.end_val = end
        self.steps = steps
        self.duration_ms = duration_ms
        self.is_running = True

    def run(self):
        # Generate steps
        if self.sweep_type == 'frequency':
            # Logarithmic sweep for frequency
            values = np.logspace(np.log10(self.start_val), np.log10(self.end_val), self.steps)
        else:
            # Linear sweep for amplitude (dB)
            values = np.linspace(self.start_val, self.end_val, self.steps)

        for i, val in enumerate(values):
            if not self.is_running:
                break

            # Set Generator
            if self.sweep_type == 'frequency':
                self.module.gen_frequency = val
            else:
                # val is dBFS, convert to linear
                self.module.gen_amplitude = 10**(val/20)

            # Wait for settling (Generator update + Audio Buffer Latency)
            # Ensure at least 300ms wait
            wait_time = max(300, self.duration_ms)
            self.msleep(wait_time)

            # Use safe capture
            self.module.request_capture()
            # Wait for capture
            timeout = 0
            while not self.module.capture_ready and timeout < 50: # 500ms timeout
                self.msleep(10)
                timeout += 1

            if self.module.capture_ready:
                data = self.module.captured_buffer
            else:
                data = self.module.input_data.copy() # Fallback

            sample_rate = self.module.audio_engine.sample_rate

            results = AudioCalc.analyze_harmonics(
                data,
                self.module.gen_frequency,
                self.module.window_type,
                sample_rate
            )

            # Add sweep parameter to results
            results['sweep_param'] = val
            self.result_ready.emit(results)
            self.progress.emit(i + 1, self.steps)

        self.finished.emit()

    def stop(self):
        self.is_running = False


class DistortionAnalyzerWidget(QWidget):
    def __init__(self, module: DistortionAnalyzer):
        super().__init__()
        self.module = module
        self.sweep_worker = None
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_realtime_analysis)
        self.timer.setInterval(100) # 10Hz update

    def init_ui(self):
        layout = QHBoxLayout()

        # --- Left Panel: Controls & Meters ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # 1. Mode Selection
        mode_group = QGroupBox(tr("Mode"))
        mode_layout = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([tr("Real-time"), tr("Frequency Sweep"), tr("Amplitude Sweep")])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        left_panel.addWidget(mode_group)

        # 2. Settings Tabs
        self.settings_tabs = QTabWidget()

        # Page 1: Real-time Controls
        rt_widget = QWidget()
        rt_layout = QFormLayout()

        # Output Mode
        self.out_mode_combo = QComboBox()
        self.out_mode_combo.addItems([tr("Off (External Source)"), tr("Sine Wave"), tr("SMPTE IMD"), tr("CCIF IMD")])
        self.out_mode_combo.currentIndexChanged.connect(self.on_out_mode_changed)
        rt_layout.addRow(tr("Signal Generator:"), self.out_mode_combo)

        # Generator Settings Stack
        self.gen_stack = QStackedWidget()

        # 1. Sine Settings
        sine_widget = QWidget()
        sine_layout = QFormLayout()
        sine_layout.setContentsMargins(0,0,0,0)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000)
        self.freq_spin.setValue(1000)
        self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(self.on_freq_changed)
        sine_layout.addRow(tr("Frequency:"), self.freq_spin)

        sine_widget.setLayout(sine_layout)
        self.gen_stack.addWidget(sine_widget)

        # 2. IMD Settings
        imd_gen_widget = QWidget()
        imd_gen_layout = QFormLayout()
        imd_gen_layout.setContentsMargins(0,0,0,0)

        self.imd_f1_spin = QDoubleSpinBox()
        self.imd_f1_spin.setRange(10, 20000)
        self.imd_f1_spin.setValue(self.module.imd_f1)
        self.imd_f1_spin.valueChanged.connect(lambda v: setattr(self.module, 'imd_f1', v))
        imd_gen_layout.addRow(tr("Freq 1 (Hz):"), self.imd_f1_spin)

        self.imd_f2_spin = QDoubleSpinBox()
        self.imd_f2_spin.setRange(10, 24000)
        self.imd_f2_spin.setValue(self.module.imd_f2)
        self.imd_f2_spin.valueChanged.connect(lambda v: setattr(self.module, 'imd_f2', v))
        imd_gen_layout.addRow(tr("Freq 2 (Hz):"), self.imd_f2_spin)

        self.imd_ratio_spin = QDoubleSpinBox()
        self.imd_ratio_spin.setRange(1, 10)
        self.imd_ratio_spin.setValue(self.module.imd_ratio)
        self.imd_ratio_spin.valueChanged.connect(lambda v: setattr(self.module, 'imd_ratio', v))
        imd_gen_layout.addRow(tr("Ratio (F1:F2):"), self.imd_ratio_spin)

        imd_gen_widget.setLayout(imd_gen_layout)
        imd_gen_widget.setLayout(imd_gen_layout)
        self.gen_stack.addWidget(imd_gen_widget)

        rt_layout.addRow(self.gen_stack)

        # Amplitude (Shared)
        amp_layout = QHBoxLayout()
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(-120, 20) # Allow positive for dBV/dBu
        self.amp_spin.setValue(-6)
        self.amp_spin.valueChanged.connect(self.on_amp_changed)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['dBFS', 'dBV', 'dBu', 'Vrms'])
        self.unit_combo.currentTextChanged.connect(self.on_unit_changed)

        amp_layout.addWidget(self.amp_spin)
        amp_layout.addWidget(self.unit_combo)
        rt_layout.addRow(tr("Amplitude:"), amp_layout)

        rt_widget.setLayout(rt_layout)
        self.settings_tabs.addTab(rt_widget, tr("Signal"))

        # Page 2: Sweep Controls
        sweep_widget = QWidget()
        sweep_layout = QFormLayout()

        self.sweep_start_spin = QDoubleSpinBox()
        self.sweep_start_spin.setRange(-120, 20000)
        self.sweep_start_spin.setValue(20)
        sweep_layout.addRow(tr("Start:"), self.sweep_start_spin)

        self.sweep_end_spin = QDoubleSpinBox()
        self.sweep_end_spin.setRange(-120, 20000)
        self.sweep_end_spin.setValue(20000)
        sweep_layout.addRow(tr("End:"), self.sweep_end_spin)

        self.sweep_steps_spin = QSpinBox()
        self.sweep_steps_spin.setRange(2, 1000)
        self.sweep_steps_spin.setValue(30)
        sweep_layout.addRow(tr("Steps:"), self.sweep_steps_spin)

        sweep_widget.setLayout(sweep_layout)
        self.settings_tabs.addTab(sweep_widget, tr("Sweep"))

        # Settings Tab
        common_widget = QWidget()
        common_layout = QFormLayout()

        self.in_channel_combo = QComboBox()
        self.in_channel_combo.addItems([tr("Left (Ch 1)"), tr("Right (Ch 2)")])
        self.in_channel_combo.currentIndexChanged.connect(self.on_in_channel_changed)
        common_layout.addRow(tr("Input Ch:"), self.in_channel_combo)

        self.channel_combo = QComboBox()
        self.channel_combo.addItems([tr("Left (Ch 1)"), tr("Right (Ch 2)")])
        self.channel_combo.currentIndexChanged.connect(self.on_channel_changed)
        common_layout.addRow(tr("Output Ch:"), self.channel_combo)

        # Averaging (Exponential)
        self.avg_label = QLabel(tr("Avg: 0%"))
        self.avg_slider = QSlider(Qt.Orientation.Horizontal)
        self.avg_slider.setRange(0, 95)
        self.avg_slider.setValue(0)
        self.avg_slider.setFixedWidth(120)
        self.avg_slider.valueChanged.connect(self.on_avg_changed)

        avg_row = QHBoxLayout()
        avg_row.addWidget(self.avg_label)
        avg_row.addWidget(self.avg_slider)
        common_layout.addRow(tr("Averaging:"), avg_row)

        common_widget.setLayout(common_layout)
        self.settings_tabs.addTab(common_widget, tr("Settings"))

        left_panel.addWidget(self.settings_tabs)

        # Action Buttons
        btn_layout = QVBoxLayout()
        self.action_btn = QPushButton(tr("Start Measurement"))
        self.action_btn.setCheckable(True)
        self.action_btn.clicked.connect(self.on_action)
        self.action_btn.setStyleSheet("QPushButton:checked { background-color: #ccffcc; }")
        btn_layout.addWidget(self.action_btn)

        self.action_btn.setStyleSheet("QPushButton:checked { background-color: #ccffcc; }")
        btn_layout.addWidget(self.action_btn)

        left_panel.addLayout(btn_layout)

        # 3. Meters (Real-time only)
        self.meters_group = QGroupBox(tr("Measurements"))
        self.meters_main_layout = QVBoxLayout()
        self.meters_group.setLayout(self.meters_main_layout)

        # View switcher
        self.meters_view_stack = QStackedWidget()
        self.meters_main_layout.addWidget(self.meters_view_stack)

        # --- Basic View ---
        basic_view = QWidget()
        meters_layout = QFormLayout(basic_view)

        # THD+N row (value and dB on one line)
        thdn_row = QWidget()
        thdn_row_layout = QHBoxLayout(thdn_row)
        thdn_row_layout.setContentsMargins(0, 0, 0, 0)
        self.thdn_label = QLabel(tr("-- %"))
        self.thdn_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ff5555;")
        self.thdn_db_label = QLabel(tr("-- dB"))
        thdn_row_layout.addWidget(self.thdn_label)
        thdn_row_layout.addWidget(self.thdn_db_label)
        thdn_row_layout.addStretch()
        meters_layout.addRow(QLabel(tr("THD+N:")), thdn_row)

        # THD row
        self.thd_label = QLabel(tr("-- %"))
        self.thd_label.setStyleSheet("font-size: 16px; color: #ffaa55;")
        meters_layout.addRow(QLabel(tr("THD:")), self.thd_label)

        # SINAD row
        self.sinad_label = QLabel(tr("-- dB"))
        self.sinad_label.setStyleSheet("font-size: 16px; color: #55ffff;")
        meters_layout.addRow(QLabel(tr("SINAD:")), self.sinad_label)

        # IMD row (Hidden by default)
        self.imd_label = QLabel(tr("-- %"))
        self.imd_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ff55ff;")
        self.imd_db_label = QLabel(tr("-- dB"))
        self.imd_row_widget = QWidget()
        imd_row_layout = QHBoxLayout(self.imd_row_widget)
        imd_row_layout.setContentsMargins(0, 0, 0, 0)
        imd_row_layout.addWidget(self.imd_label)
        imd_row_layout.addWidget(self.imd_db_label)
        imd_row_layout.addStretch()
        meters_layout.addRow(QLabel(tr("IMD:")), self.imd_row_widget)
        self.imd_row_widget.setVisible(False)

        self.meters_view_stack.addWidget(basic_view)

        # --- Detailed View ---
        detailed_view = QWidget()
        detailed_layout = QVBoxLayout(detailed_view)
        detailed_layout.setContentsMargins(0, 5, 0, 5)

        self.detailed_label = QLabel()
        self.detailed_label.setStyleSheet("font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.5;")
        self.detailed_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        detailed_layout.addWidget(self.detailed_label)

        self.meters_view_stack.addWidget(detailed_view)

        # Toggle button
        self.view_toggle_btn = QPushButton(tr("Show Detailed"))
        self.view_toggle_btn.setCheckable(True)
        self.view_toggle_btn.clicked.connect(self.on_toggle_view)
        self.meters_main_layout.addWidget(self.view_toggle_btn)

        left_panel.addWidget(self.meters_group)

        left_panel.addStretch()
        layout.addLayout(left_panel, 1)

        # --- Right Panel: Plots & Tables ---
        right_panel = QVBoxLayout()

        self.tabs = QTabWidget()

        # Tab 1: Spectrum
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel('left', tr('Amplitude'), units='dBFS')
        self.spectrum_plot.setLabel('bottom', tr('Frequency'), units='Hz')
        self.spectrum_plot.setLogMode(x=True, y=False)
        self.spectrum_plot.setYRange(-140, 0)
        self.spectrum_plot.setYRange(-140, 0)
        self.spectrum_plot.showGrid(x=True, y=True)

        # Custom Axis Ticks for Spectrum
        axis_spec = self.spectrum_plot.getPlotItem().getAxis('bottom')
        ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        ticks_log = [(np.log10(t), str(t) if t < 1000 else f"{t/1000:.0f}k") for t in ticks]
        axis_spec.setTicks([ticks_log])

        # Set Range (log domain)
        self.spectrum_plot.setXRange(np.log10(20), np.log10(20000))

        self.spectrum_curve = self.spectrum_plot.plot(pen='y')
        self.tabs.addTab(self.spectrum_plot, tr("Spectrum"))

        # Tab 2: Harmonics (Table + Bar Graph)
        harmonics_widget = QWidget()
        harmonics_layout = QVBoxLayout(harmonics_widget)

        self.harmonics_table = QTableWidget()
        self.harmonics_table.setColumnCount(4)
        self.harmonics_table.setHorizontalHeaderLabels([tr("Order"), tr("Freq (Hz)"), tr("Level (dBr)"), tr("Level (Linear)")])
        self.harmonics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        harmonics_layout.addWidget(self.harmonics_table, 1) # Stretch factor 1

        # Harmonics Bar Graph
        self.harmonics_plot = pg.PlotWidget()
        self.harmonics_plot.setLabel('left', tr('Level'), units='dBr')
        self.harmonics_plot.setLabel('bottom', tr('Harmonic Order'))
        self.harmonics_plot.showGrid(x=False, y=True)
        self.harmonics_plot.setYRange(-140, 0)

        self.harmonics_bar_item = pg.BarGraphItem(x=[], height=[], width=0.6, brush='b')
        self.harmonics_plot.addItem(self.harmonics_bar_item)

        harmonics_layout.addWidget(self.harmonics_plot, 1) # Stretch factor 1

        self.tabs.addTab(harmonics_widget, tr("Harmonics"))

        # Tab 3: Sweep Results
        self.sweep_plot = pg.PlotWidget()
        self.sweep_plot.setLabel('left', tr('THD+N'), units='dB')
        self.sweep_plot.setLabel('bottom', tr('Frequency'), units='Hz') # Dynamic label
        self.sweep_plot.setLogMode(x=True, y=False)
        self.sweep_plot.setLogMode(x=True, y=False)
        self.sweep_plot.showGrid(x=True, y=True)

        # Custom Axis Ticks for Sweep (Frequency Mode)
        # Note: If mode changes to Amplitude Sweep, we might need to reset this?
        # The user only requested "like Spectrum Analyzer", which implies Frequency domain.
        # We'll set it here, and handle mode changes if necessary.
        self.sweep_axis = self.sweep_plot.getPlotItem().getAxis('bottom')
        self.sweep_axis.setTicks([ticks_log])

        # Set Range (log domain) for Frequency Sweep default
        self.sweep_plot.setXRange(np.log10(20), np.log10(20000))

        self.sweep_curve = self.sweep_plot.plot(pen='c', symbol='o')
        self.tabs.addTab(self.sweep_plot, tr("Sweep Results"))

        right_panel.addWidget(self.tabs)
        layout.addLayout(right_panel, 3)

        self.setLayout(layout)

        # Initial update
        self.on_unit_changed(self.unit_combo.currentText())
        self.out_mode_combo.setCurrentIndex(1) # Default to Sine Wave

    def on_mode_changed(self, idx):
        # 0: Real-time, 1: Frequency Sweep, 2: Amplitude Sweep
        modes = ["Real-time", "Frequency Sweep", "Amplitude Sweep"]
        if 0 <= idx < len(modes):
            self.module.mode = modes[idx]

        if idx == 0: # Real-time
            self.settings_tabs.setCurrentIndex(0)
            self.meters_group.setVisible(True)
            self.set_meters_mode('thd')
            self.tabs.setCurrentIndex(0)
        else:
            self.settings_tabs.setCurrentIndex(1)
            self.meters_group.setVisible(False)
            self.tabs.setCurrentIndex(2)

            if idx == 1: # Frequency Sweep
                self.sweep_start_spin.setSuffix(" Hz")
                self.sweep_end_spin.setSuffix(" Hz")
                self.sweep_start_spin.setValue(20)
                self.sweep_end_spin.setValue(20000)
                self.sweep_plot.setLabel('bottom', tr('Frequency'), units='Hz')
                self.sweep_plot.setLogMode(x=True, y=False)
                # Restore custom ticks for frequency
                ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
                ticks_log = [(np.log10(t), str(t) if t < 1000 else f"{t/1000:.0f}k") for t in ticks]
                self.sweep_axis.setTicks([ticks_log])
                self.sweep_plot.setXRange(np.log10(20), np.log10(20000))
            else: # Amplitude Sweep
                self.sweep_start_spin.setSuffix(" dBFS")
                self.sweep_end_spin.setSuffix(" dBFS")
                self.sweep_start_spin.setValue(-60)
                self.sweep_end_spin.setValue(0)
                self.sweep_plot.setLabel('bottom', tr('Amplitude'), units='dBFS')
                self.sweep_plot.setLogMode(x=False, y=False)
                # Reset ticks to auto for amplitude
                self.sweep_axis.setTicks(None)

    def on_out_mode_changed(self, idx):
        # 0: Off, 1: Sine, 2: SMPTE, 3: CCIF
        if idx == 0: # Off
            self.module.output_enabled = False
            self.gen_stack.setVisible(False)
            self.amp_spin.setEnabled(False)
            self.unit_combo.setEnabled(False)
            self.module.signal_type = 'sine' # Default
        else:
            self.module.output_enabled = True
            self.gen_stack.setVisible(True)
            self.amp_spin.setEnabled(True)
            self.unit_combo.setEnabled(True)

            if idx == 1: # Sine
                self.module.signal_type = 'sine'
                self.gen_stack.setCurrentIndex(0)
                self.set_meters_mode('thd')
                self.module.reset_averaging_state()
            elif idx == 2: # SMPTE
                self.module.signal_type = 'smpte'
                self.module.imd_standard = 'smpte'
                self.gen_stack.setCurrentIndex(1)
                self.set_meters_mode('imd')
                # Update IMD params
                self.module.imd_f1 = 60.0
                self.module.imd_f2 = 7000.0
                self.imd_f1_spin.setValue(60.0)
                self.imd_f2_spin.setValue(7000.0)
                self.imd_ratio_spin.setEnabled(True)
                self.module.reset_averaging_state()
            elif idx == 3: # CCIF
                self.module.signal_type = 'ccif'
                self.module.imd_standard = 'ccif'
                self.gen_stack.setCurrentIndex(1)
                self.set_meters_mode('imd')
                # Update IMD params
                self.module.imd_f1 = 19000.0
                self.module.imd_f2 = 20000.0
                self.imd_f1_spin.setValue(19000.0)
                self.imd_f2_spin.setValue(20000.0)
                self.imd_f2_spin.setValue(20000.0)
                self.imd_ratio_spin.setEnabled(False)
                self.module.reset_averaging_state()

    def on_unit_changed(self, unit):
        # Update spin box range/value based on current amplitude
        # Current amplitude is stored in module as Linear (0-1)
        # But we need to convert it.
        # Actually, let's just update the display value.

        amp_linear = self.module.gen_amplitude
        gain = self.module.audio_engine.calibration.output_gain

        self.amp_spin.blockSignals(True)

        if unit == 'dBFS':
            val = 20 * np.log10(amp_linear + 1e-12)
        elif unit == 'dBV':
            v_peak = amp_linear * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10(v_rms + 1e-12)
        elif unit == 'dBu':
            v_peak = amp_linear * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10((v_rms + 1e-12) / 0.7746)
        elif unit == 'Vrms':
            v_peak = amp_linear * gain
            val = v_peak / np.sqrt(2)

        self.amp_spin.setValue(val)
        self.amp_spin.blockSignals(False)

    def on_amp_changed(self, val):
        unit = self.unit_combo.currentText()
        gain = self.module.audio_engine.calibration.output_gain
        amp_linear = 0.0

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
        elif unit == 'Vrms':
            v_peak = val * np.sqrt(2)
            amp_linear = v_peak / gain

        # Clamp
        if amp_linear > 1.0:
            amp_linear = 1.0
        elif amp_linear < 0.0:
            amp_linear = 0.0

        self.module.gen_amplitude = amp_linear

    def on_action(self, checked):
        idx = self.mode_combo.currentIndex()
        if idx == 0: # Real-time
            self.on_toggle_realtime(checked)
        else:
            if checked:
                self.start_sweep(idx)
            else:
                self.stop_sweep()

    def on_toggle_realtime(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.action_btn.setText(tr("Stop Measurement"))
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.action_btn.setText(tr("Start Measurement"))

    def set_meters_mode(self, mode):
        if mode == 'thd':
            self.thdn_label.setVisible(True)
            self.thdn_db_label.setVisible(True)
            self.thd_label.setVisible(True)
            self.sinad_label.setVisible(True)
            self.imd_row_widget.setVisible(False)

        else: # imd
            self.thdn_label.setVisible(False)
            self.thdn_db_label.setVisible(False)
            self.thd_label.setVisible(False)
            self.sinad_label.setVisible(False)
            self.imd_row_widget.setVisible(True)



    def start_sweep(self, mode_idx):
        self.module.start_analysis() # Ensure audio is running
        self.action_btn.setText(tr("Stop Sweep"))
        self.module.sweep_results = []
        self.sweep_curve.setData([], [])

        sweep_type = 'frequency' if mode_idx == 1 else 'amplitude'
        start = self.sweep_start_spin.value()
        end = self.sweep_end_spin.value()
        steps = self.sweep_steps_spin.value()

        if sweep_type == 'frequency':
            if start <= 0 or end <= 0:
                print("Error: Frequency sweep range must be positive.")
                self.action_btn.setChecked(False)
                self.action_btn.setText(tr("Start Measurement"))
                return

        self.sweep_worker = SweepWorker(self.module, sweep_type, start, end, steps)
        self.sweep_worker.result_ready.connect(self.on_sweep_result)
        self.sweep_worker.finished.connect(self.on_sweep_finished)
        self.sweep_worker.start()

    def stop_sweep(self):
        if self.sweep_worker:
            self.sweep_worker.stop()
            self.sweep_worker.wait()
        self.module.stop_analysis()
        self.action_btn.setText(tr("Start Measurement"))
        self.action_btn.setChecked(False)

    def on_toggle_view(self, checked):
        if checked:
            self.meters_view_stack.setCurrentIndex(1)
            self.view_toggle_btn.setText(tr("Show Basic"))
        else:
            self.meters_view_stack.setCurrentIndex(0)
            self.view_toggle_btn.setText(tr("Show Detailed"))

    def on_sweep_result(self, result):
        self.module.sweep_results.append(result)

        # Update Plot
        x_data = [r['sweep_param'] for r in self.module.sweep_results]
        y_data = [r['thdn_db'] for r in self.module.sweep_results]

        x_plot = np.array(x_data)

        self.sweep_curve.setData(x_plot, y_data)

    def on_sweep_finished(self):
        self.stop_sweep()

    def on_freq_changed(self, val):
        self.module.gen_frequency = val
        self.module.reset_averaging_state()

    def on_channel_changed(self, idx):
        self.module.output_channel = idx
        self.module.reset_averaging_state()

    def on_in_channel_changed(self, idx):
        self.module.input_channel = idx
        self.module.reset_averaging_state()

    def on_avg_changed(self, val):
        self.avg_label.setText(tr("Avg: {0}%").format(val))
        self.module.averaging = val / 100.0
        self.module.reset_averaging_state()

    def update_realtime_analysis(self):
        if not self.module.is_running:
            return

        data = self.module.input_data
        sample_rate = self.module.audio_engine.sample_rate

        # Perform Analysis
        # Check signal type instead of mode
        if self.module.signal_type in ['smpte', 'ccif']:
            window = get_window(self.module.window_type, len(data))
            fft_data = np.fft.rfft(data * window)
            mag_linear = np.abs(fft_data) * (2 / np.sum(window))
            freqs = np.fft.rfftfreq(len(data), 1/sample_rate)

            if self.module.signal_type == 'smpte':
                res = AudioCalc.calculate_imd_smpte(mag_linear, freqs, self.module.imd_f1, self.module.imd_f2)
            else:
                res = AudioCalc.calculate_imd_ccif(mag_linear, freqs, self.module.imd_f1, self.module.imd_f2)

            res = self.module._apply_imd_averaging(res)

            self.imd_label.setText(tr("{0:.4f} %").format(res['imd']))
            self.imd_db_label.setText(tr("{0:.2f} dB").format(res['imd_db']))

            # Update Detailed Label for IMD
            window_name = self.module.window_type.capitalize()
            fft_size = self.module.buffer_size
            input_level = 20 * np.log10(np.sqrt(np.mean(data**2)) + 1e-12)

            detailed_text = (
                f"{tr('Input level:'):<15} {input_level:>10.1f} dBFS   ✔\n"
                f"{tr('Window:'):<15} {window_name:>10}\n"
                f"{tr('FFT size:'):<15} {fft_size:>10}\n"
                f"{tr('Bandwidth:'):<15} {'20 kHz':>10}\n"
                "--------------------------------\n"
                f"{tr('IMD:'):<15} {res['imd']:>10.4f} %\n"
                f"{tr('IMD (dB):'):<15} {res['imd_db']:>10.1f} dB\n"
                "--------------------------------"
            )
            self.detailed_label.setText(detailed_text)

            mag_linear = self.module.apply_spectrum_averaging(mag_linear)
            mag_db = 20 * np.log10(mag_linear + 1e-12)
            self.spectrum_curve.setData(freqs[1:], mag_db[1:])

        else:
            results = AudioCalc.analyze_harmonics(
                data,
                self.module.gen_frequency,
                self.module.window_type,
                sample_rate
            )
            results = self.module._apply_result_averaging(results)
            self.module.current_result = results

            # Update Meters
            self.thdn_label.setText(tr("{0:.4f} %").format(results['thdn_percent']))
            self.thdn_db_label.setText(tr("{0:.2f} dB").format(results['thdn_db']))
            self.thd_label.setText(tr("{0:.4f} %").format(results['thd_percent']))
            self.sinad_label.setText(tr("{0:.2f} dB").format(results['sinad_db']))

            # ENOB Calculation
            # ENOB is only valid near full scale (strict check).
            # We'll use a threshold of -1.0 dBFS.
            sinad = results['sinad_db']
            enob_val = "--"
            input_level = results['basic_wave']['amplitude_dbfs']

            if input_level >= -1.0:
                 enob_calc = (sinad - 1.76) / 6.02
                 enob_str = f"{enob_calc:>10.1f} bits   ✔"
            else:
                 enob_str = f"{'--':>10} bits"

            # Update Detailed Label
            window_name = self.module.window_type.capitalize()
            fft_size = self.module.buffer_size
            bandwidth = "20 kHz" # Fixed bandwidth in current analysis

            detailed_text = (
                f"{tr('Input level:'):<15} {input_level:>10.1f} dBFS   ✔\n"
                f"{tr('Window:'):<15} {window_name:>10}\n"
                f"{tr('FFT size:'):<15} {fft_size:>10}\n"
                f"{tr('Bandwidth:'):<15} {bandwidth:>10}\n"
                "--------------------------------\n"
                f"{tr('THD+N:'):<15} {results['thdn_db']:>10.1f} dB\n"
                f"{tr('SINAD:'):<15} {results['sinad_db']:>10.1f} dB\n"
                f"{tr('ENOB:'):<15} {enob_str}\n"
                "--------------------------------"
            )
            self.detailed_label.setText(detailed_text)

            # Update Harmonics Table & Bar Graph
            self.harmonics_table.setRowCount(len(results['harmonics']))

            orders = []
            levels = []

            for i, h in enumerate(results['harmonics']):
                self.harmonics_table.setItem(i, 0, QTableWidgetItem(str(h['order'])))
                self.harmonics_table.setItem(i, 1, QTableWidgetItem(f"{h['frequency']:.1f}"))
                self.harmonics_table.setItem(i, 2, QTableWidgetItem(f"{h['amplitude_dbr']:.2f}"))
                self.harmonics_table.setItem(i, 3, QTableWidgetItem(f"{h['amplitude_linear']:.6f}"))

                orders.append(h['order'])
                levels.append(h['amplitude_dbr'])

            # Update Bar Graph
            if orders:
                floor_db = -140
                heights = [l - floor_db for l in levels]
                self.harmonics_bar_item.setOpts(x=orders, height=heights, y0=floor_db)
                # Ensure x-axis shows integer ticks for orders
                # We can just set the range to cover all orders
                self.harmonics_plot.setXRange(min(orders)-1, max(orders)+1)

            # Update Spectrum Plot
            window = get_window(self.module.window_type, len(data))
            fft_data = np.fft.rfft(data * window)
            mag_linear = np.abs(fft_data) / len(data) * 2
            mag_linear = self.module.apply_spectrum_averaging(mag_linear)
            mag = 20 * np.log10(mag_linear + 1e-12)
            freqs = np.fft.rfftfreq(len(data), 1/sample_rate)

            self.spectrum_curve.setData(freqs[1:], mag[1:])
