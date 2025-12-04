import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, 
                             QTabWidget, QStackedWidget, QTableWidget, QTableWidgetItem, 
                             QHeaderView)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.analysis import AudioCalc
from src.core.localization import tr

class AdvancedDistortionMeter(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 32768 # High resolution
        self.input_data = np.zeros(self.buffer_size)
        
        # Generator Settings
        self.output_enabled = True
        self.gen_amplitude = 0.5
        self.output_channel = 0
        self.input_channel = 0
        
        # MIM (Multitone) Settings
        self.mim_tone_count = 31
        self.mim_min_freq = 20.0
        self.mim_max_freq = 20000.0
        self._mim_freqs = None
        self._mim_phases = None
        self._mim_phase_state = None
        
        # PIM Settings
        self.pim_f1 = 1800.0
        self.pim_f2 = 2100.0
        self.pim_amp_ratio = 1.0 # Equal amplitude
        self._pim_phase_f1 = 0.0
        self._pim_phase_f2 = 0.0
        
        # Mode
        self.mode = 'MIM' # 'MIM', 'SPDR', 'PIM'
        
        self.callback_id = None
        self.current_result = None

    @property
    def name(self) -> str:
        return "Advanced Distortion Meter"

    @property
    def description(self) -> str:
        return "Advanced distortion measurements including MIM, SPDR, and PIM."

    def run(self, args):
        print("Advanced Distortion Meter running from CLI (not implemented)")

    def get_widget(self):
        return AdvancedDistortionMeterWidget(self)

    def start_analysis(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.input_data = np.zeros(self.buffer_size)
        self.current_result = None
        
        sample_rate = self.audio_engine.sample_rate
        
        # Reset generator state
        self._mim_freqs = None # Trigger regen
        self._pim_phase_f1 = 0.0
        self._pim_phase_f2 = 0.0
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
                
            # Generate Signal
            outdata.fill(0)
            if self.output_enabled:
                if self.mode == 'MIM':
                    sig = self._generate_mim(frames, sample_rate)
                elif self.mode == 'PIM':
                    sig = self._generate_pim(frames, sample_rate)
                elif self.mode == 'SPDR':
                    # SPDR typically uses a single pure tone
                    sig = self._generate_sine(frames, sample_rate)
                else:
                    sig = np.zeros(frames)
                
                if self.output_channel == 0:
                    outdata[:, 0] = sig
                elif self.output_channel == 1:
                    if outdata.shape[1] > 1:
                        outdata[:, 1] = sig
            
            # Capture Input
            capture_ch = self.input_channel
            if indata.shape[1] > capture_ch:
                new_data = indata[:, capture_ch]
            else:
                new_data = indata[:, 0]
                
            # Ring buffer
            if len(new_data) > self.buffer_size:
                self.input_data[:] = new_data[-self.buffer_size:]
            else:
                self.input_data = np.roll(self.input_data, -len(new_data))
                self.input_data[-len(new_data):] = new_data

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def _generate_mim(self, frames, sample_rate):
        # Initialize frequencies if needed
        if self._mim_freqs is None or len(self._mim_freqs) != self.mim_tone_count:
            self._mim_freqs = np.logspace(np.log10(self.mim_min_freq), np.log10(self.mim_max_freq), self.mim_tone_count)
            self._mim_phase_state = np.random.uniform(0, 2*np.pi, self.mim_tone_count)
            
        # Amplitude scaling: Peak should not exceed gen_amplitude
        # Crest factor approx 12dB (4x). 
        # RMS per tone = TotalRMS / sqrt(N)
        # Let's be safe and scale by N (very conservative) or sqrt(N) with headroom.
        # Using sqrt(N) * 4 for peak safety?
        # Let's use 1/sqrt(N) scaling for RMS, but keep peak check.
        # Simple: Amp per tone = gen_amplitude / sqrt(N)
        
        amp_per_tone = self.gen_amplitude / np.sqrt(self.mim_tone_count)
        
        signal = np.zeros(frames)
        t_idx = np.arange(frames)
        
        for i, f in enumerate(self._mim_freqs):
            inc = 2 * np.pi * f / sample_rate
            phases = self._mim_phase_state[i] + t_idx * inc
            signal += amp_per_tone * np.sin(phases)
            self._mim_phase_state[i] = (self._mim_phase_state[i] + frames * inc) % (2 * np.pi)
            
        return signal

    def _generate_pim(self, frames, sample_rate):
        # Two tones f1, f2
        amp = self.gen_amplitude / 2 # Split power
        
        inc1 = 2 * np.pi * self.pim_f1 / sample_rate
        inc2 = 2 * np.pi * self.pim_f2 / sample_rate
        
        t = np.arange(frames)
        p1 = self._pim_phase_f1 + t * inc1
        p2 = self._pim_phase_f2 + t * inc2
        
        signal = amp * np.sin(p1) + amp * np.sin(p2)
        
        self._pim_phase_f1 = (self._pim_phase_f1 + frames * inc1) % (2 * np.pi)
        self._pim_phase_f2 = (self._pim_phase_f2 + frames * inc2) % (2 * np.pi)
        
        return signal

    def _generate_sine(self, frames, sample_rate):
        # For SPDR, use 1kHz default or configurable?
        # Let's use 1kHz for now, or add a setting later.
        f = 1000.0
        if not hasattr(self, '_spdr_phase'):
            self._spdr_phase = 0.0
            
        inc = 2 * np.pi * f / sample_rate
        t = np.arange(frames)
        p = self._spdr_phase + t * inc
        signal = self.gen_amplitude * np.sin(p)
        self._spdr_phase = (self._spdr_phase + frames * inc) % (2 * np.pi)
        return signal

class AdvancedDistortionMeterWidget(QWidget):
    def __init__(self, module: AdvancedDistortionMeter):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_analysis)
        self.timer.setInterval(100) # 10Hz

    def init_ui(self):
        layout = QHBoxLayout()
        
        # --- Controls ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        # Mode
        mode_group = QGroupBox(tr("Measurement Mode"))
        mode_layout = QVBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([tr("MIM (Multitone)"), tr("SPDR"), tr("PIM")])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        left_panel.addWidget(mode_group)
        
        # Settings Stack
        self.settings_stack = QStackedWidget()
        
        # 1. MIM Settings
        mim_widget = QWidget()
        mim_layout = QFormLayout()
        
        self.mim_count_spin = QSpinBox()
        self.mim_count_spin.setRange(3, 100)
        self.mim_count_spin.setValue(self.module.mim_tone_count)
        self.mim_count_spin.valueChanged.connect(lambda v: setattr(self.module, 'mim_tone_count', v))
        mim_layout.addRow(tr("Tone Count:"), self.mim_count_spin)
        
        self.mim_min_spin = QDoubleSpinBox()
        self.mim_min_spin.setRange(10, 20000)
        self.mim_min_spin.setValue(self.module.mim_min_freq)
        self.mim_min_spin.valueChanged.connect(lambda v: setattr(self.module, 'mim_min_freq', v))
        mim_layout.addRow(tr("Min Freq:"), self.mim_min_spin)
        
        self.mim_max_spin = QDoubleSpinBox()
        self.mim_max_spin.setRange(10, 24000)
        self.mim_max_spin.setValue(self.module.mim_max_freq)
        self.mim_max_spin.valueChanged.connect(lambda v: setattr(self.module, 'mim_max_freq', v))
        mim_layout.addRow(tr("Max Freq:"), self.mim_max_spin)
        
        mim_widget.setLayout(mim_layout)
        self.settings_stack.addWidget(mim_widget)
        
        # 2. SPDR Settings
        spdr_widget = QWidget()
        spdr_layout = QFormLayout()
        spdr_layout.addRow(QLabel(tr("Standard 1kHz Tone")))
        spdr_widget.setLayout(spdr_layout)
        self.settings_stack.addWidget(spdr_widget)
        
        # 3. PIM Settings
        pim_widget = QWidget()
        pim_layout = QFormLayout()
        
        self.pim_f1_spin = QDoubleSpinBox()
        self.pim_f1_spin.setRange(10, 20000)
        self.pim_f1_spin.setValue(self.module.pim_f1)
        self.pim_f1_spin.valueChanged.connect(lambda v: setattr(self.module, 'pim_f1', v))
        pim_layout.addRow(tr("Freq 1 (Hz):"), self.pim_f1_spin)
        
        self.pim_f2_spin = QDoubleSpinBox()
        self.pim_f2_spin.setRange(10, 20000)
        self.pim_f2_spin.setValue(self.module.pim_f2)
        self.pim_f2_spin.valueChanged.connect(lambda v: setattr(self.module, 'pim_f2', v))
        pim_layout.addRow(tr("Freq 2 (Hz):"), self.pim_f2_spin)
        
        pim_widget.setLayout(pim_layout)
        self.settings_stack.addWidget(pim_widget)
        
        left_panel.addWidget(self.settings_stack)
        
        # Amplitude
        amp_group = QGroupBox(tr("Generator"))
        amp_layout = QFormLayout()
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0, 1.0)
        self.amp_spin.setSingleStep(0.01)
        self.amp_spin.setValue(self.module.gen_amplitude)
        self.amp_spin.valueChanged.connect(lambda v: setattr(self.module, 'gen_amplitude', v))
        amp_layout.addRow(tr("Amplitude (Lin):"), self.amp_spin)
        
        self.out_ch_combo = QComboBox()
        self.out_ch_combo.addItems([tr("Left"), tr("Right")])
        self.out_ch_combo.currentIndexChanged.connect(lambda i: setattr(self.module, 'output_channel', i))
        amp_layout.addRow(tr("Output Ch:"), self.out_ch_combo)
        
        amp_group.setLayout(amp_layout)
        left_panel.addWidget(amp_group)
        
        # Control Buttons
        self.start_btn = QPushButton(tr("Start Measurement"))
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.start_btn.setStyleSheet("QPushButton:checked { background-color: #ccffcc; }")
        left_panel.addWidget(self.start_btn)
        
        # Results Display
        self.results_group = QGroupBox(tr("Results"))
        results_layout = QVBoxLayout()
        
        self.main_metric_label = QLabel("--")
        self.main_metric_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff00;")
        self.main_metric_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.main_metric_label)
        
        self.sub_metric_label = QLabel("--")
        self.sub_metric_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.sub_metric_label)
        
        self.results_group.setLayout(results_layout)
        left_panel.addWidget(self.results_group)
        
        left_panel.addStretch()
        layout.addLayout(left_panel, 1)
        
        # --- Right Panel: Plots ---
        right_panel = QVBoxLayout()
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', tr('Amplitude'), units='dB')
        self.plot_widget.setLabel('bottom', tr('Frequency'), units='Hz')
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.setYRange(-140, 0)
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_curve = self.plot_widget.plot(pen='y')
        
        right_panel.addWidget(self.plot_widget)
        layout.addLayout(right_panel, 3)
        
        self.setLayout(layout)

    def on_mode_changed(self, index):
        if index == 0: # MIM
            self.module.mode = 'MIM'
            self.settings_stack.setCurrentIndex(0)
        elif index == 1: # SPDR
            self.module.mode = 'SPDR'
            self.settings_stack.setCurrentIndex(1)
        elif index == 2: # PIM
            self.module.mode = 'PIM'
            self.settings_stack.setCurrentIndex(2)
            
        # Reset results
        self.main_metric_label.setText("--")
        self.sub_metric_label.setText("--")

    def on_start_clicked(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.start_btn.setText(tr("Stop Measurement"))
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.start_btn.setText(tr("Start Measurement"))

    def update_analysis(self):
        if not self.module.is_running:
            return
            
        data = self.module.input_data
        sr = self.module.audio_engine.sample_rate
        
        # Perform FFT
        window = np.blackman(len(data))
        fft_res = np.fft.rfft(data * window)
        freqs = np.fft.rfftfreq(len(data), 1/sr)
        
        # Magnitude in dB
        mag = np.abs(fft_res) * 2 / np.sum(window)
        mag_db = 20 * np.log10(mag + 1e-12)
        
        # Update Plot
        self.plot_curve.setData(freqs, mag_db)
        
        # Calculate Metrics
        if self.module.mode == 'MIM':
            # Need expected tone freqs
            if self.module._mim_freqs is not None:
                res = AudioCalc.calculate_multitone_tdn(mag, freqs, self.module._mim_freqs)
                self.main_metric_label.setText(f"TD+N: {res['tdn_db']:.1f} dB")
                self.sub_metric_label.setText(f"{res['tdn']:.4f} %")
                
        elif self.module.mode == 'SPDR':
            # Assume 1kHz fundamental for now
            res = AudioCalc.calculate_spdr(mag, freqs, 1000.0)
            self.main_metric_label.setText(f"SPDR: {res['spdr_db']:.1f} dB")
            self.sub_metric_label.setText(f"Max Spur: {res['max_spur_freq']:.0f} Hz ({20*np.log10(res['max_spur_amp']+1e-12):.1f} dB)")
            
        elif self.module.mode == 'PIM':
            res = AudioCalc.calculate_pim(mag, freqs, self.module.pim_f1, self.module.pim_f2)
            self.main_metric_label.setText(f"PIM: {res['pim_db']:.1f} dBc")
            products_str = ", ".join([f"{p['order']}th" for p in res['products']])
            self.sub_metric_label.setText(f"Orders: {products_str}")
