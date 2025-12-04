import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QPushButton, 
                             QComboBox, QHBoxLayout, QGroupBox, QFormLayout, QCheckBox, QSlider,
                             QRadioButton, QButtonGroup, QStackedWidget, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

@dataclass
class SignalParameters:
    waveform: str = 'sine'
    frequency: float = 1000.0
    amplitude: float = 0.5
    noise_color: str = 'white'
    
    # Sweep parameters
    sweep_enabled: bool = False
    start_freq: float = 20.0
    end_freq: float = 20000.0
    sweep_duration: float = 5.0
    log_sweep: bool = True
    
    # Advanced Signal Parameters
    multitone_count: int = 10
    mls_order: int = 15
    burst_on_cycles: int = 10
    burst_off_cycles: int = 90
    
    # New Parameters
    pulse_width: float = 50.0 # %
    noise_amplitude: float = 0.1
    phase_offset: float = 0.0 # Degrees
    
    # Internal state (not shared/copied usually, but kept here for simplicity per channel)
    _phase: float = 0.0
    _sweep_time: float = 0.0
    _buffer: Optional[np.ndarray] = None
    _buffer_index: int = 0

class SignalGenerator(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        
        self.params_L = SignalParameters()
        self.params_R = SignalParameters()
        
        # Output Routing: 'L', 'R', 'STEREO'
        self.output_mode = 'STEREO' 
        
        self.is_playing = False
        self.callback_id = None

    @property
    def name(self) -> str:
        return "Signal Generator"

    @property
    def description(self) -> str:
        return "Generates advanced test signals (Sine, Square, Noise, Sweeps) with independent channel control"

    def run(self, args: argparse.Namespace):
        print("Signal Generator running from CLI (not fully implemented)")

    def get_widget(self):
        return SignalGeneratorWidget(self)

    def _generate_noise_buffer(self, params: SignalParameters, sample_rate, duration=5.0):
        """Pre-generates a buffer of colored noise."""
        num_samples = int(sample_rate * duration)
        
        # White noise base
        white = np.random.randn(num_samples)
        
        if params.noise_color == 'white':
            # Normalize
            max_val = np.max(np.abs(white))
            if max_val > 0:
                white /= max_val
            return white
            
        # FFT filtering
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(num_samples, d=1/sample_rate)
        
        scaling = np.ones_like(freqs)
        
        if params.noise_color == 'pink':
            # 1/f^0.5 (-3dB/oct)
            scaling[1:] = 1 / np.sqrt(freqs[1:])
            scaling[0] = 0
        elif params.noise_color == 'brown':
            # 1/f (-6dB/oct)
            scaling[1:] = 1 / freqs[1:]
            scaling[0] = 0
        elif params.noise_color == 'blue':
            # f^0.5 (+3dB/oct)
            scaling = np.sqrt(freqs)
        elif params.noise_color == 'violet':
            # f (+6dB/oct)
            scaling = freqs
        elif params.noise_color == 'grey':
            # Simplified inverted A-weighting
            f = freqs
            f2 = f**2
            c1 = 12194.217**2
            c2 = 20.6**2
            c3 = 107.7**2
            c4 = 737.9**2
            
            f_safe = f.copy()
            f_safe[0] = 1.0 
            f2_safe = f_safe**2
            
            num = c1 * (f2_safe**2)
            denom = (f2_safe + c2) * np.sqrt((f2_safe + c3) * (f2_safe + c4)) * (f2_safe + c1)
            a_weight = num / denom
            
            scaling = 1.0 / (a_weight + 1e-12)
            
            idx_1k = np.argmin(np.abs(freqs - 1000))
            if idx_1k < len(scaling):
                ref_gain = scaling[idx_1k]
                scaling /= ref_gain
            
            scaling = np.minimum(scaling, 100.0)
            scaling[0] = 0
            
        fft = fft * scaling
        noise = np.fft.irfft(fft)
        
        # Normalize
        max_val = np.max(np.abs(noise))
        if max_val > 0:
            noise /= max_val
            
        return noise

    def _generate_multitone(self, params: SignalParameters, sample_rate):
        """Generates a Crest-Factor optimized Multitone signal."""
        if params.start_freq >= params.end_freq:
            freqs = np.array([params.start_freq])
        else:
            freqs = np.logspace(np.log10(params.start_freq), np.log10(params.end_freq), params.multitone_count)
        
        N = int(sample_rate) # 1 second buffer
        bin_width = sample_rate / N
        freqs = np.round(freqs / bin_width) * bin_width
        
        phases = np.pi * (np.arange(len(freqs))**2) / len(freqs)
        
        t = np.arange(N) / sample_rate
        signal = np.zeros(N)
        
        for f, p in zip(freqs, phases):
            signal += np.sin(2 * np.pi * f * t + p)
            
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal /= max_val
            
        return signal

    def _generate_mls(self, params: SignalParameters, sample_rate):
        """Generates a Maximum Length Sequence (MLS)."""
        taps = {
            10: [10, 7], 11: [11, 9], 12: [12, 11, 10, 4], 13: [13, 12, 10, 9],
            14: [14, 13, 12, 2], 15: [15, 14], 16: [16, 15, 13, 4], 17: [17, 14], 18: [18, 11]
        }
        
        order = params.mls_order
        if order not in taps:
            order = 15
            
        try:
            import scipy.signal
            seq, state = scipy.signal.max_len_seq(order)
            signal = seq.astype(float) * 2 - 1
            return signal
        except:
            print("scipy.signal.max_len_seq not found/failed, using slow fallback")
            tap_indices = [x - 1 for x in taps[order]]
            N = 2**order - 1
            state = np.ones(order, dtype=int)
            signal = np.zeros(N)
            for i in range(N):
                feedback = 0
                for tap in tap_indices:
                    feedback ^= state[tap]
                output = state[-1]
                signal[i] = float(output * 2 - 1)
                state = np.roll(state, 1)
                state[0] = feedback
            return signal

    def _generate_burst(self, params: SignalParameters, sample_rate):
        """Generates a Tone Burst."""
        total_cycles = params.burst_on_cycles + params.burst_off_cycles
        cycle_duration = 1.0 / params.frequency
        total_duration = total_cycles * cycle_duration
        
        num_samples = int(total_duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        
        sine = np.sin(2 * np.pi * params.frequency * t)
        
        on_duration = params.burst_on_cycles * cycle_duration
        on_samples = int(on_duration * sample_rate)
        
        envelope = np.zeros(num_samples)
        envelope[:on_samples] = 1.0
        
        return sine * envelope

    def _prepare_buffer(self, params: SignalParameters, sample_rate):
        if params.waveform == 'noise':
            params._buffer = self._generate_noise_buffer(params, sample_rate)
        elif params.waveform == 'multitone':
            params._buffer = self._generate_multitone(params, sample_rate)
        elif params.waveform == 'mls':
            params._buffer = self._generate_mls(params, sample_rate)
        elif params.waveform == 'burst':
            params._buffer = self._generate_burst(params, sample_rate)
        else:
            params._buffer = None
        params._buffer_index = 0

    def start_generation(self):
        if self.is_playing:
            return

        self.is_playing = True
        sample_rate = self.audio_engine.sample_rate
        
        # Reset states
        for params in [self.params_L, self.params_R]:
            params._phase = 0
            params._sweep_time = 0
            self._prepare_buffer(params, sample_rate)
        
        def generate_channel_signal(params: SignalParameters, frames, t_global):
            signal = np.zeros(frames)
            
            if params._buffer is not None:
                # Buffer based generation
                chunk_size = frames
                buf_len = len(params._buffer)
                current_idx = 0
                
                while current_idx < chunk_size:
                    remaining = chunk_size - current_idx
                    available = buf_len - params._buffer_index
                    
                    to_copy = min(remaining, available)
                    signal[current_idx:current_idx+to_copy] = params._buffer[params._buffer_index:params._buffer_index+to_copy]
                    
                    params._buffer_index += to_copy
                    current_idx += to_copy
                    
                    if params._buffer_index >= buf_len:
                        params._buffer_index = 0
                
                return signal * params.amplitude

            if params.sweep_enabled:
                # Sweep generation
                current_times = params._sweep_time + t_global
                current_times = np.mod(current_times, params.sweep_duration)
                
                if params.log_sweep:
                    k = np.log(params.end_freq / params.start_freq) / params.sweep_duration
                    if k == 0: phase = 2 * np.pi * params.start_freq * current_times
                    else: phase = 2 * np.pi * params.start_freq * (np.exp(k * current_times) - 1) / k
                else:
                    k = (params.end_freq - params.start_freq) / params.sweep_duration
                    phase = 2 * np.pi * (params.start_freq * current_times + 0.5 * k * current_times**2)
                
                signal = params.amplitude * np.sin(phase)
                params._sweep_time += frames / sample_rate
                return signal
            
            # Standard waveforms
            phase_t = (np.arange(frames) + params._phase) / sample_rate
            params._phase += frames
            
            # Apply Phase Offset
            # We add phase offset to the argument of sin/cos functions.
            # phase_t * freq gives cycles. 2*pi converts to radians.
            # We add offset in radians.
            offset_rad = np.radians(params.phase_offset)
            
            if params.waveform == 'sine':
                signal = params.amplitude * np.sin(2 * np.pi * params.frequency * phase_t + offset_rad)
            elif params.waveform == 'square':
                signal = params.amplitude * np.sign(np.sin(2 * np.pi * params.frequency * phase_t + offset_rad))
            elif params.waveform == 'triangle':
                # Triangle wave depends on phase.
                # (2 * abs(2 * ((t * f + off) % 1) - 1) - 1)
                # We need to add offset to the cycle count part.
                # offset in cycles = offset_deg / 360
                off_cycles = params.phase_offset / 360.0
                signal = params.amplitude * (2 * np.abs(2 * ((phase_t * params.frequency + off_cycles) % 1) - 1) - 1)
            elif params.waveform == 'sawtooth':
                off_cycles = params.phase_offset / 360.0
                signal = params.amplitude * (2 * ((phase_t * params.frequency + off_cycles) % 1) - 1)
            elif params.waveform == 'pulse':
                duty = params.pulse_width / 100.0
                off_cycles = params.phase_offset / 360.0
                ramp = (phase_t * params.frequency + off_cycles) % 1
                signal = params.amplitude * np.where(ramp < duty, 1.0, -1.0)
            elif params.waveform == 'tone_noise':
                signal = params.amplitude * np.sin(2 * np.pi * params.frequency * phase_t + offset_rad)
                noise = params.noise_amplitude * np.random.uniform(-1, 1, size=frames)
                signal += noise
            
            return signal

        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
                
            t = np.arange(frames) / sample_rate
            outdata.fill(0)
            
            # Left Channel
            if self.output_mode in ['L', 'STEREO']:
                sig_l = generate_channel_signal(self.params_L, frames, t)
                if outdata.shape[1] >= 1:
                    outdata[:, 0] = sig_l
            
            # Right Channel
            if self.output_mode in ['R', 'STEREO']:
                # If we are in STEREO but want to output the SAME signal if linked?
                # The user requirement says "L and R separate signals".
                # So we always use params_R for Right channel.
                # If the user wants them same, they copy settings in UI.
                sig_r = generate_channel_signal(self.params_R, frames, t)
                if outdata.shape[1] >= 2:
                    outdata[:, 1] = sig_r

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_generation(self):
        if self.is_playing:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_playing = False

class SignalGeneratorWidget(QWidget):
    def __init__(self, module: SignalGenerator):
        super().__init__()
        self.module = module
        self.current_target = 'L' # 'L', 'R', 'LINK'
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- Top Control Bar ---
        top_bar = QHBoxLayout()
        
        # Start/Stop
        self.toggle_btn = QPushButton(tr("Start Output"))
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setMinimumHeight(40)
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.toggle_btn.setStyleSheet("QPushButton:checked { background-color: #ffcccc; font-weight: bold; }")
        top_bar.addWidget(self.toggle_btn, 2)
        
        # Output Routing
        routing_group = QGroupBox(tr("Output Routing"))
        routing_layout = QHBoxLayout()
        self.route_l = QRadioButton(tr("Left Only"))
        self.route_r = QRadioButton(tr("Right Only"))
        self.route_stereo = QRadioButton(tr("Stereo (L+R)"))
        self.route_stereo.setChecked(True)
        
        self.route_group = QButtonGroup()
        self.route_group.addButton(self.route_l)
        self.route_group.addButton(self.route_r)
        self.route_group.addButton(self.route_stereo)
        
        self.route_group.buttonClicked.connect(self.on_route_changed)
        
        routing_layout.addWidget(self.route_l)
        routing_layout.addWidget(self.route_r)
        routing_layout.addWidget(self.route_stereo)
        routing_group.setLayout(routing_layout)
        top_bar.addWidget(routing_group, 3)
        
        layout.addLayout(top_bar)
        
        # --- Signal Parameters Control ---
        # Target Selector
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel(f"<b>{tr('Edit Settings For:')}</b>"))
        
        self.target_l = QRadioButton(tr("Left Channel"))
        self.target_r = QRadioButton(tr("Right Channel"))
        self.target_link = QRadioButton(tr("Linked (Both)"))
        self.target_l.setChecked(True)
        
        self.target_group = QButtonGroup()
        self.target_group.addButton(self.target_l)
        self.target_group.addButton(self.target_r)
        self.target_group.addButton(self.target_link)
        self.target_group.buttonClicked.connect(self.on_target_changed)
        
        target_layout.addWidget(self.target_l)
        target_layout.addWidget(self.target_r)
        target_layout.addWidget(self.target_link)
        target_layout.addStretch()
        
        layout.addLayout(target_layout)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # --- Main Controls ---
        # We use the same widgets but update their values based on current_target
        
        basic_group = QGroupBox(tr("Signal Parameters"))
        basic_layout = QFormLayout()
        
        # Waveform
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(['sine', 'square', 'triangle', 'sawtooth', 'pulse', 'tone_noise', 'noise', 'multitone', 'mls', 'burst'])
        self.wave_combo.currentTextChanged.connect(self.on_wave_changed)
        basic_layout.addRow(tr("Waveform:"), self.wave_combo)
        
        # Dynamic Parameters Stack
        self.param_stack = QWidget()
        self.param_layout = QVBoxLayout(self.param_stack)
        self.param_layout.setContentsMargins(0,0,0,0)
        
        # 1. Noise Params
        self.noise_widget = QWidget()
        noise_form = QFormLayout(self.noise_widget)
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(['white', 'pink', 'brown', 'blue', 'violet', 'grey'])
        self.noise_combo.currentTextChanged.connect(lambda v: self.update_param('noise_color', v))
        noise_form.addRow(tr("Color:"), self.noise_combo)
        
        # 2. Multitone Params
        self.multitone_widget = QWidget()
        mt_form = QFormLayout(self.multitone_widget)
        self.mt_count_spin = QDoubleSpinBox()
        self.mt_count_spin.setDecimals(0); self.mt_count_spin.setRange(2, 1000); self.mt_count_spin.setValue(10)
        self.mt_count_spin.valueChanged.connect(lambda v: self.update_param('multitone_count', int(v)))
        mt_form.addRow(tr("Tone Count:"), self.mt_count_spin)
        
        # 3. MLS Params
        self.mls_widget = QWidget()
        mls_form = QFormLayout(self.mls_widget)
        self.mls_order_combo = QComboBox()
        self.mls_order_combo.addItems([str(i) for i in range(10, 19)])
        self.mls_order_combo.setCurrentText("15")
        self.mls_order_combo.currentTextChanged.connect(lambda v: self.update_param('mls_order', int(v)))
        mls_form.addRow(tr("Order (N):"), self.mls_order_combo)
        
        # 4. Burst Params
        self.burst_widget = QWidget()
        burst_form = QFormLayout(self.burst_widget)
        self.burst_on_spin = QDoubleSpinBox()
        self.burst_on_spin.setDecimals(0); self.burst_on_spin.setRange(1, 1000); self.burst_on_spin.setValue(10)
        self.burst_on_spin.valueChanged.connect(lambda v: self.update_param('burst_on_cycles', int(v)))
        burst_form.addRow(tr("On Cycles:"), self.burst_on_spin)
        self.burst_off_spin = QDoubleSpinBox()
        self.burst_off_spin.setDecimals(0); self.burst_off_spin.setRange(1, 10000); self.burst_off_spin.setValue(90)
        self.burst_off_spin.valueChanged.connect(lambda v: self.update_param('burst_off_cycles', int(v)))
        burst_form.addRow(tr("Off Cycles:"), self.burst_off_spin)

        # 5. Pulse Params
        self.pulse_widget = QWidget()
        pulse_form = QFormLayout(self.pulse_widget)
        self.pulse_width_spin = QDoubleSpinBox()
        self.pulse_width_spin.setRange(0.1, 99.9)
        self.pulse_width_spin.setValue(50.0)
        self.pulse_width_spin.setSuffix("%")
        self.pulse_width_spin.valueChanged.connect(lambda v: self.update_param('pulse_width', v))
        pulse_form.addRow(tr("Pulse Width:"), self.pulse_width_spin)
        
        # 6. Tone+Noise Params
        self.tn_widget = QWidget()
        tn_form = QFormLayout(self.tn_widget)
        self.noise_amp_spin = QDoubleSpinBox()
        self.noise_amp_spin.setRange(0.0, 1.0)
        self.noise_amp_spin.setSingleStep(0.01)
        self.noise_amp_spin.setValue(0.1)
        self.noise_amp_spin.valueChanged.connect(lambda v: self.update_param('noise_amplitude', v))
        tn_form.addRow(tr("Noise Amplitude:"), self.noise_amp_spin)
        
        self.param_layout.addWidget(self.noise_widget)
        self.param_layout.addWidget(self.multitone_widget)
        self.param_layout.addWidget(self.mls_widget)
        self.param_layout.addWidget(self.burst_widget)
        self.param_layout.addWidget(self.pulse_widget)
        self.param_layout.addWidget(self.tn_widget)
        self.noise_widget.hide()
        self.multitone_widget.hide()
        self.mls_widget.hide()
        self.burst_widget.hide()
        self.pulse_widget.hide()
        self.tn_widget.hide()
        
        basic_layout.addRow(self.param_stack)
        
        # Frequency
        freq_layout = QHBoxLayout()
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000)
        self.freq_spin.setValue(1000)
        self.freq_spin.valueChanged.connect(self.on_freq_spin_changed)
        
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(0, 1000) 
        self.freq_slider.valueChanged.connect(self.on_freq_slider_changed)
        
        freq_layout.addWidget(self.freq_spin)
        freq_layout.addWidget(self.freq_slider)
        basic_layout.addRow(tr("Frequency (Hz):"), freq_layout)
        
        # Phase
        phase_layout = QHBoxLayout()
        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(-180, 180)
        self.phase_spin.setValue(0)
        self.phase_spin.setSuffix(" deg")
        self.phase_spin.valueChanged.connect(self.on_phase_spin_changed)
        
        self.phase_slider = QSlider(Qt.Orientation.Horizontal)
        self.phase_slider.setRange(-180, 180)
        self.phase_slider.setValue(0)
        self.phase_slider.valueChanged.connect(self.on_phase_slider_changed)
        
        phase_layout.addWidget(self.phase_spin)
        phase_layout.addWidget(self.phase_slider)
        basic_layout.addRow(tr("Phase Offset:"), phase_layout)
        
        # Amplitude
        amp_layout = QHBoxLayout()
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0, 1.0)
        self.amp_spin.setSingleStep(0.1)
        self.amp_spin.valueChanged.connect(self.on_amp_spin_changed)
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['Linear (0-1)', 'dBFS', 'dBV', 'dBu', 'Vrms', 'Vpeak'])
        self.unit_combo.currentTextChanged.connect(self.on_unit_changed)
        
        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setRange(0, 100)
        self.amp_slider.valueChanged.connect(self.on_amp_slider_changed)
        
        amp_layout.addWidget(self.amp_spin)
        amp_layout.addWidget(self.unit_combo)
        amp_layout.addWidget(self.amp_slider)
        basic_layout.addRow(tr("Amplitude:"), amp_layout)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # --- Sweep Controls ---
        sweep_group = QGroupBox(tr("Frequency Sweep (Sine Only)"))
        sweep_group.setCheckable(True)
        sweep_group.setChecked(False)
        sweep_group.toggled.connect(lambda v: self.update_param('sweep_enabled', v))
        self.sweep_group = sweep_group # Store ref to update checked state
        
        sweep_layout = QFormLayout()
        
        self.start_freq_spin = QDoubleSpinBox()
        self.start_freq_spin.setRange(20, 20000)
        self.start_freq_spin.valueChanged.connect(lambda v: self.update_param('start_freq', v))
        sweep_layout.addRow(tr("Start Freq:"), self.start_freq_spin)
        
        self.end_freq_spin = QDoubleSpinBox()
        self.end_freq_spin.setRange(20, 20000)
        self.end_freq_spin.valueChanged.connect(lambda v: self.update_param('end_freq', v))
        sweep_layout.addRow(tr("End Freq:"), self.end_freq_spin)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 60.0)
        self.duration_spin.valueChanged.connect(lambda v: self.update_param('sweep_duration', v))
        sweep_layout.addRow(tr("Duration (s):"), self.duration_spin)
        
        self.log_check = QCheckBox(tr("Logarithmic Sweep"))
        self.log_check.toggled.connect(lambda v: self.update_param('log_sweep', v))
        sweep_layout.addRow(self.log_check)
        
        sweep_group.setLayout(sweep_layout)
        layout.addWidget(sweep_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Initialize UI with current target (L)
        self.load_params_to_ui(self.module.params_L)

    def get_active_params_list(self):
        if self.current_target == 'L':
            return [self.module.params_L]
        elif self.current_target == 'R':
            return [self.module.params_R]
        elif self.current_target == 'LINK':
            return [self.module.params_L, self.module.params_R]
        return []

    def update_param(self, name, value):
        for p in self.get_active_params_list():
            setattr(p, name, value)
            
        # If linked, we might need to refresh UI if we just set both? 
        # No, UI reflects the state. If linked, we assume they are now same.

    def load_params_to_ui(self, params: SignalParameters):
        # Block signals to prevent feedback loops
        self.block_all_signals(True)
        
        self.wave_combo.setCurrentText(params.waveform)
        self.noise_combo.setCurrentText(params.noise_color)
        self.mt_count_spin.setValue(params.multitone_count)
        self.mls_order_combo.setCurrentText(str(params.mls_order))
        self.burst_on_spin.setValue(params.burst_on_cycles)
        self.burst_on_spin.setValue(params.burst_on_cycles)
        self.burst_off_spin.setValue(params.burst_off_cycles)
        self.pulse_width_spin.setValue(params.pulse_width)
        self.noise_amp_spin.setValue(params.noise_amplitude)
        
        self.freq_spin.setValue(params.frequency)
        self.freq_slider.setValue(self._freq_to_slider(params.frequency))
        
        self.phase_spin.setValue(params.phase_offset)
        self.phase_slider.setValue(int(params.phase_offset))
        
        self.update_amp_display_value(params.amplitude)
        
        self.sweep_group.setChecked(params.sweep_enabled)
        self.start_freq_spin.setValue(params.start_freq)
        self.end_freq_spin.setValue(params.end_freq)
        self.duration_spin.setValue(params.sweep_duration)
        self.log_check.setChecked(params.log_sweep)
        
        self.on_wave_changed(params.waveform) # Update visibility
        
        self.block_all_signals(False)

    def block_all_signals(self, block):
        widgets = [
            self.wave_combo, self.noise_combo, self.mt_count_spin, self.mls_order_combo,
            self.burst_on_spin, self.burst_off_spin, self.pulse_width_spin, self.noise_amp_spin,
            self.freq_spin, self.freq_slider, self.phase_spin, self.phase_slider,
            self.amp_spin, self.amp_slider, self.sweep_group, self.start_freq_spin,
            self.end_freq_spin, self.duration_spin, self.log_check
        ]
        for w in widgets:
            w.blockSignals(block)

    def on_target_changed(self, btn):
        if self.target_l.isChecked():
            self.current_target = 'L'
            self.load_params_to_ui(self.module.params_L)
        elif self.target_r.isChecked():
            self.current_target = 'R'
            self.load_params_to_ui(self.module.params_R)
        elif self.target_link.isChecked():
            self.current_target = 'LINK'
            # When switching to link, copy L to R (or vice versa, let's say L is master)
            # Or just load L to UI, and next edit updates both.
            # Let's copy L to R immediately to ensure consistency
            self.copy_params(self.module.params_L, self.module.params_R)
            self.load_params_to_ui(self.module.params_L)

    def copy_params(self, src, dst):
        dst.waveform = src.waveform
        dst.frequency = src.frequency
        dst.amplitude = src.amplitude
        dst.noise_color = src.noise_color
        dst.sweep_enabled = src.sweep_enabled
        dst.start_freq = src.start_freq
        dst.end_freq = src.end_freq
        dst.sweep_duration = src.sweep_duration
        dst.log_sweep = src.log_sweep
        dst.multitone_count = src.multitone_count
        dst.mls_order = src.mls_order
        dst.burst_on_cycles = src.burst_on_cycles
        dst.burst_off_cycles = src.burst_off_cycles
        dst.pulse_width = src.pulse_width
        dst.noise_amplitude = src.noise_amplitude
        dst.phase_offset = src.phase_offset

    def on_route_changed(self, btn):
        if self.route_l.isChecked():
            self.module.output_mode = 'L'
        elif self.route_r.isChecked():
            self.module.output_mode = 'R'
        elif self.route_stereo.isChecked():
            self.module.output_mode = 'STEREO'

    def on_wave_changed(self, val):
        self.update_param('waveform', val)
        
        self.noise_widget.hide()
        self.multitone_widget.hide()
        self.mls_widget.hide()
        self.mls_widget.hide()
        self.burst_widget.hide()
        self.pulse_widget.hide()
        self.tn_widget.hide()
        
        if val == 'noise': self.noise_widget.show()
        elif val == 'multitone': self.multitone_widget.show()
        elif val == 'mls': self.mls_widget.show()
        elif val == 'burst': self.burst_widget.show()
        elif val == 'pulse': self.pulse_widget.show()
        elif val == 'tone_noise': self.tn_widget.show()
        
        use_freq = val not in ['noise', 'mls']
        self.freq_spin.setEnabled(use_freq)
        self.freq_slider.setEnabled(use_freq)

    # --- Frequency Helpers ---
    def _freq_to_slider(self, freq):
        return int(1000 * (np.log10(freq) - np.log10(20)) / (np.log10(20000) - np.log10(20)))

    def _slider_to_freq(self, val):
        log_freq = np.log10(20) + (val / 1000) * (np.log10(20000) - np.log10(20))
        return 10**log_freq

    def on_freq_spin_changed(self, val):
        self.update_param('frequency', val)
        self.freq_slider.blockSignals(True)
        self.freq_slider.setValue(self._freq_to_slider(val))
        self.freq_slider.blockSignals(False)

    def on_freq_slider_changed(self, val):
        freq = self._slider_to_freq(val)
        self.update_param('frequency', freq)
        self.freq_spin.blockSignals(True)
        self.freq_spin.setValue(freq)
        self.freq_spin.blockSignals(False)

    def on_phase_spin_changed(self, val):
        self.update_param('phase_offset', val)
        self.phase_slider.blockSignals(True)
        self.phase_slider.setValue(int(val))
        self.phase_slider.blockSignals(False)

    def on_phase_slider_changed(self, val):
        self.update_param('phase_offset', float(val))
        self.phase_spin.blockSignals(True)
        self.phase_spin.setValue(float(val))
        self.phase_spin.blockSignals(False)

    # --- Amplitude Helpers ---
    def on_unit_changed(self, unit):
        # Refresh display with current amplitude in new unit
        # We need to know the current amplitude. 
        # Since we might be in LINK mode, we take from L (assuming synced) or just the first active.
        params = self.get_active_params_list()[0]
        self.update_amp_display_value(params.amplitude)

    def update_amp_display_value(self, amp_0_1):
        unit = self.unit_combo.currentText()
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
        
        self.amp_slider.blockSignals(True)
        self.amp_slider.setValue(int(amp_0_1 * 100))
        self.amp_slider.blockSignals(False)

    def on_amp_spin_changed(self, val):
        unit = self.unit_combo.currentText()
        gain = self.module.audio_engine.calibration.output_gain
        amp_0_1 = 0.0
        
        if unit == 'Linear (0-1)':
            amp_0_1 = val
        elif unit == 'dBFS':
            amp_0_1 = 10**(val/20)
        elif unit == 'dBV':
            v_rms = 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_0_1 = v_peak / gain
        elif unit == 'dBu':
            v_rms = 0.7746 * 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_0_1 = v_peak / gain
        elif unit == 'Vrms':
            v_peak = val * np.sqrt(2)
            amp_0_1 = v_peak / gain
        elif unit == 'Vpeak':
            amp_0_1 = val / gain
            
        if amp_0_1 > 1.0: amp_0_1 = 1.0
        elif amp_0_1 < 0.0: amp_0_1 = 0.0
            
        self.update_param('amplitude', amp_0_1)
        
        self.amp_slider.blockSignals(True)
        self.amp_slider.setValue(int(amp_0_1 * 100))
        self.amp_slider.blockSignals(False)

    def on_amp_slider_changed(self, val):
        amp = val / 100.0
        self.update_param('amplitude', amp)
        self.update_amp_display_value(amp)

    def on_toggle(self, checked):
        if checked:
            self.module.start_generation()
            self.toggle_btn.setText(tr("Stop Output"))
        else:
            self.module.stop_generation()
            self.toggle_btn.setText(tr("Start Output"))
