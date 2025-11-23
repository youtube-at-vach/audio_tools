import argparse
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QPushButton, 
                             QComboBox, QHBoxLayout, QGroupBox, QFormLayout, QCheckBox, QSlider)
from PyQt6.QtCore import Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class SignalGenerator(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.frequency = 1000.0
        self.amplitude = 0.5
        self.waveform = 'sine'
        self.noise_color = 'white'
        self.is_playing = False
        
        # Sweep parameters
        self.sweep_enabled = False
        self.start_freq = 20.0
        self.end_freq = 20000.0
        self.sweep_duration = 5.0
        self.log_sweep = True
        
        # Internal state
        self._phase = 0
        self._sweep_time = 0
        self._noise_buffer = None
        self._noise_index = 0
        self._buffer = None
        self._buffer_index = 0
        
        # Advanced Signal Parameters
        self.multitone_count = 10
        self.mls_order = 15
        self.burst_on_cycles = 10
        self.burst_off_cycles = 90
        
        self.callback_id = None

    @property
    def name(self) -> str:
        return "Signal Generator"

    @property
    def description(self) -> str:
        return "Generates advanced test signals (Sine, Square, Noise, Sweeps)"

    def run(self, args: argparse.Namespace):
        print("Signal Generator running from CLI (not fully implemented)")

    def get_widget(self):
        return SignalGeneratorWidget(self)

    def _generate_noise_buffer(self, sample_rate, duration=5.0):
        """Pre-generates a buffer of colored noise."""
        num_samples = int(sample_rate * duration)
        
        # White noise base
        white = np.random.randn(num_samples)
        
        if self.noise_color == 'white':
            # Normalize
            max_val = np.max(np.abs(white))
            if max_val > 0:
                white /= max_val
            return white
            
        # FFT filtering
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(num_samples, d=1/sample_rate)
        
        # Avoid division by zero at DC (index 0)
        # We'll handle DC separately or just set scaling[0] = 0
        
        scaling = np.ones_like(freqs)
        
        if self.noise_color == 'pink':
            # 1/f^0.5 (-3dB/oct)
            # Handle DC: set to 0 or same as first bin
            scaling[1:] = 1 / np.sqrt(freqs[1:])
            scaling[0] = 0
        elif self.noise_color == 'brown':
            # 1/f (-6dB/oct)
            scaling[1:] = 1 / freqs[1:]
            scaling[0] = 0
        elif self.noise_color == 'blue':
            # f^0.5 (+3dB/oct)
            scaling = np.sqrt(freqs)
        elif self.noise_color == 'violet':
            # f (+6dB/oct)
            scaling = freqs
        elif self.noise_color == 'grey':
            # Inverted A-weighting curve approximation
            # A-weighting (approx):
            # Ra(f) = 12200^2 * f^4 / ((f^2 + 20.6^2) * sqrt((f^2 + 107.7^2)(f^2 + 737.9^2)) * (f^2 + 12200^2))
            # Grey noise should be 1/Ra(f) roughly? Or just equal loudness contour.
            # ITU-R 468 is better but complex.
            # Let's use a simplified inverted A-weighting.
            
            f = freqs
            f2 = f**2
            # Constants
            c1 = 12194.217**2
            c2 = 20.6**2
            c3 = 107.7**2
            c4 = 737.9**2
            
            # A-weighting magnitude squared
            # Num = c1 * f^4
            # Denom = (f^2 + c2) * sqrt((f^2 + c3)*(f^2 + c4)) * (f^2 + c1)
            # We want inverse.
            
            # Avoid DC
            f_safe = f.copy()
            f_safe[0] = 1.0 # Dummy
            f2_safe = f_safe**2
            
            num = c1 * (f2_safe**2)
            denom = (f2_safe + c2) * np.sqrt((f2_safe + c3) * (f2_safe + c4)) * (f2_safe + c1)
            
            a_weight = num / denom
            
            # Grey noise = White / A-weighting ? No, Grey noise sounds "flat" to human ear.
            # So it should be boosted where ear is insensitive (low/high freq) and cut where sensitive (mid).
            # This is roughly Inverse A-weighting.
            
            scaling = 1.0 / (a_weight + 1e-12)
            
            # Clamp the boost to avoid infrasound dominating the signal
            # Normalize to 1kHz gain first
            # Find index closest to 1000Hz
            idx_1k = np.argmin(np.abs(freqs - 1000))
            if idx_1k < len(scaling):
                ref_gain = scaling[idx_1k]
                scaling /= ref_gain
            
            # Limit max boost to e.g. 40dB (100x) relative to 1kHz
            scaling = np.minimum(scaling, 100.0)
            
            scaling[0] = 0
            
        
        fft = fft * scaling
        noise = np.fft.irfft(fft)
        
        # Normalize
        max_val = np.max(np.abs(noise))
        if max_val > 0:
            noise /= max_val
            
        return noise

    def _generate_multitone(self, sample_rate):
        """Generates a Crest-Factor optimized Multitone signal."""
        # Logarithmic spacing
        if self.start_freq >= self.end_freq:
            freqs = np.array([self.start_freq])
        else:
            # num_tones points from start to end
            freqs = np.logspace(np.log10(self.start_freq), np.log10(self.end_freq), self.multitone_count)
        
        # Snap frequencies to FFT bins for perfect looping if we wanted to be strict,
        # but for general playback, exact frequencies are fine.
        # However, to make it loop perfectly without clicks, we should ensure integer cycles in the buffer.
        # Let's define a buffer length, say 1 second (or closest power of 2).
        N = int(sample_rate) # 1 second buffer
        
        # Adjust frequencies to be integer multiples of fs/N (bin centers)
        bin_width = sample_rate / N
        freqs = np.round(freqs / bin_width) * bin_width
        
        # Newman Phase for Crest Factor Minimization
        # phi_k = pi * k^2 / N_tones
        phases = np.pi * (np.arange(len(freqs))**2) / len(freqs)
        
        t = np.arange(N) / sample_rate
        signal = np.zeros(N)
        
        for f, p in zip(freqs, phases):
            signal += np.sin(2 * np.pi * f * t + p)
            
        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal /= max_val
            
        return signal

    def _generate_mls(self, sample_rate):
        """Generates a Maximum Length Sequence (MLS)."""
        # MLS generation using Linear Feedback Shift Register (LFSR)
        # Polynomials for common orders
        taps = {
            10: [10, 7],
            11: [11, 9],
            12: [12, 11, 10, 4],
            13: [13, 12, 10, 9],
            14: [14, 13, 12, 2],
            15: [15, 14],
            16: [16, 15, 13, 4],
            17: [17, 14],
            18: [18, 11]
        }
        
        if self.mls_order not in taps:
            self.mls_order = 15 # Fallback
            
        tap_indices = [x - 1 for x in taps[self.mls_order]] # 0-indexed
        N = 2**self.mls_order - 1
        
        # Python implementation of LFSR might be slow for large N.
        # For N=18, len=262143. It's okay.
        
        # Faster approach: use scipy.signal.max_len_seq if available, but it's not standard in all versions.
        # Let's try to use scipy.signal.max_len_seq
        try:
            import scipy.signal
            seq, state = scipy.signal.max_len_seq(self.mls_order)
            # seq is 0/1. Convert to -1/1
            signal = seq.astype(float) * 2 - 1
            return signal
        except:
            # Fallback manual implementation (slow)
            print("scipy.signal.max_len_seq not found/failed, using slow fallback")
            state = np.ones(self.mls_order, dtype=int)
            signal = np.zeros(N)
            
            for i in range(N):
                # Feedback
                feedback = 0
                for tap in tap_indices:
                    feedback ^= state[tap]
                
                output = state[-1]
                signal[i] = float(output * 2 - 1)
                
                # Shift
                state = np.roll(state, 1)
                state[0] = feedback
                
            return signal

    def _generate_burst(self, sample_rate):
        """Generates a Tone Burst."""
        # Total cycle length = On + Off
        total_cycles = self.burst_on_cycles + self.burst_off_cycles
        cycle_duration = 1.0 / self.frequency
        total_duration = total_cycles * cycle_duration
        
        num_samples = int(total_duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        
        # Continuous sine
        sine = np.sin(2 * np.pi * self.frequency * t)
        
        # Envelope
        on_duration = self.burst_on_cycles * cycle_duration
        on_samples = int(on_duration * sample_rate)
        
        envelope = np.zeros(num_samples)
        envelope[:on_samples] = 1.0
        
        return sine * envelope

    def start_generation(self):
        if self.is_playing:
            return

        self.is_playing = True
        self._phase = 0
        self._sweep_time = 0
        sample_rate = self.audio_engine.sample_rate
        
        # Pre-generate buffer for buffer-based signals
        self._buffer = None
        self._buffer_index = 0
        
        if self.waveform == 'noise':
            self._buffer = self._generate_noise_buffer(sample_rate)
        elif self.waveform == 'multitone':
            self._buffer = self._generate_multitone(sample_rate)
        elif self.waveform == 'mls':
            self._buffer = self._generate_mls(sample_rate)
        elif self.waveform == 'burst':
            self._buffer = self._generate_burst(sample_rate)
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
                
            t = np.arange(frames) / sample_rate
            
            if self._buffer is not None:
                # Loop through pre-generated buffer
                chunk_size = frames
                buf_len = len(self._buffer)
                
                # If buffer is shorter than chunk (unlikely for reasonable settings but possible for burst)
                # We need to tile it or handle it.
                # Simplified: assume buffer is long enough or we loop carefully.
                
                out_chunk = np.zeros(chunk_size)
                current_idx = 0
                
                while current_idx < chunk_size:
                    remaining = chunk_size - current_idx
                    available = buf_len - self._buffer_index
                    
                    to_copy = min(remaining, available)
                    out_chunk[current_idx:current_idx+to_copy] = self._buffer[self._buffer_index:self._buffer_index+to_copy]
                    
                    self._buffer_index += to_copy
                    current_idx += to_copy
                    
                    if self._buffer_index >= buf_len:
                        self._buffer_index = 0
                
                outdata[:, 0] = out_chunk * self.amplitude
                if outdata.shape[1] > 1:
                    outdata[:, 1] = outdata[:, 0]
                return

            if self.sweep_enabled:
                # Calculate instantaneous frequency for sweep
                # Linear: f(t) = start + (end-start)/duration * t
                # Log: f(t) = start * (end/start)^(t/duration)
                
                # We need global time for sweep
                current_times = self._sweep_time + t
                # Wrap time for continuous sweeping
                current_times = np.mod(current_times, self.sweep_duration)
                
                if self.log_sweep:
                    freqs = self.start_freq * (self.end_freq / self.start_freq) ** (current_times / self.sweep_duration)
                else:
                    freqs = self.start_freq + (self.end_freq - self.start_freq) * (current_times / self.sweep_duration)
                
                if self.log_sweep:
                    # Integral of A * B^t is A * B^t / ln(B)
                    # Phase = 2*pi * integral(f(t))
                    k = np.log(self.end_freq / self.start_freq) / self.sweep_duration
                    if k == 0: phase = 2 * np.pi * self.start_freq * current_times
                    else: phase = 2 * np.pi * self.start_freq * (np.exp(k * current_times) - 1) / k
                else:
                    # Integral of start + k*t is start*t + 0.5*k*t^2
                    k = (self.end_freq - self.start_freq) / self.sweep_duration
                    phase = 2 * np.pi * (self.start_freq * current_times + 0.5 * k * current_times**2)
                
                signal = self.amplitude * np.sin(phase)
                self._sweep_time += frames / sample_rate
                
            else:
                # Standard waveforms
                phase_t = (np.arange(frames) + self._phase) / sample_rate
                self._phase += frames
                
                if self.waveform == 'sine':
                    signal = self.amplitude * np.sin(2 * np.pi * self.frequency * phase_t)
                elif self.waveform == 'square':
                    signal = self.amplitude * np.sign(np.sin(2 * np.pi * self.frequency * phase_t))
                elif self.waveform == 'triangle':
                    signal = self.amplitude * (2 * np.abs(2 * ((phase_t * self.frequency) % 1) - 1) - 1)
                elif self.waveform == 'sawtooth':
                    signal = self.amplitude * (2 * ((phase_t * self.frequency) % 1) - 1)
                else:
                    signal = np.zeros(frames)

            # Output
            num_channels = outdata.shape[1]
            
            if num_channels >= 1:
                outdata[:, 0] = signal
            
            if num_channels >= 2:
                outdata[:, 1] = signal

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
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- Basic Controls ---
        basic_group = QGroupBox("Basic Controls")
        basic_layout = QFormLayout()
        
        # Waveform
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(['sine', 'square', 'triangle', 'sawtooth', 'noise', 'multitone', 'mls', 'burst'])
        self.wave_combo.currentTextChanged.connect(self.on_wave_changed)
        basic_layout.addRow("Waveform:", self.wave_combo)
        
        # Dynamic Parameters Stack
        self.param_stack = QWidget()
        self.param_layout = QVBoxLayout(self.param_stack)
        self.param_layout.setContentsMargins(0,0,0,0)
        
        # 1. Noise Params
        self.noise_widget = QWidget()
        noise_form = QFormLayout(self.noise_widget)
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(['white', 'pink', 'brown', 'blue', 'violet', 'grey'])
        self.noise_combo.currentTextChanged.connect(lambda v: setattr(self.module, 'noise_color', v))
        noise_form.addRow("Color:", self.noise_combo)
        
        # 2. Multitone Params
        self.multitone_widget = QWidget()
        mt_form = QFormLayout(self.multitone_widget)
        self.mt_count_spin = QDoubleSpinBox()
        self.mt_count_spin.setDecimals(0); self.mt_count_spin.setRange(2, 1000); self.mt_count_spin.setValue(10)
        self.mt_count_spin.valueChanged.connect(lambda v: setattr(self.module, 'multitone_count', int(v)))
        mt_form.addRow("Tone Count:", self.mt_count_spin)
        
        # 3. MLS Params
        self.mls_widget = QWidget()
        mls_form = QFormLayout(self.mls_widget)
        self.mls_order_combo = QComboBox()
        self.mls_order_combo.addItems([str(i) for i in range(10, 19)])
        self.mls_order_combo.setCurrentText("15")
        self.mls_order_combo.currentTextChanged.connect(lambda v: setattr(self.module, 'mls_order', int(v)))
        mls_form.addRow("Order (N):", self.mls_order_combo)
        
        # 4. Burst Params
        self.burst_widget = QWidget()
        burst_form = QFormLayout(self.burst_widget)
        self.burst_on_spin = QDoubleSpinBox()
        self.burst_on_spin.setDecimals(0); self.burst_on_spin.setRange(1, 1000); self.burst_on_spin.setValue(10)
        self.burst_on_spin.valueChanged.connect(lambda v: setattr(self.module, 'burst_on_cycles', int(v)))
        burst_form.addRow("On Cycles:", self.burst_on_spin)
        self.burst_off_spin = QDoubleSpinBox()
        self.burst_off_spin.setDecimals(0); self.burst_off_spin.setRange(1, 10000); self.burst_off_spin.setValue(90)
        self.burst_off_spin.valueChanged.connect(lambda v: setattr(self.module, 'burst_off_cycles', int(v)))
        burst_form.addRow("Off Cycles:", self.burst_off_spin)
        
        # Add all to layout but hide initially
        self.param_layout.addWidget(self.noise_widget)
        self.param_layout.addWidget(self.multitone_widget)
        self.param_layout.addWidget(self.mls_widget)
        self.param_layout.addWidget(self.burst_widget)
        self.noise_widget.hide()
        self.multitone_widget.hide()
        self.mls_widget.hide()
        self.burst_widget.hide()
        
        basic_layout.addRow(self.param_stack)
        
        # Frequency (Shared)
        freq_layout = QHBoxLayout()
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000)
        self.freq_spin.setValue(self.module.frequency)
        self.freq_spin.valueChanged.connect(self.on_freq_spin_changed)
        
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(0, 1000) # Log scale mapping
        self.freq_slider.setValue(self._freq_to_slider(self.module.frequency))
        self.freq_slider.valueChanged.connect(self.on_freq_slider_changed)
        
        freq_layout.addWidget(self.freq_spin)
        freq_layout.addWidget(self.freq_slider)
        basic_layout.addRow("Frequency (Hz):", freq_layout)
        
        # Amplitude
        amp_layout = QHBoxLayout()
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0, 1.0)
        self.amp_spin.setSingleStep(0.1)
        self.amp_spin.setValue(self.module.amplitude)
        self.amp_spin.valueChanged.connect(self.on_amp_spin_changed)
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['Linear (0-1)', 'dBFS', 'dBV', 'dBu', 'Vrms', 'Vpeak'])
        self.unit_combo.currentTextChanged.connect(self.on_unit_changed)
        
        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setRange(0, 100)
        self.amp_slider.setValue(int(self.module.amplitude * 100))
        self.amp_slider.valueChanged.connect(self.on_amp_slider_changed)
        
        amp_layout.addWidget(self.amp_spin)
        amp_layout.addWidget(self.unit_combo)
        amp_layout.addWidget(self.amp_slider)
        basic_layout.addRow("Amplitude:", amp_layout)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # --- Sweep Controls ---
        sweep_group = QGroupBox("Frequency Sweep (Sine Only)")
        sweep_group.setCheckable(True)
        sweep_group.setChecked(False)
        sweep_group.toggled.connect(self.on_sweep_toggled)
        sweep_layout = QFormLayout()
        
        self.start_freq_spin = QDoubleSpinBox()
        self.start_freq_spin.setRange(20, 20000)
        self.start_freq_spin.setValue(self.module.start_freq)
        self.start_freq_spin.valueChanged.connect(lambda v: setattr(self.module, 'start_freq', v))
        sweep_layout.addRow("Start Freq:", self.start_freq_spin)
        
        self.end_freq_spin = QDoubleSpinBox()
        self.end_freq_spin.setRange(20, 20000)
        self.end_freq_spin.setValue(self.module.end_freq)
        self.end_freq_spin.valueChanged.connect(lambda v: setattr(self.module, 'end_freq', v))
        sweep_layout.addRow("End Freq:", self.end_freq_spin)
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 60.0)
        self.duration_spin.setValue(self.module.sweep_duration)
        self.duration_spin.valueChanged.connect(lambda v: setattr(self.module, 'sweep_duration', v))
        sweep_layout.addRow("Duration (s):", self.duration_spin)
        
        self.log_check = QCheckBox("Logarithmic Sweep")
        self.log_check.setChecked(self.module.log_sweep)
        self.log_check.toggled.connect(lambda v: setattr(self.module, 'log_sweep', v))
        sweep_layout.addRow(self.log_check)
        
        sweep_group.setLayout(sweep_layout)
        layout.addWidget(sweep_group)
        
        # Start/Stop Button
        self.toggle_btn = QPushButton("Start")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.toggle_btn.setStyleSheet("QPushButton:checked { background-color: #ffcccc; }")
        layout.addWidget(self.toggle_btn)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Initial state update
        self.on_wave_changed(self.wave_combo.currentText())

    def _freq_to_slider(self, freq):
        # Log mapping: 20Hz -> 0, 20000Hz -> 1000
        return int(1000 * (np.log10(freq) - np.log10(20)) / (np.log10(20000) - np.log10(20)))

    def _slider_to_freq(self, val):
        # Log mapping inverse
        log_freq = np.log10(20) + (val / 1000) * (np.log10(20000) - np.log10(20))
        return 10**log_freq

    def on_freq_spin_changed(self, val):
        self.module.frequency = val
        self.freq_slider.blockSignals(True)
        self.freq_slider.setValue(self._freq_to_slider(val))
        self.freq_slider.blockSignals(False)

    def on_freq_slider_changed(self, val):
        freq = self._slider_to_freq(val)
        self.module.frequency = freq
        self.freq_spin.blockSignals(True)
        self.freq_spin.setValue(freq)
        self.freq_spin.blockSignals(False)

    def on_unit_changed(self, unit):
        # Update spin box range and value based on current amplitude
        self.update_amp_display()

    def update_amp_display(self):
        unit = self.unit_combo.currentText()
        amp_0_1 = self.module.amplitude
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
            # Vpeak = amp_0_1 * gain
            # Vrms = Vpeak / sqrt(2)
            # dBV = 20 * log10(Vrms)
            # Wait, dBV is usually defined as 20*log10(Vrms).
            # But let's stick to the definition in CalibrationManager if it exists. 
            # CalibrationManager.dbfs_to_dbv uses input_sensitivity.
            # Here we use output_gain.
            
            v_peak = amp_0_1 * gain
            v_rms = v_peak / np.sqrt(2)
            val = 20 * np.log10(v_rms + 1e-12)
            
            self.amp_spin.setRange(-120, 20) # Arbitrary upper limit
            self.amp_spin.setSingleStep(1.0)
            self.amp_spin.setValue(val)
            
        elif unit == 'dBu':
            # dBu = 20 * log10(Vrms / 0.7746)
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

    def on_amp_spin_changed(self, val):
        unit = self.unit_combo.currentText()
        gain = self.module.audio_engine.calibration.output_gain
        amp_0_1 = 0.0
        
        if unit == 'Linear (0-1)':
            amp_0_1 = val
        elif unit == 'dBFS':
            amp_0_1 = 10**(val/20)
        elif unit == 'dBV':
            # val = 20 * log10(Vrms)
            v_rms = 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_0_1 = v_peak / gain
        elif unit == 'dBu':
            # val = 20 * log10(Vrms / 0.7746)
            v_rms = 0.7746 * 10**(val/20)
            v_peak = v_rms * np.sqrt(2)
            amp_0_1 = v_peak / gain
        elif unit == 'Vrms':
            v_peak = val * np.sqrt(2)
            amp_0_1 = v_peak / gain
        elif unit == 'Vpeak':
            amp_0_1 = val / gain
            
        # Clamp
        if amp_0_1 > 1.0:
            amp_0_1 = 1.0
            # Optionally warn or update spin to max possible
        elif amp_0_1 < 0.0:
            amp_0_1 = 0.0
            
        self.module.amplitude = amp_0_1
        
        # Update slider
        self.amp_slider.blockSignals(True)
        self.amp_slider.setValue(int(amp_0_1 * 100))
        self.amp_slider.blockSignals(False)

    def on_amp_slider_changed(self, val):
        amp = val / 100.0
        self.module.amplitude = amp
        self.update_amp_display()

    def on_wave_changed(self, val):
        self.module.waveform = val
        
        # Hide all specific params
        self.noise_widget.hide()
        self.multitone_widget.hide()
        self.mls_widget.hide()
        self.burst_widget.hide()
        
        # Show relevant params
        if val == 'noise':
            self.noise_widget.show()
        elif val == 'multitone':
            self.multitone_widget.show()
        elif val == 'mls':
            self.mls_widget.show()
        elif val == 'burst':
            self.burst_widget.show()
            
        # Enable/Disable frequency control
        # Noise and MLS don't use single frequency
        use_freq = val not in ['noise', 'mls']
        self.freq_spin.setEnabled(use_freq)
        self.freq_slider.setEnabled(use_freq)

    def on_sweep_toggled(self, checked):
        self.module.sweep_enabled = checked
        # Disable fixed frequency if sweep is on
        enabled = not checked and self.module.waveform not in ['noise', 'mls']
        self.freq_spin.setEnabled(enabled)
        self.freq_slider.setEnabled(enabled)

    def on_toggle(self, checked):
        if checked:
            self.module.start_generation()
            self.toggle_btn.setText("Stop")
        else:
            self.module.stop_generation()
            self.toggle_btn.setText("Start")
