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
        
        if self.noise_color == 'white':
            return np.random.uniform(-1, 1, num_samples)
            
        # For other colors, use FFT filtering (simplified from legacy)
        white = np.random.randn(num_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(num_samples, d=1/sample_rate)
        
        if self.noise_color == 'pink':
            # 1/f
            scaling = 1 / np.sqrt(freqs + 1e-10)
        elif self.noise_color == 'brown':
            # 1/f^2
            scaling = 1 / (freqs + 1e-10)
        else:
            scaling = np.ones_like(freqs)
            
        fft = fft * scaling
        noise = np.fft.irfft(fft)
        
        # Normalize
        max_val = np.max(np.abs(noise))
        if max_val > 0:
            noise /= max_val
            
        return noise

    def start_generation(self):
        if self.is_playing:
            return

        self.is_playing = True
        self._phase = 0
        self._sweep_time = 0
        sample_rate = self.audio_engine.sample_rate
        
        # Pre-generate noise if needed
        if self.waveform == 'noise':
            self._noise_buffer = self._generate_noise_buffer(sample_rate)
            self._noise_index = 0
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
                
            t = np.arange(frames) / sample_rate
            
            if self.waveform == 'noise':
                # Loop through pre-generated buffer
                chunk_size = frames
                if self._noise_index + chunk_size > len(self._noise_buffer):
                    # Wrap around
                    remainder = len(self._noise_buffer) - self._noise_index
                    outdata[:remainder, 0] = self._noise_buffer[self._noise_index:] * self.amplitude
                    outdata[remainder:, 0] = self._noise_buffer[:chunk_size-remainder] * self.amplitude
                    self._noise_index = chunk_size - remainder
                else:
                    outdata[:, 0] = self._noise_buffer[self._noise_index:self._noise_index+chunk_size] * self.amplitude
                    self._noise_index += chunk_size
                    
                # Copy to second channel if available
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
                
                # Phase integration is tricky for variable frequency in blocks.
                # Simplified approach: use instantaneous phase accumulation
                # This might have discontinuities at block boundaries if not careful.
                # Better: Calculate phase directly from integral of frequency function.
                
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
            # Check how many channels we have in outdata
            num_channels = outdata.shape[1]
            
            if num_channels >= 1:
                outdata[:, 0] = signal
            
            if num_channels >= 2:
                outdata[:, 1] = signal
                
            # If more channels, we could replicate or leave silence. 
            # For now, just filling 1 and 2 is enough for Mono/Stereo/Left/Right mappings.
            # If mapping is 'Left' only, num_channels will be 1.
            # If mapping is 'Stereo', num_channels will be 2.

        self.audio_engine.start_stream(callback, channels=2) # Request 2, but engine might map fewer

    def stop_generation(self):
        if self.is_playing:
            self.audio_engine.stop_stream()
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
        self.wave_combo.addItems(['sine', 'square', 'triangle', 'sawtooth', 'noise'])
        self.wave_combo.currentTextChanged.connect(self.on_wave_changed)
        basic_layout.addRow("Waveform:", self.wave_combo)
        
        # Noise Color (Visible only when noise is selected)
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(['white', 'pink', 'brown'])
        self.noise_combo.currentTextChanged.connect(self.on_noise_changed)
        basic_layout.addRow("Noise Color:", self.noise_combo)
        
        # Frequency
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
        
        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setRange(0, 100)
        self.amp_slider.setValue(int(self.module.amplitude * 100))
        self.amp_slider.valueChanged.connect(self.on_amp_slider_changed)
        
        amp_layout.addWidget(self.amp_spin)
        amp_layout.addWidget(self.amp_slider)
        basic_layout.addRow("Amplitude (0-1):", amp_layout)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # --- Sweep Controls ---
        sweep_group = QGroupBox("Frequency Sweep")
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

    def on_amp_spin_changed(self, val):
        self.module.amplitude = val
        self.amp_slider.blockSignals(True)
        self.amp_slider.setValue(int(val * 100))
        self.amp_slider.blockSignals(False)

    def on_amp_slider_changed(self, val):
        amp = val / 100.0
        self.module.amplitude = amp
        self.amp_spin.blockSignals(True)
        self.amp_spin.setValue(amp)
        self.amp_spin.blockSignals(False)

    def on_wave_changed(self, val):
        self.module.waveform = val
        # Enable/Disable frequency control based on noise
        is_noise = val == 'noise'
        self.freq_spin.setEnabled(not is_noise)
        self.freq_slider.setEnabled(not is_noise)
        self.noise_combo.setEnabled(is_noise)

    def on_noise_changed(self, val):
        self.module.noise_color = val

    def on_sweep_toggled(self, checked):
        self.module.sweep_enabled = checked
        # Disable fixed frequency if sweep is on
        enabled = not checked and self.module.waveform != 'noise'
        self.freq_spin.setEnabled(enabled)
        self.freq_slider.setEnabled(enabled)

    def on_toggle(self, checked):
        if checked:
            self.module.start_generation()
            self.toggle_btn.setText("Stop")
        else:
            self.module.stop_generation()
            self.toggle_btn.setText("Start")

