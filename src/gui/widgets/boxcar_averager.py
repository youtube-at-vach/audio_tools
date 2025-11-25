import argparse
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QDoubleSpinBox, QSpinBox)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class BoxcarAverager(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        
        # Parameters
        self.mode = 'Internal Pulse' # 'Internal Pulse', 'Internal Step', 'External Rising', 'External Falling'
        self.period_samples = 4800 # 100ms at 48k
        self.pulse_width_samples = 48 # 1ms
        self.trigger_level = 0.0
        self.trigger_channel = 0 # 0: Left, 1: Right
        self.input_channel = 'Stereo' # 'Left', 'Right', 'Stereo'
        
        # State
        self.accumulator = None
        self.count = 0
        self.max_averages = 0 # 0 = Infinite
        
        # Buffers
        self.input_ring_buffer = None
        self.input_write_pos = 0
        self.input_read_pos = 0
        
        # Internal Gen State
        self.phase = 0
        
        self.callback_id = None
        
    @property
    def name(self) -> str:
        return "Boxcar Averager"

    @property
    def description(self) -> str:
        return "High-precision signal averaging for transient analysis."

    def run(self, args: argparse.Namespace):
        print("Boxcar Averager CLI not implemented")

    def get_widget(self):
        return BoxcarAveragerWidget(self)

    def start_analysis(self):
        if self.is_running: return
        self.is_running = True
        
        # Initialize Buffers
        self.reset_average()
        
        # Ring buffer for raw input (large enough to hold a few periods)
        # 1 second buffer
        self.input_ring_buffer = np.zeros((self.audio_engine.sample_rate * 2, 2))
        self.input_write_pos = 0
        self.input_read_pos = 0
        self.phase = 0
        
        self.callback_id = self.audio_engine.register_callback(self._callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def reset_average(self):
        self.accumulator = np.zeros((self.period_samples, 2))
        self.count = 0
        self.phase = 0 # Reset phase for internal gen sync

    def _callback(self, indata, outdata, frames, time, status):
        if status: print(status)
        
        # 1. Store Input
        # Handle Ring Buffer Wrap
        buf_len = len(self.input_ring_buffer)
        write_idx = self.input_write_pos
        
        if write_idx + frames <= buf_len:
            if indata.shape[1] >= 2:
                self.input_ring_buffer[write_idx:write_idx+frames] = indata[:, :2]
            else:
                self.input_ring_buffer[write_idx:write_idx+frames, 0] = indata[:, 0]
                self.input_ring_buffer[write_idx:write_idx+frames, 1] = indata[:, 0]
        else:
            # Wrap around
            first_part = buf_len - write_idx
            second_part = frames - first_part
            
            if indata.shape[1] >= 2:
                self.input_ring_buffer[write_idx:] = indata[:first_part, :2]
                self.input_ring_buffer[:second_part] = indata[first_part:, :2]
            else:
                self.input_ring_buffer[write_idx:, 0] = indata[:first_part, 0]
                self.input_ring_buffer[write_idx:, 1] = indata[:first_part, 0]
                self.input_ring_buffer[:second_part, 0] = indata[first_part:, 0]
                self.input_ring_buffer[:second_part, 1] = indata[first_part:, 0]
                
        self.input_write_pos = (write_idx + frames) % buf_len
        
        # 2. Generate Output (Internal Mode)
        outdata.fill(0)
        if 'Internal' in self.mode:
            # Generate Pulse/Step
            # Phase goes from 0 to period_samples
            t = np.arange(frames) + self.phase
            t_mod = t % self.period_samples
            
            signal = np.zeros(frames)
            
            if self.mode == 'Internal Pulse':
                # High for pulse_width
                signal = np.where(t_mod < self.pulse_width_samples, 1.0, 0.0)
            elif self.mode == 'Internal Step':
                # High for half period
                signal = np.where(t_mod < (self.period_samples // 2), 1.0, -1.0)
                
            self.phase = (self.phase + frames) % self.period_samples
            
            # Output to both channels? Or just Left? Let's do Left.
            # Output to selected channel(s)
            if outdata.shape[1] >= 1 and self.input_channel in ['Left', 'Stereo']:
                outdata[:, 0] = signal * 0.5 # -6dB
            if outdata.shape[1] >= 2 and self.input_channel in ['Right', 'Stereo']:
                outdata[:, 1] = signal * 0.5

    def process(self):
        """
        Called periodically to process input buffer and update average.
        """
        if not self.is_running: return
        
        # Determine available data
        buf_len = len(self.input_ring_buffer)
        write_pos = self.input_write_pos
        read_pos = self.input_read_pos
        
        if write_pos >= read_pos:
            available = write_pos - read_pos
        else:
            available = buf_len - read_pos + write_pos
            
        if available == 0: return
        
        # Extract data (linearized)
        if write_pos >= read_pos:
            data = self.input_ring_buffer[read_pos:write_pos]
        else:
            data = np.concatenate((self.input_ring_buffer[read_pos:], self.input_ring_buffer[:write_pos]))
            
        self.input_read_pos = write_pos
        
        # Process Data
        if 'Internal' in self.mode:
            # Synchronous Folding
            # We assume the input data corresponds to the generated output phase.
            # BUT there is latency.
            # However, for "Folding", we just wrap the input data by period.
            # The "Trigger Point" (T=0 of pulse) will appear at T=Latency in the averaged buffer.
            # We need to maintain the "Global Phase" of the INPUT data relative to the OUTPUT generation.
            # Since we started generation at phase=0 when we started capture (roughly),
            # we can track input_phase.
            
            # Actually, self.phase in callback tracks the NEXT sample to be generated.
            # The data we just read corresponds to samples generated 'latency' ago?
            # No, we just need to fold it continuously.
            # We need a persistent 'current_fold_index' for the input processing.
            
            if not hasattr(self, 'fold_idx'):
                self.fold_idx = 0
                
            num_samples = len(data)
            current_idx = 0
            
            while current_idx < num_samples:
                remaining_in_period = self.period_samples - self.fold_idx
                chunk_size = min(num_samples - current_idx, remaining_in_period)
                
                # Add to accumulator
                chunk = data[current_idx : current_idx + chunk_size]
                
                # Ensure accumulator size matches period (if changed)
                if self.accumulator.shape[0] != self.period_samples:
                    self.reset_average()
                    self.fold_idx = 0
                    return # Abort this cycle
                
                self.accumulator[self.fold_idx : self.fold_idx + chunk_size] += chunk
                self.fold_idx += chunk_size
                current_idx += chunk_size
                
                if self.fold_idx >= self.period_samples:
                    self.fold_idx = 0
                    self.count += 1
                    
        else:
            # External Trigger
            # We need to find triggers in 'data'
            # And add windows to accumulator.
            # This is tricky because a window might span across 'data' chunks.
            # We need a state machine.
            pass # TODO: Implement External Trigger Logic

class BoxcarAveragerWidget(QWidget):
    def __init__(self, module: BoxcarAverager):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.setInterval(50) # 20 FPS

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()
        
        self.toggle_btn = QPushButton("Start")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        controls_layout.addWidget(self.toggle_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.on_reset)
        controls_layout.addWidget(self.reset_btn)
        
        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Internal Pulse', 'Internal Step']) # External TODO
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        controls_layout.addWidget(QLabel("Mode:"))
        controls_layout.addWidget(self.mode_combo)
        
        # Period
        self.period_spin = QDoubleSpinBox()
        self.period_spin.setRange(1, 1000) # ms
        self.period_spin.setValue(100)
        self.period_spin.setSuffix(" ms")
        self.period_spin.valueChanged.connect(self.on_period_changed)
        controls_layout.addWidget(QLabel("Period:"))
        controls_layout.addWidget(self.period_spin)
        
        # Channel
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(['Stereo', 'Left', 'Right'])
        self.channel_combo.currentTextChanged.connect(self.on_channel_changed)
        controls_layout.addWidget(QLabel("Channel:"))
        controls_layout.addWidget(self.channel_combo)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Plot
        self.plot = pg.PlotWidget(title="Averaged Signal")
        self.plot.setLabel('left', "Amplitude")
        self.plot.setLabel('bottom', "Time", units='s')
        self.plot.showGrid(x=True, y=True)
        self.curve_l = self.plot.plot(pen='g', name='Left')
        self.curve_r = self.plot.plot(pen='r', name='Right')
        
        layout.addWidget(self.plot)
        self.setLayout(layout)
        
    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText("Stop")
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText("Start")
            
    def on_reset(self):
        self.module.reset_average()
        
    def on_mode_changed(self, val):
        self.module.mode = val
        self.module.reset_average()
        
    def on_period_changed(self, val):
        # val is ms
        sr = self.module.audio_engine.sample_rate
        self.module.period_samples = int(val / 1000 * sr)
        self.module.reset_average()
        
    def on_channel_changed(self, val):
        self.module.input_channel = val
        self.module.reset_average()
        
    def update_plot(self):
        if not self.module.is_running: return
        
        self.module.process()
        
        if self.module.count > 0:
            avg = self.module.accumulator / self.module.count
            t = np.linspace(0, self.module.period_samples / self.module.audio_engine.sample_rate, len(avg))
            self.curve_l.setData(t, avg[:, 0])
            self.curve_r.setData(t, avg[:, 1])
            
            # Visibility
            ch = self.module.input_channel
            if ch == 'Left':
                self.curve_l.setVisible(True)
                self.curve_r.setVisible(False)
            elif ch == 'Right':
                self.curve_l.setVisible(False)
                self.curve_r.setVisible(True)
            else:
                self.curve_l.setVisible(True)
                self.curve_r.setVisible(True)
                
            self.plot.setTitle(f"Averaged Signal (N={self.module.count})")
