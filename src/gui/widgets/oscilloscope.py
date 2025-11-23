import argparse
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QDoubleSpinBox)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class Oscilloscope(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        # Buffer enough for low frequency analysis, but we'll display a subset
        self.buffer_size = 8192 
        self.input_data = np.zeros((self.buffer_size, 2))
        
        # Settings
        self.timebase = 0.01 # Seconds per division (approx) -> Total view window
        self.gain = 1.0
        self.trigger_source = 0 # 0: Left, 1: Right
        self.trigger_mode = 'Auto' # 'Auto', 'Normal'
        self.trigger_slope = 'Rising' # 'Rising', 'Falling'
        self.trigger_level = 0.0
        self.show_left = True
        self.show_right = True
        
    @property
    def name(self) -> str:
        return "Oscilloscope"

    @property
    def description(self) -> str:
        return "Time-domain waveform monitor."

    def run(self, args: argparse.Namespace):
        print("Oscilloscope running from CLI (not fully implemented)")

    def get_widget(self):
        return OscilloscopeWidget(self)

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        self.input_data = np.zeros((self.buffer_size, 2))
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
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
            
            outdata.fill(0)

        self.audio_engine.start_stream(callback, channels=2)

    def stop_analysis(self):
        if self.is_running:
            self.audio_engine.stop_stream()
            self.is_running = False

    def get_display_data(self, window_duration):
        """
        Get triggered data for display.
        window_duration: float, seconds of data to display
        """
        sample_rate = self.audio_engine.sample_rate
        required_samples = int(window_duration * sample_rate)
        
        if required_samples > self.buffer_size:
            required_samples = self.buffer_size
            
        data = self.input_data
        trigger_channel = data[:, self.trigger_source]
        
        # Simple Trigger Search
        # Look for crossing of trigger_level with correct slope
        # We search in the range [0, buffer_size - required_samples] to ensure we have enough data after trigger
        
        search_end = self.buffer_size - required_samples
        if search_end <= 0:
            # Buffer too small for requested window, just return what we have
            return data[-required_samples:]
            
        # Limit search to recent history to be responsive (e.g. last 50% of possible range)
        # But we need enough pre-trigger data? 
        # Usually oscilloscope shows trigger point at center or left. Let's put it at the left for now.
        
        # We search backwards from the end-required_samples to find the most recent trigger event
        # Or search forwards?
        # Let's search in the last 'search_window' samples
        search_window = 2048 # Limit search to avoid high CPU
        start_idx = max(0, search_end - search_window)
        
        subset = trigger_channel[start_idx:search_end]
        
        # Find crossings
        # Rising: previous < level <= current
        # Falling: previous > level >= current
        
        if self.trigger_slope == 'Rising':
            crossings = np.where((subset[:-1] < self.trigger_level) & (subset[1:] >= self.trigger_level))[0]
        else:
            crossings = np.where((subset[:-1] > self.trigger_level) & (subset[1:] <= self.trigger_level))[0]
            
        if len(crossings) > 0:
            # Pick the last one for most recent update
            trigger_idx = start_idx + crossings[-1] + 1 # +1 because crossing is between i and i+1
            return data[trigger_idx : trigger_idx + required_samples]
        else:
            # No trigger found
            if self.trigger_mode == 'Auto':
                # Return latest data
                return data[-required_samples:]
            else:
                # Normal mode: return None (keep last frame)
                return None

class OscilloscopeWidget(QWidget):
    def __init__(self, module: Oscilloscope):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.setInterval(30) # 30ms refresh

    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- Controls ---
        controls_group = QGroupBox("Oscilloscope Controls")
        controls_layout = QHBoxLayout()
        
        # Start/Stop
        self.toggle_btn = QPushButton("Start")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; } QPushButton:checked { background-color: #ffcccc; }")
        controls_layout.addWidget(self.toggle_btn)
        
        # Timebase
        controls_layout.addWidget(QLabel("Time/Div:"))
        self.timebase_combo = QComboBox()
        # Assuming 10 divisions horizontally. 
        # Options: 1ms, 2ms, 5ms, 10ms, 20ms, 50ms, 100ms
        self.timebase_options = {
            "1 ms": 0.001, "2 ms": 0.002, "5 ms": 0.005, 
            "10 ms": 0.01, "20 ms": 0.02, "50 ms": 0.05, "100 ms": 0.1
        }
        self.timebase_combo.addItems(self.timebase_options.keys())
        self.timebase_combo.setCurrentText("10 ms")
        self.timebase_combo.currentTextChanged.connect(self.on_timebase_changed)
        controls_layout.addWidget(self.timebase_combo)

        # Vertical Scale (Gain)
        controls_layout.addWidget(QLabel("Vertical Scale:"))
        self.vscale_combo = QComboBox()
        self.vscale_options = {
            "0.1x": 0.1, "0.2x": 0.2, "0.5x": 0.5, 
            "1.0x": 1.0, "2.0x": 2.0, "5.0x": 5.0, "10.0x": 10.0
        }
        self.vscale_combo.addItems(self.vscale_options.keys())
        self.vscale_combo.setCurrentText("1.0x")
        self.vscale_combo.currentTextChanged.connect(self.on_vscale_changed)
        controls_layout.addWidget(self.vscale_combo)
        
        # Channels
        self.chk_left = QCheckBox("L")
        self.chk_left.setChecked(True)
        self.chk_left.toggled.connect(lambda x: setattr(self.module, 'show_left', x))
        controls_layout.addWidget(self.chk_left)
        
        self.chk_right = QCheckBox("R")
        self.chk_right.setChecked(True)
        self.chk_right.toggled.connect(lambda x: setattr(self.module, 'show_right', x))
        controls_layout.addWidget(self.chk_right)

        # Trigger Controls
        trigger_group = QGroupBox("Trigger")
        trigger_layout = QHBoxLayout()
        
        trigger_layout.addWidget(QLabel("Source:"))
        self.trig_source_combo = QComboBox()
        self.trig_source_combo.addItems(["Left", "Right"])
        self.trig_source_combo.currentIndexChanged.connect(self.on_trig_source_changed)
        trigger_layout.addWidget(self.trig_source_combo)
        
        trigger_layout.addWidget(QLabel("Slope:"))
        self.trig_slope_combo = QComboBox()
        self.trig_slope_combo.addItems(["Rising", "Falling"])
        self.trig_slope_combo.currentTextChanged.connect(self.on_trig_slope_changed)
        trigger_layout.addWidget(self.trig_slope_combo)
        
        trigger_layout.addWidget(QLabel("Mode:"))
        self.trig_mode_combo = QComboBox()
        self.trig_mode_combo.addItems(["Auto", "Normal"])
        self.trig_mode_combo.currentTextChanged.connect(self.on_trig_mode_changed)
        trigger_layout.addWidget(self.trig_mode_combo)
        
        trigger_layout.addWidget(QLabel("Level:"))
        self.trig_level_spin = QDoubleSpinBox()
        self.trig_level_spin.setRange(-1.0, 1.0)
        self.trig_level_spin.setSingleStep(0.1)
        self.trig_level_spin.setValue(0.0)
        self.trig_level_spin.valueChanged.connect(self.on_trig_level_changed)
        trigger_layout.addWidget(self.trig_level_spin)
        
        trigger_group.setLayout(trigger_layout)
        controls_layout.addWidget(trigger_group)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # --- Plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude', units='V')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setYRange(-1.1, 1.1)
        self.plot_widget.showGrid(x=True, y=True)
        
        self.curve_l = self.plot_widget.plot(pen=pg.mkPen('#00ff00', width=2), name="Left")
        self.curve_r = self.plot_widget.plot(pen=pg.mkPen('#ff0000', width=2), name="Right")
        
        # Trigger Level Line
        self.trig_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine))
        self.trig_line.setPos(0.0)
        self.trig_line.sigPositionChanged.connect(self.on_trig_line_moved)
        self.plot_widget.addItem(self.trig_line)
        
        layout.addWidget(self.plot_widget)
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

    def on_timebase_changed(self, text):
        val = self.timebase_options[text]
        # We display 10 divisions usually. So window is 10 * val
        self.module.timebase = val * 10 
        self.plot_widget.setXRange(0, self.module.timebase)

    def on_vscale_changed(self, text):
        scale = self.vscale_options[text]
        # Scale 1.0x -> Range -1.1 to 1.1
        # Scale 2.0x -> Range -0.55 to 0.55 (Zoom In)
        # Scale 0.5x -> Range -2.2 to 2.2 (Zoom Out)
        
        base_range = 1.1
        new_range = base_range / scale
        self.plot_widget.setYRange(-new_range, new_range)

    def on_trig_source_changed(self, index):
        self.module.trigger_source = index

    def on_trig_slope_changed(self, text):
        self.module.trigger_slope = text

    def on_trig_mode_changed(self, text):
        self.module.trigger_mode = text

    def on_trig_level_changed(self, val):
        self.module.trigger_level = val
        self.trig_line.setPos(val)

    def on_trig_line_moved(self):
        val = self.trig_line.value()
        self.trig_level_spin.setValue(val)
        self.module.trigger_level = val

    def update_plot(self):
        if not self.module.is_running:
            return
            
        window_duration = self.module.timebase
        data = self.module.get_display_data(window_duration)
        
        if data is not None:
            # Create time axis
            t = np.linspace(0, window_duration, len(data))
            
            if self.module.show_left:
                self.curve_l.setData(t, data[:, 0])
            else:
                self.curve_l.setData([], [])
                
            if self.module.show_right:
                self.curve_r.setData(t, data[:, 1])
            else:
                self.curve_r.setData([], [])
