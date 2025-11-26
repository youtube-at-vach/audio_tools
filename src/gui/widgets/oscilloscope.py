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
        
        # Math Mode
        self.math_mode = 'Off' # 'Off', 'Derivative', 'Integral'
        
        self.callback_id = None
        
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

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
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
            "10 us": 0.00001, "20 us": 0.00002, "50 us": 0.00005,
            "100 us": 0.0001, "200 us": 0.0002, "500 us": 0.0005,
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
            "1.0x": 1.0, "2.0x": 2.0, "5.0x": 5.0, "10.0x": 10.0,
            "20.0x": 20.0, "50.0x": 50.0, "100.0x": 100.0, 
            "200.0x": 200.0, "500.0x": 500.0, "1000.0x": 1000.0
        }
        self.vscale_keys = list(self.vscale_options.keys())
        self.vscale_combo.addItems(self.vscale_keys)
        self.vscale_combo.setCurrentText("1.0x")
        self.vscale_combo.currentTextChanged.connect(self.on_vscale_changed)
        controls_layout.addWidget(self.vscale_combo)
        
        # Vertical Scale Slider
        self.vscale_slider = QSlider(Qt.Orientation.Horizontal)
        self.vscale_slider.setRange(0, len(self.vscale_keys) - 1)
        self.vscale_slider.setFixedWidth(100)
        self.vscale_slider.valueChanged.connect(self.on_vscale_slider_changed)
        # Set initial value
        if "1.0x" in self.vscale_keys:
            self.vscale_slider.setValue(self.vscale_keys.index("1.0x"))
        controls_layout.addWidget(self.vscale_slider)
        
        # Channels
        self.chk_left = QCheckBox("L")
        self.chk_left.setChecked(True)
        self.chk_left.toggled.connect(lambda x: setattr(self.module, 'show_left', x))
        controls_layout.addWidget(self.chk_left)
        
        self.chk_right = QCheckBox("R")
        self.chk_right.setChecked(True)
        self.chk_right.toggled.connect(lambda x: setattr(self.module, 'show_right', x))
        controls_layout.addWidget(self.chk_right)
        
        # Math Mode
        controls_layout.addWidget(QLabel("Math:"))
        self.math_combo = QComboBox()
        self.math_combo.addItems(['Off', 'Derivative', 'Integral'])
        self.math_combo.currentTextChanged.connect(self.on_math_changed)
        controls_layout.addWidget(self.math_combo)
        
        # Cursors
        self.chk_cursors = QCheckBox("Cursors")
        self.chk_cursors.toggled.connect(self.on_cursors_toggled)
        controls_layout.addWidget(self.chk_cursors)

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
        
        # --- Measurements ---
        meas_group = QGroupBox("Measurements")
        meas_layout = QHBoxLayout()
        
        self.meas_l_label = QLabel("L: Vrms: 0.000 V  Vpp: 0.000 V")
        self.meas_l_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #00ff00;")
        meas_layout.addWidget(self.meas_l_label)
        
        self.meas_r_label = QLabel("R: Vrms: 0.000 V  Vpp: 0.000 V")
        self.meas_r_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #ff0000;")
        meas_layout.addWidget(self.meas_r_label)
        
        meas_layout.addStretch()
        meas_group.setLayout(meas_layout)
        layout.addWidget(meas_group)
        
        # --- Cursor Info ---
        self.cursor_info_label = QLabel("Cursors: Off")
        self.cursor_info_label.setStyleSheet("font-family: monospace; font-weight: bold; color: yellow;")
        layout.addWidget(self.cursor_info_label)
        
        # --- Plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude', units='V')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setYRange(-1.1, 1.1)
        self.plot_widget.showGrid(x=True, y=True)
        
        self.curve_l = self.plot_widget.plot(pen=pg.mkPen('#00ff00', width=2), name="Left")
        self.curve_r = self.plot_widget.plot(pen=pg.mkPen('#ff0000', width=2), name="Right")
        
        # # Trigger Level Line
        # self.trig_line.sigPositionChanged.connect(self.on_trig_line_moved)
        # self.plot_widget.addItem(self.trig_line)
        
        # Cursors (Hidden by default)
        self.cursor_1 = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('c', width=1), label='C1', labelOpts={'position':0.1})
        self.cursor_2 = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('m', width=1), label='C2', labelOpts={'position':0.1})
        
        self.cursor_1.sigPositionChanged.connect(self.update_cursor_info)
        self.cursor_2.sigPositionChanged.connect(self.update_cursor_info)
        
        self.plot_widget.addItem(self.cursor_1)
        self.plot_widget.addItem(self.cursor_2)
        self.cursor_1.setVisible(False)
        self.cursor_2.setVisible(False)
        
        # Math Curve
        self.curve_math = self.plot_widget.plot(pen=pg.mkPen('w', width=2, style=Qt.PenStyle.DotLine), name="Math")
        
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

    def on_vscale_slider_changed(self, idx):
        if 0 <= idx < len(self.vscale_keys):
            key = self.vscale_keys[idx]
            # Block signals to prevent feedback loop if needed, 
            # but usually setCurrentText emits signal which calls on_vscale_changed
            # which sets slider value again. To avoid loop:
            if self.vscale_combo.currentText() != key:
                self.vscale_combo.setCurrentText(key)

    def on_vscale_changed(self, text):
        if text not in self.vscale_options:
            return
            
        scale = self.vscale_options[text]
        
        # Sync slider
        if text in self.vscale_keys:
            idx = self.vscale_keys.index(text)
            if self.vscale_slider.value() != idx:
                self.vscale_slider.setValue(idx)
        
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
        # self.trig_line.setPos(val)

        self.module.trigger_level = val
        
    def on_math_changed(self, val):
        self.module.math_mode = val
        
    def on_cursors_toggled(self, checked):
        self.cursor_1.setVisible(checked)
        self.cursor_2.setVisible(checked)
        if checked:
            # Initialize positions if needed
            if self.cursor_1.value() == 0 and self.cursor_2.value() == 0:
                x_range = self.plot_widget.viewRange()[0]
                center = (x_range[1] + x_range[0]) / 2
                width = x_range[1] - x_range[0]
                self.cursor_1.setPos(center - width/4)
                self.cursor_2.setPos(center + width/4)
        self.update_cursor_info()

    def update_cursor_info(self):
        if not self.chk_cursors.isChecked():
            self.cursor_info_label.setText("Cursors: Off")
            return
            
        t1 = self.cursor_1.value()
        t2 = self.cursor_2.value()
        dt = t2 - t1
        freq = 1.0 / abs(dt) if dt != 0 else 0.0
        
        # Get Voltage at cursors (Interpolate)
        # We need the current data to do this. 
        # Since this is called on move, we might not have the exact latest data object here easily 
        # without storing it. Let's store the latest displayed data in self.latest_data
        
        v1_str = ""
        v2_str = ""
        dv_str = ""
        
        if hasattr(self, 'latest_data') and self.latest_data is not None:
            data = self.latest_data
            t = self.latest_t
            
            # Interpolate
            # Assuming Channel 0 (Left) is primary for cursor measurement if both active, 
            # or use Trigger Source? Let's use Trigger Source or just Left.
            # Let's use the first visible channel.
            
            target_data = None
            if self.module.show_left:
                target_data = data[:, 0]
            elif self.module.show_right:
                target_data = data[:, 1]
                
            if target_data is not None:
                v1 = np.interp(t1, t, target_data)
                v2 = np.interp(t2, t, target_data)
                dv = v2 - v1
                v1_str = f"V1: {v1:.3f}V"
                v2_str = f"V2: {v2:.3f}V"
                dv_str = f"dV: {dv:.3f}V"
        
        self.cursor_info_label.setText(f"T1: {t1*1000:.2f}ms {v1_str} | T2: {t2*1000:.2f}ms {v2_str} | dT: {dt*1000:.2f}ms ({freq:.1f}Hz) | {dv_str}")

    def update_plot(self):
        if not self.module.is_running:
            return
            
        window_duration = self.module.timebase
        data = self.module.get_display_data(window_duration)
        
        if data is not None:
            # Measurements
            l_data = data[:, 0]
            r_data = data[:, 1]
            
            l_rms = np.sqrt(np.mean(l_data**2))
            l_vpp = np.max(l_data) - np.min(l_data)
            
            r_rms = np.sqrt(np.mean(r_data**2))
            r_vpp = np.max(r_data) - np.min(r_data)
            
            self.meas_l_label.setText(f"L: Vrms: {l_rms:.3f} V  Vpp: {l_vpp:.3f} V")
            self.meas_r_label.setText(f"R: Vrms: {r_rms:.3f} V  Vpp: {r_vpp:.3f} V")

            # Create time axis
            t = np.linspace(0, window_duration, len(data))
            
            # Store for cursor interpolation
            self.latest_data = data
            self.latest_t = t
            
            if self.module.show_left:
                self.curve_l.setData(t, data[:, 0])
            else:
                self.curve_l.setData([], [])
                
            if self.module.show_right:
                self.curve_r.setData(t, data[:, 1])
            else:
                self.curve_r.setData([], [])
                
            # Math Processing
            if self.module.math_mode != 'Off':
                # Use Left channel for Math for now, or sum?
                # Usually Math is Ch1 - Ch2 or similar.
                # Here we do Diff/Int of Ch1 (Left) or Ch2 (Right) if Left is off.
                
                source_data = None
                if self.module.show_left:
                    source_data = data[:, 0]
                elif self.module.show_right:
                    source_data = data[:, 1]
                    
                if source_data is not None:
                    if self.module.math_mode == 'Derivative':
                        # diff = dV / dt
                        # dt = window_duration / len(data)
                        dt = t[1] - t[0] if len(t) > 1 else 1e-6
                        math_data = np.gradient(source_data, dt)
                        # Scale for visibility? Derivative of sine is cosine * w.
                        # Can be large.
                        
                    elif self.module.math_mode == 'Integral':
                        # int = sum(V * dt)
                        dt = t[1] - t[0] if len(t) > 1 else 1e-6
                        # cumulative trapezoid
                        math_data = np.cumsum(source_data) * dt
                        # Remove DC offset from integral (drift)
                        math_data = math_data - np.mean(math_data)
                        
                    self.curve_math.setData(t, math_data)
                else:
                    self.curve_math.setData([], [])
            else:
                self.curve_math.setData([], [])
                
            # Update cursor info if they are on (to update voltage readings)
            if self.chk_cursors.isChecked():
                self.update_cursor_info()
