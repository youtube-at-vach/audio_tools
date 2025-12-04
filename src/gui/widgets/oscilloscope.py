import argparse
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QDoubleSpinBox, QStackedWidget, QFormLayout, QApplication)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.analysis import AudioCalc
from src.core.localization import tr

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
        
        # Filter Settings
        self.filter_type = 'None' # 'None', 'LPF', 'HPF', 'BPF'
        self.filter_cutoff = 1000.0 # For LPF/HPF
        self.filter_low = 1000.0 # For BPF
        self.filter_high = 2000.0 # For BPF
        
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
        main_layout = QHBoxLayout()
        
        # --- Left Panel (Display) ---
        left_layout = QVBoxLayout()
        
        # Measurements
        meas_group = QGroupBox(tr("Measurements"))
        meas_layout = QHBoxLayout()
        
        self.meas_l_label = QLabel("L: Vrms: 0.000 V  Vpp: 0.000 V")
        self.meas_l_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #00ff00;")
        meas_layout.addWidget(self.meas_l_label)
        
        self.meas_r_label = QLabel("R: Vrms: 0.000 V  Vpp: 0.000 V")
        self.meas_r_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #ff0000;")
        meas_layout.addWidget(self.meas_r_label)
        
        meas_layout.addStretch()
        meas_group.setLayout(meas_layout)
        left_layout.addWidget(meas_group)
        
        # Cursor Info
        self.cursor_info_label = QLabel("Cursors: Off")
        self.cursor_info_label.setStyleSheet("font-family: monospace; font-weight: bold; color: yellow;")
        left_layout.addWidget(self.cursor_info_label)
        
        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', tr('Amplitude'), units='V')
        self.plot_widget.setLabel('bottom', tr('Time'), units='s')
        self.plot_widget.setYRange(-1.1, 1.1)
        self.plot_widget.showGrid(x=True, y=True)
        
        self.curve_l = self.plot_widget.plot(pen=pg.mkPen('#00ff00', width=2), name="Left")
        self.curve_r = self.plot_widget.plot(pen=pg.mkPen('#ff0000', width=2), name="Right")
        
        # Cursors
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
        
        left_layout.addWidget(self.plot_widget)
        main_layout.addLayout(left_layout, stretch=1) # Give priority to plot
        
        # --- Right Panel (Controls) ---
        right_widget = QWidget()
        right_widget.setFixedWidth(250) # Fixed width for controls
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. General Controls
        gen_group = QGroupBox(tr("General"))
        gen_layout = QVBoxLayout()
        
        # Start/Stop
        self.toggle_btn = QPushButton(tr("Start"))
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        
        # Theme handling
        self.app = QApplication.instance()
        if hasattr(self.app, 'theme_manager'):
            self.app.theme_manager.theme_changed.connect(self.apply_theme)
            self.apply_theme(self.app.theme_manager.get_current_theme())
        else:
            self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; } QPushButton:checked { background-color: #ffcccc; }")
            
        gen_layout.addWidget(self.toggle_btn)
        
        # Timebase
        hbox_tb = QHBoxLayout()
        hbox_tb.addWidget(QLabel(tr("Time/Div:")))
        self.timebase_combo = QComboBox()
        self.timebase_options = {
            "10 us": 0.00001, "20 us": 0.00002, "50 us": 0.00005,
            "100 us": 0.0001, "200 us": 0.0002, "500 us": 0.0005,
            "1 ms": 0.001, "2 ms": 0.002, "5 ms": 0.005, 
            "10 ms": 0.01, "20 ms": 0.02, "50 ms": 0.05, "100 ms": 0.1
        }
        self.timebase_keys = list(self.timebase_options.keys())
        self.timebase_combo.addItems(self.timebase_keys)
        self.timebase_combo.setCurrentText("10 ms")
        self.timebase_combo.currentTextChanged.connect(self.on_timebase_changed)
        hbox_tb.addWidget(self.timebase_combo)
        gen_layout.addLayout(hbox_tb)
        
        # Timebase Slider
        self.timebase_slider = QSlider(Qt.Orientation.Horizontal)
        self.timebase_slider.setRange(0, len(self.timebase_keys) - 1)
        self.timebase_slider.valueChanged.connect(self.on_timebase_slider_changed)
        if "10 ms" in self.timebase_keys:
            self.timebase_slider.setValue(self.timebase_keys.index("10 ms"))
        gen_layout.addWidget(self.timebase_slider)
        
        gen_group.setLayout(gen_layout)
        right_layout.addWidget(gen_group)
        
        # 2. Vertical Controls
        vert_group = QGroupBox(tr("Vertical"))
        vert_layout = QVBoxLayout()
        
        hbox_scale = QHBoxLayout()
        hbox_scale.addWidget(QLabel(tr("Scale:")))
        self.vscale_combo = QComboBox()
        self.vscale_options = {
            "0.01x": 0.01, "0.02x": 0.02, "0.05x": 0.05,
            "0.1x": 0.1, "0.2x": 0.2, "0.5x": 0.5, 
            "1.0x": 1.0, "2.0x": 2.0, "5.0x": 5.0, "10.0x": 10.0,
            "20.0x": 20.0, "50.0x": 50.0, "100.0x": 100.0, 
            "200.0x": 200.0, "500.0x": 500.0, "1000.0x": 1000.0,
            "2000.0x": 2000.0, "5000.0x": 5000.0, "10000.0x": 10000.0
        }
        self.vscale_keys = list(self.vscale_options.keys())
        self.vscale_combo.addItems(self.vscale_keys)
        self.vscale_combo.setCurrentText("1.0x")
        self.vscale_combo.currentTextChanged.connect(self.on_vscale_changed)
        hbox_scale.addWidget(self.vscale_combo)
        vert_layout.addLayout(hbox_scale)
        
        self.vscale_slider = QSlider(Qt.Orientation.Horizontal)
        self.vscale_slider.setRange(0, len(self.vscale_keys) - 1)
        self.vscale_slider.valueChanged.connect(self.on_vscale_slider_changed)
        if "1.0x" in self.vscale_keys:
            self.vscale_slider.setValue(self.vscale_keys.index("1.0x"))
        vert_layout.addWidget(self.vscale_slider)
        
        hbox_ch = QHBoxLayout()
        self.chk_left = QCheckBox(tr("Left Ch"))
        self.chk_left.setChecked(True)
        self.chk_left.toggled.connect(lambda x: setattr(self.module, 'show_left', x))
        hbox_ch.addWidget(self.chk_left)
        
        self.chk_right = QCheckBox(tr("Right Ch"))
        self.chk_right.setChecked(True)
        self.chk_right.toggled.connect(lambda x: setattr(self.module, 'show_right', x))
        hbox_ch.addWidget(self.chk_right)
        vert_layout.addLayout(hbox_ch)
        
        vert_group.setLayout(vert_layout)
        right_layout.addWidget(vert_group)
        
        # 3. Trigger Controls
        trig_group = QGroupBox(tr("Trigger"))
        trig_layout = QVBoxLayout()
        
        hbox_src = QHBoxLayout()
        hbox_src.addWidget(QLabel(tr("Source:")))
        self.trig_source_combo = QComboBox()
        self.trig_source_combo.addItems([tr("Left"), tr("Right")])
        self.trig_source_combo.currentIndexChanged.connect(self.on_trig_source_changed)
        hbox_src.addWidget(self.trig_source_combo)
        trig_layout.addLayout(hbox_src)
        
        hbox_slope = QHBoxLayout()
        hbox_slope.addWidget(QLabel(tr("Slope:")))
        self.trig_slope_combo = QComboBox()
        self.trig_slope_combo.addItems([tr("Rising"), tr("Falling")])
        self.trig_slope_combo.currentTextChanged.connect(self.on_trig_slope_changed)
        hbox_slope.addWidget(self.trig_slope_combo)
        trig_layout.addLayout(hbox_slope)
        
        hbox_mode = QHBoxLayout()
        hbox_mode.addWidget(QLabel("Mode:"))
        self.trig_mode_combo = QComboBox()
        self.trig_mode_combo.addItems([tr("Auto"), tr("Normal")])
        self.trig_mode_combo.currentTextChanged.connect(self.on_trig_mode_changed)
        hbox_mode.addWidget(self.trig_mode_combo)
        trig_layout.addLayout(hbox_mode)
        
        hbox_lvl = QHBoxLayout()
        hbox_lvl.addWidget(QLabel(tr("Level:")))
        self.trig_level_spin = QDoubleSpinBox()
        self.trig_level_spin.setRange(-1.0, 1.0)
        self.trig_level_spin.setSingleStep(0.1)
        self.trig_level_spin.setValue(0.0)
        self.trig_level_spin.valueChanged.connect(self.on_trig_level_changed)
        hbox_lvl.addWidget(self.trig_level_spin)
        trig_layout.addLayout(hbox_lvl)
        
        trig_group.setLayout(trig_layout)
        right_layout.addWidget(trig_group)
        
        # 4. Tools
        tools_group = QGroupBox(tr("Tools"))
        tools_layout = QVBoxLayout()
        
        hbox_math = QHBoxLayout()
        hbox_math.addWidget(QLabel(tr("Math:")))
        self.math_combo = QComboBox()
        self.math_combo.addItems([tr('Off'), tr('A + B'), tr('A - B'), tr('A * B'), tr('A / B'), tr('Derivative'), tr('Integral')])
        self.math_combo.currentTextChanged.connect(self.on_math_changed)
        hbox_math.addWidget(self.math_combo)
        tools_layout.addLayout(hbox_math)
        
        self.chk_cursors = QCheckBox(tr("Enable Cursors"))
        self.chk_cursors.toggled.connect(self.on_cursors_toggled)
        tools_layout.addWidget(self.chk_cursors)
        
        tools_group.setLayout(tools_layout)
        right_layout.addWidget(tools_group)
        
        # 5. Filter Controls
        filter_group = QGroupBox(tr("Filter"))
        filter_layout = QVBoxLayout()
        
        hbox_ft = QHBoxLayout()
        hbox_ft.addWidget(QLabel(tr("Type:")))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(['None', 'LPF', 'HPF', 'BPF'])
        self.filter_combo.currentTextChanged.connect(self.on_filter_type_changed)
        hbox_ft.addWidget(self.filter_combo)
        filter_layout.addLayout(hbox_ft)
        
        self.filter_stack = QStackedWidget()
        
        # None Page
        self.filter_stack.addWidget(QWidget())
        
        # LPF/HPF Page
        lpf_widget = QWidget()
        lpf_layout = QFormLayout()
        lpf_layout.setContentsMargins(0,0,0,0)
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(10, 24000)
        self.cutoff_spin.setValue(self.module.filter_cutoff)
        self.cutoff_spin.valueChanged.connect(lambda v: setattr(self.module, 'filter_cutoff', v))
        lpf_layout.addRow(tr("Cutoff (Hz):"), self.cutoff_spin)
        lpf_widget.setLayout(lpf_layout)
        self.filter_stack.addWidget(lpf_widget)
        
        # BPF Page
        bpf_widget = QWidget()
        bpf_layout = QFormLayout()
        bpf_layout.setContentsMargins(0,0,0,0)
        self.bpf_low_spin = QDoubleSpinBox()
        self.bpf_low_spin.setRange(10, 24000)
        self.bpf_low_spin.setValue(self.module.filter_low)
        self.bpf_low_spin.valueChanged.connect(lambda v: setattr(self.module, 'filter_low', v))
        bpf_layout.addRow(tr("Low (Hz):"), self.bpf_low_spin)
        
        self.bpf_high_spin = QDoubleSpinBox()
        self.bpf_high_spin.setRange(10, 24000)
        self.bpf_high_spin.setValue(self.module.filter_high)
        self.bpf_high_spin.valueChanged.connect(lambda v: setattr(self.module, 'filter_high', v))
        bpf_layout.addRow(tr("High (Hz):"), self.bpf_high_spin)
        bpf_widget.setLayout(bpf_layout)
        self.filter_stack.addWidget(bpf_widget)
        
        filter_layout.addWidget(self.filter_stack)
        filter_group.setLayout(filter_layout)
        right_layout.addWidget(filter_group)
        
        # Math Curve
        # Create a new ViewBox for Math
        self.math_view = pg.ViewBox()
        self.plot_widget.plotItem.scene().addItem(self.math_view)
        # Link X axis
        self.math_view.setXLink(self.plot_widget.plotItem)
        
        # Add Right Axis
        # Remove default right axis if it exists, then add our custom one
        if self.plot_widget.plotItem.getAxis('right') is not None:
            self.plot_widget.plotItem.layout.removeItem(self.plot_widget.plotItem.getAxis('right'))
        self.axis_math = pg.AxisItem('right')
        self.axis_math.linkToView(self.math_view)
        self.axis_math.setLabel('Math', color='#ffffff')
        self.plot_widget.plotItem.layout.addItem(self.axis_math, 2, 2) # Row 2, Col 2 for right axis
        self.axis_math.hide() # Hide by default
        
        # Update View Geometry on resize
        self.plot_widget.plotItem.vb.sigResized.connect(self.update_math_view_geometry)
        
        self.curve_math = pg.PlotCurveItem(pen=pg.mkPen('w', width=2, style=Qt.PenStyle.DotLine), name="Math")
        self.math_view.addItem(self.curve_math)
        
        right_layout.addStretch()
        main_layout.addWidget(right_widget)
        
        self.setLayout(main_layout)

    def update_math_view_geometry(self):
        # This function ensures the math_view's geometry matches the main plot's viewbox
        # so that the linked X-axis works correctly and the math curve overlays properly.
        self.math_view.setGeometry(self.plot_widget.plotItem.vb.sceneBoundingRect())
        # This line is crucial for the linked X-axis to update its range when the main plot's X-axis changes.
        self.math_view.linkedViewChanged(self.plot_widget.plotItem.vb, self.math_view.XAxis)

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText(tr("Stop"))
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText(tr("Start"))

    def on_timebase_changed(self, text):
        val = self.timebase_options[text]
        # We display 10 divisions usually. So window is 10 * val
        self.module.timebase = val * 10 
        self.plot_widget.setXRange(0, self.module.timebase)
        
        # Sync slider
        if text in self.timebase_keys:
            idx = self.timebase_keys.index(text)
            if self.timebase_slider.value() != idx:
                self.timebase_slider.setValue(idx)

    def on_timebase_slider_changed(self, idx):
        if 0 <= idx < len(self.timebase_keys):
            key = self.timebase_keys[idx]
            if self.timebase_combo.currentText() != key:
                self.timebase_combo.setCurrentText(key)

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
        if val == 'Off':
            self.axis_math.hide()
            self.curve_math.setData([], []) # Clear math curve when off
        else:
            self.axis_math.show()
            self.axis_math.setLabel(f"Math ({val})", color='#ffffff')
        
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

    def on_filter_type_changed(self, text):
        self.module.filter_type = text
        if text == 'None':
            self.filter_stack.setCurrentIndex(0)
        elif text == 'BPF':
            self.filter_stack.setCurrentIndex(2)
        else:
            self.filter_stack.setCurrentIndex(1) # LPF/HPF share same widget

    def update_plot(self):
        if not self.module.is_running:
            return
            
        window_duration = self.module.timebase
        data = self.module.get_display_data(window_duration)
        
        if data is not None:
            # Apply Filter if enabled
            sr = self.module.audio_engine.sample_rate
            if self.module.filter_type != 'None':
                # Filter both channels
                # Note: filtering short segments might have transient artifacts at edges.
                # Ideally we filter the continuous buffer, but for visualization this might be acceptable
                # if the segment is long enough or if we accept the edge effects.
                # get_display_data returns a copy, so we can modify it.
                
                # To reduce edge artifacts, we could fetch a bit more data, filter, then trim?
                # get_display_data logic is complex with trigger.
                # Let's try direct filtering first.
                
                if self.module.filter_type == 'LPF':
                    data[:, 0] = AudioCalc.lowpass_filter(data[:, 0], sr, self.module.filter_cutoff)
                    data[:, 1] = AudioCalc.lowpass_filter(data[:, 1], sr, self.module.filter_cutoff)
                elif self.module.filter_type == 'HPF':
                    data[:, 0] = AudioCalc.highpass_filter(data[:, 0], sr, self.module.filter_cutoff)
                    data[:, 1] = AudioCalc.highpass_filter(data[:, 1], sr, self.module.filter_cutoff)
                elif self.module.filter_type == 'BPF':
                    data[:, 0] = AudioCalc.bandpass_filter(data[:, 0], sr, self.module.filter_low, self.module.filter_high)
                    data[:, 1] = AudioCalc.bandpass_filter(data[:, 1], sr, self.module.filter_low, self.module.filter_high)

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
                math_data = None
                
                # A = Left, B = Right
                A = data[:, 0]
                B = data[:, 1]
                
                mode = self.module.math_mode
                
                if mode == 'A + B':
                    math_data = A + B
                elif mode == 'A - B':
                    math_data = A - B
                elif mode == 'A * B':
                    math_data = A * B
                elif mode == 'A / B':
                    # Avoid division by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        math_data = np.divide(A, B)
                        math_data[~np.isfinite(math_data)] = 0 # Replace inf/nan with 0
                elif mode == 'Derivative': # Derivative of A (Left)
                    dt = t[1] - t[0] if len(t) > 1 else 1e-6
                    math_data = np.gradient(A, dt)
                elif mode == 'Integral': # Integral of A (Left)
                    dt = t[1] - t[0] if len(t) > 1 else 1e-6
                    math_data = np.cumsum(A) * dt
                    math_data = math_data - np.mean(math_data)
                    
                if math_data is not None:
                    self.curve_math.setData(t, math_data)
                    # Auto-scale Math View
                    mn, mx = np.min(math_data), np.max(math_data)
                    if mn == mx:
                        mn -= 0.1
                        mx += 0.1
                    padding = (mx - mn) * 0.1
                    self.math_view.setYRange(mn - padding, mx + padding)
                else:
                    self.curve_math.setData([], [])
            else:
                self.curve_math.setData([], [])
                
            # Update cursor info if they are on (to update voltage readings)
            if self.chk_cursors.isChecked():
                self.update_cursor_info()

    def apply_theme(self, theme_name):
        if theme_name == 'system' and hasattr(self.app, 'theme_manager'):
            theme_name = self.app.theme_manager.get_effective_theme()
            
        if theme_name == 'dark':
            # Dark Theme
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; border: 1px solid #555; border-radius: 4px; padding: 5px; }"
                "QPushButton:checked { background-color: #c62828; color: white; border: 1px solid #555; border-radius: 4px; padding: 5px; }"
                "QPushButton:hover { background-color: #388e3c; }"
                "QPushButton:checked:hover { background-color: #d32f2f; }"
            )
            self.meas_l_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #00ff00;")
            self.meas_r_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #ff0000;")
            self.cursor_info_label.setStyleSheet("font-family: monospace; font-weight: bold; color: yellow;")
        else:
            # Light Theme
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #ccffcc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 5px; }"
                "QPushButton:checked { background-color: #ffcccc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 5px; }"
                "QPushButton:hover { background-color: #bbfebb; }"
                "QPushButton:checked:hover { background-color: #ffbbbb; }"
            )
            self.meas_l_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #008800;")
            self.meas_r_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #cc0000;")
            self.cursor_info_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #888800;")
