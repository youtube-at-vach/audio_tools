import argparse
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QGroupBox, QSlider, QProgressBar, QCheckBox, QApplication)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class Goniometer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        
        # Settings
        self.buffer_size = 4096 # Increased for better density in phosphor mode
        self.gain = 1.0
        self.auto_gain = False
        self.decay = 0.90 # Persistence factor (0-1)
        self.display_mode = 'Line' # 'Line', 'Phosphor'
        self.color_palette = 'Green' # 'Green', 'Fire', 'Ice', 'Rainbow'
        
        # State
        self.audio_buffer = np.zeros((self.buffer_size, 2))
        self.correlation = 0.0
        self.callback_id = None
        
        # Phosphor state
        self.heatmap_size = 400
        self.heatmap = np.zeros((self.heatmap_size, self.heatmap_size))
        
    @property
    def name(self) -> str:
        return "Goniometer"

    @property
    def description(self) -> str:
        return "Stereo image visualizer (Lissajous) and Phase Correlation."

    def run(self, args: argparse.Namespace):
        print("Goniometer CLI not implemented")

    def get_widget(self):
        return GoniometerWidget(self)

    def start_analysis(self):
        if self.is_running: return
        self.is_running = True
        self.audio_buffer = np.zeros((self.buffer_size, 2))
        self.heatmap = np.zeros((self.heatmap_size, self.heatmap_size))
        self.callback_id = self.audio_engine.register_callback(self._callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def _callback(self, indata, outdata, frames, time, status):
        if status: print(status)
        
        # Get stereo data
        if indata.shape[1] >= 2:
            new_data = indata[:, :2]
        else:
            # Mono input - duplicate to both channels
            new_data = np.column_stack((indata[:, 0], indata[:, 0]))
            
        # Update buffer (Roll)
        if frames >= self.buffer_size:
            self.audio_buffer[:] = new_data[-self.buffer_size:]
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -frames, axis=0)
            self.audio_buffer[-frames:] = new_data
            
        # Calculate Correlation (Instantaneous for this block)
        l = new_data[:, 0]
        r = new_data[:, 1]
        
        dot = np.sum(l * r)
        mag_l = np.sum(l**2)
        mag_r = np.sum(r**2)
        
        if mag_l > 1e-9 and mag_r > 1e-9:
            self.correlation = dot / np.sqrt(mag_l * mag_r)
        elif mag_l < 1e-9 and mag_r < 1e-9:
            self.correlation = 0.0 # Silence
        else:
            self.correlation = 0.0
            
        outdata.fill(0)

class GoniometerWidget(QWidget):
    def __init__(self, module: Goniometer):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.setInterval(30) # 30 FPS
        
        self.update_palette()

    def init_ui(self):
        layout = QHBoxLayout()
        
        # --- Left: Display ---
        display_layout = QVBoxLayout()
        
        # 1. Goniometer Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setXRange(-1.1, 1.1)
        self.plot_widget.setYRange(-1.1, 1.1)
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground('#111')
        
        # Add background lines for reference
        self.plot_widget.addItem(pg.InfiniteLine(pos=(0,0), angle=45, pen=pg.mkPen('#444', style=Qt.PenStyle.DashLine)))
        self.plot_widget.addItem(pg.InfiniteLine(pos=(0,0), angle=-45, pen=pg.mkPen('#444', style=Qt.PenStyle.DashLine)))
        
        # Items
        # Line Trace
        self.trace = self.plot_widget.plot(pen=pg.mkPen(color=(0, 255, 200, 150), width=1))
        
        # Phosphor Image
        self.img_item = pg.ImageItem()
        self.img_item.setImage(self.module.heatmap.T, autoLevels=False, levels=[0, 50]) # Set initial data and levels
        self.img_item.setRect(pg.QtCore.QRectF(-1.1, -1.1, 2.2, 2.2))
        self.plot_widget.addItem(self.img_item)
        self.img_item.setZValue(-1) # Behind grid? Or in front?
        
        display_layout.addWidget(self.plot_widget, stretch=1)
        
        # 2. Correlation Meter
        corr_layout = QHBoxLayout()
        corr_layout.addWidget(QLabel("-1"))
        
        self.corr_bar = QProgressBar()
        self.corr_bar.setRange(-100, 100) # Map -1.0..1.0 to -100..100
        self.corr_bar.setTextVisible(True)
        self.corr_bar.setFormat("%v") 
        self.corr_bar.setFixedHeight(20)
        corr_layout.addWidget(self.corr_bar, stretch=1)
        
        corr_layout.addWidget(QLabel("+1"))
        display_layout.addLayout(corr_layout)
        
        layout.addLayout(display_layout, stretch=3)
        
        # --- Right: Controls ---
        controls_group = QGroupBox(tr("Controls"))
        controls_layout = QVBoxLayout()
        
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
            
        controls_layout.addWidget(self.toggle_btn)
        
        # Display Mode
        controls_layout.addWidget(QLabel(tr("Display Mode:")))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Line', 'Phosphor'])
        self.mode_combo.setCurrentText(self.module.display_mode)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        controls_layout.addWidget(self.mode_combo)
        
        # Color Palette
        controls_layout.addWidget(QLabel(tr("Color Palette:")))
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(['Green', 'Fire', 'Ice', 'Rainbow'])
        self.palette_combo.setCurrentText(self.module.color_palette)
        self.palette_combo.currentTextChanged.connect(self.on_palette_changed)
        controls_layout.addWidget(self.palette_combo)
        
        # Persistence (Decay)
        controls_layout.addWidget(QLabel(tr("Persistence:")))
        self.decay_slider = QSlider(Qt.Orientation.Horizontal)
        self.decay_slider.setRange(0, 99) # 0.0 to 0.99
        self.decay_slider.setValue(int(self.module.decay * 100))
        self.decay_slider.valueChanged.connect(self.on_decay_changed)
        controls_layout.addWidget(self.decay_slider)
        
        # Gain
        controls_layout.addWidget(QLabel(tr("Gain:")))
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(1, 100) # 0.1x to 10.0x
        self.gain_slider.setValue(10) # 1.0x
        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        controls_layout.addWidget(self.gain_slider)
        
        self.gain_label = QLabel("1.0x")
        controls_layout.addWidget(self.gain_label, alignment=Qt.AlignmentFlag.AlignRight)
        
        # Auto Gain
        self.auto_gain_chk = QCheckBox(tr("Auto Gain"))
        self.auto_gain_chk.toggled.connect(lambda x: setattr(self.module, 'auto_gain', x))
        controls_layout.addWidget(self.auto_gain_chk)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        
        layout.addWidget(controls_group, stretch=1)
        
        self.setLayout(layout)
        
        # Init visibility
        self.on_mode_changed(self.module.display_mode)

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText(tr("Stop"))
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText(tr("Start"))

    def on_gain_changed(self, val):
        gain = val / 10.0
        self.module.gain = gain
        self.gain_label.setText(f"{gain:.1f}x")
        
    def on_mode_changed(self, text):
        self.module.display_mode = text
        if text == 'Line':
            self.trace.setVisible(True)
            self.img_item.setVisible(False)
        else:
            self.trace.setVisible(False)
            self.img_item.setVisible(True)
            
    def on_palette_changed(self, text):
        self.module.color_palette = text
        self.update_palette()
        
    def on_decay_changed(self, val):
        self.module.decay = val / 100.0

    def update_palette(self):
        # Define colormaps
        # 256 colors
        pos = np.linspace(0, 1, 256)
        colors = np.zeros((256, 4), dtype=np.ubyte)
        
        name = self.module.color_palette
        
        if name == 'Green':
            # Black -> Green -> White
            # 0.0 -> (0,0,0)
            # 0.8 -> (0,255,0)
            # 1.0 -> (255,255,255)
            for i in range(256):
                val = i / 255.0
                if val < 0.8:
                    g = int((val / 0.8) * 255)
                    colors[i] = [0, g, 0, 255]
                else:
                    rem = (val - 0.8) / 0.2
                    colors[i] = [int(rem*255), 255, int(rem*255), 255]
                    
        elif name == 'Fire':
            # Black -> Red -> Yellow -> White
            for i in range(256):
                val = i / 255.0
                if val < 0.33:
                    r = int((val / 0.33) * 255)
                    colors[i] = [r, 0, 0, 255]
                elif val < 0.66:
                    g = int(((val - 0.33) / 0.33) * 255)
                    colors[i] = [255, g, 0, 255]
                else:
                    b = int(((val - 0.66) / 0.34) * 255)
                    colors[i] = [255, 255, b, 255]
                    
        elif name == 'Ice':
            # Black -> Blue -> Cyan -> White
            for i in range(256):
                val = i / 255.0
                if val < 0.5:
                    b = int((val / 0.5) * 255)
                    colors[i] = [0, 0, b, 255]
                else:
                    g = int(((val - 0.5) / 0.5) * 255)
                    colors[i] = [0, g, 255, 255]
                    
        elif name == 'Rainbow':
            # HSL rainbow
            import colorsys
            for i in range(256):
                val = i / 255.0
                # Hue goes 0..1, Sat 1, Val 1
                # But we want intensity to map to brightness too?
                # Usually rainbow maps value to hue.
                # Let's map low intensity to blue, high to red?
                # Or just standard heatmap: Blue -> Cyan -> Green -> Yellow -> Red
                h = (1.0 - val) * 0.66 # Blue(0.66) to Red(0)
                r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
                # Apply intensity fade for low values
                alpha = min(1.0, val * 2)
                colors[i] = [int(r*255*alpha), int(g*255*alpha), int(b*255*alpha), 255]

        # Apply to ImageItem
        # pyqtgraph colormap
        # We can just set lookup table
        self.img_item.setLookupTable(colors)
        
        # Also update line trace color if in line mode?
        # Maybe just keep line trace simple.
        if name == 'Green':
            self.trace.setPen(pg.mkPen(color=(0, 255, 0, 200), width=1))
        elif name == 'Fire':
            self.trace.setPen(pg.mkPen(color=(255, 100, 0, 200), width=1))
        elif name == 'Ice':
            self.trace.setPen(pg.mkPen(color=(0, 200, 255, 200), width=1))
        elif name == 'Rainbow':
            self.trace.setPen(pg.mkPen(color=(200, 0, 255, 200), width=1))

    def update_display(self):
        if not self.module.is_running: return
        
        # Get data
        data = self.module.audio_buffer
        l = data[:, 0]
        r = data[:, 1]
        
        # Apply Gain
        if self.module.auto_gain:
            peak = np.max(np.abs(data))
            if peak > 1e-6:
                gain = 0.8 / peak # Target 0.8 peak
                self.module.gain = self.module.gain * 0.9 + gain * 0.1
        
        l = l * self.module.gain
        r = r * self.module.gain
        
        # Transform to M/S (Rotated 45 deg)
        # X = (R - L) * 0.707 (Side, L on left)
        # Y = (L + R) * 0.707 (Mid)
        
        m = (l + r) * 0.707
        s = (r - l) * 0.707
        
        if self.module.display_mode == 'Line':
            self.trace.setData(s, m)
        else:
            # Phosphor Mode
            # Decay existing
            self.module.heatmap *= self.module.decay
            
            # Bin new data
            # Map s, m (-1.1..1.1) to (0..size)
            # We use histogram2d
            
            # Range
            rng = [[-1.1, 1.1], [-1.1, 1.1]]
            
            # We only bin the NEWEST frames to avoid double counting if buffer overlaps?
            # self.module.audio_buffer contains the last N frames.
            # If we run at 30FPS, we get ~1600 frames per update (48k/30).
            # Buffer is 4096. So we are re-binning some old data?
            # Ideally we should only bin new data.
            # But for simplicity, let's just bin the whole buffer but weight it?
            # Or just bin the whole buffer and add to heatmap?
            # If we add whole buffer every frame, the intensity will explode if decay is slow.
            # Better approach:
            # Heatmap represents "Energy density".
            # We can just compute the histogram of the CURRENT buffer, and blend it with the previous heatmap.
            # Heatmap = Heatmap * Decay + Current_Histogram * (1-Decay)?
            # Or Heatmap = Heatmap * Decay + Current_Histogram.
            # Let's try accumulation.
            
            h, _, _ = np.histogram2d(s, m, bins=self.module.heatmap_size, range=rng)
            
            # Log compression for better dynamic range?
            # h = np.log1p(h)
            
            self.module.heatmap += h * 0.5 # Scale factor
            
            # Clamp?
            # self.module.heatmap = np.clip(self.module.heatmap, 0, 100)
            
            # Normalize for display?
            # ImageItem expects 0..1 or scaled values if LUT is used.
            # Our LUT is 0..255 indices if we pass int, or 0..1 floats.
            # Let's pass raw values and set levels.
            
            self.img_item.setImage(self.module.heatmap.T, autoLevels=False)
            self.img_item.setLevels([0, 50]) # Adjust max level for sensitivity
            
        
        # Update Correlation
        corr = self.module.correlation
        val = int(corr * 100)
        self.corr_bar.setValue(val)
        self.corr_bar.setFormat(f"{corr:.2f}")
        
        if corr >= 0:
            color = "#00ff00"
        else:
            color = "#ff0000"
            
        self.corr_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

    def apply_theme(self, theme_name):
        if theme_name == 'system' and hasattr(self.app, 'theme_manager'):
            theme_name = self.app.theme_manager.get_effective_theme()
            
        if theme_name == 'dark':
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; border: 1px solid #555; border-radius: 4px; padding: 5px; }"
                "QPushButton:checked { background-color: #c62828; color: white; border: 1px solid #555; border-radius: 4px; padding: 5px; }"
                "QPushButton:hover { background-color: #388e3c; }"
                "QPushButton:checked:hover { background-color: #d32f2f; }"
            )
        else:
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #ccffcc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 5px; }"
                "QPushButton:checked { background-color: #ffcccc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 5px; }"
                "QPushButton:hover { background-color: #bbfebb; }"
                "QPushButton:checked:hover { background-color: #ffbbbb; }"
            )
