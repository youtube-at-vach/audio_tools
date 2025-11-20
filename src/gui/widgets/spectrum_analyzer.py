import argparse
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QFormLayout)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class SpectrumAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 4096
        self.input_data = np.zeros(self.buffer_size)
        
        # Analysis parameters
        self.window_type = 'hanning'
        self.averaging = 0.0 # 0.0 to 0.95
        self.peak_hold = False
        self.octave_smoothing = 'None' # None, 1/1, 1/3, 1/6, 1/12, 1/24
        
        # State
        self._avg_magnitude = None
        self._peak_magnitude = None
        self.overall_rms = 0.0

    @property
    def name(self) -> str:
        return "Spectrum Analyzer"

    @property
    def description(self) -> str:
        return "Real-time frequency spectrum analysis."

    def run(self, args: argparse.Namespace):
        print("Spectrum Analyzer running from CLI (not fully implemented)")

    def get_widget(self):
        return SpectrumAnalyzerWidget(self)

    def set_buffer_size(self, size):
        self.buffer_size = size
        self.input_data = np.zeros(self.buffer_size)
        self._avg_magnitude = None
        self._peak_magnitude = None

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        self._avg_magnitude = None
        self._peak_magnitude = None
        self.overall_rms = 0.0
        self.input_data = np.zeros(self.buffer_size)
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
            # Shift buffer and append new data
            # Check channels
            if indata.shape[1] >= 2:
                # Average stereo to mono for analysis
                new_data = np.mean(indata, axis=1)
            else:
                # Mono
                new_data = indata[:, 0]
            
            # Efficient ring buffer or just roll
            if len(new_data) > self.buffer_size:
                self.input_data[:] = new_data[-self.buffer_size:]
            else:
                self.input_data = np.roll(self.input_data, -len(new_data))
                self.input_data[-len(new_data):] = new_data
            
            outdata.fill(0)

        self.audio_engine.start_stream(callback, channels=2)

    def stop_analysis(self):
        if self.is_running:
            self.audio_engine.stop_stream()
            self.is_running = False

class SpectrumAnalyzerWidget(QWidget):
    def __init__(self, module: SpectrumAnalyzer):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.setInterval(30) 

    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- Controls ---
        controls_group = QGroupBox("Analysis Settings")
        main_controls_layout = QVBoxLayout()
        
        # Row 1: Basic Controls
        row1_layout = QHBoxLayout()
        
        # Start/Stop
        self.toggle_btn = QPushButton("Start Analysis")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.toggle_btn.setStyleSheet("QPushButton:checked { background-color: #ccffcc; }")
        row1_layout.addWidget(self.toggle_btn)
        
        # FFT Size
        row1_layout.addWidget(QLabel("FFT Size:"))
        self.fft_combo = QComboBox()
        self.fft_combo.addItems(['1024', '2048', '4096', '8192', '16384', '32768'])
        self.fft_combo.setCurrentText(str(self.module.buffer_size))
        self.fft_combo.currentTextChanged.connect(self.on_fft_size_changed)
        row1_layout.addWidget(self.fft_combo)

        # Window Selection
        row1_layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(['hanning', 'hamming', 'blackman', 'bartlett', 'rect'])
        self.window_combo.currentTextChanged.connect(self.on_window_changed)
        row1_layout.addWidget(self.window_combo)
        
        main_controls_layout.addLayout(row1_layout)
        
        # Row 2: Advanced Controls
        row2_layout = QHBoxLayout()
        
        # Smoothing
        row2_layout.addWidget(QLabel("Smoothing:"))
        self.smooth_combo = QComboBox()
        self.smooth_combo.addItems(['None', '1/1 Octave', '1/3 Octave', '1/6 Octave', '1/12 Octave', '1/24 Octave'])
        self.smooth_combo.currentTextChanged.connect(self.on_smooth_changed)
        row2_layout.addWidget(self.smooth_combo)

        # Averaging
        self.avg_label = QLabel("Avg: 0%")
        row2_layout.addWidget(self.avg_label)
        self.avg_slider = QSlider(Qt.Orientation.Horizontal)
        self.avg_slider.setRange(0, 95)
        self.avg_slider.setValue(0)
        self.avg_slider.setFixedWidth(100)
        self.avg_slider.valueChanged.connect(self.on_avg_changed)
        row2_layout.addWidget(self.avg_slider)
        
        # Peak Hold
        self.peak_check = QCheckBox("Peak Hold")
        self.peak_check.toggled.connect(self.on_peak_changed)
        row2_layout.addWidget(self.peak_check)
        
        # Clear Peak
        self.clear_peak_btn = QPushButton("Clear Peak")
        self.clear_peak_btn.clicked.connect(self.on_clear_peak)
        row2_layout.addWidget(self.clear_peak_btn)
        
        main_controls_layout.addLayout(row2_layout)
        
        controls_group.setLayout(main_controls_layout)
        layout.addWidget(controls_group)
        
        # --- Info Display ---
        info_layout = QHBoxLayout()
        
        # Overall Value
        self.overall_label = QLabel("Overall: -- dB")
        self.overall_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #00ff00;")
        info_layout.addWidget(self.overall_label)
        
        # Cursor Value
        self.cursor_label = QLabel("Cursor: -- Hz, -- dB")
        self.cursor_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #00ffff;")
        info_layout.addWidget(self.cursor_label)
        
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        # --- Plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Magnitude', units='dB')
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.setYRange(-120, 0)
        self.plot_widget.showGrid(x=True, y=True)
        
        # Crosshair
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)
        
        # Mouse movement proxy
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)
        
        # Curves
        self.peak_curve = self.plot_widget.plot(pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
        self.plot_curve = self.plot_widget.plot(pen='y')
        
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def mouse_moved(self, evt):
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            # Log mode x-axis handling
            # If log mode is True, x is log10(freq)
            # But pyqtgraph's mapSceneToView handles the log mapping if setLogMode is used?
            # Actually, setLogMode(x=True) means the view expects log10 values for X.
            # But the data we plot is linear frequency? No, if we use setLogMode(x=True), 
            # we usually pass x values as log10(x) OR we pass linear x and pyqtgraph handles it?
            # Wait, pyqtgraph setLogMode documentation says:
            # "If True, the axis will be logarithmic. The data plotted must be logarithmic."
            # So if I plotted linear frequencies, I might have been doing it wrong or pyqtgraph is smart.
            # Let's check update_plot. 
            # freqs = np.fft.rfftfreq... (linear)
            # self.plot_curve.setData(freqs[1:], magnitude[1:])
            # If setLogMode(x=True) is on, pyqtgraph expects log10(x).
            # Ah, actually, if I pass linear data to a log plot, it might not show correctly unless I convert it.
            # Let's verify. If I see the plot working, maybe I am passing linear and it's just squashed?
            # Standard pyqtgraph usage for log plot is to pass log10(x).
            # However, PlotWidget might handle it if I didn't convert.
            # Let's assume for now I need to handle the coordinate conversion for the label.
            
            # Actually, let's fix the plotting to be correct for log mode if it isn't.
            # But first, the cursor.
            
            x = mouse_point.x()
            y = mouse_point.y()
            
            # If x is log scale, convert back to linear for display
            freq = 10**x
            
            self.cursor_label.setText(f"Cursor: {freq:.1f} Hz, {y:.1f} dB")
            self.v_line.setPos(x)
            self.h_line.setPos(y)

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText("Stop Analysis")
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText("Start Analysis")

    def on_fft_size_changed(self, val):
        self.module.set_buffer_size(int(val))

    def on_window_changed(self, val):
        self.module.window_type = val

    def on_smooth_changed(self, val):
        self.module.octave_smoothing = val

    def on_avg_changed(self, val):
        self.module.averaging = val / 100.0
        self.avg_label.setText(f"Avg: {val}%")

    def on_peak_changed(self, checked):
        self.module.peak_hold = checked
        if not checked:
            self.module._peak_magnitude = None
            self.peak_curve.setData([], [])

    def on_clear_peak(self):
        self.module._peak_magnitude = None
        self.peak_curve.setData([], [])

    def apply_octave_smoothing(self, freqs, magnitude, fraction):
        """
        Apply fractional octave smoothing to the spectrum.
        fraction: 1 for 1/1 octave, 3 for 1/3 octave, etc.
        """
        if fraction is None:
            return freqs, magnitude

        # Define octave bands
        # Start from a low frequency, e.g., 20Hz
        f_min = 20
        f_max = freqs[-1]
        
        smoothed_freqs = []
        smoothed_mags = []
        
        current_f = f_min
        while current_f < f_max:
            factor = 2**(1/(2*fraction))
            lower = current_f / factor
            upper = current_f * factor
            
            indices = np.where((freqs >= lower) & (freqs < upper))[0]
            
            if len(indices) > 0:
                linear_mags = 10**(magnitude[indices]/20)
                avg_linear = np.mean(linear_mags)
                avg_db = 20 * np.log10(avg_linear + 1e-12)
                
                smoothed_freqs.append(current_f)
                smoothed_mags.append(avg_db)
            
            current_f *= 2**(1/fraction)
            
        return np.array(smoothed_freqs), np.array(smoothed_mags)

    def update_plot(self):
        if not self.module.is_running:
            return
            
        data = self.module.input_data
        
        # Calculate Overall RMS (dBFS)
        # RMS = sqrt(mean(data^2))
        rms = np.sqrt(np.mean(data**2))
        overall_db = 20 * np.log10(rms + 1e-12)
        self.overall_label.setText(f"Overall: {overall_db:.1f} dB")
        
        # Apply window
        if self.module.window_type == 'rect':
            window = np.ones(len(data))
        else:
            window = getattr(np, self.module.window_type)(len(data))
        windowed_data = data * window
        
        # FFT
        fft_data = np.fft.rfft(windowed_data)
        magnitude = np.abs(fft_data)
        
        # Convert to dB
        magnitude = 20 * np.log10(magnitude / len(data) + 1e-12)
        
        # Averaging
        if self.module._avg_magnitude is None or len(self.module._avg_magnitude) != len(magnitude):
            self.module._avg_magnitude = magnitude
        else:
            alpha = self.module.averaging
            self.module._avg_magnitude = alpha * self.module._avg_magnitude + (1 - alpha) * magnitude
            magnitude = self.module._avg_magnitude
            
        # Peak Hold
        if self.module.peak_hold:
            if self.module._peak_magnitude is None or len(self.module._peak_magnitude) != len(magnitude):
                self.module._peak_magnitude = magnitude
            else:
                self.module._peak_magnitude = np.maximum(self.module._peak_magnitude, magnitude)
        
        # Frequency axis
        sample_rate = self.module.audio_engine.sample_rate
        freqs = np.fft.rfftfreq(len(data), 1/sample_rate)
        
        # Smoothing
        fraction_map = {
            '1/1 Octave': 1,
            '1/3 Octave': 3,
            '1/6 Octave': 6,
            '1/12 Octave': 12,
            '1/24 Octave': 24
        }
        fraction = fraction_map.get(self.module.octave_smoothing)
        
        if fraction:
            plot_freqs, plot_mags = self.apply_octave_smoothing(freqs, magnitude, fraction)
            if self.module.peak_hold and self.module._peak_magnitude is not None:
                _, peak_mags = self.apply_octave_smoothing(freqs, self.module._peak_magnitude, fraction)
            else:
                peak_mags = None
        else:
            plot_freqs = freqs[1:]
            plot_mags = magnitude[1:]
            if self.module.peak_hold and self.module._peak_magnitude is not None:
                peak_mags = self.module._peak_magnitude[1:]
            else:
                peak_mags = None
        
        # Update curves
        # IMPORTANT: For LogMode(x=True), we must pass log10(x) to setData?
        # Or does pyqtgraph handle it? 
        # "When log mode is enabled, the value of the axis is calculated as log10(value). 
        #  The data provided to the plot must be pre-transformed."
        # So I need to take log10 of freqs.
        
        # Filter out zero frequency if present (already done with [1:])
        # But smoothed_freqs might start at 20, which is fine.
        
        log_freqs = np.log10(plot_freqs + 1e-12)
        self.plot_curve.setData(log_freqs, plot_mags)
        
        if peak_mags is not None:
            self.peak_curve.setData(log_freqs, peak_mags)
        else:
            self.peak_curve.setData([], [])
