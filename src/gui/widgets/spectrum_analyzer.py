import argparse
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QApplication)
from PyQt6.QtCore import QTimer, Qt
from scipy.signal.windows import dpss
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class SpectrumAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 4096
        # Store stereo data: (frames, 2)
        self.input_data = np.zeros((self.buffer_size, 2))
        
        # Analysis parameters
        self.window_type = 'hanning'
        self.averaging = 0.0 # 0.0 to 0.95
        self.peak_hold = False
        self.octave_smoothing = 'None' # None, 1/1, 1/3, 1/6, 1/12, 1/24
        self.analysis_mode = 'Spectrum' # 'Spectrum', 'Cross Spectrum'
        self.channel_mode = 'Average' # 'Left', 'Right', 'Average', 'Dual'
        self.multitaper_enabled = False
        self.display_unit = 'dBFS' # 'dBFS', 'dBV', 'dB SPL'
        self.weighting = 'Z' # 'Z', 'A', 'C'
        
        # Multitaper cache
        self._dpss_windows = None
        self._dpss_cache_key = None # (N, NW, K)
        
        # State
        self._avg_magnitude = None
        self._avg_cross_spectrum = None # Complex average for Cross Spectrum
        self._peak_magnitude = None
        self.overall_rms = 0.0
        
        self.callback_id = None

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
        self.input_data = np.zeros((self.buffer_size, 2))
        self._avg_magnitude = None
        self._avg_cross_spectrum = None
        self._peak_magnitude = None
        # Reset DPSS cache as N changed
        self._dpss_windows = None
        self._dpss_cache_key = None

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        self._avg_magnitude = None
        self._avg_cross_spectrum = None
        self._peak_magnitude = None
        self.overall_rms = 0.0
        self.input_data = np.zeros((self.buffer_size, 2))
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
            # Shift buffer and append new data
            # We always capture 2 channels now if available
            if indata.shape[1] >= 2:
                new_data = indata[:, :2]
            else:
                # If mono, duplicate to stereo for simplicity or handle gracefully
                new_data = np.column_stack((indata[:, 0], indata[:, 0]))
            
            # Efficient ring buffer or just roll
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

    def _get_dpss_windows(self, N, NW=3, Kmax=None):
        """
        Get DPSS windows, caching them for performance.
        """
        if Kmax is None:
            Kmax = 2 * NW - 1
        
        key = (N, NW, Kmax)
        if self._dpss_windows is None or self._dpss_cache_key != key:
            # Generate windows
            # dpss returns (K, N) array
            self._dpss_windows = dpss(N, NW, int(Kmax))
            self._dpss_cache_key = key
            
        return self._dpss_windows

    def compute_weighting(self, freqs, weighting_type):
        """
        Compute weighting gain in dB for given frequencies.
        """
        if weighting_type == 'Z':
            return np.zeros_like(freqs)
        
        f = freqs.copy()
        # Avoid division by zero or log of zero issues at DC
        f[f == 0] = 1e-9
        
        f2 = f**2
        
        if weighting_type == 'A':
            # A-weighting
            # RA(f) = (12194^2 * f^4) / ((f^2 + 20.6^2) * sqrt((f^2 + 107.7^2)(f^2 + 737.9^2)) * (f^2 + 12194^2))
            # Gain = 20*log10(RA(f)) + 2.00
            
            const = 12194**2 * f**4
            denom = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2)
            R_A = const / denom
            gain = 20 * np.log10(R_A) + 2.00
            return gain
            
        elif weighting_type == 'C':
            # C-weighting
            # RC(f) = (12194^2 * f^2) / ((f^2 + 20.6^2) * (f^2 + 12194^2))
            # Gain = 20*log10(RC(f)) + 0.06
            
            const = 12194**2 * f2
            denom = (f2 + 20.6**2) * (f2 + 12194**2)
            R_C = const / denom
            gain = 20 * np.log10(R_C) + 0.06
            return gain
            
        return np.zeros_like(freqs)

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
        controls_group = QGroupBox(tr("Analysis Settings"))
        main_controls_layout = QVBoxLayout()
        
        # Row 1: Basic Controls
        row1_layout = QHBoxLayout()
        
        # Start/Stop
        self.toggle_btn = QPushButton(tr("Start Analysis"))
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; color: black; } QPushButton:checked { background-color: #ffcccc; color: black; }")
            
        row1_layout.addWidget(self.toggle_btn)
        
        # Mode Selection
        row1_layout.addWidget(QLabel(tr("Mode:")))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr('Spectrum'), 'Spectrum')
        self.mode_combo.addItem(tr('PSD'), 'PSD')
        self.mode_combo.addItem(tr('Cross Spectrum'), 'Cross Spectrum')
        
        # Set initial selection
        index = self.mode_combo.findData(self.module.analysis_mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
            
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        row1_layout.addWidget(self.mode_combo)

        # Channel Selection
        row1_layout.addWidget(QLabel(tr("Channel:")))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(['Left', 'Right', 'Average', 'Dual'])
        self.channel_combo.setCurrentText(self.module.channel_mode)
        self.channel_combo.currentTextChanged.connect(self.on_channel_changed)
        row1_layout.addWidget(self.channel_combo)

        # FFT Size
        row1_layout.addWidget(QLabel(tr("FFT Size:")))
        self.fft_combo = QComboBox()
        self.fft_combo.addItems(['1024', '2048', '4096', '8192', '16384', '32768', '65536', '131072', '262144'])
        self.fft_combo.setCurrentText(str(self.module.buffer_size))
        self.fft_combo.currentTextChanged.connect(self.on_fft_size_changed)
        row1_layout.addWidget(self.fft_combo)

        # Window Selection
        row1_layout.addWidget(QLabel(tr("Window:")))
        self.window_combo = QComboBox()
        self.window_combo.addItems(['hanning', 'hamming', 'blackman', 'bartlett', 'rect'])
        self.window_combo.currentTextChanged.connect(self.on_window_changed)
        row1_layout.addWidget(self.window_combo)

        # Weighting Selection
        row1_layout.addWidget(QLabel(tr("Weighting:")))
        self.weighting_combo = QComboBox()
        self.weighting_combo.addItems(['Z', 'A', 'C'])
        self.weighting_combo.currentTextChanged.connect(self.on_weighting_changed)
        row1_layout.addWidget(self.weighting_combo)

        # Unit Selection (Replaces Physical Units Checkbox)
        row1_layout.addWidget(QLabel(tr("Unit:")))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['dBFS', 'dBV', 'dB SPL'])
        self.unit_combo.setCurrentText(self.module.display_unit)
        self.unit_combo.currentTextChanged.connect(self.on_unit_changed)
        row1_layout.addWidget(self.unit_combo)
        
        main_controls_layout.addLayout(row1_layout)
        
        # Row 2: Advanced Controls
        row2_layout = QHBoxLayout()
        
        # Smoothing
        row2_layout.addWidget(QLabel(tr("Smoothing:")))
        self.smooth_combo = QComboBox()
        self.smooth_combo.addItem(tr('None'), 'None')
        self.smooth_combo.addItem(tr('1/1 Octave'), '1/1 Octave')
        self.smooth_combo.addItem(tr('1/3 Octave'), '1/3 Octave')
        self.smooth_combo.addItem(tr('1/6 Octave'), '1/6 Octave')
        self.smooth_combo.addItem(tr('1/12 Octave'), '1/12 Octave')
        self.smooth_combo.addItem(tr('1/24 Octave'), '1/24 Octave')
        
        index = self.smooth_combo.findData(self.module.octave_smoothing)
        if index >= 0:
            self.smooth_combo.setCurrentIndex(index)
            
        self.smooth_combo.currentIndexChanged.connect(self.on_smooth_changed)
        row2_layout.addWidget(self.smooth_combo)

        # Averaging
        self.avg_label = QLabel(tr("Avg: 0%"))
        row2_layout.addWidget(self.avg_label)
        self.avg_slider = QSlider(Qt.Orientation.Horizontal)
        self.avg_slider.setRange(0, 99) # Allow up to 99% for heavy averaging
        self.avg_slider.setValue(0)
        self.avg_slider.setFixedWidth(100)
        self.avg_slider.valueChanged.connect(self.on_avg_changed)
        row2_layout.addWidget(self.avg_slider)
        
        # Multitaper
        self.multitaper_check = QCheckBox(tr("Multitaper"))
        self.multitaper_check.toggled.connect(self.on_multitaper_changed)
        row2_layout.addWidget(self.multitaper_check)

        # Peak Hold
        self.peak_check = QCheckBox(tr("Peak Hold"))
        self.peak_check.toggled.connect(self.on_peak_changed)
        row2_layout.addWidget(self.peak_check)
        
        # Clear Peak
        self.clear_peak_btn = QPushButton(tr("Clear Peak"))
        self.clear_peak_btn.clicked.connect(self.on_clear_peak)
        row2_layout.addWidget(self.clear_peak_btn)
        
        main_controls_layout.addLayout(row2_layout)
        
        # Row 3: Calibration (Removed Physical Units from here)
        # row3_layout = QHBoxLayout()
        # row3_layout.addStretch()
        # main_controls_layout.addLayout(row3_layout)
        
        controls_group.setLayout(main_controls_layout)
        layout.addWidget(controls_group)
        
        # --- Info Display ---
        info_layout = QHBoxLayout()
        
        # Overall Value
        self.overall_label = QLabel(tr("Overall: -- dB"))
        self.overall_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #00ff00;")
        info_layout.addWidget(self.overall_label)
        
        # Cursor Value
        self.cursor_label = QLabel(tr("Cursor: -- Hz, -- dB"))
        self.cursor_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #00ffff;")
        info_layout.addWidget(self.cursor_label)
        
        info_layout.addStretch()
        layout.addLayout(info_layout)
        
        # --- Plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', tr('Magnitude'), units='dB')
        self.plot_widget.setLabel('bottom', tr('Frequency'), units='Hz')
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.setYRange(-120, 0)
        self.plot_widget.showGrid(x=True, y=True)
        
        # Custom Axis Ticks
        axis = self.plot_widget.getPlotItem().getAxis('bottom')
        ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        # Since setLogMode(x=True) is used, the view coordinates are log10(freq).
        # We need to specify ticks at log positions.
        ticks_log = [(np.log10(t), str(t) if t < 1000 else f"{t/1000:.0f}k") for t in ticks]
        axis.setTicks([ticks_log])
        
        # Set Range (log domain)
        self.plot_widget.setXRange(np.log10(20), np.log10(20000))
        
        # Crosshair
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)
        
        # Mouse movement proxy
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)
        
        # Curves
        self.peak_curve = self.plot_widget.plot(pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
        self.plot_curve = self.plot_widget.plot(pen='y', name='Main')
        self.plot_curve_2 = self.plot_widget.plot(pen='g', name='Secondary') # For Dual mode (Left=Green, Right=Red usually, but let's stick to standard)
        # Let's use: Main (Yellow) for single/avg. 
        # For Dual: Left (Green), Right (Red).
        # So we might need to change pen colors dynamically.
        
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def format_si(self, value, unit):
        if value == 0:
            return f"0.0 {unit}"
            
        exponent = int(np.floor(np.log10(abs(value)) / 3) * 3)
        exponent = max(min(exponent, 9), -15)
        
        scaled_value = value / (10**exponent)
        
        prefixes = {
            -15: 'f', -12: 'p', -9: 'n', -6: 'µ', -3: 'm', 
            0: '', 3: 'k', 6: 'M', 9: 'G'
        }
        
        prefix = prefixes.get(exponent, '')
        return f"{scaled_value:.3g} {prefix}{unit}"

    def mouse_moved(self, evt):
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            
            x = mouse_point.x()
            y = mouse_point.y()
            
            # x is log10(freq)
            freq = 10**x
            
            # x is log10(freq)
            freq = 10**x
            
            unit_db = self.module.display_unit
            unit_linear = "" 
            
            if self.module.display_unit == 'dBV':
                 unit_linear = "V"
            elif self.module.display_unit == 'dB SPL':
                 unit_linear = "Pa"

            if self.module.analysis_mode == 'PSD':
                unit_db += "/√Hz"
                if unit_linear:
                    unit_linear += "/√Hz"
            
            # Calculate linear value
            linear_val = 10**(y/20)
            
            # Format linear value
            if self.module.display_unit == 'dB SPL':
                 # For SPL, y is dB SPL. Linear is 10^(y/20) * 20uPa.
                 val_pa = (10**(y/20)) * 20e-6
                 linear_str = self.format_si(val_pa, "Pa")
                 cursor_text = f"Cursor: {freq:.1f} Hz, {y:.1f} {unit_db} ({linear_str})"
            elif self.module.display_unit == 'dBV': 
                 linear_str = self.format_si(linear_val, unit_linear)
                 cursor_text = f"Cursor: {freq:.1f} Hz, {y:.1f} {unit_db} ({linear_str})"
            else: # dBFS
                 cursor_text = f"Cursor: {freq:.1f} Hz, {y:.1f} {unit_db} ({linear_val:.4g})"

            self.cursor_label.setText(cursor_text)
            self.v_line.setPos(x)
            self.h_line.setPos(y)

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText(tr("Stop Analysis"))
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText(tr("Start Analysis"))

    def on_mode_changed(self, index):
        val = self.mode_combo.itemData(index)
        if val is None:
            return
        self.module.analysis_mode = val
        # Reset averages when mode changes
        self.module._avg_magnitude = None
        self.module._avg_cross_spectrum = None
        self.module._peak_magnitude = None
        self.peak_curve.setData([], [])
        
        # Disable channel selection in Cross Spectrum mode?
        # Cross Spectrum inherently uses L and R.
        if val == 'Cross Spectrum':
            self.channel_combo.setEnabled(False)
        else:
            self.channel_combo.setEnabled(True)
            
        # Update Y-axis label
        unit = self.module.display_unit
        if val == 'PSD':
            unit += "/√Hz"
        self.plot_widget.setLabel('left', 'Magnitude', units=unit)

    def on_channel_changed(self, val):
        self.module.channel_mode = val
        self.module._avg_magnitude = None # Reset average
        self.peak_curve.setData([], [])

    def on_fft_size_changed(self, val):
        self.module.set_buffer_size(int(val))

    def on_window_changed(self, val):
        self.module.window_type = val

    def on_weighting_changed(self, val):
        self.module.weighting = val
        # Reset peak when weighting changes
        self.module._peak_magnitude = None
        self.peak_curve.setData([], [])

    def on_smooth_changed(self, index):
        val = self.smooth_combo.itemData(index)
        if val is None:
            return
        self.module.octave_smoothing = val

    def on_avg_changed(self, val):
        self.module.averaging = val / 100.0
        self.avg_label.setText(tr("Avg: {}%").format(val))

    def on_multitaper_changed(self, checked):
        self.module.multitaper_enabled = checked
        # Disable window selection if multitaper is on (it uses its own windows)
        self.window_combo.setEnabled(not checked)

    def on_peak_changed(self, checked):
        self.module.peak_hold = checked
        if not checked:
            self.module._peak_magnitude = None
            self.peak_curve.setData([], [])

    def on_clear_peak(self):
        self.module._peak_magnitude = None
        self.peak_curve.setData([], [])

    def on_unit_changed(self, val):
        self.module.display_unit = val
        unit = val
        if self.module.analysis_mode == 'PSD':
            unit += "/√Hz"
        self.plot_widget.setLabel('left', 'Magnitude', units=unit)
        # Reset peak to avoid mixing units
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
                # Use axis=0 to preserve channel dimension if present (Dual mode)
                avg_linear = np.mean(linear_mags, axis=0)
                avg_db = 20 * np.log10(avg_linear + 1e-12)
                
                smoothed_freqs.append(current_f)
                smoothed_mags.append(avg_db)
            
            current_f *= 2**(1/fraction)
            
        return np.array(smoothed_freqs), np.array(smoothed_mags)

    def update_plot(self):
        if not self.module.is_running:
            return
            
        data = self.module.input_data
        # data shape is (buffer_size, 2)
        
        # Calculate Overall RMS (dBFS) - Raw Time Domain (Unweighted)
        # This is calculated for reference, but we will overwrite it with weighted value later
        rms = np.sqrt(np.mean(data**2))
        overall_db = 20 * np.log10(rms + 1e-12)
        
        if self.module.display_unit == 'dBV':
            # Convert to dBV
            offset = self.module.audio_engine.calibration.get_input_offset_db()
            overall_db += offset
        elif self.module.display_unit == 'dB SPL':
             # Convert to SPL
             # We need to use the method from calibration that includes offset
             # But here we have the "Unweighted" raw dBFS.
             # Ideally "Overall" for SPL should be C-weighted or whatever the user calibrated with?
             # Usually Sound Level Meters show A-weighted or C-weighted overall.
             # The user calibrated "dBFS_C" -> "SPL".
             # If we just add offset to unweighted dBFS, it's "Unweighted SPL" (Linear).
             # That's probably fine for "Overall" if we qualify it.
             # Or we should apply C-weighting to overall? 
             # Let's stick to adding offset for now.
             
             spl_offset = self.module.audio_engine.calibration.get_spl_offset_db()
             if spl_offset is not None:
                 overall_db += spl_offset
        else:
            pass
            
        # Frequency axis
        sample_rate = self.module.audio_engine.sample_rate
        freqs = np.fft.rfftfreq(len(data), 1/sample_rate)
        
        # Calculate Weighting Curve
        weighting_db = self.module.compute_weighting(freqs, self.module.weighting)
        
        magnitude = None
        
        if self.module.multitaper_enabled:
            # --- Multitaper Method ---
            # Get DPSS windows
            windows = self.module._get_dpss_windows(len(data)) # (K, N)
            K = windows.shape[0]
            
            if self.module.analysis_mode == 'Spectrum' or self.module.analysis_mode == 'PSD':
                # --- Spectrum or PSD Mode ---
                # Calculate PSD for each channel and each window
                # psd = |FFT(x*w)|^2
                
                psd_accum_0 = np.zeros(len(freqs))
                psd_accum_1 = np.zeros(len(freqs))
                
                for k in range(K):
                    w = windows[k]
                    
                    # Channel 0
                    fft_0 = np.fft.rfft(data[:, 0] * w)
                    psd_accum_0 += np.abs(fft_0)**2
                    
                    # Channel 1
                    fft_1 = np.fft.rfft(data[:, 1] * w)
                    psd_accum_1 += np.abs(fft_1)**2
                    
                # Average over K windows
                psd_0 = psd_accum_0 / K
                psd_1 = psd_accum_1 / K
                
                # Apply Channel Selection
                if self.module.channel_mode == 'Left':
                    psd_target = psd_0
                    psd_second = None
                elif self.module.channel_mode == 'Right':
                    psd_target = psd_1
                    psd_second = None
                elif self.module.channel_mode == 'Average':
                    psd_target = (psd_0 + psd_1) / 2
                    psd_second = None
                elif self.module.channel_mode == 'Dual':
                    psd_target = psd_0
                    psd_second = psd_1
                else:
                    psd_target = (psd_0 + psd_1) / 2
                    psd_second = None
                    
                # Convert to Magnitude (Linear)
                if self.module.analysis_mode == 'PSD':
                    # PSD (V/rtHz)
                    # mag = sqrt(PSD * 2 / fs)
                    # Note: PSD here is Power per Bin (approx A^2*N)
                    # Correct normalization to V/rtHz:
                    # using the formula from previous implementation: sqrt(PSD * 2 / fs)
                    norm_factor_sq = 2 / sample_rate
                else:
                    # Spectrum (Peak Amplitude)
                    # mag = sqrt(PSD) / sqrt(N)
                    norm_factor_sq = 1 / len(data)

                magnitudes = []
                
                # Target
                mag_target = np.sqrt(psd_target * norm_factor_sq)
                magnitudes.append(mag_target)
                
                # Second (if Dual)
                if psd_second is not None:
                    mag_second = np.sqrt(psd_second * norm_factor_sq)
                    magnitudes.append(mag_second)
                
                # Combine
                if len(magnitudes) == 1:
                    mag_linear = magnitudes[0]
                else:
                    mag_linear = np.column_stack(magnitudes)
                
                # Peak -> RMS conversion if Physical Units or SPL
                # For PSD, we already handle it differently.
                if self.module.analysis_mode == 'Spectrum' and self.module.display_unit in ['dBV', 'dB SPL']:
                    mag_linear /= np.sqrt(2)
                
                # Temporal Averaging
                if self.module._avg_magnitude is None or self.module._avg_magnitude.shape != mag_linear.shape:
                    self.module._avg_magnitude = mag_linear
                else:
                    alpha = self.module.averaging
                    self.module._avg_magnitude = alpha * self.module._avg_magnitude + (1 - alpha) * mag_linear
                
                magnitude = 20 * np.log10(self.module._avg_magnitude + 1e-12)

                # Apply API/SPL adjustments
                if self.module.display_unit == 'dBV':
                    offset = self.module.audio_engine.calibration.get_input_offset_db()
                    magnitude += offset
                elif self.module.display_unit == 'dB SPL':
                    spl_offset = self.module.audio_engine.calibration.get_spl_offset_db()
                    if spl_offset is not None:
                         magnitude += spl_offset
                
            elif self.module.analysis_mode == 'Cross Spectrum':
                # Average Cross Spectrum over K windows
                cs_accum = np.zeros(len(freqs), dtype=complex)
                
                for k in range(K):
                    w = windows[k]
                    fft_0 = np.fft.rfft(data[:, 0] * w)
                    fft_1 = np.fft.rfft(data[:, 1] * w)
                    cs_accum += fft_0 * np.conj(fft_1)
                
                cs_avg = cs_accum / K
                
                # Complex Temporal Averaging
                if self.module._avg_cross_spectrum is None or self.module._avg_cross_spectrum.shape != cs_avg.shape:
                    self.module._avg_cross_spectrum = cs_avg
                else:
                    alpha = self.module.averaging
                    self.module._avg_cross_spectrum = alpha * self.module._avg_cross_spectrum + (1 - alpha) * cs_avg
                
                avg_cs = self.module._avg_cross_spectrum
                
                # Normalize and Magnitude
                mag_linear = np.sqrt(np.abs(avg_cs)) / np.sqrt(len(data))
                
                if self.module.display_unit in ['dBV', 'dB SPL']:
                    mag_linear /= np.sqrt(2)

                magnitude = 20 * np.log10(mag_linear + 1e-12)

                # Apply API/SPL adjustments
                if self.module.display_unit == 'dBV':
                     offset = self.module.audio_engine.calibration.get_input_offset_db()
                     magnitude += offset
                elif self.module.display_unit == 'dB SPL':
                     spl_offset = self.module.audio_engine.calibration.get_spl_offset_db()
                     if spl_offset is not None:
                         magnitude += spl_offset

        else:
            # --- Standard Method ---
            # Apply window
            if self.module.window_type == 'rect':
                window = np.ones(len(data))
            else:
                window = getattr(np, self.module.window_type)(len(data))
            
            # Calculate Window Correction Factor (Amplitude Correction)
            # Factor = 1 / mean(window)
            # This compensates for the coherent gain loss due to windowing
            window_correction = 1.0 / np.mean(window)
            
            # Broadcast window to stereo
            windowed_data = data * window[:, np.newaxis]
            
            # FFT
            # rfft on axis 0
            fft_data = np.fft.rfft(windowed_data, axis=0)
            
            # Normalization Factor for Peak Amplitude
            # 2/N for one-sided spectrum (DC and Nyquist need special handling but usually ignored for general audio display)
            # * window_correction
            norm_factor = (2.0 / len(data)) * window_correction
            
            if self.module.analysis_mode == 'Spectrum':
                # Standard Spectrum
                mag_stereo = np.abs(fft_data)
                
                # Channel Selection Logic
                if self.module.channel_mode == 'Left':
                    mag_mono = mag_stereo[:, 0]
                    mag_second = None
                elif self.module.channel_mode == 'Right':
                    mag_mono = mag_stereo[:, 1]
                    mag_second = None
                elif self.module.channel_mode == 'Average':
                    mag_mono = np.mean(mag_stereo, axis=1)
                    mag_second = None
                elif self.module.channel_mode == 'Dual':
                    mag_mono = mag_stereo[:, 0] # Left
                    mag_second = mag_stereo[:, 1] # Right
                else:
                    mag_mono = np.mean(mag_stereo, axis=1)
                    mag_second = None
                
                # Normalize to Peak Amplitude
                mag_mono = mag_mono * norm_factor
                if mag_second is not None:
                    mag_second = mag_second * norm_factor
                
                # If Physical Units (dBV) or SPL are used, we want RMS reading for sine waves
                # to match the "Overall" RMS reading.
                # Peak to RMS for sine is 1/sqrt(2)
                if self.module.display_unit in ['dBV', 'dB SPL']:
                    mag_mono /= np.sqrt(2)
                    if mag_second is not None:
                        mag_second /= np.sqrt(2)
                
                # Averaging
                # Note: Averaging Dual channels separately might require separate state.
                # For simplicity, let's apply same averaging factor to both but only store one state if not Dual?
                # Actually, if we switch modes, we reset.
                # If Dual, we need two average states.
                # Current self.module._avg_magnitude is one array.
                # Let's make it handle (N, 2) if Dual? Or just (N,) and we only average the primary?
                # To do it properly for Dual, we need to change how _avg_magnitude is stored or use a new variable.
                # Let's try to store whatever shape we have.
                
                current_mag = mag_mono
                if mag_second is not None:
                    current_mag = np.column_stack((mag_mono, mag_second))
                
                if self.module._avg_magnitude is None or self.module._avg_magnitude.shape != current_mag.shape:
                    self.module._avg_magnitude = current_mag
                else:
                    alpha = self.module.averaging
                    self.module._avg_magnitude = alpha * self.module._avg_magnitude + (1 - alpha) * current_mag
                
                magnitude_linear = self.module._avg_magnitude
                magnitude = 20 * np.log10(magnitude_linear + 1e-12)
                
                # Apply dBV / SPL offsets
                if self.module.display_unit == 'dBV':
                    offset = self.module.audio_engine.calibration.get_input_offset_db()
                    magnitude += offset
                elif self.module.display_unit == 'dB SPL':
                     spl_offset = self.module.audio_engine.calibration.get_spl_offset_db()
                     if spl_offset is not None:
                         magnitude += spl_offset
                
            elif self.module.analysis_mode == 'PSD':
                # Power Spectral Density (Voltage Noise Density)
                # We want V/rtHz.
                # Currently mag_mono is Peak Amplitude (V_peak).
                # We need to convert to V_rms/rtHz.
                
                # 1. Convert Peak to RMS
                # mag_rms = mag_mono / sqrt(2)
                
                # 2. Normalize by Noise Bandwidth (NBW)
                # NBW = fs * sum(w^2) / (sum(w)^2)
                # LSD = mag_rms / sqrt(NBW)
                
                # Combining with existing normalization:
                # mag_mono = |X| * (2 / sum(w))
                # LSD = (|X| * (2 / sum(w)) / sqrt(2)) / sqrt(fs * sum(w^2) / sum(w)^2)
                #     = |X| * sqrt(2)/sum(w) * sum(w) / sqrt(fs * sum(w^2))
                #     = |X| * sqrt(2) / sqrt(fs * sum(w^2))
                #     = |X| * sqrt(2 / (fs * sum(w^2)))
                
                # Alternatively, using mag_mono directly:
                # LSD = (mag_mono / sqrt(2)) / sqrt(fs * sum(w^2) / sum(w)^2)
                #     = mag_mono * sum(w) / sqrt(2 * fs * sum(w^2))
                
                sum_w = np.sum(window)
                sum_w2 = np.sum(window**2)
                fs = sample_rate
                
                # Conversion factor from Peak Amplitude to V/rtHz
                psd_factor = sum_w / np.sqrt(2 * fs * sum_w2)
                
                mag_stereo = np.abs(fft_data)
                
                # Apply standard normalization first to get Peak Amplitude
                mag_stereo = mag_stereo * norm_factor
                
                # Apply PSD factor
                mag_stereo = mag_stereo * psd_factor
                
                # Channel Selection
                if self.module.channel_mode == 'Left':
                    mag_mono = mag_stereo[:, 0]
                elif self.module.channel_mode == 'Right':
                    mag_mono = mag_stereo[:, 1]
                elif self.module.channel_mode == 'Average':
                    # Average the Power (V^2/Hz), then sqrt
                    # mag_stereo is V/rtHz. Square to get V^2/Hz.
                    pow_stereo = mag_stereo**2
                    avg_pow = np.mean(pow_stereo, axis=1)
                    mag_mono = np.sqrt(avg_pow)
                elif self.module.channel_mode == 'Dual':
                    mag_mono = mag_stereo
                else:
                    mag_mono = mag_stereo[:, 0]
                
                # Averaging
                if self.module._avg_magnitude is None or self.module._avg_magnitude.shape != mag_mono.shape:
                    self.module._avg_magnitude = mag_mono
                else:
                    alpha = self.module.averaging
                    self.module._avg_magnitude = alpha * self.module._avg_magnitude + (1 - alpha) * mag_mono
                
                magnitude_linear = self.module._avg_magnitude
                magnitude = 20 * np.log10(magnitude_linear + 1e-12)

                # Apply API/SPL adjustments
                if self.module.display_unit == 'dBV':
                     offset = self.module.audio_engine.calibration.get_input_offset_db()
                     magnitude += offset
                elif self.module.display_unit == 'dB SPL':
                     spl_offset = self.module.audio_engine.calibration.get_spl_offset_db()
                     if spl_offset is not None:
                         magnitude += spl_offset
                
            elif self.module.analysis_mode == 'Cross Spectrum':
                # Cross Spectrum
                F1 = fft_data[:, 0]
                F2 = fft_data[:, 1]
                Sxy = F1 * np.conj(F2)
                
                # Normalize
                # For Power/Cross Spectrum, normalization is usually (1/N)^2 or similar.
                # But we want Magnitude of Cross Spectrum to be comparable to Spectrum.
                # Let's normalize components first or result.
                # |Sxy| = |F1|*|F2|. If |F1| and |F2| are Peak Amplitudes (unnormalized FFT),
                # then |Sxy| is proportional to Peak^2 * (N/2)^2 / window_gain^2 ?
                
                # Let's apply normalization to the magnitude of Sxy
                # If we normalized F1 and F2 with norm_factor, then |Sxy_norm| = |F1_norm| * |F2_norm|
                # So we can multiply Sxy by norm_factor^2
                
                Sxy = Sxy * (norm_factor**2)
                
                # Complex Averaging
                if self.module._avg_cross_spectrum is None or len(self.module._avg_cross_spectrum) != len(Sxy):
                    self.module._avg_cross_spectrum = Sxy
                else:
                    alpha = self.module.averaging
                    self.module._avg_cross_spectrum = alpha * self.module._avg_cross_spectrum + (1 - alpha) * Sxy
                
                # Magnitude
                avg_Sxy = self.module._avg_cross_spectrum
                magnitude_linear = np.sqrt(np.abs(avg_Sxy))
                
                if self.module.display_unit in ['dBV', 'dB SPL']:
                    magnitude_linear /= np.sqrt(2)
                
                magnitude = 20 * np.log10(magnitude_linear + 1e-12)

                # Apply API/SPL adjustments
                if self.module.display_unit == 'dBV':
                     offset = self.module.audio_engine.calibration.get_input_offset_db()
                     magnitude += offset
                elif self.module.display_unit == 'dB SPL':
                     spl_offset = self.module.audio_engine.calibration.get_spl_offset_db()
                     if spl_offset is not None:
                         magnitude += spl_offset

        # Calibration is already applied in the blocks above.
        # Removing redundant/crashing block.
        # Apply Weighting
        if magnitude.ndim == 2 and weighting_db.ndim == 1:
            magnitude += weighting_db[:, np.newaxis]
        else:
            magnitude += weighting_db
        
        # Calculate Weighted RMS from Spectrum
        # We need to sum the power in frequency domain.
        # Magnitude is in dB (Peak or RMS depending on units/mode).
        # Let's convert back to linear power.
        
        # If magnitude is dB Peak: Power ~ (10^(mag/20))^2 / 2  (for sine)
        # If magnitude is dB RMS: Power ~ (10^(mag/20))^2
        
        # Note: 'magnitude' array here is already processed (averaged, normalized).
        # To get accurate Overall RMS, we should ideally sum the raw power spectrum * weighting^2.
        # But for display purposes, summing the processed spectrum is usually "good enough" 
        # provided we handle the window correction and normalization correctly.
        
        # However, we have 'magnitude' which is the displayed curve.
        # Let's use it to estimate the Overall Weighted RMS.
        
        # Convert dB to Linear Amplitude
        mag_linear_for_rms = 10**(magnitude/20)
        
        # If we are in Physical Units, mag_linear is already RMS (we divided by sqrt(2)).
        # If not, it's Peak.
        
        if self.module.display_unit in ['dBV', 'dB SPL']:
            power_spectrum = mag_linear_for_rms**2
        else:
            # Convert Peak to RMS
            power_spectrum = (mag_linear_for_rms / np.sqrt(2))**2
            
        # Sum power
        # We need to be careful about the window correction factor which was applied to amplitude.
        # The sum of bin powers should equal total time-domain power (Parseval's theorem),
        # but windowing and zero-padding affect this.
        # For a simple estimate consistent with the plot:
        
        # We normalized by 2/N * window_correction.
        # This normalization makes a sine wave peak at 1.0 (0dB).
        # The sum of squares of bins will not directly equal time domain power without un-normalizing.
        
        # Easier approach:
        # 1. Calculate weighting in linear domain: W_lin = 10^(weighting_db/20)
        # 2. Apply W_lin to the FFT of the raw windowed data.
        # 3. Compute RMS from that.
        
        # But we want to use the already computed 'magnitude' to reflect what is shown.
        # Let's just sum the power of the displayed bins.
        # This is an approximation but aligns with "what you see is what you get".
        
        # Only consider 20Hz - 20kHz for Overall Weighted calculation as requested
        mask = (freqs >= 20) & (freqs <= 20000)
        if np.any(mask):
            total_power = np.sum(power_spectrum[mask])
            # We need to account for the fact that we are summing bins.
            # If we have a pure sine, it might be split across bins (leakage).
            # Summing power preserves energy.
            
            # However, we applied a window correction factor for AMPLITUDE (coherent gain).
            # For POWER summation (incoherent gain), the correction factor is different (S2 = sum(w^2)).
            # Since we scaled by 1/mean(w), we boosted noise power.
            # This is a known trade-off. For "Overall" reading, time-domain is best.
            # But for "Weighted Overall", we must use frequency domain.
            
            # Let's stick to the sum of the displayed spectrum power for consistency.
            overall_weighted_rms = np.sqrt(total_power)
            overall_weighted_db = 20 * np.log10(overall_weighted_rms + 1e-12)
        else:
            overall_weighted_db = -120

        unit_suffix = ""
        if self.module.weighting == 'A': unit_suffix = "A"
        elif self.module.weighting == 'C': unit_suffix = "C"
        elif self.module.weighting == 'Z': unit_suffix = "Z"
        
        if self.module.display_unit == 'dB SPL':
             unit_display = f"dB SPL({unit_suffix})"
        elif self.module.display_unit == 'dBV':
             unit_display = f"dBV({unit_suffix})"
        else:
            unit_display = f"dBFS({unit_suffix})"

        self.overall_label.setText(f"Overall: {overall_weighted_db:.1f} {unit_display}")

        # Peak Hold
        if self.module.peak_hold:
            if self.module._peak_magnitude is None or len(self.module._peak_magnitude) != len(magnitude):
                self.module._peak_magnitude = magnitude
            else:
                self.module._peak_magnitude = np.maximum(self.module._peak_magnitude, magnitude)
        
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
        # When setLogMode(x=True) is active, we must pass LINEAR x values to setData.
        # pyqtgraph handles the log conversion.
        # We should exclude 0Hz to avoid log(0) issues inside pyqtgraph.
        
        plot_freqs_linear = plot_freqs + 1e-12 # Avoid exact 0
        
        # Handle Dual Mode Plotting
        if self.module.analysis_mode in ['Spectrum', 'PSD'] and self.module.channel_mode == 'Dual':
            # plot_mags should be (N, 2)
            if plot_mags.ndim == 2 and plot_mags.shape[1] >= 2:
                # Curve 1 (Left) - Green
                self.plot_curve.setData(plot_freqs_linear, plot_mags[:, 0], pen='g')
                # Curve 2 (Right) - Red
                self.plot_curve_2.setData(plot_freqs_linear, plot_mags[:, 1], pen='r')
            else:
                # Fallback
                self.plot_curve.setData(plot_freqs_linear, plot_mags, pen='y')
                self.plot_curve_2.setData([], [])
        else:
            # Single Curve
            # Ensure 1D
            if plot_mags.ndim == 2:
                plot_mags = plot_mags[:, 0] # Should not happen if logic above is correct for non-Dual
            
            self.plot_curve.setData(plot_freqs_linear, plot_mags, pen='y')
            self.plot_curve_2.setData([], [])
        
        if peak_mags is not None:
            # Peak hold usually just max of whatever we are displaying.
            # If Dual, peak hold might be complex. Let's just show peak of primary (Left) or max of both?
            # For simplicity, if Dual, let's just not show Peak Hold or show it for Left.
            if peak_mags.ndim == 2:
                peak_mags = peak_mags[:, 0]
            self.peak_curve.setData(plot_freqs_linear, peak_mags)
        else:
            self.peak_curve.setData([], [])

    def apply_theme(self, theme_name):
        # If theme_name is 'system', resolve it
        if theme_name == 'system' and hasattr(self.app, 'theme_manager'):
            theme_name = self.app.theme_manager.get_effective_theme()
            
        if theme_name == 'dark':
            # Dark Theme: Darker colors, White text
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; border: 1px solid #555; border-radius: 4px; padding: 5px; }"
                "QPushButton:checked { background-color: #c62828; color: white; border: 1px solid #555; border-radius: 4px; padding: 5px; }"
                "QPushButton:hover { background-color: #388e3c; }"
                "QPushButton:checked:hover { background-color: #d32f2f; }"
            )
            self.overall_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #00ff00;")
            self.cursor_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #00ffff;")
        else:
            # Light Theme: Pastel colors, Black text
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #ccffcc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 5px; }"
                "QPushButton:checked { background-color: #ffcccc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 5px; }"
                "QPushButton:hover { background-color: #bbfebb; }"
                "QPushButton:checked:hover { background-color: #ffbbbb; }"
            )
            self.overall_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #008800;")
            self.cursor_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #0000aa;")

        # Theme handling
        self.app = QApplication.instance()
        if hasattr(self.app, 'theme_manager'):
            self.app.theme_manager.theme_changed.connect(self.apply_theme)
            self.apply_theme(self.app.theme_manager.get_current_theme())
