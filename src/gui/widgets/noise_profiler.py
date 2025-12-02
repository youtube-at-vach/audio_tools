import argparse
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox)
from PyQt6.QtCore import QTimer, Qt
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.analysis import AudioCalc

class NoiseProfiler(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 16384 # Large buffer for better low-freq resolution
        self.input_data = np.zeros((self.buffer_size, 2))
        
        # Settings
        self.window_type = 'hanning'
        self.averaging = 0.0
        self.lna_gain_db = 0.0
        self.temperature_c = 25.0
        self.input_impedance = 50.0
        
        # State
        self._avg_magnitude = None
        self.callback_id = None
        self.manual_corner_enabled = False
        self.manual_corner_freq = 100.0
        
        # Results
        self.last_results = {}

    @property
    def name(self) -> str:
        return "Noise Profiler"

    @property
    def description(self) -> str:
        return "Noise characterization and analysis tool."

    def run(self, args: argparse.Namespace):
        print("Noise Profiler running from CLI (not fully implemented)")

    def get_widget(self):
        return NoiseProfilerWidget(self)

    def set_buffer_size(self, size):
        self.buffer_size = size
        self.input_data = np.zeros((self.buffer_size, 2))
        self._avg_magnitude = None

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        self._avg_magnitude = None
        self.input_data = np.zeros((self.buffer_size, 2))
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
            if indata.shape[1] >= 2:
                new_data = indata[:, :2]
            else:
                new_data = np.column_stack((indata[:, 0], indata[:, 0]))
            
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

class NoiseProfilerWidget(QWidget):
    def __init__(self, module: NoiseProfiler):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_analysis)
        self.timer.setInterval(100) # 10Hz update

    def init_ui(self):
        layout = QHBoxLayout()
        
        # --- Left Panel: Controls ---
        left_panel = QVBoxLayout()
        
        # Analysis Control
        ctrl_group = QGroupBox("Control")
        ctrl_layout = QVBoxLayout()
        self.toggle_btn = QPushButton("Start Profiling")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; color: black; } QPushButton:checked { background-color: #ffcccc; color: black; }")
        ctrl_layout.addWidget(self.toggle_btn)
        
        ctrl_group.setLayout(ctrl_layout)
        left_panel.addWidget(ctrl_group)
        
        # LNA Settings
        lna_group = QGroupBox("LNA / Input Settings")
        lna_layout = QFormLayout()
        
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(-100, 100)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setValue(0.0)
        self.gain_spin.valueChanged.connect(self.on_lna_changed)
        lna_layout.addRow("Pre-Amp Gain:", self.gain_spin)
        
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(-273, 500)
        self.temp_spin.setSuffix(" °C")
        self.temp_spin.setValue(25.0)
        self.temp_spin.valueChanged.connect(self.on_lna_changed)
        lna_layout.addRow("Temperature:", self.temp_spin)
        
        self.imp_spin = QDoubleSpinBox()
        self.imp_spin.setRange(1, 1e6)
        self.imp_spin.setSuffix(" Ω")
        self.imp_spin.setValue(50.0)
        self.imp_spin.valueChanged.connect(self.on_lna_changed)
        lna_layout.addRow("Input Z:", self.imp_spin)
        
        lna_group.setLayout(lna_layout)
        left_panel.addWidget(lna_group)
        
        # Manual Override
        manual_group = QGroupBox("Manual Analysis")
        manual_layout = QFormLayout()
        
        self.manual_corner_chk = QCheckBox("Manual 1/f Corner")
        self.manual_corner_chk.toggled.connect(self.on_manual_corner_toggled)
        manual_layout.addRow(self.manual_corner_chk)
        
        self.corner_spin = QDoubleSpinBox()
        self.corner_spin.setRange(1.0, 20000.0)
        self.corner_spin.setSuffix(" Hz")
        self.corner_spin.setValue(100.0)
        self.corner_spin.setEnabled(False)
        self.corner_spin.valueChanged.connect(self.on_manual_corner_changed)
        manual_layout.addRow("Corner Freq:", self.corner_spin)
        
        manual_group.setLayout(manual_layout)
        left_panel.addWidget(manual_group)
        
        left_panel.addStretch()
        layout.addLayout(left_panel, 1)
        
        # --- Center Panel: Visualization ---
        center_panel = QVBoxLayout()
        
        # FFT Plot
        self.plot_widget = pg.PlotWidget(title="Noise Spectrum (Log-Log)")
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.setLabel('left', 'Noise Density', units='V/√Hz')
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setYRange(-160, -60) # Typical noise floor range
        self.plot_widget.setXRange(np.log10(10), np.log10(20000))
        
        # Custom Axis Ticks for Log Scale
        axis = self.plot_widget.getPlotItem().getAxis('bottom')
        ticks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        ticks_log = [(np.log10(t), str(t) if t < 1000 else f"{t/1000:.0f}k") for t in ticks]
        axis.setTicks([ticks_log])
        
        self.plot_curve = self.plot_widget.plot(pen='y', name='PSD')
        self.fit_curve = self.plot_widget.plot(pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2), name='1/f Fit')
        self.hum_curve = self.plot_widget.plot(pen=None, symbol='o', symbolBrush='c', symbolSize=8, name='Hum')
        self.white_curve = self.plot_widget.plot(pen=pg.mkPen('g', style=Qt.PenStyle.DotLine), name='White Floor')
        
        center_panel.addWidget(self.plot_widget, 2)
        
        # Radar Chart (Simulated with PlotWidget)
        self.radar_widget = pg.PlotWidget(title="Noise Contribution")
        self.radar_widget.setAspectLocked()
        self.radar_widget.hideAxis('left')
        self.radar_widget.hideAxis('bottom')
        self.radar_widget.setXRange(-1.2, 1.2)
        self.radar_widget.setYRange(-1.2, 1.2)
        self.radar_widget.disableAutoRange()
        # Lock mouse interaction to prevent user from accidentally shifting the view
        self.radar_widget.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        self.radar_widget.getPlotItem().getViewBox().setMenuEnabled(False)
        
        # Initialize Radar Chart Background
        self.radar_axes = ['Hum', 'White', '1/f Corner', 'Total Noise']
        n = len(self.radar_axes)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        self.radar_angles = angles
        
        # Draw Web
        for r in [0.25, 0.5, 0.75, 1.0]:
            x_web = r * np.cos(angles)
            y_web = r * np.sin(angles)
            x_web = np.append(x_web, x_web[0])
            y_web = np.append(y_web, y_web[0])
            self.radar_widget.plot(x_web, y_web, pen=pg.mkPen('#444', style=Qt.PenStyle.DotLine))
            
        # Draw Axes and Labels
        for i in range(n):
            self.radar_widget.plot([0, np.cos(angles[i])], [0, np.sin(angles[i])], pen=pg.mkPen('#444'))
            # Labels
            text = pg.TextItem(self.radar_axes[i], anchor=(0.5, 0.5), color='#aaa')
            text.setPos(1.1*np.cos(angles[i]), 1.1*np.sin(angles[i]))
            self.radar_widget.addItem(text)
            
        # Data Polygon (Initialize empty)
        self.radar_curve = self.radar_widget.plot(pen=pg.mkPen('c', width=2), fillLevel=0, brush=pg.mkBrush(0, 255, 255, 50))
        
        center_panel.addWidget(self.radar_widget, 1)
        
        layout.addLayout(center_panel, 3)
        
        # --- Right Panel: Report ---
        right_panel = QVBoxLayout()
        report_group = QGroupBox("Noise Report")
        self.report_label = QLabel("Waiting for data...")
        self.report_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.report_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.report_label.setWordWrap(True)
        
        report_layout = QVBoxLayout()
        report_layout.addWidget(self.report_label)
        report_group.setLayout(report_layout)
        
        right_panel.addWidget(report_group)
        layout.addLayout(right_panel, 1)
        
        self.setLayout(layout)

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText("Stop Profiling")
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText("Start Profiling")

    def on_lna_changed(self):
        self.module.lna_gain_db = self.gain_spin.value()
        self.module.temperature_c = self.temp_spin.value()
        self.module.input_impedance = self.imp_spin.value()

    def on_manual_corner_toggled(self, checked):
        self.corner_spin.setEnabled(checked)
        self.module.manual_corner_enabled = checked
        if checked:
            self.module.manual_corner_freq = self.corner_spin.value()

    def on_manual_corner_changed(self):
        self.module.manual_corner_freq = self.corner_spin.value()

    def update_analysis(self):
        if not self.module.is_running:
            return
            
        try:
            data = self.module.input_data
            if len(data) < self.module.buffer_size:
                return

            # 1. Compute PSD (V/rtHz)
            # Use Hanning window
            window = np.hanning(len(data))
            
            fs = self.module.audio_engine.sample_rate
            sum_w2 = np.sum(window**2)
            psd_factor = np.sqrt(2 / (fs * sum_w2))
            
            # FFT
            fft_data = np.fft.rfft(data[:, 0] * window) # Use Left channel for now
            mag_v_rthz = np.abs(fft_data) * psd_factor
            
            freqs = np.fft.rfftfreq(len(data), 1/fs)
            
            # Averaging
            if self.module._avg_magnitude is None:
                self.module._avg_magnitude = mag_v_rthz
            else:
                alpha = 0.8 # Fixed smoothing
                self.module._avg_magnitude = alpha * self.module._avg_magnitude + (1 - alpha) * mag_v_rthz
                
            avg_mag = self.module._avg_magnitude
            
            # 2. Analyze Noise
            results = AudioCalc.calculate_noise_profile(avg_mag, freqs, fs)
            
            # Apply Manual Override
            if self.module.manual_corner_enabled:
                results['corner_freq'] = self.module.manual_corner_freq
                # Recalculate slope if needed? Or just use the manual corner for display/radar?
                # For now, just override the corner frequency value.
                
            self.module.last_results = results
            
            # 3. Update Plots
            # Main Spectrum
            mag_db = 20 * np.log10(avg_mag + 1e-15)
            self.plot_curve.setData(freqs[1:], mag_db[1:]) # Skip DC
            
            # 1/f Fit Line
            if results['flicker_slope'] != 0:
                # Generate line points
                f_fit = np.logspace(0, 2, 100) # 1Hz to 100Hz
                y_fit = results['flicker_slope'] * np.log10(f_fit) + results['flicker_intercept']
                self.fit_curve.setData(f_fit, y_fit)
            else:
                self.fit_curve.setData([], [])
                
            # Hum Markers
            hum_freqs = [h[0] for h in results['hum_components']]
            hum_amps = []
            for f in hum_freqs:
                idx = np.argmin(np.abs(freqs - f))
                hum_amps.append(mag_db[idx])
            self.hum_curve.setData(hum_freqs, hum_amps)
            
            # White Noise Floor
            white_level_db = 20 * np.log10(results['white_density'] + 1e-15)
            self.white_curve.setData([10, 20000], [white_level_db, white_level_db])
            
            # 4. Update Radar Chart
            self.update_radar_chart(results)
            
            # 5. Update Report
            self.update_report(results)
            
        except Exception as e:
            print(f"Error in NoiseProfiler update: {e}")
            import traceback
            traceback.print_exc()
        
    def update_radar_chart(self, results):
        # Categories: Hum, White, 1/f (Corner), Total RMS
        # Normalize values to 0-1 range for display
        
        # Hum: Relative to Total RMS?
        hum_rms = results['hum_rms']
        white_rms = results['noise_rms_20k']
        # 1/f contribution? Maybe Corner Freq relative to 1kHz?
        corner = results['corner_freq']
        
        # Let's define axes
        # axes = ['Hum', 'White', '1/f Corner', 'Total Noise']
        values = []
        
        # Normalize logic (heuristic)
        # Hum: Log scale, -120dB to -60dB -> 0 to 1
        hum_db = 20 * np.log10(hum_rms + 1e-15)
        val_hum = np.clip((hum_db + 120) / 60, 0, 1)
        
        # White: Log scale density, -160dB to -100dB -> 0 to 1
        white_db = 20 * np.log10(results['white_density'] + 1e-15)
        val_white = np.clip((white_db + 160) / 60, 0, 1)
        
        # Corner: Log scale, 1Hz to 1000Hz -> 0 to 1
        val_corner = np.clip(np.log10(corner + 1e-1) / 3, 0, 1)
        
        # Total: Log scale, -100dB to -40dB -> 0 to 1
        total_db = 20 * np.log10(results['noise_rms_20k'] + 1e-15)
        val_total = np.clip((total_db + 100) / 60, 0, 1)
        
        values = np.array([val_hum, val_white, val_corner, val_total])
        # Handle NaNs
        values = np.nan_to_num(values)
        
        # Draw Polygon
        angles = self.radar_angles
        x = values * np.cos(angles)
        y = values * np.sin(angles)
        
        # Close loop
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        
        # Update Data
        self.radar_curve.setData(x, y)
        
    def update_report(self, results):
        # Calculate Input Referred Noise
        gain_db = self.module.lna_gain_db
        gain_linear = 10**(gain_db/20)
        
        # Thermal Noise
        # V_thermal = sqrt(4 * k * T * R * BW)
        # Density = sqrt(4 * k * T * R)
        k = 1.38e-23
        T = self.module.temperature_c + 273.15
        R = self.module.input_impedance
        thermal_density = np.sqrt(4 * k * T * R)
        thermal_density_db = 20 * np.log10(thermal_density)
        
        # Input Referred
        # V_in = V_out / Gain
        white_density_in = results['white_density'] / gain_linear
        white_density_in_db = 20 * np.log10(white_density_in + 1e-15)
        
        # NF (Noise Figure) = SNR_in / SNR_out = Total_Noise_In / Thermal_Noise
        # Or simply Excess Noise Ratio?
        # Let's just show Input Referred Density vs Thermal Density
        
        txt = f"""
        <b>Noise Report</b><br>
        <br>
        <b>Hum ({results['hum_freq']:.0f}Hz):</b><br>
        RMS: {results['hum_rms']*1e6:.2f} µV<br>
        THD+N (Hum): {20*np.log10(results['hum_rms']/results['noise_rms_20k']+1e-15):.1f} dB<br>
        <br>
        <b>White Noise:</b><br>
        Density: {results['white_density']*1e9:.2f} nV/√Hz<br>
        (Input Ref: {white_density_in*1e9:.2f} nV/√Hz)<br>
        <br>
        <b>1/f Noise:</b><br>
        Corner Freq: {results['corner_freq']:.1f} Hz<br>
        Slope: {results['flicker_slope']:.2f} dB/dec<br>
        <br>
        <b>Integrated RMS (20k):</b><br>
        Total: {results['noise_rms_20k']*1e6:.2f} µV<br>
        <br>
        <b>Thermal Limit ({R}Ω):</b><br>
        {thermal_density*1e9:.2f} nV/√Hz ({thermal_density_db:.1f} dBV)
        """
        self.report_label.setText(txt)
