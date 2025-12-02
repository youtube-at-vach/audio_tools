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

        # Display Options
        disp_group = QGroupBox("Display")
        disp_layout = QVBoxLayout()
        
        self.res_mode_chk = QCheckBox("Show as Resistance (Ω)")
        self.res_mode_chk.toggled.connect(self.update_analysis)
        disp_layout.addWidget(self.res_mode_chk)
        
        self.thermal_chk = QCheckBox("Show Thermal Limit")
        self.thermal_chk.setChecked(True)
        self.thermal_chk.toggled.connect(self.update_analysis)
        disp_layout.addWidget(self.thermal_chk)
        
        disp_group.setLayout(disp_layout)
        left_panel.addWidget(disp_group)
        
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
        self.thermal_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('m', style=Qt.PenStyle.DashDotLine, width=1), label='Thermal Limit', labelOpts={'position':0.9, 'color': (200,0,200), 'movable': True})
        self.plot_widget.addItem(self.thermal_line)
        
        center_panel.addWidget(self.plot_widget, 2)
        
        # Stacked Bar Chart (Noise Contribution)
        self.stack_widget = pg.PlotWidget(title="Noise Contribution (%)")
        self.stack_widget.setMouseEnabled(x=False, y=False)
        self.stack_widget.setMenuEnabled(False)
        self.stack_widget.hideAxis('left')
        self.stack_widget.setXRange(0, 100)
        self.stack_widget.setYRange(0, 1)
        self.stack_widget.getPlotItem().hideButtons()
        
        # Add Legend
        self.stack_legend = self.stack_widget.addLegend(offset=(10, 10))
        
        # Bars (using BarGraphItem logic manually or stacked curves)
        # We will use 3 BarGraphItems for Hum, White, 1/f
        # Horizontal bars: x0 (left), width (length), y (center), height (thickness)
        # Note: pyqtgraph BarGraphItem arguments: x, height (vertical) OR x0, x1, y, height?
        # Let's check docs or common usage. usually x, height, width, brush.
        # For horizontal: y, height, width (length), x0 (start).
        
        self.bar_hum = pg.BarGraphItem(x0=[0], y=[0.5], height=[0.6], width=[0], brush='c', name='Hum')
        self.bar_white = pg.BarGraphItem(x0=[0], y=[0.5], height=[0.6], width=[0], brush='g', name='White')
        self.bar_flicker = pg.BarGraphItem(x0=[0], y=[0.5], height=[0.6], width=[0], brush='r', name='1/f')
        
        self.stack_widget.addItem(self.bar_hum)
        self.stack_widget.addItem(self.bar_white)
        self.stack_widget.addItem(self.bar_flicker)
        
        center_panel.addWidget(self.stack_widget, 1)
        
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
            
            # Constants
            k = 1.380649e-23
            T = self.module.temperature_c + 273.15
            R_in = self.module.input_impedance
            
            # Thermal Noise Density (V/rtHz)
            thermal_density = np.sqrt(4 * k * T * R_in)
            
            # Resistance Mode Logic
            is_res_mode = self.res_mode_chk.isChecked()
            
            if is_res_mode:
                # Convert to Ohms: R = V^2 / (4kT)
                # Avoid division by zero
                denom = 4 * k * T
                mag_plot = (avg_mag**2) / denom
                
                # Thermal Limit (Ohms) -> R_in
                thermal_limit_val = R_in
                
                # Update Labels
                self.plot_widget.setLabel('left', 'Equivalent Resistance', units='Ω')
                self.plot_widget.setTitle("Noise Resistance (Log-Log)")
                
                # Update Curves
                self.plot_curve.setData(freqs[1:], mag_plot[1:])
                
                # Fit Line (Convert V fit to R fit)
                if results['flicker_slope'] != 0:
                    f_fit = np.logspace(0, 2, 100)
                    # V density fit (log10)
                    y_fit_v_log = results['flicker_slope'] * np.log10(f_fit) + results['flicker_intercept']
                    y_fit_v = 10**(y_fit_v_log) # Convert log10(V) to Linear V
                    y_fit_r = (y_fit_v**2) / denom
                    self.fit_curve.setData(f_fit, y_fit_r)
                else:
                    self.fit_curve.setData([], [])
                
                # Hum Markers
                hum_freqs = [h[0] for h in results['hum_components']]
                hum_vals = []
                for f in hum_freqs:
                    idx = np.argmin(np.abs(freqs - f))
                    hum_vals.append(mag_plot[idx])
                self.hum_curve.setData(hum_freqs, hum_vals)
                
                # White Noise Floor
                white_r = (results['white_density']**2) / denom
                self.white_curve.setData([10, 20000], [white_r, white_r])
                
            else:
                # Voltage Mode (PSD dBV)
                mag_plot = 20 * np.log10(avg_mag + 1e-15)
                
                # Thermal Limit (dBV)
                thermal_limit_val = 20 * np.log10(thermal_density + 1e-15)
                
                # Update Labels
                self.plot_widget.setLabel('left', 'Noise Density', units='dBV/√Hz')
                self.plot_widget.setTitle("Noise PSD (Log-Log)")
                
                self.plot_curve.setData(freqs[1:], mag_plot[1:])
                
                # Fit Line
                if results['flicker_slope'] != 0:
                    f_fit = np.logspace(0, 2, 100)
                    # fit is log10(V), we need 20*log10(V)
                    y_fit_log = results['flicker_slope'] * np.log10(f_fit) + results['flicker_intercept']
                    y_fit_db = 20 * y_fit_log
                    self.fit_curve.setData(f_fit, y_fit_db)
                else:
                    self.fit_curve.setData([], [])
                    
                # Hum Markers
                hum_freqs = [h[0] for h in results['hum_components']]
                hum_vals = []
                for f in hum_freqs:
                    idx = np.argmin(np.abs(freqs - f))
                    hum_vals.append(mag_plot[idx])
                self.hum_curve.setData(hum_freqs, hum_vals)
                
                # White Noise Floor
                white_level_db = 20 * np.log10(results['white_density'] + 1e-15)
                self.white_curve.setData([10, 20000], [white_level_db, white_level_db])

            # Update Thermal Limit Line
            if self.thermal_chk.isChecked():
                self.thermal_line.show()
                self.thermal_line.setValue(thermal_limit_val)
            else:
                self.thermal_line.hide()

            # 4. Update Stacked Bar Chart
            self.update_stack_chart(results)
            
            # 5. Update Report
            self.update_report(results)
            
        except Exception as e:
            print(f"Error in NoiseProfiler update: {e}")
            import traceback
            traceback.print_exc()
        
    def update_stack_chart(self, results):
        # Calculate Power Contributions
        # Total Power = (RMS)^2
        p_total = results['noise_rms_20k']**2
        
        # Hum Power
        p_hum = results['hum_rms']**2
        
        # White Noise Power (Density^2 * BW)
        # BW is approx 20kHz
        p_white = (results['white_density']**2) * 20000
        
        # 1/f Power (Remainder)
        p_flicker = p_total - p_hum - p_white
        if p_flicker < 0: p_flicker = 0
        
        # Normalize to %
        if p_total > 0:
            pct_hum = (p_hum / p_total) * 100
            pct_white = (p_white / p_total) * 100
            pct_flicker = (p_flicker / p_total) * 100
        else:
            pct_hum = 0
            pct_white = 0
            pct_flicker = 0
            
        # Update Bars
        # Hum (Cyan) starts at 0
        self.bar_hum.setOpts(width=[pct_hum], x0=[0])
        
        # White (Green) starts after Hum
        self.bar_white.setOpts(width=[pct_white], x0=[pct_hum])
        
        # Flicker (Red) starts after White
        self.bar_flicker.setOpts(width=[pct_flicker], x0=[pct_hum + pct_white])
        
        # Update Title with Total RMS
        self.stack_widget.setTitle(f"Noise Contribution (Total: {results['noise_rms_20k']*1e6:.2f} µVrms)")
        
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
