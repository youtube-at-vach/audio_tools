import argparse
import numpy as np
import pyqtgraph as pg
from scipy.signal import hilbert
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QProgressBar, QSpinBox, QTabWidget, QMessageBox, QApplication)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import time
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine

class ImpedanceAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.buffer_size = 4096 
        self.input_data = np.zeros((self.buffer_size, 2))
        
        # Settings
        self.gen_frequency = 1000.0
        self.gen_amplitude = 0.5 
        self.output_channel = 0 # 0: Left, 1: Right, 2: Stereo
        self.voltage_channel = 0 # 0: Left, 1: Right
        self.current_channel = 1 # 0: Left, 1: Right
        self.shunt_resistance = 100.0
        
        # Calibration Data (Freq -> Complex Z)
        # Calibration Data (Freq -> Complex Z)
        self.cal_open = {} 
        self.cal_short = {}
        self.cal_load = {}
        self.load_standard_real = 100.0 # Ohm
        self.use_calibration = False
        
        # Results
        self.meas_v_complex = 0j
        self.meas_i_complex = 0j
        self.meas_z_complex = 0j
        
        # Averaging
        self.averaging_count = 1
        self.history_v = deque(maxlen=100)
        self.history_i = deque(maxlen=100)
        
        self.callback_id = None
        
    @property
    def name(self) -> str:
        return "Impedance Analyzer"

    @property
    def description(self) -> str:
        return "Measure Impedance (Z) using Dual Lock-in Amplifier."

    def run(self, args: argparse.Namespace):
        print("Impedance Analyzer running from CLI (not fully implemented)")

    def get_widget(self):
        return ImpedanceAnalyzerWidget(self)

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        self.input_data = np.zeros((self.buffer_size, 2))
        
        # Generator State
        self._phase = 0
        sample_rate = self.audio_engine.sample_rate
        
        def callback(indata, outdata, frames, time, status):
            if status:
                print(status)
            
            # --- Input Capture ---
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
            
            # --- Output Generation ---
            t = (np.arange(frames) + self._phase) / sample_rate
            self._phase += frames
            
            signal = self.gen_amplitude * np.cos(2 * np.pi * self.gen_frequency * t)
            
            outdata.fill(0)
            if self.output_channel == 2: # Stereo
                if outdata.shape[1] >= 1: outdata[:, 0] = signal
                if outdata.shape[1] >= 2: outdata[:, 1] = signal
            elif outdata.shape[1] > self.output_channel:
                outdata[:, self.output_channel] = signal

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_analysis(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False

    def process_data(self):
        """
        Perform Dual Lock-in calculation.
        """
        data = self.input_data
        
        # Extract Signals
        sig_v = data[:, self.voltage_channel]
        sig_i = data[:, self.current_channel]
        
        # Generate Reference Phasor (Internally generated sine)
        # We need to reconstruct the reference phase corresponding to the buffer.
        # However, since we are generating the signal, we know the frequency.
        # But we don't know the exact phase of the buffer relative to generation start easily 
        # without passing it from callback. 
        # ALTERNATIVE: Use one of the channels (e.g. V) as phase reference if we only care about Z phase?
        # NO, we need absolute phase relative to generator or relative to each other.
        # Relative to each other is Z phase.
        # So we can just lock-in both to a common reference.
        # Ideally we should use the Loopback (REF) if available, but here we assume 
        # internal generation is perfect and we just want relative phase V vs I.
        # Actually, if we use Hilbert on the generated signal (or just cos/sin), we need time alignment.
        # 
        # SIMPLIFICATION:
        # We can treat V as the reference for phase 0? No, Z phase is Phase(V) - Phase(I).
        # So we can calculate V phasor and I phasor independently using an arbitrary reference 
        # (e.g. the first sample of the buffer is t=0).
        # As long as V and I are from the same buffer, the relative phase is preserved.
        
        # Create a local reference sine/cosine for demodulation
        # We assume the buffer contains integer number of cycles or we use windowing/long buffer.
        # Lock-in usually multiplies by cos(wt) and sin(wt).
        # Let's generate a reference vector for the buffer duration.
        # We don't know the absolute time, so the absolute phase will be drifting if freq is not exact bin.
        # BUT, we calculate V and I using the SAME reference vector.
        # So Phase(V) will drift, Phase(I) will drift, but Phase(V) - Phase(I) will be constant.
        
        t = np.arange(len(sig_v)) / self.audio_engine.sample_rate
        ref_cos = np.cos(2 * np.pi * self.gen_frequency * t)
        ref_sin = np.sin(2 * np.pi * self.gen_frequency * t)
        ref_phasor = ref_cos - 1j * ref_sin # exp(-jwt)
        
        # Demodulate V
        # V_complex = mean(sig_v * ref_phasor) * 2
        v_raw = np.mean(sig_v * ref_phasor) * 2
        
        # Demodulate I
        # I_complex = mean(sig_i * ref_phasor) * 2
        i_raw = np.mean(sig_i * ref_phasor) * 2
        
        # Averaging
        self.history_v.append(v_raw)
        self.history_i.append(i_raw)
        
        while len(self.history_v) > self.averaging_count:
            self.history_v.popleft()
            self.history_i.popleft()
            
        avg_v = np.mean(self.history_v)
        avg_i = np.mean(self.history_i)
        
        self.meas_v_complex = avg_v
        self.meas_i_complex = avg_i
        
        # Calculate Z
        # I_actual = - I_measured / R_shunt
        # Z = V / I_actual = V / (- I_measured / R_shunt) = - V * R_shunt / I_measured
        
        if abs(avg_i) > 1e-12:
            z_raw = - (avg_v * self.shunt_resistance) / avg_i
        else:
            z_raw = 0j
            
        # Apply Calibration
        if self.use_calibration:
            self.meas_z_complex = self.apply_calibration(z_raw, self.gen_frequency)
        else:
            self.meas_z_complex = z_raw

    def apply_calibration(self, z_meas, freq):
        """
        Apply Open/Short/Load (OSL) calibration.
        Formula:
        Z_dut = Z_std * ((Z_open - Z_load) * (Z_meas - Z_short)) / ((Z_open - Z_meas) * (Z_load - Z_short))
        
        Fallback to Open/Short (OS) if Load not available:
        Z_dut = (Z_meas - Z_short) / (1 - (Z_meas - Z_short) * Y_open)
        """
        if not self.cal_short or not self.cal_open:
            return z_meas
            
        # Find nearest freq
        freqs = list(self.cal_short.keys())
        nearest_f = min(freqs, key=lambda x: abs(x - freq))
        
        # Check if nearest is reasonably close (e.g. within 5%)
        if abs(nearest_f - freq) / freq > 0.05:
            return z_meas # No valid cal data
            
        z_short = self.cal_short[nearest_f]
        z_open = self.cal_open[nearest_f]
        z_load = self.cal_load.get(nearest_f, None)
        
        # OSL Calibration
        if z_load is not None:
            z_std = self.load_standard_real
            
            # Denominator check
            term1 = z_open - z_meas
            term2 = z_load - z_short
            if abs(term1) < 1e-12 or abs(term2) < 1e-12:
                return z_meas
                
            numerator = z_std * (z_open - z_load) * (z_meas - z_short)
            denominator = term1 * term2
            
            return numerator / denominator
            
        # OS Calibration (Fallback)
        if z_open == 0: return z_meas
        y_open = 1.0 / z_open
        
        numerator = z_meas - z_short
        denominator = 1.0 - (numerator * y_open)
        
        if abs(denominator) < 1e-12:
            return z_meas
            
        return numerator / denominator


class ImpedanceSweepWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(float, complex) # freq, z_complex
    finished_sweep = pyqtSignal()
    
    def __init__(self, module: ImpedanceAnalyzer, start_f, end_f, steps, log_sweep, settle_time):
        super().__init__()
        self.module = module
        self.start_f = start_f
        self.end_f = end_f
        self.steps = steps
        self.log_sweep = log_sweep
        self.settle_time = settle_time
        self.is_cancelled = False
        
    def run(self):
        if self.log_sweep:
            freqs = np.logspace(np.log10(self.start_f), np.log10(self.end_f), self.steps)
        else:
            freqs = np.linspace(self.start_f, self.end_f, self.steps)
            
        if not self.module.is_running:
            self.module.start_analysis()
            time.sleep(0.5)
            
        for i, f in enumerate(freqs):
            if self.is_cancelled: break
            
            self.module.gen_frequency = f
            time.sleep(self.settle_time)
            
            # Clear history to avoid averaging with old freq data
            self.module.history_v.clear()
            self.module.history_i.clear()
            
            # Wait for buffer fill
            sample_rate = self.module.audio_engine.sample_rate
            buffer_duration = self.module.buffer_size / sample_rate
            wait_time = max(0.05, buffer_duration)
            
            time.sleep(wait_time) # Wait for settling in buffer
            
            # Average
            for _ in range(self.module.averaging_count):
                if self.is_cancelled: break
                time.sleep(wait_time)
                self.module.process_data()
                
            z = self.module.meas_z_complex
            self.result.emit(f, z)
            self.progress.emit(int((i+1)/self.steps * 100))
            
        self.finished_sweep.emit()
        
    def cancel(self):
        self.is_cancelled = True


class ImpedanceAnalyzerWidget(QWidget):
    def __init__(self, module: ImpedanceAnalyzer):
        super().__init__()
        self.module = module
        
        # Sweep Data
        self.sweep_freqs = []
        self.sweep_z_complex = [] # Store full complex data
        self.sweep_z_mags = []
        self.sweep_z_phases = []
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.setInterval(100) 

        self.sweep_worker = None
        self.cal_mode = None # 'open', 'short', or None (DUT)

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # --- Tab 1: Manual Control ---
        manual_widget = QWidget()
        manual_layout = QHBoxLayout(manual_widget)
        
        # Settings (Left Panel)
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()
        
        self.toggle_btn = QPushButton("Start Measurement")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; color: black; font-weight: bold; padding: 10px; } QPushButton:checked { background-color: #ffcccc; }")
            
        settings_layout.addRow(self.toggle_btn)
        
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000); self.freq_spin.setValue(1000); self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(lambda v: setattr(self.module, 'gen_frequency', v))
        settings_layout.addRow("Frequency:", self.freq_spin)
        
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0, 1.0); self.amp_spin.setValue(0.5); self.amp_spin.setSingleStep(0.1)
        self.amp_spin.valueChanged.connect(lambda v: setattr(self.module, 'gen_amplitude', v))
        settings_layout.addRow("Amplitude (0-1):", self.amp_spin)
        
        self.shunt_spin = QDoubleSpinBox()
        self.shunt_spin.setRange(0.1, 1000000); self.shunt_spin.setValue(100.0); self.shunt_spin.setSuffix(" Ohm")
        self.shunt_spin.valueChanged.connect(lambda v: setattr(self.module, 'shunt_resistance', v))
        settings_layout.addRow("Shunt R:", self.shunt_spin)
        
        self.load_std_spin = QDoubleSpinBox()
        self.load_std_spin.setRange(0.1, 1000000); self.load_std_spin.setValue(100.0); self.load_std_spin.setSuffix(" Ohm")
        self.load_std_spin.setToolTip("Resistance of the Load Standard used for OSL Calibration.")
        self.load_std_spin.valueChanged.connect(lambda v: setattr(self.module, 'load_standard_real', v))
        settings_layout.addRow("Load Std R:", self.load_std_spin)
        
        # Channels
        self.out_ch_combo = QComboBox()
        self.out_ch_combo.addItems(["Left", "Right", "Stereo"])
        self.out_ch_combo.currentIndexChanged.connect(lambda i: setattr(self.module, 'output_channel', i))
        settings_layout.addRow("Output Ch:", self.out_ch_combo)
        
        self.v_ch_combo = QComboBox()
        self.v_ch_combo.addItems(["Left (Ch 1)", "Right (Ch 2)"])
        self.v_ch_combo.setCurrentIndex(0)
        self.v_ch_combo.currentIndexChanged.connect(lambda i: setattr(self.module, 'voltage_channel', i))
        settings_layout.addRow("Voltage Ch:", self.v_ch_combo)
        
        self.i_ch_combo = QComboBox()
        self.i_ch_combo.addItems(["Left (Ch 1)", "Right (Ch 2)"])
        self.i_ch_combo.setCurrentIndex(1)
        self.i_ch_combo.currentIndexChanged.connect(lambda i: setattr(self.module, 'current_channel', i))
        settings_layout.addRow("Current Ch:", self.i_ch_combo)
        
        self.cal_check = QCheckBox("Apply Calibration")
        self.cal_check.toggled.connect(lambda c: setattr(self.module, 'use_calibration', c))
        settings_layout.addRow(self.cal_check)
        
        settings_group.setLayout(settings_layout)
        manual_layout.addWidget(settings_group, stretch=1)
        
        # Readings (Right Panel)
        readings_widget = QWidget()
        readings_layout = QVBoxLayout(readings_widget)
        
        # Impedance (Primary Result)
        z_group = QGroupBox("Impedance (Z)")
        z_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        z_layout = QHBoxLayout()
        
        # Z Magnitude & Phase
        z_mp_layout = QVBoxLayout()
        self.z_mag_label = QLabel("0.00 Ω")
        self.z_mag_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #00ff00;")
        self.z_mag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_mp_layout.addWidget(QLabel("Magnitude (|Z|)"))
        z_mp_layout.addWidget(self.z_mag_label)
        
        self.z_phase_label = QLabel("0.00°")
        self.z_phase_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ffff;")
        self.z_phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_mp_layout.addWidget(QLabel("Phase (θ)"))
        z_mp_layout.addWidget(self.z_phase_label)
        z_layout.addLayout(z_mp_layout)
        
        # Z Real & Imag
        z_ri_layout = QVBoxLayout()
        self.z_r_label = QLabel("0.00 Ω")
        self.z_r_label.setStyleSheet("font-size: 20px; color: #ffff00;")
        self.z_r_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_ri_layout.addWidget(QLabel("Resistance (R)"))
        z_ri_layout.addWidget(self.z_r_label)
        
        self.z_x_label = QLabel("0.00 Ω")
        self.z_x_label.setStyleSheet("font-size: 20px; color: #ff00ff;")
        self.z_x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_ri_layout.addWidget(QLabel("Reactance (X)"))
        z_ri_layout.addWidget(self.z_x_label)
        z_layout.addLayout(z_ri_layout)
        
        z_group.setLayout(z_layout)
        readings_layout.addWidget(z_group)
        
        # Detailed V/I
        details_layout = QHBoxLayout()
        
        # Voltage Box
        v_group = QGroupBox("Voltage (CH1)")
        v_layout = QFormLayout()
        self.v_mag_lbl = QLabel("0.000 V")
        self.v_phase_lbl = QLabel("0.00°")
        self.v_x_lbl = QLabel("0.000 V")
        self.v_y_lbl = QLabel("0.000 V")
        v_layout.addRow("Amp (A):", self.v_mag_lbl)
        v_layout.addRow("Phase:", self.v_phase_lbl)
        v_layout.addRow("X (In-phase):", self.v_x_lbl)
        v_layout.addRow("Y (Quad):", self.v_y_lbl)
        v_group.setLayout(v_layout)
        details_layout.addWidget(v_group)
        
        # Current Box
        i_group = QGroupBox("Current (CH2)")
        i_layout = QFormLayout()
        self.i_mag_lbl = QLabel("0.000 mA")
        self.i_phase_lbl = QLabel("0.00°")
        self.i_x_lbl = QLabel("0.000 mA")
        self.i_y_lbl = QLabel("0.000 mA")
        i_layout.addRow("Amp (A):", self.i_mag_lbl)
        i_layout.addRow("Phase:", self.i_phase_lbl)
        i_layout.addRow("X (In-phase):", self.i_x_lbl)
        i_layout.addRow("Y (Quad):", self.i_y_lbl)
        i_group.setLayout(i_layout)
        details_layout.addWidget(i_group)
        
        readings_layout.addLayout(details_layout)
        readings_layout.addStretch()
        
        manual_layout.addWidget(readings_widget, stretch=2)
        
        self.tabs.addTab(manual_widget, "Manual Mode")
        
        # --- Tab 2: Sweep & Calibration ---
        sweep_widget = QWidget()
        sweep_layout = QHBoxLayout(sweep_widget)
        
        # Sweep Settings
        sweep_settings = QGroupBox("Sweep Control")
        sweep_form = QFormLayout()
        
        self.sw_start = QDoubleSpinBox(); self.sw_start.setRange(20, 20000); self.sw_start.setValue(20)
        sweep_form.addRow("Start Freq:", self.sw_start)
        
        self.sw_end = QDoubleSpinBox(); self.sw_end.setRange(20, 20000); self.sw_end.setValue(20000)
        sweep_form.addRow("End Freq:", self.sw_end)
        
        self.sw_steps = QSpinBox(); self.sw_steps.setRange(10, 1000); self.sw_steps.setValue(50)
        sweep_form.addRow("Steps:", self.sw_steps)
        
        self.sw_log = QCheckBox("Log Sweep"); self.sw_log.setChecked(True)
        sweep_form.addRow(self.sw_log)
        
        self.btn_open = QPushButton("Measure OPEN (Cal)")
        self.btn_open.clicked.connect(lambda: self.start_sweep('open'))
        sweep_form.addRow(self.btn_open)
        
        self.btn_short = QPushButton("Measure SHORT (Cal)")
        self.btn_short.clicked.connect(lambda: self.start_sweep('short'))
        sweep_form.addRow(self.btn_short)
        
        self.btn_load = QPushButton("Measure LOAD (Cal)")
        self.btn_load.clicked.connect(lambda: self.start_sweep('load'))
        sweep_form.addRow(self.btn_load)
        
        self.btn_dut = QPushButton("Measure DUT")
        self.btn_dut.clicked.connect(lambda: self.start_sweep(None))
        self.btn_dut.setStyleSheet("font-weight: bold; background-color: #ccccff; color: black;")
        sweep_form.addRow(self.btn_dut)
        
        self.sw_progress = QProgressBar()
        sweep_form.addRow(self.sw_progress)
        
        sweep_settings.setLayout(sweep_form)
        sweep_layout.addWidget(sweep_settings, stretch=1)
        
        # Plot
        plot_layout = QVBoxLayout()
        
        # Plot Mode Selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Plot Mode:"))
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems(["|Z| & Phase", "R & X (ESR/ESL)", "Q Factor", "C / L"])
        self.plot_mode_combo.currentIndexChanged.connect(self.update_plot_mode)
        mode_layout.addWidget(self.plot_mode_combo)
        mode_layout.addStretch()
        plot_layout.addLayout(mode_layout)

        self.plot_widget = pg.PlotWidget(title="Impedance Z(f)")
        self.plot_widget.setLabel('bottom', "Frequency", units='Hz')
        self.plot_widget.setLabel('left', "|Z|", units='Ohm')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.getPlotItem().setLogMode(x=True, y=True)
        
        # Primary Curve (Left Axis)
        self.curve_primary = pg.PlotCurveItem(pen='g', name='|Z|')
        self.plot_widget.addItem(self.curve_primary)
        
        self.curve_secondary = pg.PlotCurveItem(pen='y', name='Secondary') # Used for R/X mode
        self.plot_widget.addItem(self.curve_secondary)
        self.curve_secondary.setVisible(False)
        
        # Secondary Axis (Right Axis) - For Phase or L/C split
        self.plot_right = pg.ViewBox()
        self.plot_widget.scene().addItem(self.plot_right)
        self.plot_widget.getPlotItem().showAxis('right')
        self.plot_widget.getPlotItem().scene().addItem(self.plot_right)
        self.plot_widget.getPlotItem().getAxis('right').linkToView(self.plot_right)
        self.plot_right.setXLink(self.plot_widget.getPlotItem())
        self.plot_widget.getPlotItem().getAxis('right').setLabel('Phase', units='deg')
        
        # Ensure Right Axis is Linear by default
        self.plot_widget.getPlotItem().getAxis('right').setLogMode(False)
        
        self.curve_right = pg.PlotCurveItem(pen='c', name='Phase')
        self.plot_right.addItem(self.curve_right)
        
        # Legend
        self.legend = self.plot_widget.addLegend()
        
        # Initial Plot Setup
        self.update_plot_mode()
        
        def update_views():
            self.plot_right.setGeometry(self.plot_widget.getPlotItem().vb.sceneBoundingRect())
            self.plot_right.linkedViewChanged(self.plot_widget.getPlotItem().vb, self.plot_right.XAxis)
        self.plot_widget.getPlotItem().vb.sigResized.connect(update_views)
        
        plot_layout.addWidget(self.plot_widget)
        sweep_layout.addLayout(plot_layout, stretch=3)
        
        self.tabs.addTab(sweep_widget, "Sweep / Calibration")
        
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText("Stop")
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText("Start Measurement")

    def update_ui(self):
        if not self.module.is_running: return
        
        self.module.process_data()
        
        z = self.module.meas_z_complex
        v = self.module.meas_v_complex
        i = self.module.meas_i_complex
        
        # Impedance
        self.z_mag_label.setText(f"{abs(z):.2f} Ω")
        self.z_phase_label.setText(f"{np.degrees(np.angle(z)):.2f}°")
        self.z_r_label.setText(f"{z.real:.2f} Ω")
        self.z_x_label.setText(f"{z.imag:.2f} Ω")
        
        # Voltage
        self.v_mag_lbl.setText(f"{abs(v):.4f} V")
        self.v_phase_lbl.setText(f"{np.degrees(np.angle(v)):.2f}°")
        self.v_x_lbl.setText(f"{v.real:.4f} V")
        self.v_y_lbl.setText(f"{v.imag:.4f} V")
        
        # Current (mA)
        # Calculate actual Current Complex from Voltage across Shunt
        if self.module.shunt_resistance > 0:
            i_complex = - self.module.meas_i_complex / self.module.shunt_resistance
        else:
            i_complex = 0j
            
        i_ma = i_complex * 1000
        self.i_mag_lbl.setText(f"{abs(i_ma):.4f} mA")
        self.i_phase_lbl.setText(f"{np.degrees(np.angle(i_ma)):.2f}°")
        self.i_x_lbl.setText(f"{i_ma.real:.4f} mA")
        self.i_y_lbl.setText(f"{i_ma.imag:.4f} mA")

    def start_sweep(self, mode):
        if self.sweep_worker is not None and self.sweep_worker.isRunning():
            return
            
        self.cal_mode = mode
        self.sweep_freqs = []
        self.sweep_z_complex = []
        self.sweep_z_mags = []
        self.sweep_z_phases = []
        
        # Clear Curves
        self.curve_primary.setData([], [])
        self.curve_secondary.setData([], [])
        self.curve_right.setData([], [])
        
        start = self.sw_start.value()
        end = self.sw_end.value()
        steps = self.sw_steps.value()
        log = self.sw_log.isChecked()
        
        # Initial Plot Setup
        self.update_plot_mode()
        
        # Reset AutoRange
        self.plot_widget.getPlotItem().enableAutoRange()
        self.plot_right.enableAutoRange()
        
        self.sweep_worker = ImpedanceSweepWorker(self.module, start, end, steps, log, 0.2)
        self.sweep_worker.progress.connect(self.sw_progress.setValue)
        self.sweep_worker.result.connect(self.on_sweep_result)
        self.sweep_worker.finished_sweep.connect(self.on_sweep_finished)
        self.sweep_worker.start()
        
    def update_plot_mode(self):
        mode = self.plot_mode_combo.currentText()
        pi = self.plot_widget.getPlotItem()
        ax_right = pi.getAxis('right')
        
        # Default Visibility
        self.curve_secondary.setVisible(False)
        self.curve_right.setVisible(True)
        ax_right.setStyle(showValues=True)
        
        # X-Axis Log Mode
        is_log_x = self.sw_log.isChecked()
        pi.setLogMode(x=is_log_x, y=False) # Reset Y log first
        
        # Clear Legend (Robust)
        if hasattr(self.legend, 'items'):
            # Create a list of labels to remove to avoid modifying list while iterating
            labels_to_remove = [label.text for sample, label in self.legend.items]
            for label_text in labels_to_remove:
                self.legend.removeItem(label_text)
        else:
            self.legend.clear()
        
        if mode == "|Z| & Phase":
            pi.setLabel('left', "|Z|", units='Ohm')
            pi.setLogMode(y=True) # Z is Log Y
            
            self.curve_primary.setData(name='|Z|', pen='g')
            self.legend.addItem(self.curve_primary, '|Z|')
            
            ax_right.setLabel('Phase', units='deg')
            ax_right.setLogMode(False)
            self.curve_right.setData(name='Phase', pen='c')
            self.legend.addItem(self.curve_right, 'Phase')
            
        elif mode == "R & X (ESR/ESL)":
            pi.setLabel('left', "Resistance (R) / Reactance (X)", units='Ohm')
            pi.setLogMode(y=True) # R/X often span large ranges
            
            self.curve_primary.setData(name='Resistance (R)', pen='y')
            self.legend.addItem(self.curve_primary, 'Resistance (R)')
            
            self.curve_secondary.setVisible(True)
            self.curve_secondary.setData(name='Reactance (X)', pen='m')
            self.legend.addItem(self.curve_secondary, 'Reactance (X)')
            
            self.curve_right.setVisible(False)
            ax_right.setStyle(showValues=False)
            ax_right.setLabel('')
            
        elif mode == "Q Factor":
            pi.setLabel('left', "Q Factor")
            pi.setLogMode(y=False) 
            
            self.curve_primary.setData(name='Q', pen='r')
            self.legend.addItem(self.curve_primary, 'Q')
            
            self.curve_right.setVisible(False)
            ax_right.setStyle(showValues=False)
            ax_right.setLabel('')
            
        elif mode == "C / L":
            pi.setLabel('left', "Capacitance", units='F')
            pi.setLogMode(y=True)
            
            self.curve_primary.setData(name='Capacitance', pen='b')
            self.legend.addItem(self.curve_primary, 'Capacitance')
            
            ax_right.setLabel('Inductance', units='H')
            ax_right.setLogMode(True) # L is Log Y
            
            self.curve_right.setData(name='Inductance', pen='r')
            self.legend.addItem(self.curve_right, 'Inductance')
            
        # Re-plot data if available
        if self.sweep_freqs:
            self.refresh_plot_data()

    def refresh_plot_data(self):
        if not self.sweep_freqs: return
        
        mode = self.plot_mode_combo.currentText()
        freqs = np.array(self.sweep_freqs)
        zs = np.array(self.sweep_z_complex)
        
        # X-Axis Data (Manual Log)
        is_log_x = self.plot_widget.getPlotItem().getAxis('bottom').logMode
        if is_log_x:
            x_data = np.log10(freqs)
        else:
            x_data = freqs
            
        if mode == "|Z| & Phase":
            # |Z| (Log Y)
            y_data = np.abs(zs)
            if self.plot_widget.getPlotItem().getAxis('left').logMode:
                y_data = np.log10(y_data)
            self.curve_primary.setData(x_data, y_data)
            
            # Phase (Linear Y)
            self.curve_right.setData(x_data, np.degrees(np.angle(zs)))
            
        elif mode == "R & X (ESR/ESL)":
            # R (Log Y)
            r_data = np.abs(zs.real)
            x_data_val = np.abs(zs.imag)
            
            if self.plot_widget.getPlotItem().getAxis('left').logMode:
                # Avoid log(0)
                r_data = np.log10(r_data + 1e-12)
                x_data_val = np.log10(x_data_val + 1e-12)
                
            self.curve_primary.setData(x_data, r_data)
            self.curve_secondary.setData(x_data, x_data_val)
            
        elif mode == "Q Factor":
            # Q = |X| / R
            rs = zs.real
            xs = zs.imag
            qs = np.zeros_like(rs)
            mask = (np.abs(rs) > 1e-12)
            qs[mask] = np.abs(xs[mask]) / np.abs(rs[mask])
            
            self.curve_primary.setData(x_data, qs)
            
        elif mode == "C / L":
            # C = -1 / (w * X) for X < 0
            # L = X / w for X > 0
            w = 2 * np.pi * freqs
            xs = zs.imag
            
            # Capacitance (valid where X < 0)
            cs = np.full_like(xs, np.nan)
            mask_c = (xs < -1e-12)
            cs[mask_c] = -1.0 / (w[mask_c] * xs[mask_c])
            
            # Inductance (valid where X > 0)
            ls = np.full_like(xs, np.nan)
            mask_l = (xs > 1e-12)
            ls[mask_l] = xs[mask_l] / w[mask_l]
            
            # Log Y for both
            # Left Axis (C)
            if self.plot_widget.getPlotItem().getAxis('left').logMode:
                 # Handle NaNs and log
                 valid_c = ~np.isnan(cs)
                 cs_plot = np.full_like(cs, np.nan)
                 cs_plot[valid_c] = np.log10(cs[valid_c])
                 self.curve_primary.setData(x_data, cs_plot)
            else:
                 self.curve_primary.setData(x_data, cs)
                 
            # Right Axis (L)
            if self.plot_widget.getPlotItem().getAxis('right').logMode:
                 valid_l = ~np.isnan(ls)
                 ls_plot = np.full_like(ls, np.nan)
                 ls_plot[valid_l] = np.log10(ls[valid_l])
                 self.curve_right.setData(x_data, ls_plot)
            else:
                 self.curve_right.setData(x_data, ls)

    def on_sweep_result(self, f, z):
        if self.cal_mode == 'open':
            self.module.cal_open[f] = z
        elif self.cal_mode == 'short':
            self.module.cal_short[f] = z
        elif self.cal_mode == 'load':
            self.module.cal_load[f] = z
        else:
            # DUT Measurement
            self.sweep_freqs.append(f)
            self.sweep_z_complex.append(z)
            self.sweep_z_mags.append(abs(z))
            self.sweep_z_phases.append(np.degrees(np.angle(z)))
            
            # Update Plot
            self.refresh_plot_data()

    def on_sweep_finished(self):
        if self.cal_mode == 'open':
            QMessageBox.information(self, "Calibration", "Open Calibration Completed")
        elif self.cal_mode == 'short':
            QMessageBox.information(self, "Calibration", "Short Calibration Completed")
        elif self.cal_mode == 'load':
            QMessageBox.information(self, "Calibration", "Load Calibration Completed")

    def apply_theme(self, theme_name):
        if theme_name == 'system' and hasattr(self.app, 'theme_manager'):
            theme_name = self.app.theme_manager.get_effective_theme()
            
        if theme_name == 'dark':
            # Dark Theme
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; border: 1px solid #555; border-radius: 4px; padding: 10px; font-weight: bold; }"
                "QPushButton:checked { background-color: #c62828; color: white; border: 1px solid #555; border-radius: 4px; padding: 10px; }"
                "QPushButton:hover { background-color: #388e3c; }"
                "QPushButton:checked:hover { background-color: #d32f2f; }"
            )
            self.z_mag_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #00ff00;")
            self.z_phase_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ffff;")
            self.z_r_label.setStyleSheet("font-size: 20px; color: #ffff00;")
            self.z_x_label.setStyleSheet("font-size: 20px; color: #ff00ff;")
            self.btn_dut.setStyleSheet("font-weight: bold; background-color: #5e35b1; color: white;")
        else:
            # Light Theme
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #ccffcc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 10px; font-weight: bold; }"
                "QPushButton:checked { background-color: #ffcccc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 10px; }"
                "QPushButton:hover { background-color: #bbfebb; }"
                "QPushButton:checked:hover { background-color: #ffbbbb; }"
            )
            self.z_mag_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #008800;")
            self.z_phase_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #008888;")
            self.z_r_label.setStyleSheet("font-size: 20px; color: #888800;")
            self.z_x_label.setStyleSheet("font-size: 20px; color: #880088;")
            self.btn_dut.setStyleSheet("font-weight: bold; background-color: #ccccff; color: black;")

        # Theme handling
        self.app = QApplication.instance()
        if hasattr(self.app, 'theme_manager'):
            self.app.theme_manager.theme_changed.connect(self.apply_theme)
            self.apply_theme(self.app.theme_manager.get_current_theme())
