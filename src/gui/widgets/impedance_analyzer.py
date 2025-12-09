import argparse
import json
import numpy as np
import pyqtgraph as pg
from scipy.signal import hilbert
from collections import deque
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QProgressBar, QSpinBox, QTabWidget, QMessageBox, QApplication, QGridLayout, QFileDialog)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
import time
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

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
        self.use_cal_interpolation = True
        
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
        return tr("Impedance Analyzer")

    @property
    def description(self) -> str:
        return tr("Measure Impedance (Z) using Dual Lock-in Amplifier.")

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
            
        # Get Calibration Data (Always Interpolate)
        z_short = self._get_interpolated_cal_value(self.cal_short, freq)
        z_open = self._get_interpolated_cal_value(self.cal_open, freq)
        if self.cal_load:
            z_load = self._get_interpolated_cal_value(self.cal_load, freq)
        else:
            z_load = None
        
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
            
    def _get_interpolated_cal_value(self, cal_dict, freq):
        """
        Get interpolated calibration value for a specific frequency.
        Uses linear interpolation on complex real/imag parts.
        If freq is outside range, uses nearest neighbor.
        """
        sorted_freqs = sorted(cal_dict.keys())
        if not sorted_freqs:
            return 0j
            
        if freq <= sorted_freqs[0]:
            return cal_dict[sorted_freqs[0]]
        if freq >= sorted_freqs[-1]:
            return cal_dict[sorted_freqs[-1]]
            
        # Find interval
        # Use binary search or simple iteration (small lists usually)
        for i in range(len(sorted_freqs) - 1):
            f_low = sorted_freqs[i]
            f_high = sorted_freqs[i+1]
            if f_low <= freq <= f_high:
                t = (freq - f_low) / (f_high - f_low)
                z_low = cal_dict[f_low]
                z_high = cal_dict[f_high]
                
                # Interpolate Real and Imag separately
                r = z_low.real + t * (z_high.real - z_low.real)
                im = z_low.imag + t * (z_high.imag - z_low.imag)
                return complex(r, im)
                
        return cal_dict[sorted_freqs[0]] # Should not reach here
            
    def save_calibration(self, filename):
        data = {
            "cal_open": self._serialize_cal(self.cal_open),
            "cal_short": self._serialize_cal(self.cal_short),
            "cal_load": self._serialize_cal(self.cal_load),
            "load_std_real": self.load_standard_real
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def load_calibration(self, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.cal_open = self._deserialize_cal(data.get("cal_open", {}))
            self.cal_short = self._deserialize_cal(data.get("cal_short", {}))
            self.cal_load = self._deserialize_cal(data.get("cal_load", {}))
            self.load_standard_real = data.get("load_std_real", 100.0)
            return True, ""
        except Exception as e:
            return False, str(e)

    def _serialize_cal(self, cal_dict):
        # Dict[float, complex] -> Dict[str, [real, imag]]
        return {str(f): [z.real, z.imag] for f, z in cal_dict.items()}

    def _deserialize_cal(self, data_dict):
        # Dict[str, [real, imag]] -> Dict[float, complex]
        new_cal = {}
        for f_str, z_list in data_dict.items():
            try:
                f = float(f_str)
                z = complex(z_list[0], z_list[1])
                new_cal[f] = z
            except:
                pass
        return new_cal


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


class ImpedanceResultsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.is_detailed = False
        self.circuit_mode = tr("Series") # "Series" or "Parallel"
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Header / Toggle ---
        header_layout = QHBoxLayout()
        self.mode_label = QLabel(f"{tr('Mode:')} {tr('Series')}")
        self.mode_label.setStyleSheet("font-weight: bold; color: #aaa;")
        header_layout.addWidget(self.mode_label)
        
        header_layout.addStretch()
        
        self.detail_btn = QPushButton(tr("Show Details"))
        self.detail_btn.setCheckable(True)
        self.detail_btn.clicked.connect(self.toggle_details)
        self.detail_btn.setStyleSheet("font-size: 10px; padding: 2px;")
        header_layout.addWidget(self.detail_btn)
        layout.addLayout(header_layout)

        # --- Simple View ---
        self.simple_widget = QWidget()
        simple_layout = QGridLayout(self.simple_widget)
        
        # Primary Z
        self.lbl_z_mag = QLabel("0.00 Ω")
        self.lbl_z_mag.setStyleSheet("font-size: 28px; font-weight: bold; color: #4caf50;") # Green
        self.lbl_z_phase = QLabel("0.00°")
        self.lbl_z_phase.setStyleSheet("font-size: 20px; font-weight: bold; color: #2196f3;") # Blue
        
        simple_layout.addWidget(QLabel(tr("|Z|:")), 0, 0)
        simple_layout.addWidget(self.lbl_z_mag, 0, 1)
        simple_layout.addWidget(QLabel(tr("θ:")), 0, 2)
        simple_layout.addWidget(self.lbl_z_phase, 0, 3)
        
        # Secondary (R/X or G/B based on mode)
        self.lbl_p1_name = QLabel(tr("Rs:"))
        self.lbl_p1_val = QLabel("0.00 Ω")
        self.lbl_p1_val.setStyleSheet("font-size: 18px; color: #ffeb3b;") # Yellow
        
        self.lbl_p2_name = QLabel(tr("Xs:"))
        self.lbl_p2_val = QLabel("0.00 Ω")
        self.lbl_p2_val.setStyleSheet("font-size: 18px; color: #e91e63;") # Pink
        
        simple_layout.addWidget(self.lbl_p1_name, 1, 0)
        simple_layout.addWidget(self.lbl_p1_val, 1, 1)
        simple_layout.addWidget(self.lbl_p2_name, 1, 2)
        simple_layout.addWidget(self.lbl_p2_val, 1, 3)
        
        # L/C/Q
        self.lbl_lc_name = QLabel(tr("L:"))
        self.lbl_lc_val = QLabel("0.00 H")
        self.lbl_q_val = QLabel(f"{tr('Q:')} 0.00")
        
        simple_layout.addWidget(self.lbl_lc_name, 2, 0)
        simple_layout.addWidget(self.lbl_lc_val, 2, 1)
        simple_layout.addWidget(self.lbl_q_val, 2, 2, 1, 2)

        layout.addWidget(self.simple_widget)

        # --- Detailed View ---
        self.detail_widget = QWidget()
        self.detail_widget.setVisible(False)
        detail_layout = QGridLayout(self.detail_widget)
        
        # Group 1: Series
        box_s = QGroupBox(tr("Series Equivalent"))
        lay_s = QFormLayout()
        self.val_rs = QLabel("-"); lay_s.addRow(tr("Rs:"), self.val_rs)
        self.val_xs = QLabel("-"); lay_s.addRow(tr("Xs:"), self.val_xs)
        self.val_ls = QLabel("-"); lay_s.addRow(tr("Ls:"), self.val_ls)
        self.val_cs = QLabel("-"); lay_s.addRow(tr("Cs:"), self.val_cs)
        box_s.setLayout(lay_s)
        detail_layout.addWidget(box_s, 0, 0)
        
        # Group 2: Parallel
        box_p = QGroupBox(tr("Parallel Equivalent"))
        lay_p = QFormLayout()
        self.val_rp = QLabel("-"); lay_p.addRow(tr("Rp:"), self.val_rp)
        self.val_xp = QLabel("-"); lay_p.addRow(tr("Xp:"), self.val_xp)
        self.val_lp = QLabel("-"); lay_p.addRow(tr("Lp:"), self.val_lp)
        self.val_cp = QLabel("-"); lay_p.addRow(tr("Cp:"), self.val_cp)
        box_p.setLayout(lay_p)
        detail_layout.addWidget(box_p, 0, 1)
        
        # Group 3: Admittance
        box_y = QGroupBox(tr("Admittance (Y)"))
        lay_y = QFormLayout()
        self.val_y_mag = QLabel("-"); lay_y.addRow(tr("|Y|:"), self.val_y_mag)
        self.val_g = QLabel("-"); lay_y.addRow(tr("G (Cond):"), self.val_g)
        self.val_b = QLabel("-"); lay_y.addRow(tr("B (Susc):"), self.val_b)
        box_y.setLayout(lay_y)
        detail_layout.addWidget(box_y, 1, 0)
        
        # Group 4: Quality / Loss
        box_q = QGroupBox(tr("Quality / Loss"))
        lay_q = QFormLayout()
        self.val_q = QLabel("-"); lay_q.addRow(tr("Q Factor:"), self.val_q)
        self.val_d = QLabel("-"); lay_q.addRow(tr("D (Loss):"), self.val_d)
        self.val_esr = QLabel("-"); lay_q.addRow(tr("ESR:"), self.val_esr) # Same as Rs usually
        box_q.setLayout(lay_q)
        detail_layout.addWidget(box_q, 1, 1)
        
        # Group 5: Raw Signals
        box_raw = QGroupBox(tr("Raw Signals (V / I)"))
        lay_raw = QFormLayout()
        self.val_v = QLabel("-"); lay_raw.addRow(tr("Voltage:"), self.val_v)
        self.val_i = QLabel("-"); lay_raw.addRow(tr("Current:"), self.val_i)
        self.val_v_phase = QLabel("-"); lay_raw.addRow(tr("V Phase:"), self.val_v_phase)
        self.val_i_phase = QLabel("-"); lay_raw.addRow(tr("I Phase:"), self.val_i_phase)
        box_raw.setLayout(lay_raw)
        detail_layout.addWidget(box_raw, 2, 0, 1, 2)
        
        layout.addWidget(self.detail_widget)
        layout.addStretch()

    def toggle_details(self, checked):
        self.is_detailed = checked
        self.detail_widget.setVisible(checked)
        self.detail_btn.setText(tr("Hide Details") if checked else tr("Show Details"))

    def update_data(self, z: complex, v: complex, i: complex, freq: float):
        if freq <= 0: return
        w = 2 * np.pi * freq
        
        # Basic Z
        z_mag = abs(z)
        z_phase = np.degrees(np.angle(z))
        
        self.lbl_z_mag.setText(f"{z_mag:.4g} Ω")
        self.lbl_z_phase.setText(f"{z_phase:.2f}°")
        
        # Series
        rs = z.real
        xs = z.imag
        ls = xs / w if w > 0 else 0
        cs = -1 / (w * xs) if (w > 0 and abs(xs) > 1e-12) else float('inf')
        
        # Parallel
        # Y = 1/Z = G + jB
        if z_mag > 1e-12:
            y = 1.0 / z
            g = y.real
            b = y.imag
            
            rp = 1.0 / g if abs(g) > 1e-12 else float('inf')
            xp = -1.0 / b if abs(b) > 1e-12 else float('inf')
            
            lp = -1.0 / (w * b) if (w > 0 and abs(b) > 1e-12) else float('inf')
            cp = b / w if w > 0 else 0
        else:
            y = 0j; g=0; b=0; rp=0; xp=0; lp=0; cp=0
            
        # Q / D
        q = abs(xs) / abs(rs) if abs(rs) > 1e-12 else float('inf')
        d = 1.0 / q if q > 1e-12 else float('inf')
        
        # --- Update Detailed View ---
        self.val_rs.setText(f"{rs:.4g} Ω")
        self.val_xs.setText(f"{xs:.4g} Ω")
        self.val_ls.setText(f"{ls*1e6:.4g} µH" if abs(ls) < 1 else f"{ls:.4g} H")
        self.val_cs.setText(f"{cs*1e9:.4g} nF" if abs(cs) < 1e-6 else f"{cs*1e6:.4g} µF")
        
        self.val_rp.setText(f"{rp:.4g} Ω")
        self.val_xp.setText(f"{xp:.4g} Ω")
        self.val_lp.setText(f"{lp*1e6:.4g} µH" if abs(lp) < 1 else f"{lp:.4g} H")
        self.val_cp.setText(f"{cp*1e9:.4g} nF" if abs(cp) < 1e-6 else f"{cp*1e6:.4g} µF")
        
        self.val_y_mag.setText(f"{abs(y):.4g} S")
        self.val_g.setText(f"{g:.4g} S")
        self.val_b.setText(f"{b:.4g} S")
        
        self.val_q.setText(f"{q:.4g}")
        self.val_d.setText(f"{d:.4g}")
        self.val_esr.setText(f"{rs:.4g} Ω")
        
        self.val_v.setText(f"{abs(v):.4g} V")
        self.val_i.setText(f"{abs(i)*1000:.4g} mA")
        self.val_v_phase.setText(f"{np.degrees(np.angle(v)):.2f}°")
        self.val_i_phase.setText(f"{np.degrees(np.angle(i)):.2f}°")
        
        # --- Update Simple View ---
        self.mode_label.setText(f"{tr('Mode:')} {self.circuit_mode}")
        
        if self.circuit_mode == tr("Series"):
            self.lbl_p1_name.setText(tr("Rs:"))
            self.lbl_p1_val.setText(f"{rs:.4g} Ω")
            self.lbl_p2_name.setText(tr("Xs:"))
            self.lbl_p2_val.setText(f"{xs:.4g} Ω")
            
            if xs > 0: # Inductive
                self.lbl_lc_name.setText(tr("Ls:"))
                self.lbl_lc_val.setText(f"{ls*1e6:.4g} µH" if abs(ls) < 1e-3 else f"{ls*1e3:.4g} mH")
            else: # Capacitive
                self.lbl_lc_name.setText(tr("Cs:"))
                self.lbl_lc_val.setText(f"{cs*1e9:.4g} nF" if abs(cs) < 1e-6 else f"{cs*1e6:.4g} µF")
                
        else: # Parallel
            self.lbl_p1_name.setText(tr("Rp:"))
            self.lbl_p1_val.setText(f"{rp:.4g} Ω")
            self.lbl_p2_name.setText(tr("Xp:"))
            self.lbl_p2_val.setText(f"{xp:.4g} Ω")
            
            if b < 0: # Inductive (B is negative for Inductor in Admittance? Y = 1/jwL = -j/wL -> B < 0)
                self.lbl_lc_name.setText(tr("Lp:"))
                self.lbl_lc_val.setText(f"{lp*1e6:.4g} µH" if abs(lp) < 1e-3 else f"{lp*1e3:.4g} mH")
            else: # Capacitive (Y = jwC -> B > 0)
                self.lbl_lc_name.setText(tr("Cp:"))
                self.lbl_lc_val.setText(f"{cp*1e9:.4g} nF" if abs(cp) < 1e-6 else f"{cp*1e6:.4g} µF")

        self.lbl_q_val.setText(f"{tr('Q:')} {q:.4g}")


class ImpedanceAnalyzerWidget(QWidget):
    def __init__(self, module: ImpedanceAnalyzer):
        super().__init__()
        self.module = module
        
        # Sweep Data
        self.sweep_freqs = []
        self.sweep_z_complex = [] # Store full complex data
        self.sweep_z_mags = []
        self.sweep_z_phases = []
        
        # Resonance Marker
        self.resonance_line = None
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.setInterval(100) 

        self.sweep_worker = None
        self.cal_mode = None # 'open', 'short', or None (DUT)

    def init_ui(self):
        nyquist_freq = self.module.audio_engine.sample_rate / 2.0
        main_layout = QHBoxLayout(self)
        
        # --- Left Panel: Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(320)
        
        # Tabs
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs)
        
        # --- Tab 1: Manual (Measurement & Config) ---
        tab_manual = QWidget()
        manual_layout = QVBoxLayout(tab_manual)
        
        # 1. Measurement Control
        grp_meas = QGroupBox(tr("Measurement"))
        lay_meas = QFormLayout()
        
        self.toggle_btn = QPushButton(tr("Start Measurement"))
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.on_toggle)
        self.toggle_btn.setStyleSheet("QPushButton { background-color: #ccffcc; color: black; font-weight: bold; padding: 10px; } QPushButton:checked { background-color: #ffcccc; }")
        lay_meas.addRow(self.toggle_btn)
        
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, nyquist_freq); self.freq_spin.setValue(1000); self.freq_spin.setSuffix(" Hz")
        self.freq_spin.valueChanged.connect(lambda v: setattr(self.module, 'gen_frequency', v))
        lay_meas.addRow(tr("Frequency:"), self.freq_spin)
        
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0, 1.0); self.amp_spin.setValue(0.5); self.amp_spin.setSingleStep(0.1)
        self.amp_spin.valueChanged.connect(lambda v: setattr(self.module, 'gen_amplitude', v))
        lay_meas.addRow(tr("Amplitude:"), self.amp_spin)
        
        self.avg_spin = QSpinBox()
        self.avg_spin.setRange(1, 100); self.avg_spin.setValue(self.module.averaging_count)
        self.avg_spin.valueChanged.connect(lambda v: setattr(self.module, 'averaging_count', v))
        lay_meas.addRow(tr("Averages:"), self.avg_spin)
        
        self.circuit_combo = QComboBox()
        self.circuit_combo.addItems([tr("Series"), tr("Parallel")])
        self.circuit_combo.currentTextChanged.connect(self.on_circuit_mode_changed)
        lay_meas.addRow(tr("Circuit Model:"), self.circuit_combo)
        
        grp_meas.setLayout(lay_meas)
        manual_layout.addWidget(grp_meas)
        
        # 2. Configuration
        grp_conf = QGroupBox(tr("Configuration"))
        lay_conf = QFormLayout()
        
        self.shunt_spin = QDoubleSpinBox()
        self.shunt_spin.setRange(0.1, 1000000); self.shunt_spin.setValue(100.0); self.shunt_spin.setSuffix(" Ω")
        self.shunt_spin.valueChanged.connect(lambda v: setattr(self.module, 'shunt_resistance', v))
        lay_conf.addRow(tr("Shunt R:"), self.shunt_spin)
        
        self.load_std_spin = QDoubleSpinBox()
        self.load_std_spin.setRange(0.1, 1000000); self.load_std_spin.setValue(100.0); self.load_std_spin.setSuffix(" Ω")
        self.load_std_spin.valueChanged.connect(lambda v: setattr(self.module, 'load_standard_real', v))
        lay_conf.addRow(tr("Load Std R:"), self.load_std_spin)
        
        grp_conf.setLayout(lay_conf)
        manual_layout.addWidget(grp_conf)
        
        manual_layout.addStretch()
        self.tabs.addTab(tab_manual, tr("Manual"))
        
        # --- Tab 2: Sweep / Cal ---
        tab_sweep = QWidget()
        sweep_layout = QVBoxLayout(tab_sweep)
        
        # 3. Sweep & Cal Actions
        grp_sweep = QGroupBox(tr("Sweep / Calibration"))
        lay_sweep = QFormLayout()

        self.cal_check = QCheckBox(tr("Apply Calibration"))
        self.cal_check.toggled.connect(lambda c: setattr(self.module, 'use_calibration', c))
        lay_sweep.addRow(self.cal_check)
        
        self.sw_start = QDoubleSpinBox(); self.sw_start.setRange(20, nyquist_freq); self.sw_start.setValue(20)
        lay_sweep.addRow(tr("Start:"), self.sw_start)
        self.sw_end = QDoubleSpinBox(); self.sw_end.setRange(20, nyquist_freq); self.sw_end.setValue(min(20000, nyquist_freq))
        lay_sweep.addRow(tr("End:"), self.sw_end)
        self.sw_steps = QSpinBox(); self.sw_steps.setRange(10, 1000); self.sw_steps.setValue(50)
        lay_sweep.addRow(tr("Steps:"), self.sw_steps)
        self.sw_log = QCheckBox(tr("Log Sweep")); self.sw_log.setChecked(True)
        lay_sweep.addRow(self.sw_log)
        
        self.chk_resonance = QCheckBox(tr("Find Resonance"))
        self.chk_resonance.setToolTip(tr("Find the resonance frequency (Points where X=0 / Phase=0)"))
        lay_sweep.addRow(self.chk_resonance)
        
        self.lbl_resonance_result = QLabel("")
        self.lbl_resonance_result.setStyleSheet("color: blue; font-weight: bold;")
        lay_sweep.addRow(self.lbl_resonance_result)
        
        self.sw_progress = QProgressBar()
        lay_sweep.addRow(self.sw_progress)
        
        # Buttons Grid
        btn_grid = QGridLayout()
        self.btn_open = QPushButton(tr("Open Cal"))
        self.btn_open.clicked.connect(lambda: self.start_sweep('open'))
        self.btn_short = QPushButton(tr("Short Cal"))
        self.btn_short.clicked.connect(lambda: self.start_sweep('short'))
        self.btn_load = QPushButton(tr("Load Cal"))
        self.btn_load.clicked.connect(lambda: self.start_sweep('load'))
        self.btn_dut = QPushButton(tr("Sweep DUT"))
        self.btn_dut.clicked.connect(lambda: self.start_sweep(None))
        self.btn_dut.setStyleSheet("font-weight: bold; background-color: #ccccff; color: black;")
        
        btn_grid.addWidget(self.btn_open, 0, 0)
        btn_grid.addWidget(self.btn_short, 0, 1)
        btn_grid.addWidget(self.btn_load, 1, 0)
        btn_grid.addWidget(self.btn_dut, 1, 1)

        self.btn_stop = QPushButton(tr("Stop Sweep"))
        self.btn_stop.clicked.connect(self.stop_sweep)
        self.btn_stop.setStyleSheet("font-weight: bold; background-color: #ffcccc; color: black;")
        self.btn_stop.setEnabled(False) # Default disabled
        btn_grid.addWidget(self.btn_stop, 2, 0, 1, 2) # Span 2 columns
        
        self.btn_save_cal_file = QPushButton(tr("Save Cal File"))
        self.btn_save_cal_file.clicked.connect(self.on_save_cal)
        self.btn_load_cal_file = QPushButton(tr("Load Cal File"))
        self.btn_load_cal_file.clicked.connect(self.on_load_cal)
        
        btn_grid.addWidget(self.btn_save_cal_file, 3, 0)
        btn_grid.addWidget(self.btn_load_cal_file, 3, 1)
        
        lay_sweep.addRow(btn_grid)
        
        grp_sweep.setLayout(lay_sweep)
        sweep_layout.addWidget(grp_sweep)
        
        sweep_layout.addStretch()
        self.tabs.addTab(tab_sweep, tr("Sweep / Cal"))
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # --- Right Panel: Results ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 1. Plot
        self.plot_widget = pg.PlotWidget(title=tr("Impedance Z(f)"))
        self.plot_widget.setLabel('bottom', tr("Frequency"), units='Hz')
        self.plot_widget.setLabel('left', tr("|Z|"), units='Ohm')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.getPlotItem().setLogMode(x=True, y=True)
        
        self.curve_primary = pg.PlotCurveItem(pen='g', name=tr('|Z|'))
        self.plot_widget.addItem(self.curve_primary)
        self.curve_secondary = pg.PlotCurveItem(pen='y', name=tr('Secondary'))
        self.plot_widget.addItem(self.curve_secondary)
        self.curve_secondary.setVisible(False)
        
        # Secondary Axis
        self.plot_right = pg.ViewBox()
        self.plot_widget.scene().addItem(self.plot_right)
        self.plot_widget.getPlotItem().showAxis('right')
        self.plot_widget.getPlotItem().getAxis('right').linkToView(self.plot_right)
        self.plot_right.setXLink(self.plot_widget.getPlotItem())
        self.plot_widget.getPlotItem().getAxis('right').setLabel(tr('Phase'), units='deg')
        
        self.curve_right = pg.PlotCurveItem(pen='c', name=tr('Phase'))
        self.plot_right.addItem(self.curve_right)
        
        self.legend = self.plot_widget.addLegend()
        
        def update_views():
            self.plot_right.setGeometry(self.plot_widget.getPlotItem().vb.sceneBoundingRect())
            self.plot_right.linkedViewChanged(self.plot_widget.getPlotItem().vb, self.plot_right.XAxis)
        self.plot_widget.getPlotItem().vb.sigResized.connect(update_views)
        
        right_layout.addWidget(self.plot_widget, stretch=2)
        
        # Plot Mode Selector (Overlay or below plot)
        pm_layout = QHBoxLayout()
        pm_layout.addWidget(QLabel(tr("Plot Mode:")))
        self.plot_mode_combo = QComboBox()
        self.plot_mode_combo.addItems([tr("|Z| & Phase"), tr("R & X (ESR/ESL)"), tr("D (Tan δ)"), tr("C / L"), tr("Nyquist Plot")])
        self.plot_mode_combo.currentIndexChanged.connect(self.update_plot_mode)
        pm_layout.addWidget(self.plot_mode_combo)
        pm_layout.addStretch()
        right_layout.addLayout(pm_layout)
        
        # 2. Detailed Results Widget
        self.results_widget = ImpedanceResultsWidget()
        right_layout.addWidget(self.results_widget, stretch=1)
        
        main_layout.addWidget(right_panel)
        self.setLayout(main_layout)
        
        # Initial Setup
        self.update_plot_mode()

    def on_toggle(self, checked):
        if checked:
            self.module.start_analysis()
            self.timer.start()
            self.toggle_btn.setText(tr("Stop"))
        else:
            self.module.stop_analysis()
            self.timer.stop()
            self.toggle_btn.setText(tr("Start Measurement"))

    def on_circuit_mode_changed(self, mode):
        self.results_widget.circuit_mode = mode
        # Trigger update if running
        if not self.module.is_running:
            # Manually update with last known data if available
            self.results_widget.update_data(self.module.meas_z_complex, self.module.meas_v_complex, self.module.meas_i_complex, self.module.gen_frequency)

    def update_ui(self):
        if not self.module.is_running: return
        
        self.module.process_data()
        self.results_widget.update_data(self.module.meas_z_complex, self.module.meas_v_complex, self.module.meas_i_complex, self.module.gen_frequency)

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
        
        # Clear Resonance Marker
        if self.resonance_line:
            self.plot_widget.removeItem(self.resonance_line)
            self.resonance_line = None
        self.lbl_resonance_result.setText("")
        
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
        
        # Update UI state
        self.btn_open.setEnabled(False)
        self.btn_short.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.btn_dut.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_sweep(self):
        if self.sweep_worker and self.sweep_worker.isRunning():
            self.sweep_worker.cancel()
            # UI update will happen in on_sweep_finished
        
    def on_save_cal(self):
        filename, _ = QFileDialog.getSaveFileName(self, tr("Save Calibration"), "", tr("JSON Files (*.json)"))
        if filename:
            self.module.save_calibration(filename)
            QMessageBox.information(self, tr("Success"), tr("Calibration saved successfully."))
            
    def on_load_cal(self):
        filename, _ = QFileDialog.getOpenFileName(self, tr("Load Calibration"), "", tr("JSON Files (*.json)"))
        if filename:
            success, msg = self.module.load_calibration(filename)
            if success:
                # Update UI elements that might depend on loaded flags (optional)
                self.load_std_spin.setValue(self.module.load_standard_real)
                QMessageBox.information(self, tr("Success"), tr("Calibration loaded successfully."))
            else:
                QMessageBox.critical(self, tr("Error"), tr("Failed to load calibration: ") + msg)

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
            labels_to_remove = [label.text for sample, label in self.legend.items]
            for label_text in labels_to_remove:
                self.legend.removeItem(label_text)
        else:
            self.legend.clear()
        
        if mode == tr("|Z| & Phase"):
            pi.setLabel('left', tr("|Z|"), units='Ohm')
            pi.setLabel('bottom', tr("Frequency"), units='Hz')
            pi.setLogMode(y=True) # Z is Log Y
            
            self.curve_primary.setData(name=tr('|Z|'), pen='g')
            self.legend.addItem(self.curve_primary, tr('|Z|'))
            
            ax_right.setLabel(tr('Phase'), units='deg')
            ax_right.setLogMode(False)
            self.curve_right.setData(name=tr('Phase'), pen='c')
            self.legend.addItem(self.curve_right, tr('Phase'))
            
        elif mode == tr("R & X (ESR/ESL)"):
            pi.setLabel('left', tr("Resistance (R) / Reactance (X)"), units='Ohm')
            pi.setLabel('bottom', tr("Frequency"), units='Hz')
            pi.setLogMode(y=True) # R/X often span large ranges
            
            self.curve_primary.setData(name=tr('Resistance (R)'), pen='y')
            self.legend.addItem(self.curve_primary, tr('Resistance (R)'))
            
            self.curve_secondary.setVisible(True)
            self.curve_secondary.setData(name=tr('Reactance (X)'), pen='m')
            self.legend.addItem(self.curve_secondary, tr('Reactance (X)'))
            
            self.curve_right.setVisible(False)
            ax_right.setStyle(showValues=False)
            ax_right.setLabel('')
            
        elif mode == tr("D (Tan δ)"):
            pi.setLabel('left', tr("D (Tan δ)"))
            pi.setLabel('bottom', tr("Frequency"), units='Hz')
            pi.setLogMode(y=False) 
            
            self.curve_primary.setData(name=tr('D'), pen='r')
            self.legend.addItem(self.curve_primary, tr('D'))
            
            self.curve_right.setVisible(False)
            ax_right.setStyle(showValues=False)
            ax_right.setLabel('')
            
        elif mode == tr("C / L"):
            pi.setLabel('left', tr("Capacitance"), units='F')
            pi.setLabel('bottom', tr("Frequency"), units='Hz')
            pi.setLogMode(y=True)
            
            self.curve_primary.setData(name=tr('Capacitance'), pen='b')
            self.legend.addItem(self.curve_primary, tr('Capacitance'))
            
            ax_right.setLabel(tr('Inductance'), units='H')
            ax_right.setLogMode(True) # L is Log Y
            
            self.curve_right.setData(name=tr('Inductance'), pen='r')
            self.legend.addItem(self.curve_right, tr('Inductance'))
            
        elif mode == tr("Nyquist Plot"):
            pi.setLabel('left', tr("-Imag (Z)"), units='Ohm')
            pi.setLabel('bottom', tr("Real (Z)"), units='Ohm')
            pi.setLogMode(x=False, y=False) # Linear for Nyquist
            
            self.curve_primary.setData(name=tr('Nyquist'), pen='w', symbol='o', symbolSize=5)
            self.legend.addItem(self.curve_primary, tr('Nyquist'))
            
            self.curve_right.setVisible(False)
            ax_right.setLabel('')
            ax_right.setStyle(showValues=False)
            
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
        if is_log_x and mode != tr("Nyquist Plot"):
            x_data = np.log10(freqs)
        else:
            x_data = freqs
            
        if mode == tr("|Z| & Phase"):
            # |Z| (Log Y)
            y_data = np.abs(zs)
            if self.plot_widget.getPlotItem().getAxis('left').logMode:
                y_data = np.log10(y_data)
            self.curve_primary.setData(x_data, y_data)
            
            # Phase (Linear Y)
            self.curve_right.setData(x_data, np.degrees(np.angle(zs)))
            
        elif mode == tr("R & X (ESR/ESL)"):
            # R (Log Y)
            r_data = np.abs(zs.real)
            x_data_val = np.abs(zs.imag)
            
            if self.plot_widget.getPlotItem().getAxis('left').logMode:
                # Avoid log(0)
                r_data = np.log10(r_data + 1e-12)
                x_data_val = np.log10(x_data_val + 1e-12)
                
            self.curve_primary.setData(x_data, r_data)
            self.curve_secondary.setData(x_data, x_data_val)
            
        elif mode == tr("D (Tan δ)"):
            # D = 1/Q = |R| / |X|
            rs = zs.real
            xs = zs.imag
            ds = np.zeros_like(rs)
            mask = (np.abs(xs) > 1e-12)
            ds[mask] = np.abs(rs[mask]) / np.abs(xs[mask])
            
            self.curve_primary.setData(x_data, ds)
            
        elif mode == tr("C / L"):
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

        elif mode == tr("Nyquist Plot"):
            # Nyquist: X = Real(Z), Y = -Imag(Z)
            # Standard EIS Convention
            x_nyquist = zs.real
            y_nyquist = -zs.imag
            
            self.curve_primary.setData(x_nyquist, y_nyquist)

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
            QMessageBox.information(self, tr("Calibration"), tr("Open Calibration Completed"))
        elif self.cal_mode == 'short':
            QMessageBox.information(self, tr("Calibration"), tr("Short Calibration Completed"))
        elif self.cal_mode == 'load':
            QMessageBox.information(self, tr("Calibration"), tr("Load Calibration Completed"))

        # Restore UI state
        self.btn_open.setEnabled(True)
        self.btn_short.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_dut.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if self.cal_mode is None and self.chk_resonance.isChecked():
            self.calculate_resonance()

    def calculate_resonance(self):
        if not self.sweep_freqs or not self.sweep_z_complex:
            return

        freqs = np.array(self.sweep_freqs)
        zs = np.array(self.sweep_z_complex)
        xs = zs.imag
        
        # 1. Find Zero Crossings (Sign changes in Reactance)
        # indices where sign changes between i and i+1
        zero_crossings = []
        
        for i in range(len(xs) - 1):
            x1 = xs[i]
            x2 = xs[i+1]
            
            if (x1 <= 0 and x2 >= 0) or (x1 >= 0 and x2 <= 0):
                # Found crossing
                f1 = freqs[i]
                f2 = freqs[i+1]
                
                # Linear Interpolation for X=0
                # 0 = x1 + slope * (t) -> t = -x1 / (x2 - x1)
                # f_res = f1 + t * (f2 - f1)
                
                if x2 != x1:
                    t = -x1 / (x2 - x1)
                    res_freq = f1 + t * (f2 - f1)
                    
                    # Interpolate Z magnitude at this frequency as well
                    z1 = zs[i]
                    z2 = zs[i+1]
                    res_z_complex = z1 + t * (z2 - z1) # Linear interp of complex
                    res_z_mag = abs(res_z_complex)
                    
                    zero_crossings.append((res_freq, res_z_mag))
                else:
                    # Rare case x1=x2=0
                    zero_crossings.append((f1, abs(zs[i])))

        # 2. Select Best Candidate
        if not zero_crossings:
            # Fallback: Just return min(|Z|) if no crossing found (e.g. over-damped or out of range)
            mags = np.abs(zs)
            min_idx = np.argmin(mags)
            res_freq = freqs[min_idx]
            self.lbl_resonance_result.setText(f"{tr('Resonance:')} {res_freq:.4f} Hz (Min |Z|)")
        else:
            # Pick the crossing with the minimum Impedance Magnitude (Series Resonance)
            # (Parallel resonance would be Max |Z| at X=0)
            # Assumption: User looks for Series Resonance mostly or general resonance points.
            # Let's pick min |Z| for now.
            best_res = min(zero_crossings, key=lambda p: p[1])
            res_freq = best_res[0]
            self.lbl_resonance_result.setText(f"{tr('Resonance:')} {res_freq:.4f} Hz")

        # 3. Visualize
        if self.resonance_line:
            self.plot_widget.removeItem(self.resonance_line)
        
        # Draw explicit vertical line
        # Check if Log X is on
        is_log_x = self.sw_log.isChecked()
        x_val = np.log10(res_freq) if is_log_x else res_freq
        
        self.resonance_line = pg.InfiniteLine(pos=x_val, angle=90, pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.resonance_line)
        
        # Optional: Add label to the line
        # self.resonance_line.label = pg.InfLineLabel(f"{res_freq:.1f}Hz", position=0.8, rotateAxis=(1,0), anchor=(1, 1))



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
            self.btn_dut.setStyleSheet("font-weight: bold; background-color: #5e35b1; color: white;")
        else:
            # Light Theme
            self.toggle_btn.setStyleSheet(
                "QPushButton { background-color: #ccffcc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 10px; font-weight: bold; }"
                "QPushButton:checked { background-color: #ffcccc; color: black; border: 1px solid #ccc; border-radius: 4px; padding: 10px; }"
                "QPushButton:hover { background-color: #bbfebb; }"
                "QPushButton:checked:hover { background-color: #ffbbbb; }"
            )
            self.btn_dut.setStyleSheet("font-weight: bold; background-color: #ccccff; color: black;")

        # Theme handling
        self.app = QApplication.instance()
        if hasattr(self.app, 'theme_manager'):
            self.app.theme_manager.theme_changed.connect(self.apply_theme)
            self.apply_theme(self.app.theme_manager.get_current_theme())
