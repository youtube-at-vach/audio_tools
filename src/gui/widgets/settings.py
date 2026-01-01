import numpy as np
import scipy.signal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, 
                             QFormLayout, QGroupBox, QMessageBox, QLineEdit, QDialog,
                             QDoubleSpinBox, QHBoxLayout, QTabWidget, QCheckBox, QFileDialog)
from PyQt6.QtCore import QTimer
from src.core.audio_engine import AudioEngine
from src.core.config_manager import ConfigManager
from src.core.localization import tr, get_manager


def _design_c_weighting(sr: float):
    """Design digital C-weighting filter (IEC 61672) for sample rate sr."""
    sr = float(sr)
    if sr <= 0:
        raise ValueError("Invalid sample rate")

    # Analog poles (rad/s)
    w1 = 2 * np.pi * 20.6
    w2 = 2 * np.pi * 12194.0

    # C-weighting analog transfer function: H(s) = K * s^2 / ((s + w1)^2 (s + w2)^2)
    zeros = np.array([0.0, 0.0])
    poles = np.array([-w1, -w1, -w2, -w2])
    gain = 1.0

    # Normalize to 0 dB at 1 kHz
    s = 1j * 2 * np.pi * 1000.0
    h = gain * (s**2) / ((s + w1)**2 * (s + w2)**2)
    gain = 1.0 / np.abs(h)

    z, p, k = scipy.signal.bilinear_zpk(zeros, poles, gain, fs=sr)
    b, a = scipy.signal.zpk2tf(z, p, k)
    return b.astype(np.float64), a.astype(np.float64)


class _PinkNoise:
    """Stateful pink-noise generator (Paul Kellet filter)."""

    def __init__(self):
        self.b0 = 0.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.b3 = 0.0
        self.b4 = 0.0
        self.b5 = 0.0
        self.b6 = 0.0

    def generate(self, n: int):
        white = np.random.randn(n).astype(np.float32)

        out = np.empty(n, dtype=np.float32)
        b0 = self.b0
        b1 = self.b1
        b2 = self.b2
        b3 = self.b3
        b4 = self.b4
        b5 = self.b5
        b6 = self.b6

        # Coefficients from Paul Kellet's refined pink noise filter.
        for i in range(n):
            w = float(white[i])
            b0 = 0.99886 * b0 + w * 0.0555179
            b1 = 0.99332 * b1 + w * 0.0750759
            b2 = 0.96900 * b2 + w * 0.1538520
            b3 = 0.86650 * b3 + w * 0.3104856
            b4 = 0.55000 * b4 + w * 0.5329522
            b5 = -0.7616 * b5 - w * 0.0168980
            y = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362
            b6 = w * 0.115926
            out[i] = y * 0.11

        self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6 = b0, b1, b2, b3, b4, b5, b6
        return out


class SplCalibrationDialog(QDialog):
    def __init__(self, audio_engine: AudioEngine, parent=None):
        super().__init__(parent)
        self.audio_engine = audio_engine
        self.setWindowTitle(tr("SPL Calibration Wizard"))
        self.resize(460, 360)

        self.is_running = False
        self.callback_id = None

        # Measurement state
        self.current_dbfs_c = -120.0
        self._ema_power = None

        # DSP state
        self._pink = _PinkNoise()
        self._bpf_sos = None
        self._bpf_zi = None
        self._noise_ref_rms = None
        self._target_fs_rms = None
        self._c_b = None
        self._c_a = None
        self._c_zi = None
        self._measure_bw_sos = None
        self._measure_bw_zi = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_level)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel(f"<b>1.</b> {tr('Prepare a speaker and measurement microphone.')}"))
        layout.addWidget(QLabel(f"<b>2.</b> {tr('Play band-limited pink noise, measure SPL with an external meter.')}"))
        layout.addWidget(QLabel(f"<b>3.</b> {tr('Enter the SPL shown on the meter.')}"))
        layout.addWidget(QLabel(f"<b>4.</b> {tr('Store the conversion between input dBFS and SPL.')}"))

        form = QFormLayout()

        self.profile_combo = QComboBox()
        self.profile_combo.addItem(tr("Speaker (500–2000 Hz)"), "speaker")
        self.profile_combo.addItem(tr("Subwoofer (30–80 Hz)"), "subwoofer")
        form.addRow(tr("Test Signal Band") + ":", self.profile_combo)

        self.level_spin = QDoubleSpinBox()
        self.level_unit_combo = QComboBox()
        self._populate_level_units()
        self.level_unit_combo.currentIndexChanged.connect(self._on_level_unit_changed)

        self.level_spin.setSingleStep(1.0)
        self._configure_level_spin_for_unit(self.level_unit_combo.currentText())

        level_layout = QHBoxLayout()
        level_layout.setContentsMargins(0, 0, 0, 0)
        level_layout.addWidget(self.level_spin)
        level_layout.addWidget(self.level_unit_combo)
        level_layout.addStretch(1)
        form.addRow(tr("Output Level (RMS)") + ":", level_layout)

        self.avg_spin = QDoubleSpinBox()
        self.avg_spin.setRange(0.2, 10.0)
        self.avg_spin.setValue(1.0)
        self.avg_spin.setSingleStep(0.2)
        form.addRow(tr("Averaging Time (s)") + ":", self.avg_spin)

        # Measurement bandwidth (matches Sound Level Meter presets, plus custom)
        self.measure_bw_combo = QComboBox()
        self.measure_bw_combo.addItem(tr("20 Hz - 20 kHz (Wide)"), "wide")
        self.measure_bw_combo.addItem(tr("20 Hz - 12.5 kHz"), "12k5")
        self.measure_bw_combo.addItem(tr("20 Hz - 8 kHz (Normal)"), "normal")
        self.measure_bw_combo.addItem(tr("Custom"), "custom")
        self.measure_bw_combo.setCurrentIndex(2)  # Default to "Normal"
        self.measure_bw_combo.currentIndexChanged.connect(self._on_measure_bw_mode_changed)

        self.measure_bw_low_spin = QDoubleSpinBox()
        self.measure_bw_low_spin.setDecimals(1)
        self.measure_bw_low_spin.setRange(0.1, 48000.0)
        self.measure_bw_low_spin.setValue(20.0)
        self.measure_bw_low_spin.setSuffix(" Hz")
        self.measure_bw_low_spin.valueChanged.connect(self._on_custom_bw_changed)

        self.measure_bw_high_spin = QDoubleSpinBox()
        self.measure_bw_high_spin.setDecimals(1)
        self.measure_bw_high_spin.setRange(1.0, 48000.0)
        self.measure_bw_high_spin.setValue(8000.0)
        self.measure_bw_high_spin.setSuffix(" Hz")
        self.measure_bw_high_spin.valueChanged.connect(self._on_custom_bw_changed)

        bw_layout = QHBoxLayout()
        bw_layout.setContentsMargins(0, 0, 0, 0)
        bw_layout.addWidget(self.measure_bw_combo)
        bw_layout.addWidget(self.measure_bw_low_spin)
        bw_layout.addWidget(QLabel("-"))
        bw_layout.addWidget(self.measure_bw_high_spin)
        bw_layout.addStretch(1)
        form.addRow(tr("Measurement Bandwidth") + ":", bw_layout)

        self._on_measure_bw_mode_changed()

        layout.addLayout(form)

        self.start_btn = QPushButton(tr("Start"))
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self.on_start_toggle)
        layout.addWidget(self.start_btn)

        self.level_label = QLabel(tr("Input Level (C-weighted): -- dBFS"))
        self.level_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.level_label)

        spl_row = QHBoxLayout()
        self.spl_spin = QDoubleSpinBox()
        self.spl_spin.setRange(0.0, 160.0)
        self.spl_spin.setDecimals(1)
        self.spl_spin.setValue(80.0)
        spl_row.addWidget(self.spl_spin)
        self.spl_unit = QLabel(tr("dB SPL"))
        spl_row.addWidget(self.spl_unit)
        spl_row.addStretch(1)

        layout.addWidget(QLabel(tr("Measured SPL from external meter:")))
        layout.addLayout(spl_row)

        self.save_btn = QPushButton(tr("Calculate & Save"))
        self.save_btn.clicked.connect(self.on_save)
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

    def _populate_level_units(self):
        self.level_unit_combo.clear()
        self.level_unit_combo.addItem("dBFS")

        cal = getattr(self.audio_engine, 'calibration', None)
        has_output_cal = False
        try:
            has_output_cal = bool(getattr(cal, 'output_gain_is_calibrated', False))
        except Exception:
            has_output_cal = False

        if has_output_cal:
            self.level_unit_combo.addItems(["Vrms", "mVrms", "dBV", "dBu"])

        # Default
        idx = self.level_unit_combo.findText("dBFS")
        if idx >= 0:
            self.level_unit_combo.setCurrentIndex(idx)

    def _configure_level_spin_for_unit(self, unit: str):
        unit = str(unit)
        if unit == "dBFS":
            self.level_spin.setDecimals(1)
            self.level_spin.setRange(-80.0, 0.0)
            self.level_spin.setValue(-20.0)
            self.level_spin.setSingleStep(1.0)
        elif unit == "Vrms":
            self.level_spin.setDecimals(3)
            self.level_spin.setRange(0.001, 20.0)
            self.level_spin.setValue(0.100)
            self.level_spin.setSingleStep(0.010)
        elif unit == "mVrms":
            self.level_spin.setDecimals(1)
            self.level_spin.setRange(1.0, 20000.0)
            self.level_spin.setValue(100.0)
            self.level_spin.setSingleStep(10.0)
        elif unit == "dBV":
            self.level_spin.setDecimals(1)
            self.level_spin.setRange(-80.0, 20.0)
            self.level_spin.setValue(-20.0)
            self.level_spin.setSingleStep(1.0)
        elif unit == "dBu":
            self.level_spin.setDecimals(1)
            self.level_spin.setRange(-80.0, 20.0)
            self.level_spin.setValue(-20.0)
            self.level_spin.setSingleStep(1.0)
        else:
            # Fallback
            self.level_spin.setDecimals(1)
            self.level_spin.setRange(-80.0, 0.0)
            self.level_spin.setValue(-20.0)

    def _on_level_unit_changed(self):
        self._configure_level_spin_for_unit(self.level_unit_combo.currentText())

    def _on_measure_bw_mode_changed(self):
        is_custom = self.measure_bw_combo.currentData() == "custom"
        self.measure_bw_low_spin.setEnabled(is_custom)
        self.measure_bw_high_spin.setEnabled(is_custom)

    def _on_custom_bw_changed(self):
        # Keep upper cutoff above lower cutoff when the user drags quickly.
        low = self.measure_bw_low_spin.value()
        high = self.measure_bw_high_spin.value()
        if high <= low:
            self.measure_bw_high_spin.blockSignals(True)
            self.measure_bw_high_spin.setValue(low + 10.0)
            self.measure_bw_high_spin.blockSignals(False)

    def _get_measurement_band_hz(self):
        mode = self.measure_bw_combo.currentData()
        presets = {
            "wide": (20.0, 20000.0),
            "12k5": (20.0, 12500.0),
            "normal": (20.0, 8000.0),
        }

        if mode == "custom":
            low = float(self.measure_bw_low_spin.value())
            high = float(self.measure_bw_high_spin.value())
        else:
            low, high = presets.get(mode, presets["normal"])

        if high <= low:
            raise ValueError(tr("Invalid measurement bandwidth: upper cutoff must be higher than lower cutoff."))

        return low, high

    def _design_measurement_bandpass(self, sr: float, band_hz):
        nyq = 0.5 * sr
        low_norm = max(0.1, band_hz[0]) / nyq
        high_norm = min(nyq * 0.99, band_hz[1]) / nyq
        if not (0 < low_norm < high_norm < 1):
            raise ValueError(tr("Measurement bandwidth is outside the valid range for this sample rate."))

        sos = scipy.signal.butter(4, [low_norm, high_norm], btype='bandpass', output='sos')
        zi = np.zeros((sos.shape[0], 2), dtype=np.float64)
        return sos, zi

    def _get_target_fs_rms(self) -> float:
        """Returns desired output RMS level in full-scale units (FS RMS)."""
        unit = self.level_unit_combo.currentText()
        val = float(self.level_spin.value())

        if unit == "dBFS":
            return float(10 ** (val / 20.0))

        # Voltage-based units require output calibration
        cal = self.audio_engine.calibration
        out_gain = float(getattr(cal, 'output_gain', 0.0) or 0.0)
        if out_gain <= 0 or not np.isfinite(out_gain):
            raise ValueError(tr("Output calibration is required for voltage units."))

        if unit == "Vrms":
            v_rms = val
        elif unit == "mVrms":
            v_rms = val / 1000.0
        elif unit == "dBV":
            v_rms = 10 ** (val / 20.0)
        elif unit == "dBu":
            # 0 dBu = 0.7745966692 Vrms
            v_rms = 0.7745966692 * (10 ** (val / 20.0))
        else:
            raise ValueError(tr("Invalid level unit"))

        return float(v_rms / out_gain)

    def _prepare_filters(self):
        sr = float(self.audio_engine.sample_rate)
        if sr <= 0:
            raise ValueError("Invalid sample rate")

        profile = self.profile_combo.currentData()
        if profile == "speaker":
            band = (500.0, 2000.0)
        else:
            band = (30.0, 80.0)

        # Band-pass for noise output (Butterworth)
        nyq = 0.5 * sr
        low = max(1.0, band[0]) / nyq
        high = min(nyq * 0.99, band[1]) / nyq
        if not (0 < low < high < 1):
            raise ValueError("Invalid bandpass settings")

        # Use SOS for numerical stability, especially at very low normalized frequencies.
        bpf_sos = scipy.signal.butter(4, [low, high], btype='bandpass', output='sos')
        self._bpf_sos = bpf_sos
        # Streaming state for sosfilt: shape (n_sections, 2)
        self._bpf_zi = np.zeros((bpf_sos.shape[0], 2), dtype=np.float64)

        # Reference RMS for unscaled output noise (used to map desired RMS -> scale).
        try:
            sim_pink = _PinkNoise()
            sim_zi = np.zeros((bpf_sos.shape[0], 2), dtype=np.float64)
            n = int(max(1, sr * 2.0))
            x = sim_pink.generate(n).astype(np.float64)
            y, _ = scipy.signal.sosfilt(bpf_sos, x, zi=sim_zi)
            discard = int(min(n - 1, sr * 0.5))
            y2 = y[discard:]
            rms = float(np.sqrt(np.mean(y2 * y2) + 1e-24))
            self._noise_ref_rms = float(max(rms, 1e-12))
        except Exception:
            self._noise_ref_rms = 1.0

        # C-weighting for input measurement
        c_b, c_a = _design_c_weighting(sr)
        self._c_b = c_b
        self._c_a = c_a
        self._c_zi = scipy.signal.lfilter_zi(c_b, c_a).astype(np.float64)

        # Measurement bandwidth (input side)
        meas_band = self._get_measurement_band_hz()
        self._measure_bw_sos, self._measure_bw_zi = self._design_measurement_bandpass(sr, meas_band)

        self._ema_power = None
        self.current_dbfs_c = -120.0

        return band

    def on_start_toggle(self, checked):
        if checked:
            try:
                self.start_measurement()
                self.start_btn.setText(tr("Stop"))
                self.timer.start(100)
                # Avoid changing settings mid-stream.
                self.profile_combo.setEnabled(False)
                self.level_spin.setEnabled(False)
                self.level_unit_combo.setEnabled(False)
                self.avg_spin.setEnabled(False)
                self.measure_bw_combo.setEnabled(False)
                self.measure_bw_low_spin.setEnabled(False)
                self.measure_bw_high_spin.setEnabled(False)
            except Exception as e:
                self.start_btn.setChecked(False)
                QMessageBox.critical(self, tr("Error"), str(e))
        else:
            self.stop_measurement()
            self.start_btn.setText(tr("Start"))
            self.timer.stop()
            self.profile_combo.setEnabled(True)
            self.level_spin.setEnabled(True)
            self.level_unit_combo.setEnabled(True)
            self.avg_spin.setEnabled(True)
            self.measure_bw_combo.setEnabled(True)
            self._on_measure_bw_mode_changed()

    def start_measurement(self):
        self._prepare_filters()

        self._target_fs_rms = self._get_target_fs_rms()
        if not np.isfinite(self._target_fs_rms) or self._target_fs_rms <= 0:
            raise ValueError(tr("Invalid output level"))

        tau = float(self.avg_spin.value())
        sr = float(self.audio_engine.sample_rate)

        def callback(indata, outdata, frames, time, status):
            # --- Output: band-limited pink noise ---
            pink = self._pink.generate(frames).astype(np.float64)
            y, self._bpf_zi = scipy.signal.sosfilt(self._bpf_sos, pink, zi=self._bpf_zi)

            # Scale so that output RMS ~= target FS RMS.
            ref = float(self._noise_ref_rms or 1.0)
            scale = float(self._target_fs_rms) / max(1e-12, ref)
            y = y * scale
            # Prevent hard overflow if user requests too much level.
            y = np.clip(y, -0.99, 0.99).astype(np.float32)

            if outdata.shape[1] >= 2:
                outdata[:, 0] = y
                outdata[:, 1] = y
            else:
                outdata[:, 0] = y

            # --- Input: C-weighted RMS (channel 0) ---
            if indata.shape[1] > 0:
                x = indata[:, 0].astype(np.float64)
                if self._measure_bw_sos is not None and self._measure_bw_zi is not None:
                    x, self._measure_bw_zi = scipy.signal.sosfilt(self._measure_bw_sos, x, zi=self._measure_bw_zi)
                xw, self._c_zi = scipy.signal.lfilter(self._c_b, self._c_a, x, zi=self._c_zi)
                p = float(np.mean(xw * xw) + 1e-24)

                # EMA on power for stable reading
                dt = frames / sr
                alpha = float(np.exp(-dt / max(1e-3, tau)))
                if self._ema_power is None:
                    self._ema_power = p
                else:
                    self._ema_power = alpha * self._ema_power + (1.0 - alpha) * p

                self.current_dbfs_c = 10.0 * np.log10(self._ema_power + 1e-24)

        self.callback_id = self.audio_engine.register_callback(callback)
        self.is_running = True
        # No popup here (per UX request). The dialog already explains the steps.

    def stop_measurement(self):
        if self.is_running:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_running = False
            self.profile_combo.setEnabled(True)
            self.level_spin.setEnabled(True)
            self.level_unit_combo.setEnabled(True)
            self.avg_spin.setEnabled(True)

    def update_level(self):
        self.level_label.setText(
            tr("Input Level (C-weighted): {0:.1f} dBFS").format(self.current_dbfs_c)
        )

    def on_save(self):
        try:
            if not np.isfinite(self.current_dbfs_c) or self.current_dbfs_c < -110:
                raise ValueError(tr("No signal detected. Please start measurement and check connections."))

            spl = float(self.spl_spin.value())
            profile = self.profile_combo.currentData()
            if profile == "speaker":
                pass
            else:
                pass

            self.audio_engine.calibration.set_spl_calibration(
                measured_dbfs_c=float(self.current_dbfs_c),
                measured_spl_db=spl,
            )

            off = self.audio_engine.calibration.get_spl_offset_db()
            QMessageBox.information(
                self,
                tr("Success"),
                tr("SPL calibration saved. Offset = {0:.2f} dB (SPL = dBFS + offset)").format(off if off is not None else 0.0),
            )

            # Stop test signal immediately after saving.
            self.stop_measurement()
            self.start_btn.setChecked(False)
            self.start_btn.setText(tr("Start"))
            self.timer.stop()
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), str(e))

    def closeEvent(self, event):
        self.stop_measurement()
        super().closeEvent(event)

class OutputCalibrationDialog(QDialog):
    def __init__(self, audio_engine: AudioEngine, parent=None):
        super().__init__(parent)
        self.audio_engine = audio_engine
        self.setWindowTitle(tr("Output Calibration Wizard"))
        self.resize(400, 300)
        self.init_ui()
        self.is_playing = False
        self.callback_id = None

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Step 1
        layout.addWidget(QLabel(f"<b>{tr('Step 1:')}</b> {tr('Connect a voltmeter to the output.')}"))
        
        # Step 2
        layout.addWidget(QLabel(f"<b>{tr('Step 2:')}</b> {tr('Set Test Tone.')}"))
        form = QFormLayout()
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000); self.freq_spin.setValue(1000)
        form.addRow(f"{tr('Frequency (Hz):')}", self.freq_spin)
        
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setRange(-60, 0); self.level_spin.setValue(-12)
        form.addRow(f"{tr('Level (dBFS):')}", self.level_spin)
        layout.addLayout(form)
        
        # Step 3
        layout.addWidget(QLabel(f"<b>{tr('Step 3:')}</b> {tr('Play Tone.')}"))
        self.play_btn = QPushButton(tr("Start Tone"))
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.on_play_toggle)
        layout.addWidget(self.play_btn)
        
        # Step 4
        layout.addWidget(QLabel(f"<b>{tr('Step 4:')}</b> {tr('Enter measured voltage.')}"))
        meas_layout = QHBoxLayout()
        self.meas_spin = QDoubleSpinBox()
        self.meas_spin.setRange(-200, 1000); self.meas_spin.setDecimals(4)
        meas_layout.addWidget(self.meas_spin)
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Vrms", "mVrms", "dBV", "dBu"])
        meas_layout.addWidget(self.unit_combo)
        layout.addLayout(meas_layout)
        
        # Step 5
        layout.addWidget(QLabel(f"<b>{tr('Step 5:')}</b> {tr('Save.')}"))
        self.save_btn = QPushButton(tr("Calculate & Save"))
        self.save_btn.clicked.connect(self.on_save)
        layout.addWidget(self.save_btn)
        
        self.setLayout(layout)

    def on_play_toggle(self, checked):
        if checked:
            self.start_tone()
            self.play_btn.setText(tr("Stop Tone"))
        else:
            self.stop_tone()
            self.play_btn.setText(tr("Start Tone"))

    def start_tone(self):
        freq = self.freq_spin.value()
        dbfs = self.level_spin.value()
        amp = 10**(dbfs/20)
        sr = self.audio_engine.sample_rate
        
        def callback(indata, outdata, frames, time, status):
            t = (np.arange(frames) + callback.t_start) / sr
            callback.t_start += frames
            tone = amp * np.sin(2 * np.pi * freq * t)
            # Stereo output
            if outdata.shape[1] >= 2:
                outdata[:, 0] = tone
                outdata[:, 1] = tone
            else:
                outdata[:, 0] = tone
        
        callback.t_start = 0
        callback.t_start = 0
        self.callback_id = self.audio_engine.register_callback(callback)
        self.is_playing = True

    def stop_tone(self):
        if self.is_playing:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_playing = False

    def on_save(self):
        try:
            val = self.meas_spin.value()
            unit = self.unit_combo.currentText()
            dbfs = self.level_spin.value()
            
            # Convert to Vpeak
            if unit == "Vrms":
                v_peak = val * np.sqrt(2)
            elif unit == "mVrms":
                v_peak = (val / 1000.0) * np.sqrt(2)
            elif unit == "dBV":
                v_rms = 10**(val/20)
                v_peak = v_rms * np.sqrt(2)
            elif unit == "dBu":
                v_rms = 10**((val - 2.218)/20) # 0dBu = 0.7746V
                v_peak = v_rms * np.sqrt(2)
            
            # Calculate Gain (V/FS)
            # V_out_peak = Gain * 10^(dBFS/20)
            # Gain = V_out_peak / 10^(dBFS/20)
            gain = v_peak / (10**(dbfs/20))
            
            self.audio_engine.calibration.set_output_gain(gain)
            QMessageBox.information(
                self,
                tr("Success"),
                tr("Output Gain calibrated to {0:.4f} V/FS").format(gain),
            )
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), str(e))

    def closeEvent(self, event):
        self.stop_tone()
        super().closeEvent(event)

class InputCalibrationDialog(QDialog):
    def __init__(self, audio_engine: AudioEngine, parent=None):
        super().__init__(parent)
        self.audio_engine = audio_engine
        self.setWindowTitle(tr("Input Calibration Wizard"))
        self.resize(400, 300)
        self.init_ui()
        self.is_measuring = False
        self.callback_id = None
        self.current_rms_dbfs = -100.0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_level)

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Step 1
        layout.addWidget(QLabel(f"<b>{tr('Step 1:')}</b> {tr('Connect a known signal source to the input.')}"))
        
        # Step 2
        layout.addWidget(QLabel(f"<b>{tr('Step 2:')}</b> {tr('Measure Input Level.')}"))
        self.measure_btn = QPushButton(tr("Start Measurement"))
        self.measure_btn.setCheckable(True)
        self.measure_btn.clicked.connect(self.on_measure_toggle)
        layout.addWidget(self.measure_btn)
        
        self.level_label = QLabel(tr("Current Level: -- dBFS"))
        self.level_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.level_label)
        
        # Step 3
        layout.addWidget(QLabel(f"<b>{tr('Step 3:')}</b> {tr('Enter known source voltage.')}"))
        meas_layout = QHBoxLayout()
        self.meas_spin = QDoubleSpinBox()
        self.meas_spin.setRange(-200, 1000); self.meas_spin.setDecimals(4)
        meas_layout.addWidget(self.meas_spin)
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Vrms", "mVrms", "dBV", "dBu"])
        meas_layout.addWidget(self.unit_combo)
        layout.addLayout(meas_layout)
        
        # Step 4
        layout.addWidget(QLabel(f"<b>{tr('Step 4:')}</b> {tr('Save.')}"))
        self.save_btn = QPushButton(tr("Calculate & Save"))
        self.save_btn.clicked.connect(self.on_save)
        layout.addWidget(self.save_btn)
        
        self.setLayout(layout)

    def on_measure_toggle(self, checked):
        if checked:
            self.start_measurement()
            self.measure_btn.setText(tr("Stop Measurement"))
            self.timer.start(100)
        else:
            self.stop_measurement()
            self.measure_btn.setText(tr("Start Measurement"))
            self.timer.stop()

    def start_measurement(self):
        def callback(indata, outdata, frames, time, status):
            # Calculate RMS of first channel
            if indata.shape[1] > 0:
                rms = np.sqrt(np.mean(indata[:, 0]**2))
                db = 20 * np.log10(rms + 1e-12)
                self.current_rms_dbfs = db
            outdata.fill(0)
            
        self.callback_id = self.audio_engine.register_callback(callback)
        self.is_measuring = True

    def stop_measurement(self):
        if self.is_measuring:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.is_measuring = False

    def update_level(self):
        self.level_label.setText(tr("Current Level: {0:.1f} dBFS").format(self.current_rms_dbfs))

    def on_save(self):
        try:
            val = self.meas_spin.value()
            unit = self.unit_combo.currentText()
            measured_dbfs = self.current_rms_dbfs
            
            if measured_dbfs < -100:
                raise ValueError(tr("No signal detected. Please check connections."))
            
            # Convert Known Voltage to Vpeak
            if unit == "Vrms":
                v_peak = val * np.sqrt(2)
            elif unit == "mVrms":
                v_peak = (val / 1000.0) * np.sqrt(2)
            elif unit == "dBV":
                v_rms = 10**(val/20)
                v_peak = v_rms * np.sqrt(2)
            elif unit == "dBu":
                v_rms = 10**((val - 2.218)/20)
                v_peak = v_rms * np.sqrt(2)
            
            # Calculate Sensitivity (V/FS)
            # Measured_FS_Peak = 10^(measured_dbfs/20) * sqrt(2)
            
            measured_fs_peak = (10**(measured_dbfs/20)) * np.sqrt(2)
            
            # Sensitivity = Volts / FS
            # We want Measured_FS_Peak * Sensitivity = V_peak
            sensitivity = v_peak / measured_fs_peak
            
            self.audio_engine.calibration.set_input_sensitivity(sensitivity)
            QMessageBox.information(
                self,
                tr("Success"),
                tr("Input Sensitivity calibrated to {0:.4f} V/FS").format(sensitivity),
            )
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), str(e))

    def closeEvent(self, event):
        self.stop_measurement()
        super().closeEvent(event)

class SettingsWidget(QWidget):
    def __init__(self, audio_engine: AudioEngine, config_manager: ConfigManager):
        super().__init__()
        self.audio_engine = audio_engine
        self.config_manager = config_manager
        self.init_ui()

    def _format_device_label(self, device_index: int, dev: dict) -> str:
        """Build a human-friendly device label for the combo boxes.

        On Windows especially, the same physical device can appear multiple times
        under different host APIs (ASIO/WASAPI/DirectSound/etc). We surface that
        info to help users pick the right one.
        """
        base = f"{device_index}: {dev.get('name', '')}"
        hostapi_name = dev.get('hostapi_name')
        if hostapi_name:
            return f"{base} ({hostapi_name})"
        return base

    def _get_device_name_for_config(self, device_index: int, fallback_text: str) -> str:
        """Return the raw PortAudio device name for config persistence."""
        try:
            if device_index is not None and int(device_index) >= 0:
                devices = self.audio_engine.list_devices()
                idx = int(device_index)
                if 0 <= idx < len(devices):
                    name = devices[idx].get('name')
                    if name:
                        return str(name)
        except Exception:
            pass

        # Fallback to prior behavior (best-effort, strips our appended hostapi).
        try:
            raw = fallback_text.split(": ", 1)[1]
        except Exception:
            raw = fallback_text
        raw = str(raw).strip()
        if raw.endswith(')') and ' (' in raw:
            raw = raw.rsplit(' (', 1)[0]
        return raw

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel(tr("Audio Device Settings"))
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # --- Tab 1: General ---
        general_tab = QWidget()
        general_layout = QVBoxLayout()
        
        # Language Settings
        lang_group = QGroupBox(tr("Language"))
        lang_layout = QFormLayout()
        
        self.lang_combo = QComboBox()
        # Load available languages
        manager = get_manager()
        # Sort: en first, then others
        langs = sorted(manager.available_languages.keys())
        if 'en' in langs:
            langs.remove('en')
            langs.insert(0, 'en')
            
        for lang in langs:
            self.lang_combo.addItem(lang, lang)
            
        # Set current
        idx = self.lang_combo.findData(manager.language)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
            
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        lang_layout.addRow(tr("Language") + ":", self.lang_combo)
        
        lang_group.setLayout(lang_layout)
        general_layout.addWidget(lang_group)

        # Appearance Settings
        appearance_group = QGroupBox(tr("Appearance"))
        appearance_layout = QFormLayout()
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItem(tr("System"), "system")
        self.theme_combo.addItem(tr("Light"), "light")
        self.theme_combo.addItem(tr("Dark"), "dark")
        
        # Set current theme
        current_theme = self.config_manager.get_theme()
        idx = self.theme_combo.findData(current_theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        
        self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)
        appearance_layout.addRow(tr("Theme") + ":", self.theme_combo)
        
        appearance_group.setLayout(appearance_layout)
        general_layout.addWidget(appearance_group)

        # Screenshot Settings
        screenshot_group = QGroupBox(tr("Screenshots"))
        screenshot_layout = QFormLayout()

        self.screenshot_dir_edit = QLineEdit()
        self.screenshot_dir_edit.setText(self.config_manager.get_screenshot_output_dir())
        self.screenshot_dir_edit.editingFinished.connect(self.on_screenshot_dir_changed)

        self.screenshot_browse_btn = QPushButton(tr("Browse..."))
        self.screenshot_browse_btn.clicked.connect(self.browse_screenshot_dir)

        screenshot_dir_row = QWidget()
        screenshot_dir_row_layout = QHBoxLayout(screenshot_dir_row)
        screenshot_dir_row_layout.setContentsMargins(0, 0, 0, 0)
        screenshot_dir_row_layout.addWidget(self.screenshot_dir_edit)
        screenshot_dir_row_layout.addWidget(self.screenshot_browse_btn)

        screenshot_layout.addRow(tr("Output Folder") + ":", screenshot_dir_row)
        screenshot_group.setLayout(screenshot_layout)
        general_layout.addWidget(screenshot_group)
        
        general_layout.addStretch()
        general_tab.setLayout(general_layout)
        self.tabs.addTab(general_tab, tr("General"))

        # --- Tab 2: Audio ---
        audio_tab = QWidget()
        audio_layout = QVBoxLayout()

        # Device Selection Group
        dev_group = QGroupBox(tr("Audio Devices"))
        dev_layout = QFormLayout()
        
        self.input_combo = QComboBox()
        dev_layout.addRow(tr("Input Device:"), self.input_combo)
        
        self.output_combo = QComboBox()
        dev_layout.addRow(tr("Output Device:"), self.output_combo)
        
        self.refresh_btn = QPushButton(tr("Refresh Devices"))
        self.refresh_btn.clicked.connect(self.refresh_devices)
        dev_layout.addRow(self.refresh_btn)
        
        # Active Device Info
        self.active_in_label = QLabel(tr("None"))
        self.active_out_label = QLabel(tr("None"))
        dev_layout.addRow(tr("Active Input:"), self.active_in_label)
        dev_layout.addRow(tr("Active Output:"), self.active_out_label)
        
        dev_group.setLayout(dev_layout)
        audio_layout.addWidget(dev_group)
        
        # Audio Configuration Group
        conf_group = QGroupBox(tr("Audio Configuration"))
        conf_layout = QFormLayout()

        # PipeWire/JACK resident mode (keep PortAudio stream open)
        self.pipewire_jack_resident_check = QCheckBox(tr("PipeWire / JACK Mode (Resident)"))
        self.pipewire_jack_resident_check.setChecked(self.config_manager.get_pipewire_jack_resident())
        self.pipewire_jack_resident_check.toggled.connect(self.on_pipewire_jack_resident_toggled)
        conf_layout.addRow(self.pipewire_jack_resident_check)
        
        # Sample Rate
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(['44100', '48000', '88200', '96000', '192000'])
        self.sr_combo.setCurrentText(str(self.audio_engine.sample_rate))
        self.sr_combo.currentTextChanged.connect(self.on_sr_changed)
        conf_layout.addRow(tr("Sample Rate:"), self.sr_combo)
        
        # Buffer Size
        self.bs_combo = QComboBox()
        # Include larger buffers for stability testing; host/driver will reject unsupported sizes.
        self.bs_combo.addItems(['256', '512', '1024', '2048', '4096', '8192', '16384'])
        self.bs_combo.setCurrentText(str(self.audio_engine.block_size))
        self.bs_combo.currentTextChanged.connect(self.on_bs_changed)
        
        self.bs_duration_label = QLabel()
        self.bs_duration_label.setStyleSheet("color: #888888; font-style: italic; margin-left: 10px;")
        
        bs_layout = QHBoxLayout()
        bs_layout.setContentsMargins(0, 0, 0, 0)
        bs_layout.addWidget(self.bs_combo)
        bs_layout.addWidget(self.bs_duration_label)
        bs_layout.addStretch() # Keep it to the left
        conf_layout.addRow(tr("Buffer Size:"), bs_layout)
        
        # Input Channels
        self.in_ch_combo = QComboBox()
        self.in_ch_combo.addItems([tr('Stereo'), tr('Left'), tr('Right')])
        self.in_ch_combo.setCurrentText(self.audio_engine.input_channel_mode.capitalize())
        self.in_ch_combo.currentTextChanged.connect(self.on_ch_mode_changed)
        conf_layout.addRow(tr("Input Channels:"), self.in_ch_combo)

        # Output Channels
        self.out_ch_combo = QComboBox()
        self.out_ch_combo.addItems([tr('Stereo'), tr('Left'), tr('Right')])
        self.out_ch_combo.setCurrentText(self.audio_engine.output_channel_mode.capitalize())
        self.out_ch_combo.currentTextChanged.connect(self.on_ch_mode_changed)
        conf_layout.addRow(tr("Output Channels:"), self.out_ch_combo)
        
        conf_group.setLayout(conf_layout)
        audio_layout.addWidget(conf_group)
        
        audio_layout.addStretch()
        audio_tab.setLayout(audio_layout)
        self.tabs.addTab(audio_tab, tr("Audio"))

        # --- Tab 3: Calibration ---
        calibration_tab = QWidget()
        calibration_layout = QVBoxLayout()

        # Calibration Group
        cal_group = QGroupBox(tr("Calibration"))
        cal_layout = QFormLayout()
        
        # Input Sensitivity
        self.in_sens_edit = QLineEdit()
        # self.in_sens_edit.setText(f"{self.audio_engine.calibration.input_sensitivity:.4f}") # Moved to update_in_sens_display
        self.in_sens_edit.editingFinished.connect(self.on_in_sens_changed)
        
        self.in_sens_unit = QComboBox()
        self.in_sens_unit.addItems(["V/FS", "mV/FS"])
        self.in_sens_unit.currentIndexChanged.connect(self.update_in_sens_display)

        in_cal_btn = QPushButton(tr("Wizard"))
        in_cal_btn.clicked.connect(self.open_input_calibration)
        in_cal_layout = QHBoxLayout()
        in_cal_layout.setContentsMargins(0, 0, 0, 0)
        in_cal_layout.addWidget(self.in_sens_edit)
        in_cal_layout.addWidget(self.in_sens_unit)
        in_cal_layout.addWidget(in_cal_btn)
        
        cal_layout.addRow(tr("Input Sensitivity:"), in_cal_layout)
        
        # Output Gain
        self.out_gain_edit = QLineEdit()
        # self.out_gain_edit.setText(f"{self.audio_engine.calibration.output_gain:.4f}") # Moved to update_out_gain_display
        self.out_gain_edit.editingFinished.connect(self.on_out_gain_changed)
        
        self.out_gain_unit = QComboBox()
        self.out_gain_unit.addItems(["V/FS", "mV/FS"])
        self.out_gain_unit.currentIndexChanged.connect(self.update_out_gain_display)

        out_cal_btn = QPushButton(tr("Wizard"))
        out_cal_btn.clicked.connect(self.open_output_calibration)
        out_cal_layout = QHBoxLayout()
        out_cal_layout.setContentsMargins(0, 0, 0, 0)
        out_cal_layout.addWidget(self.out_gain_edit)
        out_cal_layout.addWidget(self.out_gain_unit)
        out_cal_layout.addWidget(out_cal_btn)
        
        cal_layout.addRow(tr("Output Gain:"), out_cal_layout)

        # SPL Calibration (Measurement Mic)
        self.spl_offset_edit = QLineEdit()
        self.spl_offset_edit.editingFinished.connect(self.on_spl_offset_changed)

        self.spl_offset_unit = QComboBox()
        self.spl_offset_unit.addItems(["dB SPL/FS"])
        self.spl_offset_unit.setEnabled(True)

        spl_btn = QPushButton(tr("Wizard"))
        spl_btn.clicked.connect(self.open_spl_calibration)

        spl_layout = QHBoxLayout()
        spl_layout.setContentsMargins(0, 0, 0, 0)
        spl_layout.addWidget(self.spl_offset_edit)
        spl_layout.addWidget(self.spl_offset_unit)
        spl_layout.addWidget(spl_btn)

        cal_layout.addRow(tr("SPL Offset:"), spl_layout)
        
        cal_group.setLayout(cal_layout)
        calibration_layout.addWidget(cal_group)
        
        calibration_layout.addStretch()
        calibration_tab.setLayout(calibration_layout)
        self.tabs.addTab(calibration_tab, tr("Calibration"))
        
        self.setLayout(main_layout)
        
        # Initialize
        self.refresh_devices()
        self.update_buffer_duration()
        self.update_in_sens_display()
        self.update_out_gain_display()
        self.update_spl_display()

    def on_pipewire_jack_resident_toggled(self, checked: bool):
        self.config_manager.set_pipewire_jack_resident(bool(checked))
        self.audio_engine.set_pipewire_jack_resident(bool(checked))

    def open_spl_calibration(self):
        dlg = SplCalibrationDialog(self.audio_engine, self)
        if dlg.exec():
            self.update_spl_display()

    def update_spl_display(self):
        off = self.audio_engine.calibration.get_spl_offset_db()
        self.spl_offset_edit.setText("" if off is None else f"{off:.2f}")

    def on_spl_offset_changed(self):
        try:
            txt = self.spl_offset_edit.text().strip()
            if not txt:
                self.audio_engine.calibration.spl_offset_db = None
                self.audio_engine.calibration.save()
                self.update_spl_display()
                return
            val = float(txt)
            self.audio_engine.calibration.spl_offset_db = float(val)
            self.audio_engine.calibration.save()
            self.update_spl_display()
        except ValueError:
            self.update_spl_display()

    def on_language_changed(self):
        lang = self.lang_combo.currentData()
        if lang:
            # Only save if changed
            if lang != self.config_manager.get_language():
                self.config_manager.set_language(lang)
                QMessageBox.information(self, tr("Restart Required"), tr("Please restart the application to apply language changes."))

    def on_theme_changed(self):
        theme = self.theme_combo.currentData()
        if theme:
            # Only save if changed
            if theme != self.config_manager.get_theme():
                self.config_manager.set_theme(theme)
                # Apply theme immediately if ThemeManager is available
                from PyQt6.QtWidgets import QApplication
                app = QApplication.instance()
                if hasattr(app, 'theme_manager'):
                    app.theme_manager.set_theme(theme)

    def on_screenshot_dir_changed(self):
        out_dir = self.screenshot_dir_edit.text().strip()
        if not out_dir:
            out_dir = "screenshots"
            self.screenshot_dir_edit.setText(out_dir)
        self.config_manager.set_screenshot_output_dir(out_dir)

    def browse_screenshot_dir(self):
        current = self.screenshot_dir_edit.text().strip()
        if not current:
            current = self.config_manager.get_screenshot_output_dir()
        selected = QFileDialog.getExistingDirectory(self, tr("Select Folder"), current)
        if not selected:
            return
        self.screenshot_dir_edit.setText(selected)
        self.config_manager.set_screenshot_output_dir(selected)


    def open_input_calibration(self):
        dlg = InputCalibrationDialog(self.audio_engine, self)
        if dlg.exec():
            self.update_in_sens_display()

    def open_output_calibration(self):
        dlg = OutputCalibrationDialog(self.audio_engine, self)
        if dlg.exec():
            self.update_out_gain_display()

    def update_in_sens_display(self):
        val = self.audio_engine.calibration.input_sensitivity
        if self.in_sens_unit.currentText() == "mV/FS":
            val *= 1000.0
        self.in_sens_edit.setText(f"{val:.4f}")

    def update_out_gain_display(self):
        val = self.audio_engine.calibration.output_gain
        if self.out_gain_unit.currentText() == "mV/FS":
            val *= 1000.0
        self.out_gain_edit.setText(f"{val:.4f}")

    def on_in_sens_changed(self):
        try:
            val = float(self.in_sens_edit.text())
            if self.in_sens_unit.currentText() == "mV/FS":
                val /= 1000.0
            self.audio_engine.calibration.set_input_sensitivity(val)
            # Refresh display to show canonical format or if rounding happened
            self.update_in_sens_display() 
        except ValueError:
            # Revert if invalid
            self.update_in_sens_display()

    def on_out_gain_changed(self):
        try:
            val = float(self.out_gain_edit.text())
            if self.out_gain_unit.currentText() == "mV/FS":
                val /= 1000.0
            self.audio_engine.calibration.set_output_gain(val)
            self.update_out_gain_display()
        except ValueError:
            self.update_out_gain_display()

    def refresh_devices(self):
        devices = self.audio_engine.list_devices()
        self.input_combo.clear()
        self.output_combo.clear()
        
        default_in = self.audio_engine.input_device
        default_out = self.audio_engine.output_device
        
        for i, dev in enumerate(devices):
            name = self._format_device_label(i, dev)
            if dev['max_input_channels'] > 0:
                self.input_combo.addItem(name, i)
            if dev['max_output_channels'] > 0:
                self.output_combo.addItem(name, i)
                
        # Restore selection if possible
        if default_in is not None:
            idx = self.input_combo.findData(default_in)
            if idx >= 0: 
                self.input_combo.setCurrentIndex(idx)
                self.active_in_label.setText(self.input_combo.itemText(idx))
            
        if default_out is not None:
            idx = self.output_combo.findData(default_out)
            if idx >= 0: 
                self.output_combo.setCurrentIndex(idx)
                self.active_out_label.setText(self.output_combo.itemText(idx))
            
        # Connect signals after populating to avoid triggering during setup
        try:
            self.input_combo.currentIndexChanged.disconnect()
            self.output_combo.currentIndexChanged.disconnect()
        except TypeError:
            pass # Not connected yet
            
        self.input_combo.currentIndexChanged.connect(self.on_device_changed)
        self.output_combo.currentIndexChanged.connect(self.on_device_changed)

    def on_device_changed(self):
        input_idx = self.input_combo.currentData()
        output_idx = self.output_combo.currentData()
        
        if input_idx is not None and output_idx is not None:
            try:
                self.audio_engine.set_devices(input_idx, output_idx)
                self.active_in_label.setText(self.input_combo.currentText())
                self.active_out_label.setText(self.output_combo.currentText())
                
                # Save to config
                in_name = self._get_device_name_for_config(input_idx, self.input_combo.currentText())
                out_name = self._get_device_name_for_config(output_idx, self.output_combo.currentText())
                
                self.config_manager.set_audio_config(
                    in_name, 
                    out_name, 
                    self.audio_engine.sample_rate,
                    self.audio_engine.block_size,
                    self.audio_engine.input_channel_mode,
                    self.audio_engine.output_channel_mode
                )
                
                QMessageBox.information(self, tr("Success"), f"{tr('Devices set to Input:')} {input_idx}, {tr('Output:')} {output_idx}")
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to set devices:')} {e}")

    def update_buffer_duration(self):
        try:
            sr = int(self.sr_combo.currentText())
            bs = int(self.bs_combo.currentText())
            duration_ms = (bs / sr) * 1000
            self.bs_duration_label.setText(tr("Latency: {0:.1f} ms").format(duration_ms))
        except ValueError:
            self.bs_duration_label.setText("")

    def on_sr_changed(self, text):
        try:
            rate = int(text)
            self.audio_engine.set_sample_rate(rate)
            self.update_buffer_duration()
            
            # Save config
            if self.input_combo.currentIndex() >= 0:
                in_id = self.input_combo.currentData()
                out_id = self.output_combo.currentData()
                in_name = self._get_device_name_for_config(in_id, self.input_combo.currentText())
                out_name = self._get_device_name_for_config(out_id, self.output_combo.currentText())
                self.config_manager.set_audio_config(
                    in_name, out_name, rate, 
                    self.audio_engine.block_size,
                    self.audio_engine.input_channel_mode,
                    self.audio_engine.output_channel_mode
                )
        except ValueError:
            pass

    def on_bs_changed(self, text):
        try:
            size = int(text)
            self.audio_engine.set_block_size(size)
            self.update_buffer_duration()
            
            # Save config
            if self.input_combo.currentIndex() >= 0:
                in_id = self.input_combo.currentData()
                out_id = self.output_combo.currentData()
                in_name = self._get_device_name_for_config(in_id, self.input_combo.currentText())
                out_name = self._get_device_name_for_config(out_id, self.output_combo.currentText())
                self.config_manager.set_audio_config(
                    in_name, out_name, 
                    self.audio_engine.sample_rate,
                    size,
                    self.audio_engine.input_channel_mode,
                    self.audio_engine.output_channel_mode
                )
        except ValueError:
            pass

    def on_ch_mode_changed(self):
        in_mode = self.in_ch_combo.currentText().lower()
        out_mode = self.out_ch_combo.currentText().lower()
        self.audio_engine.set_channel_mode(in_mode, out_mode)
        
        # Save config
        if self.input_combo.currentIndex() >= 0:
            in_id = self.input_combo.currentData()
            out_id = self.output_combo.currentData()
            in_name = self._get_device_name_for_config(in_id, self.input_combo.currentText())
            out_name = self._get_device_name_for_config(out_id, self.output_combo.currentText())
            self.config_manager.set_audio_config(
                in_name, out_name, 
                self.audio_engine.sample_rate,
                self.audio_engine.block_size,
                in_mode, out_mode
            )
