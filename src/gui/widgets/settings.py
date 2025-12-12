import numpy as np
import scipy.signal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, 
                             QFormLayout, QGroupBox, QMessageBox, QLineEdit, QDialog,
                             QDialogButtonBox, QDoubleSpinBox, QHBoxLayout)
from PyQt6.QtCore import QTimer, Qt
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
        self._c_b = None
        self._c_a = None
        self._c_zi = None

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
        self.level_spin.setRange(-60.0, -3.0)
        self.level_spin.setValue(-20.0)
        self.level_spin.setSingleStep(1.0)
        form.addRow(tr("Noise Level (dBFS)") + ":", self.level_spin)

        self.avg_spin = QDoubleSpinBox()
        self.avg_spin.setRange(0.2, 10.0)
        self.avg_spin.setValue(1.0)
        self.avg_spin.setSingleStep(0.2)
        form.addRow(tr("Averaging Time (s)") + ":", self.avg_spin)

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

        # C-weighting for input measurement
        c_b, c_a = _design_c_weighting(sr)
        self._c_b = c_b
        self._c_a = c_a
        self._c_zi = scipy.signal.lfilter_zi(c_b, c_a).astype(np.float64)

        self._ema_power = None
        self.current_dbfs_c = -120.0

        return band

    def on_start_toggle(self, checked):
        if checked:
            try:
                self.start_measurement()
                self.start_btn.setText(tr("Stop"))
                self.timer.start(100)
            except Exception as e:
                self.start_btn.setChecked(False)
                QMessageBox.critical(self, tr("Error"), str(e))
        else:
            self.stop_measurement()
            self.start_btn.setText(tr("Start"))
            self.timer.stop()

    def start_measurement(self):
        band = self._prepare_filters()

        amp = 10 ** (float(self.level_spin.value()) / 20.0)
        tau = float(self.avg_spin.value())
        sr = float(self.audio_engine.sample_rate)

        def callback(indata, outdata, frames, time, status):
            # --- Output: band-limited pink noise ---
            pink = self._pink.generate(frames).astype(np.float64)
            y, self._bpf_zi = scipy.signal.sosfilt(self._bpf_sos, pink, zi=self._bpf_zi)
            y = (y * amp).astype(np.float32)

            if outdata.shape[1] >= 2:
                outdata[:, 0] = y
                outdata[:, 1] = y
            else:
                outdata[:, 0] = y

            # --- Input: C-weighted RMS (channel 0) ---
            if indata.shape[1] > 0:
                x = indata[:, 0].astype(np.float64)
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
                band = (500.0, 2000.0)
            else:
                band = (30.0, 80.0)

            self.audio_engine.calibration.set_spl_calibration(
                measured_dbfs_c=float(self.current_dbfs_c),
                measured_spl_db=spl,
                band_hz=band,
                weighting='C',
                notes='Band-limited pink noise; C-weighted input; EMA power average',
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
        layout.addWidget(QLabel(f"<b>Step 1:</b> {tr('Connect a voltmeter to the output.')}"))
        
        # Step 2
        layout.addWidget(QLabel(f"<b>Step 2:</b> {tr('Set Test Tone.')}"))
        form = QFormLayout()
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(20, 20000); self.freq_spin.setValue(1000)
        form.addRow(f"{tr('Frequency (Hz):')}", self.freq_spin)
        
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setRange(-60, 0); self.level_spin.setValue(-12)
        form.addRow(f"{tr('Level (dBFS):')}", self.level_spin)
        layout.addLayout(form)
        
        # Step 3
        layout.addWidget(QLabel(f"<b>Step 3:</b> {tr('Play Tone.')}"))
        self.play_btn = QPushButton(tr("Start Tone"))
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.on_play_toggle)
        layout.addWidget(self.play_btn)
        
        # Step 4
        layout.addWidget(QLabel(f"<b>Step 4:</b> {tr('Enter measured voltage.')}"))
        meas_layout = QHBoxLayout()
        self.meas_spin = QDoubleSpinBox()
        self.meas_spin.setRange(-200, 1000); self.meas_spin.setDecimals(4)
        meas_layout.addWidget(self.meas_spin)
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Vrms", "mVrms", "dBV", "dBu"])
        meas_layout.addWidget(self.unit_combo)
        layout.addLayout(meas_layout)
        
        # Step 5
        layout.addWidget(QLabel(f"<b>Step 5:</b> {tr('Save.')}"))
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
            QMessageBox.information(self, tr("Success"), f"Output Gain calibrated to {gain:.4f} V/FS")
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
        layout.addWidget(QLabel(f"<b>Step 1:</b> {tr('Connect a known signal source to the input.')}"))
        
        # Step 2
        layout.addWidget(QLabel(f"<b>Step 2:</b> {tr('Measure Input Level.')}"))
        self.measure_btn = QPushButton(tr("Start Measurement"))
        self.measure_btn.setCheckable(True)
        self.measure_btn.clicked.connect(self.on_measure_toggle)
        layout.addWidget(self.measure_btn)
        
        self.level_label = QLabel(tr("Current Level: -- dBFS"))
        self.level_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.level_label)
        
        # Step 3
        layout.addWidget(QLabel(f"<b>Step 3:</b> {tr('Enter known source voltage.')}"))
        meas_layout = QHBoxLayout()
        self.meas_spin = QDoubleSpinBox()
        self.meas_spin.setRange(-200, 1000); self.meas_spin.setDecimals(4)
        meas_layout.addWidget(self.meas_spin)
        
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Vrms", "mVrms", "dBV", "dBu"])
        meas_layout.addWidget(self.unit_combo)
        layout.addLayout(meas_layout)
        
        # Step 4
        layout.addWidget(QLabel(f"<b>Step 4:</b> {tr('Save.')}"))
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
                raise ValueError("No signal detected. Please check connections.")
            
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
            QMessageBox.information(self, tr("Success"), f"Input Sensitivity calibrated to {sensitivity:.4f} V/FS")
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

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(tr("Audio Device Settings"))
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

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
        layout.addWidget(lang_group)

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
        layout.addWidget(appearance_group)

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
        layout.addWidget(dev_group)
        
        # Audio Configuration Group
        conf_group = QGroupBox(tr("Audio Configuration"))
        conf_layout = QFormLayout()
        
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
        layout.addWidget(conf_group)
        
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
        self.spl_offset_unit.addItems(["dB"])
        self.spl_offset_unit.setEnabled(False)

        spl_btn = QPushButton(tr("Wizard"))
        spl_btn.clicked.connect(self.open_spl_calibration)

        spl_layout = QHBoxLayout()
        spl_layout.setContentsMargins(0, 0, 0, 0)
        spl_layout.addWidget(self.spl_offset_edit)
        spl_layout.addWidget(self.spl_offset_unit)
        spl_layout.addWidget(spl_btn)

        cal_layout.addRow(tr("SPL Offset:"), spl_layout)
        
        cal_group.setLayout(cal_layout)
        layout.addWidget(cal_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Initialize
        self.refresh_devices()
        self.update_buffer_duration()
        self.update_in_sens_display()
        self.update_out_gain_display()
        self.update_spl_display()

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
            name = f"{i}: {dev['name']}"
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
                in_name = self.input_combo.currentText().split(": ", 1)[1]
                out_name = self.output_combo.currentText().split(": ", 1)[1]
                
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
                in_name = self.input_combo.currentText().split(": ", 1)[1]
                out_name = self.output_combo.currentText().split(": ", 1)[1]
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
                in_name = self.input_combo.currentText().split(": ", 1)[1]
                out_name = self.output_combo.currentText().split(": ", 1)[1]
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
            in_name = self.input_combo.currentText().split(": ", 1)[1]
            out_name = self.output_combo.currentText().split(": ", 1)[1]
            self.config_manager.set_audio_config(
                in_name, out_name, 
                self.audio_engine.sample_rate,
                self.audio_engine.block_size,
                in_mode, out_mode
            )
