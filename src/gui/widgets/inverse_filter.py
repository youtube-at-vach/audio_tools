import argparse
import numpy as np
import pyqtgraph as pg
import soundfile as sf
import os
import time
from scipy.interpolate import interp1d
from scipy import signal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
                             QComboBox, QCheckBox, QSlider, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QProgressBar, QSpinBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from src.measurement_modules.base import MeasurementModule
from src.core.localization import tr

class InverseFilter(MeasurementModule):
    def __init__(self, audio_engine):
        self.audio_engine = audio_engine
        self.is_running = False
        
    @property
    def name(self) -> str:
        return "Inverse Filter"

    @property
    def description(self) -> str:
        return "Apply inverse calibration filter to audio files."

    def run(self, args: argparse.Namespace):
        pass

    def get_widget(self):
        return InverseFilterWidget(self)

    def start_analysis(self):
        self.is_running = True

    def stop_analysis(self):
        self.is_running = False


class ProcessingWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str) # success, message
    
    def __init__(self, input_path, output_path, calibration_map, max_gain_db, taps, smoothing, normalize_gain):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.calibration_map = calibration_map
        self.max_gain_db = max_gain_db
        self.taps = taps
        self.smoothing = smoothing
        self.normalize_gain = normalize_gain
        self.is_cancelled = False
        
    def run(self):
        try:
            # 1. Prepare Filter Kernel
            if not self.calibration_map:
                self.finished.emit(False, tr("No calibration map content."))
                return

            self.progress.emit(5)

            # Read File Info
            try:
                info = sf.info(self.input_path)
                sr = info.samplerate
            except Exception as e:
                self.finished.emit(False, f"Failed to read input file: {e}")
                return
            
            # Create Inverse Filter Response
            # Target Frequencies for FIR (0 to sr/2, taps//2 + 1 points)
            num_freqs = self.taps // 2 + 1
            target_freqs = np.linspace(0, sr/2, num_freqs)
            
            # Extract Map Data
            map_data = np.array(self.calibration_map)
            map_freqs = map_data[:, 0]
            map_mags = map_data[:, 1]
            map_phases = map_data[:, 2]

            # Unwrap phase to avoid 180/360 degree jumps that create spurious ringing
            map_phases = np.degrees(np.unwrap(np.radians(map_phases)))
            
            # Interpolate
            # Use linear interpolation. Fill_value behavior: clamp
            interp_mag = interp1d(map_freqs, map_mags, kind='linear', fill_value=(map_mags[0], map_mags[-1]), bounds_error=False)
            interp_phase = interp1d(map_freqs, map_phases, kind='linear', fill_value=(map_phases[0], map_phases[-1]), bounds_error=False)
            
            interp_mags_db = interp_mag(target_freqs)
            interp_phases_deg = interp_phase(target_freqs)
            
            # Apply Smoothing (Optional)
            if self.smoothing > 0:
                # Simple moving average on dB magnitude
                window_len = int(self.smoothing * num_freqs / 100) | 1 # Odd
                if window_len > 3:
                    interp_mags_db = signal.savgol_filter(interp_mags_db, window_len, 2)
            
            # Inverse: H_inv_dB = - H_sys_dB
            inv_mags_db = -interp_mags_db
            
            # Regularization (Max Gain Limit)
            inv_mags_db = np.clip(inv_mags_db, -200, self.max_gain_db)
            
            # Inverse Phase: H_inv_ang = - H_sys_ang
            # Note: Phase is in degrees
            inv_phases_rad = np.radians(-interp_phases_deg)
            
            # Construct Complex Response
            inv_mags_lin = 10**(inv_mags_db / 20)
            H_inv = inv_mags_lin * np.exp(1j * inv_phases_rad)
            
            # Create FIR Kernel via IFFT
            # irfft expects (n/2+1) points for n-point output
            kernel = np.fft.irfft(H_inv, n=self.taps)
            
            # Circular Shift to center
            # FFT assumes phase starts at t=0. If we have negative phase (delay), it wraps.
            # We want a linear phase causal filter usually, or just centered.
            # If we just used the phase from calibration, it captures the delay.
            # But the resulting impulse response might be wrapped around.
            # To make it a usable FIR kernel, we often roll it to center.
            kernel = np.roll(kernel, self.taps // 2)
            
            # Apply Window
            window = signal.windows.hamming(self.taps)
            kernel = kernel * window

            self.progress.emit(15)
            
            # Normalize Kernel Gain?
            # If we want 0dB DC gain to be preserved?
            # Usually we trust the absolute values of H_inv.
            
            # 2. Process Audio
            # Block processing
            block_size = 65536
            
            with sf.SoundFile(self.input_path) as infile:
                with sf.SoundFile(self.output_path, 'w', samplerate=sr, channels=infile.channels) as outfile:
                    
                    # Pre-fill for convolution tail if needed? 
                    # scipy.signal.convolve handles it, but for streaming we need Overlap-Add.
                    # Simpler: Use scipy.signal.fftconvolve on the whole file if it fits in memory?
                    # Or simple OLA.
                    
                    # For safety with "arbitrary" files, let's implement simple OLA or use oaconvolve.
                    # oaconvolve is efficient.
                    
                    # Check file size
                    if info.frames < 10 * 60 * sr: # < 10 mins
                         # Load all
                        data = infile.read(dtype='float32')
                        self.progress.emit(25)
                        
                        processed_channels = []
                        channel_count = 1 if data.ndim == 1 else data.shape[1]

                        if data.ndim == 1:
                            # Mono
                            out_data = signal.oaconvolve(data, kernel, mode='same')
                            processed_channels.append(out_data)
                            self.progress.emit(25 + int(60 * 1 / channel_count))
                        else:
                            # Multichannel
                            for ch in range(channel_count):
                                ch_data = data[:, ch]
                                out_data = signal.oaconvolve(ch_data, kernel, mode='same')
                                processed_channels.append(out_data)
                                # Emit incremental progress per channel
                                self.progress.emit(25 + int(60 * (ch + 1) / channel_count))
                                
                        if len(processed_channels) == 1:
                            data = processed_channels[0]
                        else:
                            data = np.column_stack(processed_channels)
                        
                        # Normalize Gain if Requested (Peak Normalize)
                        if self.normalize_gain:
                            peak = np.max(np.abs(data))
                            if peak > 0:
                                data = data / peak * 0.95 # -0.5 dB roughly
                        
                        self.progress.emit(90)
                        outfile.write(data)
                        self.progress.emit(100)
                        
                    else:
                        # Chunk processing (Naive implementation without proper overlap-add state for now, 
                        # just to avoid memory crash, but it will click at boundaries. 
                        # Implementing proper OLA is complex for this snippet.
                        # Let's trust oaconvolve with large blocks? 
                        # Actually, `scipy.signal.convolve` doesn't support streaming state.
                        
                        # Fallback: Just limit to 10 mins for now or warn.
                        # Or implementing a basic Overlap-Save.
                        
                        # Let's try loading all. Most WAVs are short.
                        # If huge, we risk MemoryError.
                        
                        self.finished.emit(False, "File too large (> 10 mins) for current implementation.")
                        return

            self.finished.emit(True, "Processing Complete.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e))
    
    def cancel(self):
        self.is_cancelled = True


class InverseFilterWidget(QWidget):
    def __init__(self, module: InverseFilter):
        super().__init__()
        self.module = module
        self.init_ui()
        self.worker = None

    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- Section 1: Calibration ---
        cal_group = QGroupBox(tr("1. Calibration Data"))
        cal_layout = QHBoxLayout()
        
        self.cal_status_label = QLabel(tr("Status: No Map Loaded"))
        self.cal_status_label.setStyleSheet("color: orange;")
        cal_layout.addWidget(self.cal_status_label)
        
        self.load_cal_btn = QPushButton(tr("Reload Calibration Data"))
        self.load_cal_btn = QPushButton(tr("Reload from Memory"))
        self.load_cal_btn.clicked.connect(self.load_calibration)
        cal_layout.addWidget(self.load_cal_btn)

        self.load_file_btn = QPushButton(tr("Load File..."))
        self.load_file_btn.clicked.connect(self.load_calibration_file)
        cal_layout.addWidget(self.load_file_btn)
        
        cal_group.setLayout(cal_layout)
        layout.addWidget(cal_group)
        
        # --- Section 2: Filter Design ---
        filter_group = QGroupBox(tr("2. Inverse Filter Design"))
        filter_layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        self.max_gain_spin = QDoubleSpinBox()
        self.max_gain_spin.setRange(0, 60)
        self.max_gain_spin.setValue(20)
        self.max_gain_spin.setSuffix(" dB")
        self.max_gain_spin.setToolTip(tr("Maximum gain applied by the inverse filter (Regularization)."))
        self.max_gain_spin.valueChanged.connect(self.update_plot)
        form_layout.addRow(tr("Max Gain (Regularization):"), self.max_gain_spin)
        
        self.taps_combo = QComboBox()
        self.taps_combo.addItems(["1024", "2048", "4096", "8192", "16384", "32768", "65536"])
        self.taps_combo.setCurrentText("8192")
        self.taps_combo.currentTextChanged.connect(self.update_plot)
        form_layout.addRow(tr("FIR Taps:"), self.taps_combo)
        
        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0, 20)
        self.smooth_spin.setValue(0)
        self.smooth_spin.setSuffix(" %")
        self.smooth_spin.valueChanged.connect(self.update_plot)
        form_layout.addRow(tr("Smoothing:"), self.smooth_spin)
        
        filter_layout.addLayout(form_layout)
        
        # Plot
        self.plot = pg.PlotWidget(title=tr("Filter Response"))
        self.plot.setLabel('bottom', "Frequency", units='Hz')
        self.plot.setLabel('left', "Magnitude", units='dB')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend()
        self.plot.getPlotItem().getAxis('bottom').setLogMode(True)
        
        self.curve_sys = self.plot.plot(pen='r', name=tr("System (Measured)"))
        self.curve_inv_raw = self.plot.plot(pen=pg.mkPen(color=(0, 100, 0), style=Qt.PenStyle.DotLine), name=tr("Inverse (Raw)"))
        self.curve_inv = self.plot.plot(pen='g', name=tr("Inverse (Final)"))
        
        filter_layout.addWidget(self.plot)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group, stretch=2)
        
        # --- Section 3: File Processing ---
        proc_group = QGroupBox(tr("3. Audio Processing"))
        proc_layout = QFormLayout()
        
        # Input
        in_layout = QHBoxLayout()
        self.in_path_edit = QLabel(tr("No file selected"))
        in_btn = QPushButton(tr("Select Input Wav..."))
        in_btn.clicked.connect(self.select_input_file)
        in_layout.addWidget(self.in_path_edit, stretch=1)
        in_layout.addWidget(in_btn)
        proc_layout.addRow(tr("Input:"), in_layout)
        
        # Output
        out_layout = QHBoxLayout()
        self.out_path_edit = QLabel(tr("No output file")) # Will auto-generate
        out_btn = QPushButton(tr("Select Output Wav..."))
        out_btn.clicked.connect(self.select_output_file)
        out_layout.addWidget(self.out_path_edit, stretch=1)
        out_layout.addWidget(out_btn)
        proc_layout.addRow(tr("Output:"), out_layout)
        
        self.norm_check = QCheckBox(tr("Normalize Output (Peak)"))
        self.norm_check.setChecked(True)
        proc_layout.addRow(self.norm_check)
        
        self.process_btn = QPushButton(tr("Process & Save"))
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        proc_layout.addRow(self.process_btn)
        
        self.progress = QProgressBar()
        proc_layout.addRow(self.progress)
        
        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)
        
        self.setLayout(layout)
        
        # Initial Load
        self.load_calibration()
    
    def load_calibration(self):
        cal = self.module.audio_engine.calibration
        if hasattr(cal, 'frequency_map') and cal.frequency_map:
            self.cal_status_label.setText(tr("Status: Calibration Loaded ({0} points)").format(len(cal.frequency_map)))
            self.cal_status_label.setStyleSheet("color: green;")
            self.update_plot()
        else:
            self.cal_status_label.setText(tr("Status: No Map Found in AudioEngine"))
            self.cal_status_label.setText(tr("Status: No Map Found in AudioEngine"))
            self.cal_status_label.setStyleSheet("color: red;")
            
    def load_calibration_file(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Load Calibration Map"), "", tr("JSON Files (*.json)"))
        if path:
            if self.module.audio_engine.calibration.load_frequency_map(path):
                self.load_calibration() # Update UI
                QMessageBox.information(self, tr("Success"), tr("Calibration loaded successfully."))
            else:
                QMessageBox.critical(self, tr("Error"), tr("Failed to load calibration file."))
            
    def update_plot(self):
        cal = self.module.audio_engine.calibration
        if not hasattr(cal, 'frequency_map') or not cal.frequency_map:
            return
            
        data = np.array(cal.frequency_map)
        freqs = data[:, 0]
        mags = data[:, 1]
        
        # Plot System
        log_freqs = np.log10(freqs)
        self.curve_sys.setData(log_freqs, mags)
        
        # Calculate Inverse Preview
        # We want to match ProcessingWorker logic: Linear Interp -> Smoothing -> Clip
        
        # Parameters
        max_gain = self.max_gain_spin.value()
        taps = int(self.taps_combo.currentText())
        smoothing = self.smooth_spin.value()
        
        # Assume 48kHz for preview or max freq in map * 2
        sr = 48000
        if freqs[-1] > 24000: sr = freqs[-1] * 2
        
        num_freqs = taps // 2 + 1
        target_freqs = np.linspace(0, sr/2, num_freqs)
        
        # Interpolate System Response to Linear Grid
        interp_func = interp1d(freqs, mags, kind='linear', fill_value=(mags[0], mags[-1]), bounds_error=False)
        sys_mags_lin = interp_func(target_freqs)
        
        # Raw Inverse
        inv_mags_raw = -sys_mags_lin
        inv_mags_raw_clipped = np.clip(inv_mags_raw, -200, max_gain)
        
        # Smoothed Inverse
        inv_mags_smooth = inv_mags_raw.copy()
        if smoothing > 0:
            window_len = int(smoothing * num_freqs / 100) | 1
            if window_len > 3:
                inv_mags_smooth = signal.savgol_filter(inv_mags_smooth, window_len, 2)
        
        inv_mags_smooth_clipped = np.clip(inv_mags_smooth, -200, max_gain)
        
        # Plot
        # Avoid log(0) without mutating the design grid used later
        target_freqs_plot = target_freqs.copy()
        target_freqs_plot[0] = target_freqs_plot[1] / 10 
        log_target_freqs = np.log10(target_freqs_plot)
        
        # Show only the span of the loaded map so smoothing does not stretch x-axis
        freq_mask = (target_freqs_plot >= freqs[0]) & (target_freqs_plot <= freqs[-1])
        self.curve_inv_raw.setData(log_target_freqs[freq_mask], inv_mags_raw_clipped[freq_mask])
        self.curve_inv.setData(log_target_freqs[freq_mask], inv_mags_smooth_clipped[freq_mask])
        self.plot.setXRange(np.log10(freqs[0]), np.log10(freqs[-1]), padding=0.02)
        
    def select_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Open Wav File"), "", tr("Wav Files (*.wav)"))
        if path:
            self.in_path_edit.setText(path)
            self._update_process_btn()
            
            # Auto-set output
            base, ext = os.path.splitext(path)
            self.out_path_edit.setText(base + "_inverted" + ext)
            
    def select_output_file(self):
        path, _ = QFileDialog.getSaveFileName(self, tr("Save Wav File"), self.out_path_edit.text(), tr("Wav Files (*.wav)"))
        if path:
            self.out_path_edit.setText(path)
            self._update_process_btn()
            
    def _update_process_btn(self):
        self.process_btn.setEnabled(
            os.path.exists(self.in_path_edit.text()) and 
            self.out_path_edit.text() != "" and
            "Loaded" in self.cal_status_label.text()
        )
        
    def start_processing(self):
        if self.worker and self.worker.isRunning():
            return
            
        input_path = self.in_path_edit.text()
        output_path = self.out_path_edit.text()
        cal_map = self.module.audio_engine.calibration.frequency_map
        max_gain = self.max_gain_spin.value()
        taps = int(self.taps_combo.currentText())
        smoothing = self.smooth_spin.value()
        norm = self.norm_check.isChecked()
        
        self.process_btn.setEnabled(False)
        self.process_btn.setText(tr("Processing..."))
        self.progress.setValue(0)
        
        self.worker = ProcessingWorker(input_path, output_path, cal_map, max_gain, taps, smoothing, norm)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()
        
    def on_processing_finished(self, success, msg):
        self.process_btn.setEnabled(True)
        self.process_btn.setText(tr("Process & Save"))
        
        if success:
            QMessageBox.information(self, tr("Success"), msg)
        else:
            QMessageBox.critical(self, tr("Error"), msg)
