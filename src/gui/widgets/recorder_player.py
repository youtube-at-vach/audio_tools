import numpy as np
import soundfile as sf
import scipy.signal
import os
import time
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QComboBox, QCheckBox, QGroupBox, QProgressBar,
                             QStyle, QMessageBox, QProgressDialog)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class FileLoadWorker(QThread):
    finished = pyqtSignal(bool, object, str) # success, data, message

    def __init__(self, filepath, target_sr):
        super().__init__()
        self.filepath = filepath
        self.target_sr = target_sr

    def run(self):
        try:
            # First read basic info to check length/sr
            info = sf.info(self.filepath)
            file_sr = info.samplerate
            
            # Read data
            data, _ = sf.read(self.filepath, always_2d=True)
            
            msg_extra = ""
            if file_sr != self.target_sr:
                # Resample
                num_samples = int(len(data) * self.target_sr / file_sr)
                # scipy.signal.resample is Fourier method, good for quality but slow for huge files
                # For this task, we assume it's acceptable as long as it's in a thread
                data = scipy.signal.resample(data, num_samples)
                msg_extra = f" (Resampled from {file_sr}Hz)"
            
            result_msg = f"Loaded: {os.path.basename(self.filepath)} ({self.target_sr}Hz{msg_extra}, {data.shape[1]}ch, {len(data)/self.target_sr:.2f}s)"
            self.finished.emit(True, data, result_msg)
            
        except Exception as e:
            self.finished.emit(False, None, str(e))


class RecorderPlayer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        
        # State
        self.is_playing = False
        self.is_recording = False
        self.loop_playback = False
        
        # Buffers
        self.playback_buffer = None # numpy array (samples, channels)
        self.playback_pos = 0
        self.record_buffer = [] # List of numpy arrays
        self.recorded_samples = 0
        
        # Settings
        self.input_mode = 'Stereo' # Stereo, Left, Right
        self.output_mode = 'Stereo' # Stereo, Left, Right, Mono
        
        self.callback_id = None
        self.widget = None

    @property
    def name(self) -> str:
        return "Recorder & Player"

    @property
    def description(self) -> str:
        return "Record and play audio files (WAV, MP3, FLAC, etc.)"

    def run(self, args):
        pass

    def get_widget(self):
        if self.widget is None:
            self.widget = RecorderPlayerWidget(self)
        return self.widget

    def set_playback_data(self, data):
        self.playback_buffer = data
        self.playback_pos = 0

    # Deprecated synchronous load, kept for compatibility if needed, but UI should use worker
    def load_file(self, filepath):
        try:
            data, file_sr = sf.read(filepath, always_2d=True)
            engine_sr = self.audio_engine.sample_rate
            
            msg_extra = ""
            
            # Resample if needed
            if file_sr != engine_sr:
                print(f"Resampling {os.path.basename(filepath)}: {file_sr}Hz -> {engine_sr}Hz")
                # Calculate new number of samples
                num_samples = int(len(data) * engine_sr / file_sr)
                
                # Use scipy.signal.resample (Fourier method)
                # Note: For very large files, this might be slow and memory intensive.
                # But for typical measurement signals it's fine.
                data = scipy.signal.resample(data, num_samples)
                msg_extra = f" (Resampled from {file_sr}Hz)"
            
            self.playback_buffer = data
            self.playback_pos = 0
            return True, f"Loaded: {os.path.basename(filepath)} ({engine_sr}Hz{msg_extra}, {data.shape[1]}ch, {len(data)/engine_sr:.2f}s)"
        except Exception as e:
            return False, str(e)

    def save_recording(self, filepath, format=None, subtype=None):
        if not self.record_buffer:
            return False, "No recording data"
        
        try:
            data = np.concatenate(self.record_buffer, axis=0)
            sf.write(filepath, data, self.audio_engine.sample_rate, format=format, subtype=subtype)
            return True, f"Saved: {filepath}"
        except Exception as e:
            return False, str(e)

    def start_playback(self):
        if self.playback_buffer is None:
            return
        self.is_playing = True
        self._ensure_callback()

    def stop_playback(self):
        self.is_playing = False
        self._check_stop_callback()

    def start_recording(self):
        self.record_buffer = []
        self.recorded_samples = 0
        self.is_recording = True
        self._ensure_callback()

    def stop_recording(self):
        self.is_recording = False
        self._check_stop_callback()

    def _ensure_callback(self):
        if self.callback_id is None:
            self.callback_id = self.audio_engine.register_callback(self.audio_callback)

    def _check_stop_callback(self):
        if not self.is_playing and not self.is_recording:
            if self.callback_id is not None:
                self.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None

    def audio_callback(self, indata, outdata, frames, time_info, status):
        # Recording
        if self.is_recording:
            # Select channels based on input_mode
            if self.input_mode == 'Stereo':
                rec_data = indata.copy()
            elif self.input_mode == 'Left':
                rec_data = indata[:, 0:1] # Keep 2D
            elif self.input_mode == 'Right':
                if indata.shape[1] > 1:
                    rec_data = indata[:, 1:2]
                else:
                    rec_data = np.zeros((frames, 1), dtype=indata.dtype)
            
            self.record_buffer.append(rec_data)
            self.recorded_samples += frames

        # Playback
        if self.is_playing and self.playback_buffer is not None:
            pb_len = len(self.playback_buffer)
            current_idx = 0
            
            while current_idx < frames:
                remaining = frames - current_idx
                available = pb_len - self.playback_pos
                
                to_copy = min(remaining, available)
                
                # Get chunk from buffer
                chunk = self.playback_buffer[self.playback_pos : self.playback_pos + to_copy]
                
                # Target slice in outdata
                out_slice = outdata[current_idx : current_idx + to_copy]
                
                file_ch = chunk.shape[1]
                out_ch = out_slice.shape[1]
                
                if self.output_mode == 'Stereo':
                    if file_ch == 1:
                        out_slice[:, 0] = chunk[:, 0]
                        if out_ch > 1: out_slice[:, 1] = chunk[:, 0]
                    else:
                        limit = min(file_ch, out_ch)
                        out_slice[:, :limit] = chunk[:, :limit]
                elif self.output_mode == 'Left':
                    out_slice[:, 0] = chunk[:, 0]
                    if out_ch > 1: out_slice[:, 1] = 0
                elif self.output_mode == 'Right':
                    if out_ch > 1: 
                        out_slice[:, 1] = chunk[:, 0] if file_ch == 1 else chunk[:, 1] if file_ch > 1 else 0
                        out_slice[:, 0] = 0
                elif self.output_mode == 'Mono':
                    # Mix down to mono and send to all outputs
                    if file_ch > 1:
                        mono = np.mean(chunk, axis=1)
                    else:
                        mono = chunk[:, 0]
                    
                    out_slice[:, 0] = mono
                    if out_ch > 1: out_slice[:, 1] = mono
                
                self.playback_pos += to_copy
                current_idx += to_copy
                
                if self.playback_pos >= pb_len:
                    if self.loop_playback:
                        self.playback_pos = 0
                    else:
                        self.is_playing = False
                        # Fill rest with zeros
                        outdata[current_idx:] = 0
                        break
        else:
            outdata.fill(0)

class RecorderPlayerWidget(QWidget):
    def __init__(self, module: RecorderPlayer):
        super().__init__()
        self.module = module
        self.init_ui()
        
        self.load_worker = None
        self.progress_dialog = None
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)

    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- Playback Section ---
        pb_group = QGroupBox(tr("Playback"))
        pb_layout = QVBoxLayout()
        
        # File Info
        self.file_label = QLabel(tr("No file loaded"))
        self.file_label.setWordWrap(True)
        pb_layout.addWidget(self.file_label)
        
        # Controls
        ctrl_layout = QHBoxLayout()
        self.load_btn = QPushButton(tr("Load File"))
        self.load_btn.clicked.connect(self.on_load)
        self.play_btn = QPushButton(tr("Play"))
        self.play_btn.clicked.connect(self.on_play_toggle)
        self.loop_check = QCheckBox(tr("Loop"))
        self.loop_check.toggled.connect(self.on_loop_toggle)
        
        ctrl_layout.addWidget(self.load_btn)
        ctrl_layout.addWidget(self.play_btn)
        ctrl_layout.addWidget(self.loop_check)
        pb_layout.addLayout(ctrl_layout)
        
        # Progress
        self.pb_progress = QProgressBar()
        self.pb_progress.setTextVisible(True)
        pb_layout.addWidget(self.pb_progress)
        
        # Output Mode
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel(tr("Output Mode:")))
        self.out_mode_combo = QComboBox()
        self.out_mode_combo.addItem(tr("Stereo"), "Stereo")
        self.out_mode_combo.addItem(tr("Left"), "Left")
        self.out_mode_combo.addItem(tr("Right"), "Right")
        self.out_mode_combo.addItem(tr("Mono"), "Mono")
        self.out_mode_combo.currentTextChanged.connect(self.on_out_mode_changed)
        out_layout.addWidget(self.out_mode_combo)
        pb_layout.addLayout(out_layout)

        # Output Destination
        dest_layout = QHBoxLayout()
        dest_layout.addWidget(QLabel(tr("Destination:")))
        self.dest_combo = QComboBox()
        self.dest_combo.addItem(tr("Physical Output"), "physical")
        self.dest_combo.addItem(tr("Internal Loopback (Silent)"), "loopback_silent")
        self.dest_combo.addItem(tr("Loopback + Physical"), "loopback_mix")
        self.dest_combo.setToolTip(tr("Select where the signal is sent.\nLoopback routes output to input internally."))
        self.dest_combo.currentTextChanged.connect(self.on_dest_changed)
        
        # Init state
        if self.module.audio_engine.loopback:
            if self.module.audio_engine.mute_output:
                self.dest_combo.setCurrentIndex(1)
            else:
                self.dest_combo.setCurrentIndex(2)
        else:
            self.dest_combo.setCurrentIndex(0)

        dest_layout.addWidget(self.dest_combo)
        pb_layout.addLayout(dest_layout)
        
        pb_group.setLayout(pb_layout)
        layout.addWidget(pb_group)
        
        # --- Recording Section ---
        rec_group = QGroupBox(tr("Recording"))
        rec_layout = QVBoxLayout()
        
        # Controls
        rec_ctrl_layout = QHBoxLayout()
        self.rec_btn = QPushButton(tr("Record"))
        self.rec_btn.setCheckable(True)
        self.rec_btn.clicked.connect(self.on_record_toggle)
        self.save_btn = QPushButton(tr("Save Recording"))
        self.save_btn.clicked.connect(self.on_save)
        self.save_btn.setEnabled(False)
        
        rec_ctrl_layout.addWidget(self.rec_btn)
        rec_ctrl_layout.addWidget(self.save_btn)
        rec_layout.addLayout(rec_ctrl_layout)
        
        # Info
        self.rec_info_label = QLabel(tr("Recorded: 0.00s"))
        rec_layout.addWidget(self.rec_info_label)
        
        # Input Mode
        in_layout = QHBoxLayout()
        in_layout.addWidget(QLabel(tr("Input Mode:")))
        self.in_mode_combo = QComboBox()
        self.in_mode_combo.addItem(tr("Stereo"), "Stereo")
        self.in_mode_combo.addItem(tr("Left"), "Left")
        self.in_mode_combo.addItem(tr("Right"), "Right")
        self.in_mode_combo.currentTextChanged.connect(self.on_in_mode_changed)
        in_layout.addWidget(self.in_mode_combo)
        rec_layout.addLayout(in_layout)
        
        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)

        layout.addStretch()
        self.setLayout(layout)

    def on_load(self):
        fname, _ = QFileDialog.getOpenFileName(self, tr("Open Audio File"), "", tr("Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*)"))
        if not fname:
            return

        try:
            # Check sample rate first
            info = sf.info(fname)
            file_sr = info.samplerate
            engine_sr = self.module.audio_engine.sample_rate
            
            if file_sr != engine_sr:
                reply = QMessageBox.question(
                    self, 
                    tr("Resample Required"), 
                    tr("The file sample rate ({0} Hz) differs from the engine rate ({1} Hz).\n"
                    "Resampling is required to play correctly.\n\n"
                    "Do you want to proceed? (This may take a moment for large files)").format(file_sr, engine_sr),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.No:
                    return

            # Start background loading
            self.load_worker = FileLoadWorker(fname, engine_sr)
            self.load_worker.finished.connect(self.on_load_finished)
            
            # Show progress dialog
            self.progress_dialog = QProgressDialog(tr("Loading and processing audio..."), tr("Cancel"), 0, 0, self)
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.canceled.connect(self.on_load_cancel)
            self.progress_dialog.show()
            
            self.load_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), tr("Failed to read file info:\n{0}").format(e))

    def on_load_finished(self, success, data, msg):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            
        if success:
            self.module.set_playback_data(data)
            self.file_label.setText(msg)
            self.pb_progress.setValue(0)
        else:
            if msg != "Cancelled": # Don't show error if user cancelled
                QMessageBox.critical(self, tr("Error"), tr("Failed to load file:\n{0}").format(msg))
        
        self.load_worker = None

    def on_load_cancel(self):
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.terminate() # Terminate is harsh but effective for simple worker
            self.load_worker.wait()
            self.load_worker = None

    def on_play_toggle(self):
        if self.module.is_playing:
            self.module.stop_playback()
        else:
            self.module.start_playback()

    def on_loop_toggle(self, checked):
        self.module.loop_playback = checked

    def on_out_mode_changed(self, text):
        self.module.output_mode = self.out_mode_combo.currentData()

    def on_record_toggle(self):
        if self.rec_btn.isChecked():
            self.module.start_recording()
            self.rec_btn.setText(tr("Stop Recording"))
            self.rec_btn.setStyleSheet("background-color: #ffcccc; color: red; font-weight: bold;")
            self.save_btn.setEnabled(False)
        else:
            self.module.stop_recording()
            self.rec_btn.setText(tr("Record"))
            self.rec_btn.setStyleSheet("")
            self.save_btn.setEnabled(True)

    def on_save(self):
        fname, selected_filter = QFileDialog.getSaveFileName(self, tr("Save Recording"), "recording.wav", tr("WAV (*.wav);;FLAC (*.flac);;OGG (*.ogg)"))
        if fname:
            # Determine format/subtype if needed, or let soundfile guess from extension
            success, msg = self.module.save_recording(fname)
            if success:
                QMessageBox.information(self, tr("Success"), msg)
            else:
                QMessageBox.critical(self, tr("Error"), tr("Failed to save:\n{0}").format(msg))

    def on_in_mode_changed(self, text):
        self.module.input_mode = self.in_mode_combo.currentData()

    def on_dest_changed(self, text):
        data = self.dest_combo.currentData()
        if data == "physical":
            self.module.audio_engine.set_loopback(False)
            self.module.audio_engine.set_mute_output(False)
        elif data == "loopback_silent":
            self.module.audio_engine.set_loopback(True)
            self.module.audio_engine.set_mute_output(True)
        elif data == "loopback_mix":
            self.module.audio_engine.set_loopback(True)
            self.module.audio_engine.set_mute_output(False)

    def update_ui(self):
        # Update Playback UI
        if self.module.is_playing:
            self.play_btn.setText(tr("Stop"))
            if self.module.playback_buffer is not None:
                total = len(self.module.playback_buffer)
                if total > 0:
                    progress = int(100 * self.module.playback_pos / total)
                    self.pb_progress.setValue(progress)
        else:
            self.play_btn.setText(tr("Play"))
            
        # Update Recording UI
        if self.module.is_recording:
            duration = self.module.recorded_samples / self.module.audio_engine.sample_rate
            self.rec_info_label.setText(tr("Recorded: {0:.2f}s").format(duration))
        elif self.module.recorded_samples > 0:
            duration = self.module.recorded_samples / self.module.audio_engine.sample_rate
            self.rec_info_label.setText(tr("Recorded: {0:.2f}s (Stopped)").format(duration))
