
import numpy as np
import pyqtgraph as pg
import pywt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QComboBox, QGroupBox, QSpinBox, QSplitter, QProgressBar, QMessageBox)
from PyQt6.QtCore import QTimer, Qt, QRectF
from PyQt6.QtGui import QTransform
from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

class TransientAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        
        # State
        self.is_recording = False
        self.recorded_data = [] # List of chunks
        self.final_data = None # Numpy array (1D) after stop
        self.fs = 48000
        
        # Settings
        self.input_channel = 'Left'
        self.wavelet_name = 'cmor1.5-1.0'
        self.scale_min = 1
        self.scale_max = 128
        self.scale_step = 1
        self.min_anal_freq = 20
        self.max_anal_freq = 20000
        
        self.callback_id = None
        self.widget = None

    @property
    def name(self) -> str:
        return "Transient Analyzer"

    @property
    def description(self) -> str:
        return "Transient analysis using Wavelet Transform."

    def run(self, args):
        pass

    def get_widget(self):
        if self.widget is None:
            self.widget = TransientAnalyzerWidget(self)
        return self.widget

    def start_recording(self):
        self.recorded_data = []
        self.is_recording = True
        self.fs = self.audio_engine.sample_rate
        self.callback_id = self.audio_engine.register_callback(self._audio_callback)

    def stop_recording(self):
        self.is_recording = False
        if self.callback_id:
            self.audio_engine.unregister_callback(self.callback_id)
            self.callback_id = None
        
        # Concatenate data
        if self.recorded_data:
            full_raw = np.concatenate(self.recorded_data, axis=0)
            
            # Select channel
            if self.input_channel == 'Left':
                self.final_data = full_raw[:, 0]
            elif self.input_channel == 'Right':
                if full_raw.shape[1] > 1:
                    self.final_data = full_raw[:, 1]
                else:
                    self.final_data = full_raw[:, 0]
            else: # Average
                self.final_data = np.mean(full_raw, axis=1)
        else:
            self.final_data = None

    def _audio_callback(self, indata, outdata, frames, time, status):
        if self.is_recording:
            self.recorded_data.append(indata.copy())
        outdata.fill(0)

    def analyze(self):
        """
        Perform CWT on final_data.
        Returns: (times, frequencies, magnitude_scalogram)
        """
        if self.final_data is None or len(self.final_data) == 0:
            return None, None, None

        # Use linear frequencies for correct axis mapping in ImageItem
        num_scales = 120
        min_freq = self.min_anal_freq
        max_freq = self.max_anal_freq
        if max_freq > self.fs / 2:
            max_freq = self.fs / 2
        
        # Check integrity
        if min_freq <= 0: min_freq = 1
        if min_freq >= max_freq: min_freq = max_freq - 10
        
        # Linear space for frequencies to match linear Y-axis of plot
        freqs = np.linspace(min_freq, max_freq, num_scales)
        
        scales = []
        for f in freqs:
            s = pywt.frequency2scale(self.wavelet_name, f / self.fs)
            scales.append(s)
        
        scales = np.array(scales)
        
        # Run CWT
        cwtmatr, frequencies = pywt.cwt(self.final_data, scales, self.wavelet_name, sampling_period=1.0/self.fs)
        
        # Calculate Magnitude
        mag = np.abs(cwtmatr)
        
        times = np.arange(len(self.final_data)) / self.fs
        
        return times, frequencies, mag


class TransientAnalyzerWidget(QWidget):
    def __init__(self, module: TransientAnalyzer):
        super().__init__()
        self.module = module
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(100)

    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- Settings & Controls ---
        ctrl_group = QGroupBox(tr("Controls"))
        ctrl_layout = QHBoxLayout()
        
        # Input Channel
        ctrl_layout.addWidget(QLabel(tr("Channel:")))
        self.chan_combo = QComboBox()
        self.chan_combo.addItems(['Left', 'Right', 'Average'])
        self.chan_combo.currentTextChanged.connect(self.on_channel_changed)
        ctrl_layout.addWidget(self.chan_combo)
        
        # Wavelet
        ctrl_layout.addWidget(QLabel(tr("Wavelet:")))
        self.wavelet_combo = QComboBox()
        # Common continuous wavelets
        self.wavelet_combo.addItems(['cmor1.5-1.0', 'mexh', 'morl', 'cgau1', 'gaus1']) 
        self.wavelet_combo.setEditable(True) 
        self.wavelet_combo.currentTextChanged.connect(self.on_wavelet_changed)
        ctrl_layout.addWidget(self.wavelet_combo)

        # Param Layout (Freq Range)
        param_layout = QHBoxLayout()
        
        param_layout.addWidget(QLabel(tr("Min Freq:")))
        self.min_freq_spin = QSpinBox()
        self.min_freq_spin.setRange(1, 96000)
        self.min_freq_spin.setValue(self.module.min_anal_freq)
        self.min_freq_spin.setSuffix(" Hz")
        self.min_freq_spin.valueChanged.connect(self.on_min_freq_changed)
        param_layout.addWidget(self.min_freq_spin)

        param_layout.addWidget(QLabel(tr("Max Freq:")))
        self.max_freq_spin = QSpinBox()
        self.max_freq_spin.setRange(1, 96000)
        self.max_freq_spin.setValue(self.module.max_anal_freq)
        self.max_freq_spin.setSuffix(" Hz")
        self.max_freq_spin.valueChanged.connect(self.on_max_freq_changed)
        param_layout.addWidget(self.max_freq_spin)
        
        ctrl_layout.addLayout(param_layout)
        
        # Buttons
        self.rec_btn = QPushButton(tr("Record"))
        self.rec_btn.setCheckable(True)
        self.rec_btn.clicked.connect(self.on_record_toggle)
        ctrl_layout.addWidget(self.rec_btn)
        
        self.analyze_btn = QPushButton(tr("Analyze"))
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setToolTip(tr("Warning: Analysis can be slow for long recordings.\nComplexity ~ O(N * Scales)."))
        ctrl_layout.addWidget(self.analyze_btn)
        
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)
        
        # Complexity Note
        note_label = QLabel(tr("Note: CWT analysis is computationally intensive. Long recordings may take time."))
        note_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(note_label)
        
        # --- Visualization ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 1. Waveform Plot
        self.wave_plot = pg.PlotWidget(title=tr("Transient Waveform"))
        self.wave_plot.setLabel('left', tr("Amplitude"))
        self.wave_plot.setLabel('bottom', tr("Time"), units='s')
        self.wave_plot.showGrid(x=True, y=True)
        splitter.addWidget(self.wave_plot)
        
        # 2. Scalogram (Image)
        self.scalo_win = pg.GraphicsLayoutWidget()
        self.scalo_plot = self.scalo_win.addPlot(title=tr("Wavelet Scalogram"))
        self.scalo_plot.setLabel('left', tr("Frequency"), units='Hz')
        self.scalo_plot.setLabel('bottom', tr("Time"), units='s')
        
        self.img_item = pg.ImageItem()
        self.scalo_plot.addItem(self.img_item)
        
        # Histogram
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        self.hist.gradient.loadPreset('viridis')
        self.scalo_win.addItem(self.hist)
        
        splitter.addWidget(self.scalo_win)
        layout.addWidget(splitter)
        
        # Link X Axes
        self.scalo_plot.setXLink(self.wave_plot)
        
        self.setLayout(layout)

    def on_channel_changed(self, val):
        self.module.input_channel = val

    def on_wavelet_changed(self, val):
        self.module.wavelet_name = val

    def on_min_freq_changed(self, val):
        self.module.min_anal_freq = val

    def on_max_freq_changed(self, val):
        self.module.max_anal_freq = val

    def on_record_toggle(self):
        if self.rec_btn.isChecked():
            self.module.start_recording()
            self.rec_btn.setText(tr("Stop"))
            self.rec_btn.setStyleSheet("background-color: #ffcccc; color: red;")
            self.analyze_btn.setEnabled(False)
        else:
            self.module.stop_recording()
            self.rec_btn.setText(tr("Record"))
            self.rec_btn.setStyleSheet("")
            self.analyze_btn.setEnabled(True)
            self.update_waveform_plot() 

    def update_status(self):
        if self.module.is_recording:
             pass

    def update_waveform_plot(self):
        if self.module.final_data is None: return
        
        t = np.arange(len(self.module.final_data)) / self.module.fs
        self.wave_plot.clear()
        self.wave_plot.plot(t, self.module.final_data, pen='y')

    def on_analyze(self):
        if self.module.final_data is None: return
        
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText(tr("Analyzing..."))
        QTimer.singleShot(10, self._perform_analysis) 

    def _perform_analysis(self):
        try:
            times, freqs, mag = self.module.analyze()
            
            if times is None:
                return

            img_data = mag.T
            
            self.img_item.setImage(img_data, autoLevels=True)
            
            min_f = np.min(freqs)
            max_f = np.max(freqs)
            duration = times[-1]
            
            self.img_item.setRect(QRectF(0, min_f, duration, max_f - min_f))
            
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), str(e))
        
        finally:
            self.analyze_btn.setEnabled(True)
            self.analyze_btn.setText(tr("Analyze"))

