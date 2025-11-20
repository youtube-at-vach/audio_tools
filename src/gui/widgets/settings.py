from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, 
                             QFormLayout, QGroupBox, QMessageBox)
from src.core.audio_engine import AudioEngine

class SettingsWidget(QWidget):
    def __init__(self, audio_engine: AudioEngine):
        super().__init__()
        self.audio_engine = audio_engine
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Audio Device Settings")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Input Device Selection
        layout.addWidget(QLabel("Input Device:"))
        self.input_combo = QComboBox()
        layout.addWidget(self.input_combo)

        # Output Device Selection
        layout.addWidget(QLabel("Output Device:"))
        self.output_combo = QComboBox()
        layout.addWidget(self.output_combo)

        # Device Selection Group
        dev_group = QGroupBox("Audio Devices")
        dev_layout = QFormLayout()
        
        dev_layout.addRow("Input Device:", self.input_combo)
        dev_layout.addRow("Output Device:", self.output_combo)
        
        self.refresh_btn = QPushButton("Refresh Devices")
        self.refresh_btn.clicked.connect(self.refresh_devices)
        dev_layout.addRow(self.refresh_btn)
        
        dev_group.setLayout(dev_layout)
        layout.addWidget(dev_group)
        
        # Audio Configuration Group
        conf_group = QGroupBox("Audio Configuration")
        conf_layout = QFormLayout()
        
        # Sample Rate
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(['44100', '48000', '88200', '96000', '192000'])
        self.sr_combo.setCurrentText(str(self.audio_engine.sample_rate))
        self.sr_combo.currentTextChanged.connect(self.on_sr_changed)
        conf_layout.addRow("Sample Rate:", self.sr_combo)
        
        # Buffer Size
        self.bs_combo = QComboBox()
        self.bs_combo.addItems(['256', '512', '1024', '2048', '4096'])
        self.bs_combo.setCurrentText(str(self.audio_engine.block_size))
        self.bs_combo.currentTextChanged.connect(self.on_bs_changed)
        conf_layout.addRow("Buffer Size:", self.bs_combo)
        
        # Input Channels
        self.in_ch_combo = QComboBox()
        self.in_ch_combo.addItems(['Stereo', 'Left', 'Right'])
        self.in_ch_combo.setCurrentText(self.audio_engine.input_channel_mode.capitalize())
        self.in_ch_combo.currentTextChanged.connect(self.on_ch_mode_changed)
        conf_layout.addRow("Input Channels:", self.in_ch_combo)
        
        # Output Channels
        self.out_ch_combo = QComboBox()
        self.out_ch_combo.addItems(['Stereo', 'Left', 'Right'])
        self.out_ch_combo.setCurrentText(self.audio_engine.output_channel_mode.capitalize())
        self.out_ch_combo.currentTextChanged.connect(self.on_ch_mode_changed)
        conf_layout.addRow("Output Channels:", self.out_ch_combo)
        
        conf_group.setLayout(conf_layout)
        layout.addWidget(conf_group)
        
        layout.addStretch()
        self.setLayout(layout)

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
            if idx >= 0: self.input_combo.setCurrentIndex(idx)
            
        if default_out is not None:
            idx = self.output_combo.findData(default_out)
            if idx >= 0: self.output_combo.setCurrentIndex(idx)
            
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
                QMessageBox.information(self, "Success", f"Devices set to Input: {input_idx}, Output: {output_idx}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to set devices: {e}")

    def on_sr_changed(self, text):
        try:
            rate = int(text)
            self.audio_engine.set_sample_rate(rate)
        except ValueError:
            pass

    def on_bs_changed(self, text):
        try:
            size = int(text)
            self.audio_engine.set_block_size(size)
        except ValueError:
            pass

    def on_ch_mode_changed(self):
        in_mode = self.in_ch_combo.currentText().lower()
        out_mode = self.out_ch_combo.currentText().lower()
        self.audio_engine.set_channel_mode(in_mode, out_mode)
