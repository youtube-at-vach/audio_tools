from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QStackedWidget, QStatusBar
from PyQt6.QtCore import Qt, QTimer
from src.core.audio_engine import AudioEngine
from src.core.config_manager import ConfigManager
from src.gui.widgets.settings import SettingsWidget
from src.gui.widgets.signal_generator import SignalGenerator
from src.gui.widgets.spectrum_analyzer import SpectrumAnalyzer
from src.gui.widgets.lufs_meter import LufsMeter
from src.gui.widgets.loopback_finder import LoopbackFinder
from src.gui.widgets.network_analyzer import NetworkAnalyzer
from src.gui.widgets.distortion_analyzer import DistortionAnalyzer
from src.gui.widgets.oscilloscope import Oscilloscope
from src.gui.widgets.lock_in_amplifier import LockInAmplifier
from src.gui.widgets.welcome import WelcomeWidget
from src.gui.widgets.frequency_counter import FrequencyCounter
from src.gui.widgets.spectrogram import Spectrogram
from src.core.localization import get_manager, tr

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("Audio Measurement Suite"))
        self.resize(1000, 700)
        
        # Initialize Core Components
        self.config_manager = ConfigManager()
        
        # Initialize Localization
        lang = self.config_manager.get_language()
        get_manager().load_language(lang)
        
        self.audio_engine = AudioEngine()
        
        # Load saved config
        audio_cfg = self.config_manager.get_audio_config()
        last_in = audio_cfg.get('input_device')
        last_out = audio_cfg.get('output_device')
        
        # Default IDs
        in_id, out_id = 3, 3 # Fallback
        
        if last_in or last_out:
            # Find IDs by name
            devices = self.audio_engine.list_devices()
            found_in = False
            found_out = False
            
            for i, dev in enumerate(devices):
                if last_in and dev['name'] == last_in and dev['max_input_channels'] > 0:
                    in_id = i
                    found_in = True
                if last_out and dev['name'] == last_out and dev['max_output_channels'] > 0:
                    out_id = i
                    found_out = True
            
            if not found_in and last_in:
                print(f"Saved input device '{last_in}' not found, using default.")
            if not found_out and last_out:
                print(f"Saved output device '{last_out}' not found, using default.")

        try:
            self.audio_engine.set_devices(in_id, out_id)
            
            # Apply other settings
            sr = audio_cfg.get('sample_rate', 48000)
            self.audio_engine.set_sample_rate(sr)
            
            bs = audio_cfg.get('block_size', 1024)
            self.audio_engine.set_block_size(bs)
            
            in_ch = audio_cfg.get('input_channels', 'stereo')
            out_ch = audio_cfg.get('output_channels', 'stereo')
            self.audio_engine.set_channel_mode(in_ch, out_ch)
            
        except Exception as e:
            print(f"Failed to set devices/settings: {e}")
            # Try default if specific failed
            try:
                self.audio_engine.set_devices(None, None)
            except:
                pass
        
        # Initialize Modules
        self.modules = [
            SignalGenerator(self.audio_engine),
            SpectrumAnalyzer(self.audio_engine),
            LufsMeter(self.audio_engine),
            LoopbackFinder(self.audio_engine),
            DistortionAnalyzer(self.audio_engine),
            NetworkAnalyzer(self.audio_engine),
            Oscilloscope(self.audio_engine),
            LockInAmplifier(self.audio_engine),
            FrequencyCounter(self.audio_engine),
            Spectrogram(self.audio_engine)
        ]
        
        # Main layout container
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Sidebar for tool selection
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.addItem(tr("Welcome"))
        self.sidebar.addItem(tr("Settings")) # Add Settings item
        
        for module in self.modules:
            self.sidebar.addItem(tr(module.name))
            
        self.sidebar.currentRowChanged.connect(self.on_tool_selected)
        layout.addWidget(self.sidebar)
        
        # Main content area
        self.content_area = QStackedWidget()
        layout.addWidget(self.content_area)
        
        # Add initial welcome page (Index 0)
        self.welcome_widget = WelcomeWidget()
        self.content_area.addWidget(self.welcome_widget)
        
        # Add Settings Page (Index 1)
        self.settings_widget = SettingsWidget(self.audio_engine, self.config_manager)
        self.content_area.addWidget(self.settings_widget)
        
        # Add module widgets (Index 2+)
        for module in self.modules:
            widget = module.get_widget()
            if widget:
                self.content_area.addWidget(widget)
            else:
                self.content_area.addWidget(QLabel(f"No GUI for {module.name}"))
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status Labels
        self.status_label = QLabel("Idle")
        self.io_label = QLabel("In: - | Out: -")
        self.sr_label = QLabel("SR: -")
        self.cpu_label = QLabel("CPU: 0%")
        self.clients_label = QLabel("Clients: 0")
        
        # Add labels to status bar
        self.status_bar.addPermanentWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.io_label)
        self.status_bar.addPermanentWidget(self.sr_label)
        self.status_bar.addPermanentWidget(self.cpu_label)
        self.status_bar.addPermanentWidget(self.clients_label)
        
        # Timer for status update
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(500) # 500ms update rate

    def update_status(self):
        status = self.audio_engine.get_status()
        
        # Active State
        if status['active']:
            self.status_label.setText("ACTIVE")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText("IDLE")
            self.status_label.setStyleSheet("color: gray;")
            
        # I/O Mode
        in_mode = status['input_channels'].capitalize()
        out_mode = status['output_channels'].capitalize()
        self.io_label.setText(f"In: {in_mode} | Out: {out_mode}")
        
        # Sample Rate
        self.sr_label.setText(f"SR: {status['sample_rate']}")
        
        # CPU Load
        cpu = status['cpu_load'] * 100
        self.cpu_label.setText(f"CPU: {cpu:.1f}%")
        
        # Clients
        self.clients_label.setText(f"Clients: {status['active_clients']}")
        
    def on_tool_selected(self, index):
        self.content_area.setCurrentIndex(index)
