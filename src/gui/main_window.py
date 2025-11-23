from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QStackedWidget
from PyQt6.QtCore import Qt
from src.core.audio_engine import AudioEngine
from src.core.config_manager import ConfigManager
from src.gui.widgets.settings import SettingsWidget
from src.gui.widgets.signal_generator import SignalGenerator
from src.gui.widgets.spectrum_analyzer import SpectrumAnalyzer
from src.gui.widgets.lufs_meter import LufsMeter
from src.gui.widgets.loopback_finder import LoopbackFinder
from src.gui.widgets.loopback_finder import LoopbackFinder
from src.gui.widgets.imd_analyzer import IMDAnalyzer
from src.gui.widgets.network_analyzer import NetworkAnalyzer
from src.gui.widgets.distortion_analyzer import DistortionAnalyzer
from src.gui.widgets.distortion_analyzer import DistortionAnalyzer
from src.gui.widgets.oscilloscope import Oscilloscope
from src.gui.widgets.lock_in_amplifier import LockInAmplifier

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Measurement Tools")
        self.resize(1000, 700)
        
        # Initialize Core Components
        # Initialize Core Components
        self.config_manager = ConfigManager()
        self.audio_engine = AudioEngine()
        
        # Load saved devices
        last_in, last_out = self.config_manager.get_last_devices()
        
        # Default IDs
        in_id, out_id = 3, 3 # Fallback to UAC-232 if not found or first run
        
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
        except Exception as e:
            print(f"Failed to set devices: {e}")
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
            IMDAnalyzer(self.audio_engine),
            NetworkAnalyzer(self.audio_engine),
            DistortionAnalyzer(self.audio_engine),
            NetworkAnalyzer(self.audio_engine),
            DistortionAnalyzer(self.audio_engine),
            Oscilloscope(self.audio_engine),
            LockInAmplifier(self.audio_engine)
        ]
        
        # Main layout container
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Sidebar for tool selection
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.addItem("Welcome")
        self.sidebar.addItem("Settings") # Add Settings item
        
        for module in self.modules:
            self.sidebar.addItem(module.name)
            
        self.sidebar.currentRowChanged.connect(self.on_tool_selected)
        layout.addWidget(self.sidebar)
        
        # Main content area
        self.content_area = QStackedWidget()
        layout.addWidget(self.content_area)
        
        # Add initial welcome page (Index 0)
        welcome_label = QLabel("Select a tool from the sidebar to begin.")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_area.addWidget(welcome_label)
        
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
        
    def on_tool_selected(self, index):
        self.content_area.setCurrentIndex(index)
