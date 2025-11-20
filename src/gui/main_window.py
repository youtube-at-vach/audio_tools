from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QStackedWidget
from PyQt6.QtCore import Qt
from src.core.audio_engine import AudioEngine
from src.gui.widgets.settings import SettingsWidget
from src.gui.widgets.signal_generator import SignalGenerator
from src.gui.widgets.spectrum_analyzer import SpectrumAnalyzer
from src.gui.widgets.lufs_meter import LufsMeter
from src.gui.widgets.loopback_finder import LoopbackFinder
from src.gui.widgets.freq_response import FreqResponseAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Measurement Tools")
        self.resize(1000, 700)
        
        # Initialize Core Components
        self.audio_engine = AudioEngine()
        # Set default devices to UAC-232 (ID 3) as per user context
        # In a real app, this should be configurable via a settings menu
        self.audio_engine.set_devices(3, 3)
        
        # Initialize Modules
        self.modules = [
            SignalGenerator(self.audio_engine),
            SpectrumAnalyzer(self.audio_engine),
            LufsMeter(self.audio_engine),
            LoopbackFinder(self.audio_engine),
            FreqResponseAnalyzer(self.audio_engine)
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
        self.settings_widget = SettingsWidget(self.audio_engine)
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
