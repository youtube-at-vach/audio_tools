from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont
import os
from src.core.utils import resource_path

from src.core.localization import tr

class WelcomeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Image Container
        image_label = QLabel()
        # Load image from assets
        # We assume 'src/assets/welcome.png' relative to the bundle root or source root
        assets_path = resource_path('src/assets/welcome.png')
        
        # Fallback for dev environment if running from inside src or similar
        if not os.path.exists(assets_path):
             # Try relative to this file
             assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets', 'welcome.png')

        if os.path.exists(assets_path):
            pixmap = QPixmap(assets_path)
            # Resize to fit width, keeping aspect ratio, or just display centered
            # Let's scale it to a reasonable height, e.g., 400px
            scaled_pixmap = pixmap.scaledToHeight(400, Qt.TransformationMode.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setStyleSheet("background-color: #1e1e1e;") # Match dark theme
        else:
            image_label.setText(tr("Welcome Image Not Found"))
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
        layout.addWidget(image_label)
        
        # Text Container
        text_container = QWidget()
        text_container.setStyleSheet("background-color: #2b2b2b; color: #e0e0e0;")
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(40, 30, 40, 40)
        text_layout.setSpacing(15)
        
        # Title
        title = QLabel("MeasureLab")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_layout.addWidget(title)
        
        # Description
        desc = QLabel(
            tr("A comprehensive set of tools for precision audio analysis and measurement.\nSelect a module from the sidebar to begin.")
        )
        desc.setFont(QFont("Arial", 12))
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_layout.addWidget(desc)
        
        # Features List (Grid or Horizontal)
        features_layout = QHBoxLayout()
        features_layout.setSpacing(20)
        
        features = [
            tr("Signal Generator"), tr("Spectrum Analyzer"), tr("LUFS Meter"),
            tr("Loopback Finder"), tr("Distortion Analyzer"), tr("Network Analyzer"),
            tr("Oscilloscope"), tr("Lock-in Amplifier"), tr("Frequency Counter"),
            tr("Spectrogram"), tr("Boxcar Averager")
        ]
        
        for feat in features:
            lbl = QLabel(f"• {feat}")
            lbl.setFont(QFont("Arial", 10))
            lbl.setStyleSheet("color: #aaaaaa;")
            features_layout.addWidget(lbl)
            
        features_layout.addStretch()
        features_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # We might have too many items for one row, let's just show a few key ones or use a grid if needed.
        # For now, a simple label with bullets might be cleaner.
        
        # Let's replace the HBox with a single centered label for features
        features_str = " • ".join(features)
        features_label = QLabel(features_str)
        features_label.setFont(QFont("Arial", 10))
        features_label.setStyleSheet("color: #888888;")
        features_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        text_layout.addWidget(features_label)
        text_layout.addStretch()
        
        layout.addWidget(text_container)
        
        self.setLayout(layout)
