#!/usr/bin/env python3
import sys
import signal
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from src.core.config_manager import ConfigManager
from src.core.localization import get_manager, tr
from src.core.utils import resource_path
from src.gui.main_window import MainWindow

def main():
    """GUI Application Entry Point"""
    # Allow Ctrl+C to exit
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Load language early so the splash text matches user settings.
    # Keep this lightweight: just read config + load translations.
    try:
        config_manager = ConfigManager()
        get_manager().load_language(config_manager.get_language())
    except Exception:
        # If config or translations fail, proceed with defaults.
        pass

    app = QApplication(sys.argv)

    # Startup splash (loading screen): show immediately while MainWindow initializes.
    pixmap = QPixmap(resource_path('src/assets/welcome.png'))
    if pixmap.isNull():
        pixmap = QPixmap(720, 405)
        pixmap.fill(Qt.GlobalColor.black)

    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    splash.show()
    splash.showMessage(
        f"{tr('Loading...')}\n{tr('Initializing application...')}",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        Qt.GlobalColor.white,
    )
    app.processEvents()

    app.setApplicationName(tr("Audio Measurement Suite"))
    
    window = MainWindow()
    window.show()

    splash.finish(window)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
