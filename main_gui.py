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
        pixmap = QPixmap(520, 300)
        pixmap.fill(Qt.GlobalColor.black)
    else:
        pixmap = pixmap.scaled(
            520,
            300,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    splash.show()
    # Center on primary screen
    try:
        screen = app.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            splash_rect = splash.frameGeometry()
            splash_rect.moveCenter(geom.center())
            splash.move(splash_rect.topLeft())
    except Exception:
        pass
    splash.showMessage(
        f"{tr('Loading...')}\n{tr('Initializing application...')}",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        Qt.GlobalColor.white,
    )
    app.processEvents()

    # Brand name (do not translate)
    app.setApplicationName("MeasureLab")
    try:
        app.setApplicationDisplayName("MeasureLab")
    except Exception:
        pass
    
    window = MainWindow()

    # Preload all modules while splash is visible, so module switching feels instant.
    def _update_splash(msg: str):
        splash.showMessage(
            f"{tr('Loading...')}\n{msg}",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            Qt.GlobalColor.white,
        )
        app.processEvents()

    try:
        window.preload_all_modules(progress_callback=_update_splash)
    except Exception:
        # If preload fails, still show the window; individual pages may show errors.
        pass

    window.show()
    splash.finish(window)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
