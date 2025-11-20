#!/usr/bin/env python3
import sys
import signal
from PyQt6.QtWidgets import QApplication
from src.gui.main_window import MainWindow

def main():
    """GUI Application Entry Point"""
    # Allow Ctrl+C to exit
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setApplicationName("Audio Measurement Tools")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
