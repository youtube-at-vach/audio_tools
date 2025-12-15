import sys
import os
import pytest
from PyQt6.QtWidgets import QApplication

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.main_window import MainWindow

pytestmark = pytest.mark.skipif(
    os.environ.get("AUDIO_TOOLS_ENABLE_GUI_TESTS") != "1",
    reason="Requires GUI environment; set AUDIO_TOOLS_ENABLE_GUI_TESTS=1 to run",
)

def test_welcome_screen():
    QApplication(sys.argv)
    
    print("Initializing MainWindow...")
    window = MainWindow()
    
    print("Checking initial widget...")
    current_widget = window.content_area.currentWidget()
    print(f"Current widget type: {type(current_widget).__name__}")
    
    if type(current_widget).__name__ == 'WelcomeWidget':
        print("SUCCESS: WelcomeWidget is the initial widget.")
    else:
        print(f"FAILED: Expected WelcomeWidget, got {type(current_widget).__name__}")
        
    # Check if image loaded (indirectly)
    # We can't easily check the pixmap content in headless, but we can check if it didn't crash.
    
    print("Test Complete.")
    # Don't actually show window to avoid blocking
    # window.show()
    # sys.exit(app.exec())

if __name__ == "__main__":
    test_welcome_screen()
