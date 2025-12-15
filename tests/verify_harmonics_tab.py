import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from PyQt6.QtWidgets import QApplication, QTableWidget
import pyqtgraph as pg
from src.gui.widgets.distortion_analyzer import DistortionAnalyzer, DistortionAnalyzerWidget

def test_harmonics_tab_structure():
    # Mock AudioEngine
    class MockAudioEngine:
        def __init__(self):
            self.sample_rate = 48000
            self.calibration = type('obj', (object,), {'output_gain': 1.0})
        def register_callback(self, cb): return 1
        def unregister_callback(self, id): pass

    QApplication(sys.argv)
    engine = MockAudioEngine()
    module = DistortionAnalyzer(engine)
    widget = DistortionAnalyzerWidget(module)
    
    # Check if Harmonics tab (index 1) has the new structure
    # Note: Tab index might be different if tabs were reordered, but based on code it's 1 (Spectrum=0, Harmonics=1, Sweep=2)
    # Wait, let's check tab text to be sure
    harmonics_idx = -1
    for i in range(widget.tabs.count()):
        if widget.tabs.tabText(i) == "Harmonics":
            harmonics_idx = i
            break
            
    assert harmonics_idx != -1, "Harmonics tab not found"
    
    harmonics_tab = widget.tabs.widget(harmonics_idx)
    assert harmonics_tab is not None
    
    # It should be a QWidget with QVBoxLayout
    layout = harmonics_tab.layout()
    assert layout is not None
    # We expect 2 items: Table and Plot
    assert layout.count() == 2
    
    # First item should be the table
    table = layout.itemAt(0).widget()
    assert isinstance(table, QTableWidget)
    assert table == widget.harmonics_table
    
    # Second item should be the plot
    plot = layout.itemAt(1).widget()
    assert isinstance(plot, pg.PlotWidget)
    assert plot == widget.harmonics_plot
    
    print("Harmonics tab structure verified successfully.")

if __name__ == "__main__":
    test_harmonics_tab_structure()
