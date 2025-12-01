import sys
import unittest
from unittest.mock import MagicMock
from PyQt6.QtWidgets import QApplication
import pyqtgraph as pg

# Mock AudioEngine
sys.modules['src.core.audio_engine'] = MagicMock()

# Mock MeasurementModule
class MockMeasurementModule:
    def __init__(self, audio_engine):
        self.audio_engine = audio_engine

mock_base = MagicMock()
mock_base.MeasurementModule = MockMeasurementModule
sys.modules['src.measurement_modules.base'] = mock_base

# Import after mocking
from src.gui.widgets.impedance_analyzer import ImpedanceAnalyzer, ImpedanceAnalyzerWidget

class TestImpedanceAnalyzerLegend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.mock_engine = MagicMock()
        self.mock_engine.sample_rate = 48000
        self.module = ImpedanceAnalyzer(self.mock_engine)
        self.widget = ImpedanceAnalyzerWidget(self.module)

    def test_legend_updates(self):
        """Test that legend labels update when plot mode changes"""
        
        # Helper to get legend labels
        def get_legend_labels():
            return [label.text for sample, label in self.widget.legend.items]

        # Mode: |Z| & Phase
        self.widget.plot_mode_combo.setCurrentText("|Z| & Phase")
        labels = get_legend_labels()
        print(f"|Z| Mode Labels: {labels}")
        self.assertIn('|Z|', labels)
        self.assertIn('Phase', labels)
        
        # Mode: R & X
        self.widget.plot_mode_combo.setCurrentText("R & X (ESR/ESL)")
        labels = get_legend_labels()
        print(f"R & X Mode Labels: {labels}")
        self.assertIn('Resistance (R)', labels)
        self.assertIn('Reactance (X)', labels)
        self.assertNotIn('|Z|', labels) # Should be gone
        
        # Mode: Q Factor
        self.widget.plot_mode_combo.setCurrentText("Q Factor")
        labels = get_legend_labels()
        print(f"Q Factor Mode Labels: {labels}")
        self.assertIn('Q', labels)
        self.assertNotIn('Resistance (R)', labels)
        
        # Mode: C / L
        self.widget.plot_mode_combo.setCurrentText("C / L")
        labels = get_legend_labels()
        print(f"C / L Mode Labels: {labels}")
        self.assertIn('Capacitance', labels)
        self.assertIn('Inductance', labels)

if __name__ == '__main__':
    unittest.main()
