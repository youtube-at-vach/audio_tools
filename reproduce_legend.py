import sys
from PyQt6.QtWidgets import QApplication
import pyqtgraph as pg
import unittest

class TestLegendUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

    def test_legend_update(self):
        plot_widget = pg.PlotWidget()
        legend = plot_widget.addLegend()
        
        curve = pg.PlotCurveItem(name="Initial Name")
        plot_widget.addItem(curve)
        
        # Check initial legend
        # pyqtgraph LegendItem stores items in .items list of (sample, label) tuples
        initial_labels = [label.text for sample, label in legend.items]
        print(f"Initial Labels: {initial_labels}")
        self.assertIn("Initial Name", initial_labels)
        
        # Update name via setData
        curve.setData(name="Updated Name")
        
        # Check updated legend
        updated_labels = [label.text for sample, label in legend.items]
        print(f"Updated Labels: {updated_labels}")
        
        if "Updated Name" not in updated_labels:
            print("FAILURE: Legend did not update with setData(name=...)")
        else:
            print("SUCCESS: Legend updated.")
            
        self.assertIn("Updated Name", updated_labels)

if __name__ == '__main__':
    unittest.main()
