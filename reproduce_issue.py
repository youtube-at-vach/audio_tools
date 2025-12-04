
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import numpy as np
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800, 600)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Main Plot (Log Mode)
        self.plot = pg.PlotWidget(title="Main Plot (Log X)")
        self.plot.setLogMode(x=True, y=False)
        layout.addWidget(self.plot)
        
        # Data
        x = np.linspace(20, 20000, 100)
        y = np.sin(x)
        self.plot.plot(x, y, pen='y')

        # Overlay View (Linked)
        self.vb2 = pg.ViewBox()
        self.plot.plotItem.scene().addItem(self.vb2)
        self.plot.plotItem.layout.addItem(pg.AxisItem('right'), 2, 3)
        
        # Link X
        self.vb2.setXLink(self.plot.plotItem.vb)
        
        # Scenario 2: Manual Log
        self.vb2.setLogMode(False, False)
        
        # Plot data to Overlay
        # Manually log X
        curve2 = pg.PlotCurveItem(np.log10(x), y + 2, pen='r')
        self.vb2.addItem(curve2)
        
        self.vb2.setGeometry(self.plot.plotItem.vb.sceneBoundingRect())
        self.plot.plotItem.vb.sigResized.connect(self.update_views)

    def update_views(self):
        self.vb2.setGeometry(self.plot.plotItem.vb.sceneBoundingRect())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    # We can't see the window, but we can inspect the view ranges or just run it to ensure no crash.
    # To verify the axis labels, we need to inspect the AxisItem or ViewBox state.
    
    # Let's print the view range of the main plot
    # After a short delay to allow auto-range
    import time
    def check():
        vr = win.plot.plotItem.vb.viewRange()
        print(f"Main View Range: {vr}")
        # If range is [1.3, 4.3], it's correct (Log).
        # If range is [20, 20000], it's incorrect (Linear interpreted as Log).
        app.quit()
        
    from PyQt6.QtCore import QTimer
    QTimer.singleShot(1000, check)
    app.exec()
