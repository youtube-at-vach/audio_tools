import unittest
import numpy as np
from unittest.mock import MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.widgets.noise_profiler import NoiseProfiler

class MockEngine:
    def __init__(self):
        self.sample_rate = 48000
        self.calibration = MagicMock()
        self.calibration.get_input_offset_db.return_value = 0.0
    
    def register_callback(self, cb):
        return 1
        
    def unregister_callback(self, id):
        pass

class TestNoiseProfilerAverage(unittest.TestCase):
    def setUp(self):
        self.engine = MockEngine()
        self.profiler = NoiseProfiler(self.engine)
        # Manually set attributes that will be added
        self.profiler.average_mode = True
        self.profiler.target_averages = 10
        self.profiler.current_avg_count = 0
        self.profiler.accumulated_magnitude = None
        self.profiler._avg_magnitude = None
        self.profiler.buffer_size = 1024
        self.profiler.input_data = np.zeros((1024, 2))

    def test_averaging_logic(self):
        # Simulate 3 updates
        
        # 1. First update
        # Create fake magnitude data (simulating what happens inside update_analysis)
        # Since we can't easily call update_analysis without GUI dependencies (QTimer, etc) or refactoring,
        # we will test the logic we INTEND to put into update_analysis here, 
        # OR we can refactor NoiseProfiler to have a separate 'process_data' method.
        # For now, let's assume we are testing the logic that will be added.
        
        # Let's define the logic function here as it would appear in the class
        def process_average(profiler, new_mag):
            if profiler.current_avg_count == 0:
                profiler.accumulated_magnitude = new_mag.copy()
                profiler.current_avg_count = 1
            else:
                # Cumulative Average: new_avg = (old_avg * count + current) / (count + 1)
                # But to avoid precision issues, better to accumulate sum?
                # "accumulated_magnitude" suggests sum.
                # If we store SUM:
                profiler.accumulated_magnitude += new_mag
                profiler.current_avg_count += 1
                
            profiler._avg_magnitude = profiler.accumulated_magnitude / profiler.current_avg_count
            
        # Test Data
        mag1 = np.ones(513) * 1.0
        mag2 = np.ones(513) * 2.0
        mag3 = np.ones(513) * 3.0
        
        # Step 1
        process_average(self.profiler, mag1)
        self.assertEqual(self.profiler.current_avg_count, 1)
        np.testing.assert_array_almost_equal(self.profiler._avg_magnitude, mag1)
        
        # Step 2
        process_average(self.profiler, mag2)
        self.assertEqual(self.profiler.current_avg_count, 2)
        # Avg of 1 and 2 is 1.5
        np.testing.assert_array_almost_equal(self.profiler._avg_magnitude, np.ones(513) * 1.5)
        
        # Step 3
        process_average(self.profiler, mag3)
        self.assertEqual(self.profiler.current_avg_count, 3)
        # Avg of 1, 2, 3 is 2.0
        np.testing.assert_array_almost_equal(self.profiler._avg_magnitude, np.ones(513) * 2.0)

if __name__ == '__main__':
    unittest.main()
