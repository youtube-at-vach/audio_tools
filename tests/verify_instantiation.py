import sys
import os
sys.path.append(os.getcwd())

from src.core.audio_engine import AudioEngine
from src.gui.widgets.advanced_distortion_meter import AdvancedDistortionMeter

try:
    engine = AudioEngine()
    meter = AdvancedDistortionMeter(engine)
    print("Successfully instantiated AdvancedDistortionMeter")
except Exception as e:
    print(f"Failed to instantiate: {e}")
    sys.exit(1)
