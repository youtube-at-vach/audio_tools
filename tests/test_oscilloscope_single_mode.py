import os
import sys

sys.path.append(os.getcwd())

from src.gui.widgets.oscilloscope import Oscilloscope


class _StubAudioEngine:
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def register_callback(self, _cb):
        raise RuntimeError("Not used in this test")

    def unregister_callback(self, _cb_id):
        raise RuntimeError("Not used in this test")


def test_single_mode_stops_after_first_trigger_capture():
    engine = _StubAudioEngine(sample_rate=48000)
    scope = Oscilloscope(engine)

    scope.trigger_source = 0
    scope.trigger_slope = 'Rising'
    scope.trigger_level = 0.0
    scope.trigger_mode = 'Single'
    scope.single_shot_armed = True
    scope.single_shot_fired = False

    # Make a buffer with a clean rising crossing inside the search window.
    scope.input_data[:] = -1.0
    crossing_prev = 7700
    crossing_now = 7701
    scope.input_data[crossing_prev, 0] = -0.5
    scope.input_data[crossing_now, 0] = 0.5

    window_duration = 0.01  # 10ms
    data = scope.get_display_data(window_duration)

    assert data is not None
    assert scope.single_shot_fired is True
    assert scope.single_shot_armed is False

    # After firing, further calls should not produce new data until re-armed.
    data2 = scope.get_display_data(window_duration)
    assert data2 is None
