import numpy as np


def _chunk_iter(arr: np.ndarray, chunk_size: int):
    for i in range(0, len(arr), chunk_size):
        yield arr[i : i + chunk_size]


def test_ltc_encoder_decoder_roundtrip_single_frame():
    from src.gui.widgets.timecode_monitor import LTCDecoder, LTCEncoder

    sr = 48_000
    fps = 30.0

    enc = LTCEncoder(sr, fps)
    dec = LTCDecoder(sr, fps)

    # In a real stream the decoder may need more than one frame to lock.
    # Also, decoding the *last* frame in a finite buffer can require subsequent
    # transitions; so we provide a few frames and assert the target appears.
    frames = [enc.generate_frame(12, 34, 56, ff) for ff in (19, 20, 21)]
    stream = np.concatenate(frames)

    seen = set()
    for chunk in _chunk_iter(stream, 256):
        if dec.process_samples(chunk):
            seen.add(dec.decoded_tc)

    assert dec.locked
    assert "12:34:56:20" in seen


def test_ltc_decoder_handles_block_boundary_crossings_halfbit_chunks():
    """Regression test: decoding must not miss transitions at chunk boundaries.

    Worst case is when the audio is processed in chunks that align with LTC half-bit
    transitions (e.g. 10 samples at 48k/30fps). Without boundary handling, the
    decoder can miss all crossings and decode garbage.
    """

    from src.gui.widgets.timecode_monitor import LTCDecoder, LTCEncoder

    sr = 48_000
    fps = 30.0

    enc = LTCEncoder(sr, fps)
    dec = LTCDecoder(sr, fps)

    # Build consecutive frames and include an extra trailing frame so the
    # penultimate frame can be fully decoded (needs following transitions).
    frames = [enc.generate_frame(1, 2, 3, ff) for ff in (0, 1, 2, 3)]
    stream = np.concatenate(frames)

    # 48k/30fps => 1600 samples/frame, 20 samples/bit, 10 samples/half-bit.
    decoded_tcs = []
    for chunk in _chunk_iter(stream, 10):
        if dec.process_samples(chunk):
            decoded_tcs.append(dec.decoded_tc)

    assert dec.locked
    assert decoded_tcs
    assert "01:02:03:02" in decoded_tcs


def test_timecode_monitor_input_delay_applies_to_display():
    from src.gui.widgets.timecode_monitor import TimecodeMonitor

    class _AE:
        sample_rate = 48_000

    m = TimecodeMonitor(_AE())
    m.set_fps(30.0)

    # Interpret decoded as a time-of-day; add +1 second should advance by 1 second.
    # The implementation uses an offset in frames, not milliseconds.
    m.channels["L"].decoded_tc = "00:00:00:00"
    m.channels["L"].input_offset_frames = 30  # 30fps => 30 frames = 1 second
    m.channels["L"].display_tz_enabled = False

    assert m._get_display_timecode(key="L") == "00:00:01:00"


def test_timecode_monitor_stop_start_unregisters_callback_id_zero():
    from src.gui.widgets.timecode_monitor import TimecodeMonitor

    class _MockAE:
        sample_rate = 48_000

        def __init__(self):
            import threading

            self.lock = threading.Lock()
            self.callbacks = {}
            self.next_callback_id = 0
            self.unregistered = []

        def register_callback(self, cb):
            with self.lock:
                cid = self.next_callback_id
                self.next_callback_id += 1
                self.callbacks[cid] = cb
                return cid

        def unregister_callback(self, callback_id):
            with self.lock:
                self.callbacks.pop(callback_id, None)
                self.unregistered.append(callback_id)

    ae = _MockAE()
    m = TimecodeMonitor(ae)

    m.start_analysis()
    assert m.callback_id == 0
    assert 0 in ae.callbacks

    m.stop_analysis()
    assert m.callback_id is None
    assert ae.callbacks == {}
    assert 0 in ae.unregistered

    m.start_analysis()
    assert m.callback_id == 1
    assert 1 in ae.callbacks
