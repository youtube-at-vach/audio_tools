from __future__ import annotations

import numpy as np
import time
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
                             QComboBox, QPushButton, QFrame, QGridLayout, QCheckBox, 
                             QGroupBox, QTabWidget)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QColor

from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr

# Constants
SYNC_WORD = 0xBFFC # 1011 1111 1111 1100 (Reverse of 0011 1111 1111 1101 ?)
# SMPTE 12M sync word is 0011 1111 1111 1101 (forward) -> 0x3FFD
# When read simply from bit stream it might appear differently depending on shift direction.
# We will check patterns dynamically.

@dataclass
class _LTCGenState:
    encoder: LTCEncoder
    gen_buffer: deque = field(default_factory=deque)
    gen_current: Optional[np.ndarray] = None
    gen_pos: int = 0
    frames_generated: int = 0
    tod_epoch_base: Optional[float] = None
    free_run_start_time: float = 0.0
    jam_base_total_frames: Optional[int] = None
    jam_base_fps: Optional[float] = None

@dataclass
class _JamMemory:
    valid: bool = False
    tc_raw: str = "--:--:--:--"
    captured_at: float = 0.0
    fps: float = 30.0
    total_frames: int = 0

@dataclass
class _TimecodeChannelState:
    key: str
    input_channel: int
    output_channel: int
    decoder: LTCDecoder
    fps: float = 30.0
    fps_drop_frame: bool = False
    decoded_tc: str = "--:--:--:--"
    locked: bool = False
    input_level_db: float = -100.0
    input_offset_ms: float = 0.0
    display_tz_enabled: bool = False
    display_tz_name: str = "System"
    generator_enabled: bool = False
    generator_mode: str = "tod"
    generator_jam_slot: int = 0
    gen_offset_ms: float = 0.0
    gen: _LTCGenState = None
    estimated_fps: float = 0.0
    last_frame_time: Optional[float] = None
    fps_intervals: deque = field(default_factory=deque)

class LTCEncoder:
    """Generates LTC audio samples."""
    def __init__(self, sample_rate: int, fps: float):
        self.sample_rate = sample_rate
        self.fps = fps
        self.samples_per_frame = sample_rate / fps
        self.current_frame_samples = 0
        self.phase = 1.0 # -1.0 or 1.0
        
        # State
        self.total_frames = 0
        
    def set_fps(self, fps: float):
        self.fps = fps
        self.samples_per_frame = self.sample_rate / fps
        
    def generate_frame(self, hh: int, mm: int, ss: int, ff: int, user_bits: list = None) -> np.ndarray:
        """Generates audio samples for one LTC frame."""
        bits = [0] * 80
        
        # Helper to set bits
        def set_b(idx, val):
            if 0 <= idx < 80:
                bits[idx] = 1 if val else 0
                
        # Timecode Data (BCD)
        # IMPORTANT: these are BCD digits, not binary values.

        # Frame
        ff_u = int(ff) % 10
        ff_t = int(ff) // 10
        set_b(0, ff_u & 1)
        set_b(1, ff_u & 2)
        set_b(2, ff_u & 4)
        set_b(3, ff_u & 8)
        set_b(8, ff_t & 1)
        set_b(9, ff_t & 2)
        
        # Seconds
        ss_u = int(ss) % 10
        ss_t = int(ss) // 10
        set_b(16, ss_u & 1)
        set_b(17, ss_u & 2)
        set_b(18, ss_u & 4)
        set_b(19, ss_u & 8)
        set_b(24, ss_t & 1)
        set_b(25, ss_t & 2)
        set_b(26, ss_t & 4)
        
        # Minutes
        mm_u = int(mm) % 10
        mm_t = int(mm) // 10
        set_b(32, mm_u & 1)
        set_b(33, mm_u & 2)
        set_b(34, mm_u & 4)
        set_b(35, mm_u & 8)
        set_b(40, mm_t & 1)
        set_b(41, mm_t & 2)
        set_b(42, mm_t & 4)
        
        # Hours
        hh_u = int(hh) % 10
        hh_t = int(hh) // 10
        set_b(48, hh_u & 1)
        set_b(49, hh_u & 2)
        set_b(50, hh_u & 4)
        set_b(51, hh_u & 8)
        set_b(56, hh_t & 1)
        set_b(57, hh_t & 2)
        
        # Sync Word (Bits 64-79): 0011 1111 1111 1101
        sync_pattern = [0,0,1,1, 1,1,1,1, 1,1,1,1, 1,1,0,1]
        for i, b in enumerate(sync_pattern):
            bits[64 + i] = b
            
        # Bi-phase Mark Encoding
        # Transition at start of every bit window.
        # If '1', transition also in middle.
        
        # Calculate samples per bit (80 bits total)
        # Note: Samples per bit is not integer usually. We need sub-sample precision or just accumulate phase.
        
        samples = np.zeros(int(self.samples_per_frame + 1.0)) # Over allocate slightly
        idx = 0
        
        samples_per_bit = self.samples_per_frame / 80.0
        
        # We generate continuous samples. 
        # Ideally, we should treat time as continuous to avoid jitter accumulation over frames.
        # But for snippet generation, let's keep it simple.
        
        t = 0.0 # Time in bits
        out_idx = 0
        
        current_level = self.phase
        
        # For each bit
        buffer = []
        
        for bit_val in bits:
            # Duration of this bit is 1.0 bit-time
            # Start of bit -> transition
            current_level = -current_level
            
            # Determine transition points within this bit
            # '0': just the start transition (already did), hold for 1.0
            # '1': start transition, hold for 0.5, transition, hold for 0.5
            
            start_sample = int(out_idx)
            # How many samples for this bit?
            # We map bit_index to sample_index
            end_sample_f = (t + 1.0) * samples_per_bit
            end_sample = int(end_sample_f)
            
            mid_sample_f = (t + 0.5) * samples_per_bit
            mid_sample = int(mid_sample_f)
            
            if bit_val == 0:
                # Fill until end
                count = end_sample - start_sample
                if count > 0:
                    buffer.extend([current_level] * count)
                    out_idx += count
            else:
                # 1 -> Transition at mid
                count1 = mid_sample - start_sample
                if count1 > 0:
                    buffer.extend([current_level] * count1)
                    out_idx += count1
                    
                current_level = -current_level # Mid transition
                
                count2 = end_sample - mid_sample
                if count2 > 0:
                    buffer.extend([current_level] * count2)
                    out_idx += count2
            
            t += 1.0
            
        # Update phase for next frame
        self.phase = current_level
        
        return np.array(buffer, dtype=np.float32)

class LTCDecoder:
    """Decodes audio samples to Timecode."""
    def __init__(self, sample_rate: float, fps: float):
        self.sample_rate = sample_rate
        self.fps = fps
        self.samples_since_last_zc = 0
        self._last_sign = None
        self.bit_stream = 0
        self.bits_count = 0
        self.current_bits = []
        self.last_bit_is_one = False 
        
        # Pulse Width discrimination
        # Initial guess for half-bit (Short pulse)
        # 80 bits per frame.
        self.pulse_avg = (sample_rate / fps) / 160.0 
        
        self.decoded_bits = []
        self.decoded_tc = "--:--:--:--"
        self.locked = False
        
    def process_samples(self, samples: np.ndarray):
        """Process a chunk of audio samples. Returns True if a new frame was decoded."""
        # Vectorized zero-crossing (sign change) detection.
        # IMPORTANT: we must not miss a transition that occurs exactly at the
        # buffer boundary. We therefore track the sign of the last sample from
        # the previous call and synthesize a crossing at position 0 if needed.
        if samples is None:
            return False

        samples = np.asarray(samples)
        frames_in_chunk = int(samples.shape[0])
        if frames_in_chunk <= 0:
            return False

        decoded_any = False

        signs = np.signbit(samples)
        if self._last_sign is None:
            self._last_sign = bool(signs[0])

        crossing_positions = []

        # Boundary crossing between last sample of previous chunk and first sample of this chunk.
        if bool(signs[0]) != bool(self._last_sign):
            crossing_positions.append(0)

        # Intra-chunk crossings: position i means crossing between i-1 and i.
        intra = np.nonzero(signs[1:] != signs[:-1])[0]
        if intra.size:
            crossing_positions.extend((intra + 1).tolist())

        if crossing_positions:
            crossing_positions.sort()

            # Convert crossing positions into pulse widths.
            prev_pos = None
            for pos in crossing_positions:
                if prev_pos is None:
                    d = pos + self.samples_since_last_zc
                else:
                    d = pos - prev_pos

                if d > 0 and self._process_pulse(float(d)):
                    decoded_any = True

                prev_pos = pos

            # Residual samples since the last crossing within this chunk.
            last_pos = crossing_positions[-1]
            self.samples_since_last_zc = frames_in_chunk - last_pos
        else:
            self.samples_since_last_zc += frames_in_chunk

        self._last_sign = bool(signs[-1])
        return decoded_any

    def _process_pulse(self, d: float) -> bool:
        """Returns True if a frame completion was triggered."""
        # Adaptive discriminator
        # Long pulse ~ 2 * Short pulse
        
        # Initial guess or update
        if self.pulse_avg == 0: self.pulse_avg = d
        
        # Use simple IIR for average tracking
        # We assume we track the Short pulse duration
        
        threshold = self.pulse_avg * 1.5
        
        frame_decoded = False
        
        if d > threshold:
            # Long Pulse -> '0'
            self._push_bit(0)
            if self._check_sync():
                frame_decoded = True
                
            self.last_bit_is_one = False 
            # Update average towards Long/2 -> Short
            self.pulse_avg = 0.95 * self.pulse_avg + 0.05 * (d / 2.0) 
        else:
            # Short Pulse
            if self.last_bit_is_one:
                # Second short -> '1'
                self._push_bit(1)
                if self._check_sync():
                    frame_decoded = True
                self.last_bit_is_one = False
            else:
                s = self.last_bit_is_one
                self.last_bit_is_one = True
                
            self.pulse_avg = 0.95 * self.pulse_avg + 0.05 * d
            
        return frame_decoded

    def _push_bit(self, bit: int):
        self.decoded_bits.append(bit)
        if len(self.decoded_bits) > 160: 
             self.decoded_bits.pop(0)

    def _check_sync(self) -> bool:
        if len(self.decoded_bits) >= 16:
            # Check last 16 bits for Sync Word 0x3FFD (0011 1111 1111 1101)
            # bits are pushed 0 or 1.
            # We need to construct integer.
            
            # Optimization: could allow reverse play, but for now forward only
            
            last16 = self.decoded_bits[-16:]
            val = 0
            for b in last16:
                val = (val << 1) | b
            
            if val == 0x3FFD:
                if len(self.decoded_bits) >= 80:
                    frame_bits = self.decoded_bits[-80:]
                    self._decode_frame_bits(frame_bits)
                    return True
        return False

    def _decode_frame_bits(self, bits):
        def val(start, vid_len):
            v = 0
            for i in range(vid_len):
                if bits[start + i]:
                    v |= (1 << i)
            return v
            
        ff_ones = val(0, 4)
        ff_tens = val(8, 2)
        ff = ff_tens * 10 + ff_ones
        
        ss_ones = val(16, 4)
        ss_tens = val(24, 3)
        ss = ss_tens * 10 + ss_ones
        
        mm_ones = val(32, 4)
        mm_tens = val(40, 3)
        mm = mm_tens * 10 + mm_ones
        
        hh_ones = val(48, 4)
        hh_tens = val(56, 2)
        hh = hh_tens * 10 + hh_ones
        
        self.decoded_tc = f"{hh:02}:{mm:02}:{ss:02}:{ff:02}"
        self.locked = True

class TimecodeMonitor(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.is_running = False
        self.callback_id = None

        # Settings (legacy convenience, maps to Left channel)
        self.fps = 30.0
        
        # Audio Gate
        self.gate_threshold_db = -50.0

        self.detected_fps = 0.0

        # Output linking
        self.link_enabled = False
        self.link_source = "L"  # 'L' or 'R'

        sr = int(getattr(self.audio_engine, "sample_rate", 48000))

        dec_l = LTCDecoder(sr, self.fps)
        dec_r = LTCDecoder(sr, self.fps)
        enc_l = LTCEncoder(sr, self.fps)
        enc_r = LTCEncoder(sr, self.fps)

        self.channels: Dict[str, _TimecodeChannelState] = {
            "L": _TimecodeChannelState(
                key="L",
                input_channel=0,
                output_channel=0,
                decoder=dec_l,
                fps=self.fps,
                display_tz_enabled=False,
                display_tz_name="System",
                gen=_LTCGenState(encoder=enc_l),
            ),
            "R": _TimecodeChannelState(
                key="R",
                input_channel=1,
                output_channel=1,
                decoder=dec_r,
                fps=self.fps,
                display_tz_enabled=False,
                display_tz_name="System",
                gen=_LTCGenState(encoder=enc_r),
            ),
        }

        self.jam_memories: list[_JamMemory] = [_JamMemory() for _ in range(5)]

    def set_fps(self, fps: float):
        fps = float(fps)
        if fps <= 0:
            return

        # Legacy behavior: update both channels.
        self.fps = fps
        self.detected_fps = 0.0

        sr = int(getattr(self.audio_engine, "sample_rate", 48000))
        for ch in self.channels.values():
            ch.fps = fps
            ch.decoder = LTCDecoder(sr, fps)
            ch.decoded_tc = "--:--:--:--"
            ch.locked = False
            ch.gen.encoder.sample_rate = sr
            ch.gen.encoder.set_fps(fps)
            ch.gen.gen_buffer.clear()
            ch.gen.gen_current = None
            ch.gen.gen_pos = 0
            ch.gen.frames_generated = 0
            ch.gen.tod_epoch_base = None
            ch.gen.free_run_start_time = 0.0

    def set_channel_fps(self, key: str, fps: float):
        fps = float(fps)
        if fps <= 0:
            return

        sr = int(getattr(self.audio_engine, "sample_rate", 48000))
        ch = self.channels[key]
        ch.fps = fps
        ch.decoder = LTCDecoder(sr, fps)
        ch.decoded_tc = "--:--:--:--"
        ch.locked = False
        ch.gen.encoder.sample_rate = sr
        ch.gen.encoder.set_fps(fps)
        ch.gen.gen_buffer.clear()
        ch.gen.gen_current = None
        ch.gen.gen_pos = 0
        ch.gen.frames_generated = 0
        ch.gen.tod_epoch_base = None
        ch.gen.free_run_start_time = 0.0

        if key == "L":
            self.fps = fps

    @property
    def name(self) -> str:
        return "Timecode Monitor & Generator"
        
    @property
    def description(self) -> str:
        return "LTC (Linear Timecode) Reader and Generator."

    def get_widget(self):
        return TimecodeMonitorWidget(self)

    def run(self, args):
        print("Timecode Monitor running in CLI mode (not fully implemented)")
        self.start_analysis()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop_analysis()

    def start_analysis(self):
        if self.is_running:
            return
            
        self.is_running = True

        sr = int(getattr(self.audio_engine, "sample_rate", 48000))
        for ch in self.channels.values():
            ch.gen.encoder.sample_rate = sr
            ch.gen.encoder.set_fps(ch.fps)
            ch.gen.tod_epoch_base = None

            ch.decoder = LTCDecoder(sr, ch.fps)
            ch.decoded_tc = "--:--:--:--"
            ch.locked = False

            ch.gen.frames_generated = 0
            ch.gen.gen_buffer.clear()
            ch.gen.gen_current = None
            ch.gen.gen_pos = 0
            ch.gen.free_run_start_time = 0.0
        
        def callback(indata, outdata, frames, time_info, status):
            # Always start from silence for this module's output.
            outdata.fill(0)

            for ch in self.channels.values():
                if indata is not None and getattr(indata, "shape", None) is not None:
                    if indata.shape[1] > ch.input_channel:
                        in_sig = indata[:, ch.input_channel]

                        rms = np.sqrt(np.mean(in_sig**2))
                        ch.input_level_db = 20 * np.log10(rms + 1e-9)

                        if ch.input_level_db > self.gate_threshold_db:
                            process_sig = in_sig
                        else:
                            process_sig = np.zeros_like(in_sig)

                        if ch.decoder.process_samples(process_sig):
                            ch.decoded_tc = ch.decoder.decoded_tc
                            ch.locked = bool(ch.decoder.locked)

                            now = time.time()
                            if ch.last_frame_time is not None:
                                dt = float(now - ch.last_frame_time)
                                if 0.015 <= dt <= 0.08:
                                    ch.fps_intervals.append(dt)
                                    if len(ch.fps_intervals) > 32:
                                        ch.fps_intervals.popleft()
                                    avg = float(sum(ch.fps_intervals)) / float(len(ch.fps_intervals))
                                    if avg > 0:
                                        ch.estimated_fps = 1.0 / avg
                            ch.last_frame_time = now

            if self.link_enabled:
                src_key = self.link_source if self.link_source in self.channels else "L"
                src = self.channels[src_key]
                if src.generator_enabled:
                    gen = self._get_generator_samples(src, frames)
                    if gen is not None and len(gen) == frames:
                        gen = gen * 0.5
                        out_ch = outdata.shape[1]
                        if out_ch > 0:
                            for dst_key in ("L", "R"):
                                dst = self.channels[dst_key]
                                out_idx = int(dst.output_channel)
                                if 0 <= out_idx < out_ch:
                                    outdata[:, out_idx] = gen
            else:
                for ch in self.channels.values():
                    if ch.generator_enabled:
                        gen = self._get_generator_samples(ch, frames)
                        if gen is not None and len(gen) == frames:
                            gen = gen * 0.5

                            out_ch = outdata.shape[1]
                            if out_ch <= 0:
                                continue

                            out_idx = int(ch.output_channel)
                            if out_idx < 0 or out_idx >= out_ch:
                                continue

                            outdata[:, out_idx] = gen
        self.callback_id = self.audio_engine.register_callback(callback)

    def _get_generator_samples(self, ch: _TimecodeChannelState, frames: int) -> np.ndarray:
        """Return exactly `frames` samples of generated LTC (mono)."""
        if frames <= 0:
            return np.zeros((0,), dtype=np.float32)

        out = np.zeros((frames,), dtype=np.float32)
        out_pos = 0

        while out_pos < frames:
            if ch.gen.gen_current is None or ch.gen.gen_pos >= len(ch.gen.gen_current):
                if ch.gen.gen_buffer:
                    ch.gen.gen_current = ch.gen.gen_buffer.popleft()
                    ch.gen.gen_pos = 0
                else:
                    self._generate_next_frame(ch)
                    continue

            remaining_out = frames - out_pos
            remaining_in = len(ch.gen.gen_current) - ch.gen.gen_pos
            to_copy = remaining_out if remaining_out < remaining_in else remaining_in

            if to_copy > 0:
                out[out_pos:out_pos + to_copy] = ch.gen.gen_current[ch.gen.gen_pos:ch.gen.gen_pos + to_copy]
                out_pos += to_copy
                ch.gen.gen_pos += to_copy

        return out

    def _generate_next_frame(self, ch: _TimecodeChannelState):
        if ch.generator_mode == 'jam':
            fps = float(ch.fps) if ch.fps else 30.0
            if fps <= 0:
                fps = 30.0

            slot = int(ch.generator_jam_slot) if ch.generator_jam_slot is not None else 0
            if slot < 0:
                slot = 0
            elif slot > 4:
                slot = 4

            base = ch.gen.jam_base_total_frames
            base_fps = ch.gen.jam_base_fps
            if base is None or base_fps is None or abs(float(base_fps) - float(fps)) > 1e-6:
                mem = self.jam_memories[slot] if 0 <= slot < len(self.jam_memories) else None
                if mem is None or (not mem.valid):
                    base = 0
                else:
                    mem_fps = float(mem.fps) if mem.fps else 30.0
                    mem_nominal_fps = int(round(mem_fps))
                    if mem_nominal_fps <= 0:
                        mem_nominal_fps = 30

                    gen_nominal_fps = int(round(fps))
                    if gen_nominal_fps <= 0:
                        gen_nominal_fps = 30

                    base_seconds = float(mem.total_frames) / float(mem_nominal_fps)
                    elapsed_seconds = time.time() - float(mem.captured_at)
                    current_seconds = base_seconds + float(elapsed_seconds)
                    current_seconds = current_seconds % 86400.0

                    base = int(round(current_seconds * float(gen_nominal_fps)))
                ch.gen.jam_base_total_frames = int(base)
                ch.gen.jam_base_fps = float(fps)

            offset_frames = int(round((float(ch.gen_offset_ms) / 1000.0) * float(fps))) if ch.gen_offset_ms else 0
            total_frames = int(ch.gen.jam_base_total_frames) + int(ch.gen.frames_generated) + int(offset_frames)

            nominal_fps = int(round(fps))
            if nominal_fps <= 0:
                nominal_fps = 30

            frames_per_day = 24 * 3600 * nominal_fps
            if frames_per_day > 0:
                total_frames = total_frames % frames_per_day

            hh = int(total_frames // (3600 * nominal_fps))
            rem = int(total_frames % (3600 * nominal_fps))
            mm = int(rem // (60 * nominal_fps))
            rem = int(rem % (60 * nominal_fps))
            ss = int(rem // nominal_fps)
            ff = int(rem % nominal_fps)

            ch.gen.frames_generated += 1

        elif ch.generator_mode == 'free':
            # Relative to start
            if ch.gen.free_run_start_time == 0:
                ch.gen.free_run_start_time = time.time()
            
            # Simple frame counter
            total_frames = ch.gen.frames_generated
            # Or based on time?
            # Let's just increment frame by frame to ensure continuity.
            # But we need to initialize 'total_frames' based on 'free_run_start_time'?
            # For free run, we usually just start at 00:00:00:00 or user set value.
            # Let's implement continuous increment.
            
            # Calculate TC from total frames
            # fps
            fps = float(ch.fps) if ch.fps else 30.0
            hh = int(total_frames / (fps * 3600)) % 24
            rem = total_frames % (int(fps * 3600))
            mm = int(rem / (fps * 60))
            rem = rem % (int(fps * 60))
            ss = int(rem / fps)
            ff = int(rem % fps)
            
            ch.gen.frames_generated += 1

        else:
            # Time of Day (system local time).
            # Use a stable epoch base + frame counter so we don't jitter/jump due
            # to callback scheduling or buffer prefill.
            if ch.gen.tod_epoch_base is None:
                ch.gen.tod_epoch_base = time.time()

            fps = float(ch.fps) if ch.fps else 30.0
            if fps <= 0:
                fps = 30.0

            t_target = ch.gen.tod_epoch_base + (ch.gen.frames_generated / fps) + (ch.gen_offset_ms / 1000.0)
            # Use UTC for LTC generation always. Display converts to Local if needed.
            dt = datetime.fromtimestamp(t_target, timezone.utc)

            hh = dt.hour
            mm = dt.minute
            ss = dt.second

            frac = t_target - math.floor(t_target)
            ff = int(frac * fps)
            nominal_fps = int(round(fps))
            if nominal_fps <= 0:
                nominal_fps = 30
            if ff < 0:
                ff = 0
            elif ff >= nominal_fps:
                ff = nominal_fps - 1

            ch.gen.frames_generated += 1
            
        samples = ch.gen.encoder.generate_frame(hh, mm, ss, ff)
        ch.gen.gen_buffer.append(samples)

    def stop_analysis(self):
        if self.callback_id:
            self.audio_engine.unregister_callback(self.callback_id)
            self.callback_id = None
        self.is_running = False

    def process(self):
        l = self.channels["L"]
        r = self.channels["R"]
        return {
            "fps": float(l.fps),
            "L": {
                "fps": float(l.fps),
                "fps_est": float(l.estimated_fps),
                "tc": self._get_display_timecode(l.decoded_tc, l.input_offset_ms, key="L"),
                "tc_raw": l.decoded_tc,
                "locked": bool(l.locked),
                "level": float(l.input_level_db),
            },
            "R": {
                "fps": float(r.fps),
                "fps_est": float(r.estimated_fps),
                "tc": self._get_display_timecode(r.decoded_tc, r.input_offset_ms, key="R"),
                "tc_raw": r.decoded_tc,
                "locked": bool(r.locked),
                "level": float(r.input_level_db),
            },
        }

    def jam_capture(self, key: str, slot: int) -> bool:
        if key not in self.channels:
            return False

        s = int(slot)
        if s < 0:
            s = 0
        elif s > 4:
            s = 4

        ch = self.channels[key]
        parsed = self._parse_tc(ch.decoded_tc)
        if parsed is None:
            return False

        fps = float(ch.fps) if ch.fps else 30.0
        nominal_fps = int(round(fps))
        if nominal_fps <= 0:
            nominal_fps = 30

        hh, mm, ss, ff = parsed
        total_frames = ((hh * 3600 + mm * 60 + ss) * nominal_fps) + int(ff)

        mem = self.jam_memories[s]
        mem.valid = True
        mem.tc_raw = ch.decoded_tc
        mem.captured_at = time.time()
        mem.fps = float(fps)
        mem.total_frames = int(total_frames)
        self.jam_memories[s] = mem
        return True

    def jam_capture_auto(self, key: str) -> int:
        if key not in self.channels:
            return -1

        free_idx = None
        oldest_idx = 0
        oldest_ts = float("inf")
        for i, m in enumerate(self.jam_memories):
            if not m.valid and free_idx is None:
                free_idx = i
            if m.valid and float(m.captured_at) < oldest_ts:
                oldest_ts = float(m.captured_at)
                oldest_idx = i

        idx = int(free_idx) if free_idx is not None else int(oldest_idx)
        ok = self.jam_capture(key, idx)
        return idx if ok else -1

    def jam_get_current_tc(self, slot: int) -> str:
        s = int(slot)
        if s < 0:
            s = 0
        elif s > 4:
            s = 4

        mem = self.jam_memories[s]
        if not mem.valid:
            return "--:--:--:--"

        fps = float(mem.fps) if mem.fps else 30.0
        nominal_fps = int(round(fps))
        if nominal_fps <= 0:
            nominal_fps = 30

        now = time.time()
        elapsed_frames = int(round((now - float(mem.captured_at)) * float(fps)))
        total_frames = int(mem.total_frames) + int(elapsed_frames)
        frames_per_day = 24 * 3600 * nominal_fps
        if frames_per_day > 0:
            total_frames = total_frames % frames_per_day

        hh = int(total_frames // (3600 * nominal_fps))
        rem = int(total_frames % (3600 * nominal_fps))
        mm = int(rem // (60 * nominal_fps))
        rem = int(rem % (60 * nominal_fps))
        ss = int(rem // nominal_fps)
        ff = int(rem % nominal_fps)
        return f"{hh:02}:{mm:02}:{ss:02}:{ff:02}"

    def _parse_tc(self, tc: str):
        """Parse 'HH:MM:SS:FF' into ints. Returns (hh, mm, ss, ff) or None."""
        if not tc or tc.count(":") != 3:
            return None
        try:
            hh_s, mm_s, ss_s, ff_s = tc.split(":")
            hh = int(hh_s)
            mm = int(mm_s)
            ss = int(ss_s)
            ff = int(ff_s)
        except Exception:
            return None

        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 60 and 0 <= ff <= 99):
            return None
        return hh, mm, ss, ff

    def _get_display_timecode(self, tc: Optional[str] = None, input_offset_ms: Optional[float] = None, key: str = "L") -> str:
        """Return display timecode string.

        Applies optional input delay compensation and optional timezone display conversion.
        (LTC does not carry a date; we use today's UTC date for TZ conversion.)
        """
        ch = self.channels.get(key, self.channels["L"])

        if tc is None:
            tc = ch.decoded_tc

        if input_offset_ms is None:
            try:
                input_offset_ms = float(ch.input_offset_ms)
            except Exception:
                input_offset_ms = 0.0

        if (not ch.display_tz_enabled) and (abs(input_offset_ms) < 1e-9):
            return tc

        parsed = self._parse_tc(tc)
        if parsed is None:
            return tc

        hh, mm, ss, ff = parsed
        fps = float(ch.fps) if ch.fps else 30.0
        if fps <= 0:
            fps = 30.0

        try:
            nominal_fps = int(round(fps))
            if nominal_fps <= 0:
                nominal_fps = 30

            # Seconds-of-day from decoded LTC.
            total_seconds = (hh * 3600.0) + (mm * 60.0) + float(min(ss, 59)) + (float(ff) / fps)
            total_seconds += (input_offset_ms / 1000.0)

            if not ch.display_tz_enabled:
                # Pure offset + wrap within 24h.
                total_seconds = total_seconds % 86400.0
                disp_h = int(total_seconds // 3600)
                total_seconds -= disp_h * 3600
                disp_m = int(total_seconds // 60)
                total_seconds -= disp_m * 60
                disp_s = int(total_seconds)
                frac = total_seconds - disp_s
                disp_f = int(frac * fps)
                if disp_f < 0:
                    disp_f = 0
                elif disp_f >= nominal_fps:
                    disp_f = nominal_fps - 1
                return f"{disp_h:02}:{disp_m:02}:{disp_s:02}:{disp_f:02}"

            # TZ display enabled: interpret decoded time-of-day as UTC.
            utc_today = datetime.now(timezone.utc).date()
            base_utc = datetime(
                utc_today.year,
                utc_today.month,
                utc_today.day,
                0,
                0,
                0,
                0,
                tzinfo=timezone.utc,
            )
            dt_utc = base_utc + timedelta(seconds=total_seconds)

            tz_name = (ch.display_tz_name or "System").strip()
            if tz_name.lower() == "utc":
                tz = timezone.utc
            elif tz_name.lower() == "system":
                tz = datetime.now().astimezone().tzinfo
            else:
                if ZoneInfo is None:
                    tz = datetime.now().astimezone().tzinfo
                else:
                    tz = ZoneInfo(tz_name)

            dt_disp = dt_utc.astimezone(tz)
            frac = dt_disp.microsecond / 1_000_000.0
            ff_disp = int(frac * fps)
            if ff_disp < 0:
                ff_disp = 0
            elif ff_disp >= nominal_fps:
                ff_disp = nominal_fps - 1

            return f"{dt_disp.hour:02}:{dt_disp.minute:02}:{dt_disp.second:02}:{ff_disp:02}"
        except Exception:
            return tc

    @property
    def decoded_tc(self) -> str:
        return self.channels["L"].decoded_tc

    @decoded_tc.setter
    def decoded_tc(self, v: str):
        self.channels["L"].decoded_tc = v

    @property
    def input_offset_ms(self) -> float:
        return float(self.channels["L"].input_offset_ms)

    @input_offset_ms.setter
    def input_offset_ms(self, v: float):
        self.channels["L"].input_offset_ms = float(v)

    @property
    def gen_offset_ms(self) -> float:
        return float(self.channels["L"].gen_offset_ms)

    @gen_offset_ms.setter
    def gen_offset_ms(self, v: float):
        self.channels["L"].gen_offset_ms = float(v)

    @property
    def generator_enabled(self) -> bool:
        return bool(self.channels["L"].generator_enabled)

    @generator_enabled.setter
    def generator_enabled(self, v: bool):
        self.channels["L"].generator_enabled = bool(v)

    @property
    def generator_mode(self) -> str:
        return str(self.channels["L"].generator_mode)

    @generator_mode.setter
    def generator_mode(self, v: str):
        self.channels["L"].generator_mode = str(v)

    @property
    def display_tz_enabled(self) -> bool:
        return bool(self.channels["L"].display_tz_enabled)

    @display_tz_enabled.setter
    def display_tz_enabled(self, v: bool):
        self.channels["L"].display_tz_enabled = bool(v)

    @property
    def display_tz_name(self) -> str:
        return str(self.channels["L"].display_tz_name)

    @display_tz_name.setter
    def display_tz_name(self, v: str):
        self.channels["L"].display_tz_name = str(v)

class TimecodeMonitorWidget(QWidget):
    def __init__(self, module: TimecodeMonitor):
        super().__init__()
        self.module = module
        self._gen_buttons: Dict[str, QPushButton] = {}
        self._tz_combos: Dict[str, QComboBox] = {}
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50) # 20Hz UI update
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        display_row = QHBoxLayout()

        self.tc_label_L = QLabel("--:--:--:--")
        self.tc_label_R = QLabel("--:--:--:--")
        self.sync_led_L = QLabel(tr("SYNC"))
        self.sync_led_R = QLabel(tr("SYNC"))
        self.fps_est_label_L = QLabel(tr("FPS: --"))
        self.fps_est_label_R = QLabel(tr("FPS: --"))
        self.level_label_L = QLabel("-- dB")
        self.level_label_R = QLabel("-- dB")

        def build_display_frame(title: str, key: str, tc_label: QLabel, sync_led: QLabel, fps_label: QLabel, level_label: QLabel):
            frame = QFrame()
            frame.setStyleSheet("background-color: #111; border: 2px solid #555; border-radius: 8px;")
            v = QVBoxLayout(frame)
            v.setContentsMargins(8, 6, 8, 6)
            v.setSpacing(4)

            header = QHBoxLayout()
            hdr = QLabel(title)
            hdr.setStyleSheet("color: #888; font-weight: bold;")
            header.addWidget(hdr)
            header.addStretch()
            v.addLayout(header)

            tc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tc_label.setFont(QFont("Monospace", 44, QFont.Weight.Bold))
            tc_label.setStyleSheet("color: #ff3333;")
            v.addWidget(tc_label)

            jam_row = QHBoxLayout()
            jam_btn = QPushButton(tr("JAM"))
            jam_btn.setMinimumHeight(26)
            jam_btn.clicked.connect(lambda _=False, k=key: self._on_jam_capture_auto(k))
            jam_row.addWidget(jam_btn)
            jam_msg = QLabel("")
            jam_msg.setStyleSheet("color: #888;")
            jam_row.addWidget(jam_msg)
            jam_row.addStretch()
            v.addLayout(jam_row)

            self._jam_capture_msg[key] = jam_msg

            status = QHBoxLayout()
            sync_led.setStyleSheet("color: #333; font-weight: bold; border: 1px solid #333; padding: 2px 5px; border-radius:4px;")
            status.addWidget(sync_led)

            fps_label.setStyleSheet("color: #888;")
            status.addWidget(fps_label)
            status.addStretch()
            level_label.setStyleSheet("color: #888;")
            status.addWidget(level_label)
            v.addLayout(status)
            return frame

        self._jam_capture_msg = {}
        display_row.addWidget(build_display_frame(tr("Left"), "L", self.tc_label_L, self.sync_led_L, self.fps_est_label_L, self.level_label_L))
        display_row.addWidget(build_display_frame(tr("Right"), "R", self.tc_label_R, self.sync_led_R, self.fps_est_label_R, self.level_label_R))
        layout.addLayout(display_row)
        
        controls_group = QGroupBox(tr("Output"))
        c_layout = QGridLayout()

        self.link_check = QCheckBox(tr("Link Stereo Output"))
        self.link_check.setChecked(bool(self.module.link_enabled))
        self.link_check.toggled.connect(self.on_link_toggled)
        c_layout.addWidget(self.link_check, 0, 0, 1, 2)

        c_layout.addWidget(QLabel(tr("Link Source:")), 1, 0)
        self.link_src_combo = QComboBox()
        self.link_src_combo.addItem(tr("Left"), "L")
        self.link_src_combo.addItem(tr("Right"), "R")
        self.link_src_combo.setCurrentIndex(0 if self.module.link_source == "L" else 1)
        self.link_src_combo.currentIndexChanged.connect(self.on_link_source_changed)
        c_layout.addWidget(self.link_src_combo, 1, 1)
        self.link_src_combo.setEnabled(bool(self.module.link_enabled))
        
        controls_group.setLayout(c_layout)
        layout.addWidget(controls_group)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_channel_tab("L"), tr("Left"))
        self.tabs.addTab(self._build_channel_tab("R"), tr("Right"))
        self.tabs.addTab(self._build_jam_tab(), tr("JAM"))
        layout.addWidget(self.tabs)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Start module
        QTimer.singleShot(100, self.module.start_analysis)
        
    def on_link_toggled(self, checked: bool):
        self.module.link_enabled = bool(checked)
        self.link_src_combo.setEnabled(bool(checked))
        if checked:
            src = self.module.link_source if self.module.link_source in ("L", "R") else "L"
            other = "R" if src == "L" else "L"
            if self.module.channels[other].generator_enabled:
                self.module.channels[other].generator_enabled = False
            if self._gen_buttons.get(other) is not None:
                self._gen_buttons[other].setChecked(False)
                self._gen_buttons[other].setText(tr("Enable Generator"))

    def on_link_source_changed(self):
        self.module.link_source = self.link_src_combo.currentData() or "L"
        if self.module.link_enabled:
            for key in ("L", "R"):
                if self.module.channels[key].generator_enabled:
                    self.module.channels[key].generator_enabled = False
                if self._gen_buttons.get(key) is not None:
                    self._gen_buttons[key].setChecked(False)
                    self._gen_buttons[key].setText(tr("Enable Generator"))
        
    def update_ui(self):
        data = self.module.process()
        l = data.get("L", {})
        r = data.get("R", {})

        self.tc_label_L.setText(l.get("tc", "--:--:--:--"))
        self.tc_label_R.setText(r.get("tc", "--:--:--:--"))

        if l.get("locked", False):
            self.sync_led_L.setStyleSheet("color: #0f0; font-weight: bold; border: 1px solid #0f0; background-color: #003300; padding: 2px 5px; border-radius:4px;")
        else:
            self.sync_led_L.setStyleSheet("color: #555; font-weight: normal; border: 1px solid #555; padding: 2px 5px; border-radius:4px;")

        if r.get("locked", False):
            self.sync_led_R.setStyleSheet("color: #0f0; font-weight: bold; border: 1px solid #0f0; background-color: #003300; padding: 2px 5px; border-radius:4px;")
        else:
            self.sync_led_R.setStyleSheet("color: #555; font-weight: normal; border: 1px solid #555; padding: 2px 5px; border-radius:4px;")

        self.level_label_L.setText(tr("{0} dB").format(f"{float(l.get('level', -100.0)):.1f}"))
        self.level_label_R.setText(tr("{0} dB").format(f"{float(r.get('level', -100.0)):.1f}"))

        fpsl = float(l.get("fps_est", 0.0))
        fpsr = float(r.get("fps_est", 0.0))
        self.fps_est_label_L.setText(tr("FPS: {0}").format(self._format_fps_est("L", fpsl)))
        self.fps_est_label_R.setText(tr("FPS: {0}").format(self._format_fps_est("R", fpsr)))

        if getattr(self, "_jam_tab_index", None) is not None and self.tabs.currentIndex() == self._jam_tab_index:
            now = time.time()
            if not hasattr(self, "_jam_last_update"):
                self._jam_last_update = 0.0
            if (now - float(self._jam_last_update)) >= 0.5:
                self._jam_last_update = now
                self.update_jam_ui()

    def _on_jam_capture_auto(self, key: str):
        idx = self.module.jam_capture_auto(key)
        if idx >= 0 and self._jam_capture_msg.get(key) is not None:
            self._jam_capture_msg[key].setText(tr("Saved: Mem {0}").format(str(idx + 1)))
        elif self._jam_capture_msg.get(key) is not None:
            self._jam_capture_msg[key].setText(tr("JAM failed"))

    def _build_channel_tab(self, key: str) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        ch = self.module.channels[key]

        top_row = QHBoxLayout()

        settings = QGroupBox(tr("Channel Settings"))
        sl = QGridLayout()

        sl.addWidget(QLabel(tr("Frame Rate:")), 0, 0)
        fps_combo = QComboBox()
        fps_combo.addItems(["23.98", "24.00", "25.00", "30.0D", "30.0", "29.97D", "29.97"])
        fps_combo.setCurrentText(self._format_fps_option(float(ch.fps), bool(getattr(ch, "fps_drop_frame", False))))
        fps_combo.currentTextChanged.connect(lambda t="", k=key: self._on_fps_changed(k, t))
        sl.addWidget(fps_combo, 0, 1)

        tz_check = QCheckBox(tr("Display Local Time"))
        tz_check.setChecked(bool(ch.display_tz_enabled))
        tz_check.toggled.connect(lambda checked=False, k=key: self._on_tz_toggled(k, checked))
        sl.addWidget(tz_check, 0, 2)

        sl.addWidget(QLabel(tr("Display TZ:")), 1, 2)
        tz_combo = QComboBox()
        tz_combo.setEditable(True)
        tz_combo.addItems([
            "System",
            "UTC",
            "Asia/Tokyo",
            "Europe/London",
            "America/New_York",
        ])
        tz_combo.setCurrentText(ch.display_tz_name or "System")
        tz_combo.currentTextChanged.connect(lambda text="", k=key: self._on_tz_changed(k, text))
        tz_combo.setEnabled(bool(ch.display_tz_enabled))
        sl.addWidget(tz_combo, 1, 3)
        self._tz_combos[key] = tz_combo

        settings.setLayout(sl)
        top_row.addWidget(settings, 2)

        g = QGroupBox(tr("Generator"))
        gl = QGridLayout()

        gl.addWidget(QLabel(tr("Gen Mode:")), 0, 0)
        mode_combo = QComboBox()
        mode_combo.addItem(tr("Time of Day"), "tod")
        mode_combo.addItem(tr("Free Run"), "free")
        mode_combo.addItem(tr("JAM"), "jam")
        if ch.generator_mode == "tod":
            mode_combo.setCurrentIndex(0)
        elif ch.generator_mode == "free":
            mode_combo.setCurrentIndex(1)
        else:
            mode_combo.setCurrentIndex(2)
        mode_combo.currentIndexChanged.connect(lambda _=0, k=key, c=mode_combo: self._on_mode_changed(k, c.currentData()))
        gl.addWidget(mode_combo, 0, 1)

        gl.addWidget(QLabel(tr("JAM Mem:")), 3, 0)
        jam_combo = QComboBox()
        for i in range(5):
            jam_combo.addItem(tr("Mem {0}").format(str(i + 1)), i)
        cur_slot = int(ch.generator_jam_slot) if ch.generator_jam_slot is not None else 0
        if cur_slot < 0:
            cur_slot = 0
        elif cur_slot > 4:
            cur_slot = 4
        jam_combo.setCurrentIndex(cur_slot)
        jam_combo.currentIndexChanged.connect(lambda _=0, k=key, c=jam_combo: self._on_jam_slot_changed(k, int(c.currentData())))
        gl.addWidget(jam_combo, 3, 1)

        gen_btn = QPushButton(tr("Enable Generator"))
        gen_btn.setCheckable(True)
        gen_btn.setChecked(bool(ch.generator_enabled))
        gen_btn.clicked.connect(lambda checked=False, k=key, b=gen_btn: self._on_gen_toggle(k, checked, b))
        gen_btn.setText(tr("Stop Generator") if ch.generator_enabled else tr("Enable Generator"))
        gl.addWidget(gen_btn, 0, 2, 2, 1)
        self._gen_buttons[key] = gen_btn

        gl.addWidget(QLabel(tr("In Delay (ms):")), 1, 0)
        in_spin = QDoubleSpinBox()
        in_spin.setRange(-1000, 1000)
        in_spin.setValue(float(ch.input_offset_ms))
        in_spin.valueChanged.connect(lambda v=0.0, k=key: self._set_in_offset(k, v))
        gl.addWidget(in_spin, 1, 1)

        gl.addWidget(QLabel(tr("Out Delay (ms):")), 2, 0)
        out_spin = QDoubleSpinBox()
        out_spin.setRange(-1000, 1000)
        out_spin.setValue(float(ch.gen_offset_ms))
        out_spin.valueChanged.connect(lambda v=0.0, k=key: self._set_out_offset(k, v))
        gl.addWidget(out_spin, 2, 1)

        g.setLayout(gl)
        top_row.addWidget(g, 3)

        v.addLayout(top_row)
        v.addStretch()
        return w

    def _on_mode_changed(self, key: str, mode: str):
        ch = self.module.channels[key]
        ch.generator_mode = str(mode)
        ch.gen.frames_generated = 0
        ch.gen.gen_buffer.clear()
        ch.gen.gen_current = None
        ch.gen.gen_pos = 0
        ch.gen.free_run_start_time = 0.0
        ch.gen.tod_epoch_base = None
        ch.gen.jam_base_total_frames = None
        ch.gen.jam_base_fps = None

    def _on_jam_slot_changed(self, key: str, slot: int):
        ch = self.module.channels[key]
        ch.generator_jam_slot = int(slot)
        ch.gen.jam_base_total_frames = None
        ch.gen.jam_base_fps = None

    def _on_gen_toggle(self, key: str, checked: bool, btn: QPushButton):
        if self.module.link_enabled:
            # In link mode, only one side should generate; choose the side that was toggled.
            other = "R" if key == "L" else "L"
            self.module.link_source = key
            if self.link_src_combo.currentData() != key:
                self.link_src_combo.setCurrentIndex(0 if key == "L" else 1)

            self.module.channels[other].generator_enabled = False
            if self._gen_buttons.get(other) is not None:
                self._gen_buttons[other].setChecked(False)
                self._gen_buttons[other].setText(tr("Enable Generator"))

        ch = self.module.channels[key]
        ch.generator_enabled = bool(checked)
        if checked:
            ch.gen.frames_generated = 0
            ch.gen.gen_buffer.clear()
            ch.gen.gen_current = None
            ch.gen.gen_pos = 0
            ch.gen.tod_epoch_base = None
            ch.gen.free_run_start_time = 0.0
            ch.gen.jam_base_total_frames = None
            ch.gen.jam_base_fps = None
        btn.setText(tr("Stop Generator") if checked else tr("Enable Generator"))

    def _build_jam_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        grid_box = QGroupBox(tr("JAM Memories"))
        gl = QGridLayout()

        gl.addWidget(QLabel(tr("Slot")), 0, 0)
        gl.addWidget(QLabel(tr("Captured")), 0, 1)
        gl.addWidget(QLabel(tr("Current")), 0, 2)

        self._jam_labels = {}
        row = 1
        for slot in range(5):
            gl.addWidget(QLabel(str(slot + 1)), row, 0)
            cap = QLabel("--:--:--:--")
            cur = QLabel("--:--:--:--")
            cap.setFont(QFont("Monospace", 10))
            cur.setFont(QFont("Monospace", 10))
            gl.addWidget(cap, row, 1)
            gl.addWidget(cur, row, 2)
            self._jam_labels[(slot, "cap")] = cap
            self._jam_labels[(slot, "cur")] = cur
            row += 1

        grid_box.setLayout(gl)
        v.addWidget(grid_box)
        v.addStretch()

        self._jam_tab_index = 2
        self.update_jam_ui()
        return w

    def update_jam_ui(self):
        if not hasattr(self, "_jam_labels"):
            return

        for slot in range(5):
            mem = self.module.jam_memories[slot]
            cap = mem.tc_raw if mem.valid else "--:--:--:--"
            cur = self.module.jam_get_current_tc(slot) if mem.valid else "--:--:--:--"
            if self._jam_labels.get((slot, "cap")) is not None:
                self._jam_labels[(slot, "cap")].setText(cap)
            if self._jam_labels.get((slot, "cur")) is not None:
                self._jam_labels[(slot, "cur")].setText(cur)

    def _on_fps_changed(self, key: str, text: str):
        fps, drop_frame = self._parse_fps_option(text)
        if fps is None or float(fps) <= 0:
            return
        self.module.channels[key].fps_drop_frame = bool(drop_frame)
        self.module.set_channel_fps(key, float(fps))

    def _on_tz_toggled(self, key: str, checked: bool):
        ch = self.module.channels[key]
        ch.display_tz_enabled = bool(checked)
        if self._tz_combos.get(key) is not None:
            self._tz_combos[key].setEnabled(bool(checked))

    def _on_tz_changed(self, key: str, text: str):
        ch = self.module.channels[key]
        ch.display_tz_name = str(text).strip() if text else "System"

    def _parse_fps_option(self, text: str) -> tuple[Optional[float], bool]:
        t = (text or "").strip()
        drop_frame = False
        if t.endswith("D"):
            drop_frame = True
            t = t[:-1]

        mapping = {
            "23.98": 23.976,
            "24.00": 24.0,
            "25.00": 25.0,
            "30.0": 30.0,
            "29.97": 29.97,
        }

        if t in mapping:
            return float(mapping[t]), drop_frame

        try:
            return float(t), drop_frame
        except Exception:
            return None, drop_frame

    def _format_fps_option(self, fps: float, drop_frame: bool = False) -> str:
        presets = [23.976, 24.0, 25.0, 30.0, 29.97]
        labels = ["23.98", "24.00", "25.00", "30.0", "29.97"]
        if fps <= 0:
            base = "30.0"
        else:
            best_i = 0
            best_d = float("inf")
            for i, p in enumerate(presets):
                d = abs(float(fps) - float(p))
                if d < best_d:
                    best_d = d
                    best_i = i
            base = labels[best_i]

        if base in ("29.97", "30.0") and drop_frame:
            return f"{base}D"
        return base

    def _format_fps_est(self, key: str, fps_est: float) -> str:
        if float(fps_est) <= 0:
            return "--"
        ch = self.module.channels.get(key)
        drop_frame = bool(getattr(ch, "fps_drop_frame", False)) if ch is not None else False
        return self._format_fps_option(float(fps_est), drop_frame)

    def _set_in_offset(self, key: str, v: float):
        self.module.channels[key].input_offset_ms = float(v)

    def _set_out_offset(self, key: str, v: float):
        self.module.channels[key].gen_offset_ms = float(v)
