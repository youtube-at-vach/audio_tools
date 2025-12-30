import numpy as np
import time
import math
from collections import deque
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
                             QComboBox, QPushButton, QFrame, QGridLayout, QCheckBox, 
                             QGroupBox)
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
        # Frame
        set_b(0, ff & 1)
        set_b(1, ff & 2)
        set_b(2, ff & 4)
        set_b(3, ff & 8)
        set_b(8, (ff // 10) & 1)
        set_b(9, (ff // 10) & 2)
        
        # Seconds
        set_b(16, ss & 1)
        set_b(17, ss & 2)
        set_b(18, ss & 4)
        set_b(19, ss & 8)
        set_b(24, (ss // 10) & 1)
        set_b(25, (ss // 10) & 2)
        set_b(26, (ss // 10) & 4)
        
        # Minutes
        set_b(32, mm & 1)
        set_b(33, mm & 2)
        set_b(34, mm & 4)
        set_b(35, mm & 8)
        set_b(40, (mm // 10) & 1)
        set_b(41, (mm // 10) & 2)
        set_b(42, (mm // 10) & 4)
        
        # Hours
        set_b(48, hh & 1)
        set_b(49, hh & 2)
        set_b(50, hh & 4)
        set_b(51, hh & 8) # Bit 51 is technically BGF0 but often used for Hours tens bit 2 (10+20=30? no max 23)
                          # SMPTE: Bits 56,57 are Hours tens (wait, looking at spec)
                          # Bits 0-3: Frame unit, 8-9: Frame tens
                          # Bits 16-19: Sec unit, 24-26: Sec tens
                          # Bits 32-35: Min unit, 40-42: Min tens
                          # Bits 48-51: Hour unit, 56-57: Hour tens
        set_b(56, (hh // 10) & 1)
        set_b(57, (hh // 10) & 2)
        
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
        # Vectorized ZC detection
        # Note: We assume samples is mono
        signs = np.signbit(samples)
        diffs = np.diff(signs)
        zero_crossings = np.where(diffs)[0]
        
        frames_in_chunk = len(samples)
        decoded_any = False
        
        if len(zero_crossings) > 0:
            # Handle first ZC relative to last buffer residual
            first_dist = zero_crossings[0] + self.samples_since_last_zc
            dists = np.diff(zero_crossings)
            dists = np.insert(dists, 0, first_dist)
            
            self.samples_since_last_zc = frames_in_chunk - 1 - zero_crossings[-1]
            
            for d in dists:
                if self._process_pulse(d):
                    decoded_any = True
        else:
            self.samples_since_last_zc += frames_in_chunk
            
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
        
        # Settings
        self.fps = 30.0
        self.generator_enabled = False
        self.generator_mode = 'tod' # 'tod', 'free'
        self.free_run_start_time = 0.0 # epoch
        self.gen_offset_ms = 0.0
        self.input_offset_ms = 0.0
        
        # Audio Gate
        self.gate_threshold_db = -50.0
        
        self.selected_input_channel = 0
        self.selected_output_channel = 0
        
        self.input_level_db = -100.0
        self.decoded_tc = "--:--:--:--"
        self.display_tz_enabled = False
        # Display TZ: 'System', 'UTC', or IANA tz database name (e.g. 'Asia/Tokyo')
        self.display_tz_name = "System"
        self.detected_fps = 0.0
        self.locked = False
        
        # Encoder State
        self.encoder = LTCEncoder(48000, 30.0) # Correct SR set on run
        self.gen_buffer = deque()
        self._gen_current = None
        self._gen_pos = 0
        self.frames_generated = 0
        
        # Decoder State
        self.decoder = LTCDecoder(48000, 30.0)

    def set_fps(self, fps: float):
        fps = float(fps)
        if fps <= 0:
            return

        self.fps = fps
        self.encoder.set_fps(fps)
        # Reset decoder/generator state to avoid stale framing.
        self.decoder = LTCDecoder(self.audio_engine.sample_rate, fps)
        self.decoded_tc = "--:--:--:--"
        self.locked = False
        self.detected_fps = 0.0

        self.gen_buffer.clear()
        self._gen_current = None
        self._gen_pos = 0
        self.frames_generated = 0
        self.free_run_start_time = 0.0

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
        self.encoder.sample_rate = self.audio_engine.sample_rate
        self.encoder.set_fps(self.fps)
        
        # Reset Decoder
        self.decoder = LTCDecoder(self.audio_engine.sample_rate, self.fps)
        
        def callback(indata, outdata, frames, time_info, status):
            sr = self.audio_engine.sample_rate

            # Always start from silence for this module's output.
            outdata.fill(0)
            
            # --- Decoder (Input) ---
            if indata.shape[1] > self.selected_input_channel:
                in_sig = indata[:, self.selected_input_channel]
                
                # Input Level
                rms = np.sqrt(np.mean(in_sig**2))
                self.input_level_db = 20 * np.log10(rms + 1e-9)
                
                # Pass to decoder (Gate applied)
                if self.input_level_db > self.gate_threshold_db:
                    process_sig = in_sig
                else:
                    process_sig = np.zeros_like(in_sig)
                    
                if self.decoder.process_samples(process_sig):
                    # Flag update
                    self.decoded_tc = self.decoder.decoded_tc
                    self.locked = self.decoder.locked

            # --- Generator (Output) ---
            if self.generator_enabled:
                gen = self._get_generator_samples(frames)
                if gen is not None and len(gen) == frames:
                    # Keep headroom; LTC does not need full-scale.
                    gen = gen * 0.5

                    out_ch = outdata.shape[1]
                    if out_ch <= 0:
                        return

                    ch = int(self.selected_output_channel)
                    if ch < 0 or ch >= out_ch:
                        ch = 0

                    outdata[:, ch] = gen
        self.callback_id = self.audio_engine.register_callback(callback)

    def _get_generator_samples(self, frames: int) -> np.ndarray:
        """Return exactly `frames` samples of generated LTC (mono)."""
        if frames <= 0:
            return np.zeros((0,), dtype=np.float32)

        out = np.zeros((frames,), dtype=np.float32)
        out_pos = 0

        while out_pos < frames:
            if self._gen_current is None or self._gen_pos >= len(self._gen_current):
                if self.gen_buffer:
                    self._gen_current = self.gen_buffer.popleft()
                    self._gen_pos = 0
                else:
                    self._generate_next_frame()
                    continue

            remaining_out = frames - out_pos
            remaining_in = len(self._gen_current) - self._gen_pos
            to_copy = remaining_out if remaining_out < remaining_in else remaining_in

            if to_copy > 0:
                out[out_pos:out_pos + to_copy] = self._gen_current[self._gen_pos:self._gen_pos + to_copy]
                out_pos += to_copy
                self._gen_pos += to_copy

        return out

    def _generate_next_frame(self):
        # Determine time
        t_now = time.time()
        
        # Apply output offset (advance time)
        t_target = t_now + (self.gen_offset_ms / 1000.0)
        
        if self.generator_mode == 'free':
            # Relative to start
            if self.free_run_start_time == 0:
                self.free_run_start_time = t_now
            
            # Simple frame counter
            total_frames = self.frames_generated
            # Or based on time?
            # Let's just increment frame by frame to ensure continuity.
            # But we need to initialize 'total_frames' based on 'free_run_start_time'?
            # For free run, we usually just start at 00:00:00:00 or user set value.
            # Let's implement continuous increment.
            
            # Calculate TC from total frames
            # fps
            fps = self.fps
            hh = int(total_frames / (fps * 3600)) % 24
            rem = total_frames % (int(fps * 3600))
            mm = int(rem / (fps * 60))
            rem = rem % (int(fps * 60))
            ss = int(rem / fps)
            ff = int(rem % fps)
            
            self.frames_generated += 1
            
        else:
            # Time of Day (UTC)
            # Keep internal time base consistent (UTC). Display-time conversion is handled separately.
            dt = time.gmtime(t_target)
            hh = dt.tm_hour
            mm = dt.tm_min
            ss = dt.tm_sec
            # Calculate frame from fractional second
            frac = t_target - int(t_target)
            ff = int(frac * self.fps)
            
            # Frame continuity is tricky with TOD if we just jump.
            # But TOD generator usually just encodes current time.
            
        samples = self.encoder.generate_frame(hh, mm, ss, ff)
        self.gen_buffer.append(samples)

    def stop_analysis(self):
        if self.callback_id:
            self.audio_engine.unregister_callback(self.callback_id)
            self.callback_id = None
        self.is_running = False

    def process(self):
        tc_display = self._get_display_timecode()
        # Return monitored values for UI
        return {
            "tc": tc_display,
            "tc_raw": self.decoded_tc,
            "locked": self.locked,
            "fps": self.fps, # Or measured
            "level": self.input_level_db
        }

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

    def _get_display_timecode(self) -> str:
        """Return display timecode string.

        If enabled, we interpret the decoded time-of-day as UTC and convert for display only.
        (LTC does not carry a date; we use today's UTC date for conversion.)
        """
        if not self.display_tz_enabled:
            return self.decoded_tc

        parsed = self._parse_tc(self.decoded_tc)
        if parsed is None:
            return self.decoded_tc

        hh, mm, ss, ff = parsed
        fps = float(self.fps) if self.fps else 30.0
        if fps <= 0:
            fps = 30.0

        try:
            utc_today = datetime.now(timezone.utc).date()
            base = datetime(
                utc_today.year,
                utc_today.month,
                utc_today.day,
                hh,
                mm,
                min(ss, 59),
                0,
                tzinfo=timezone.utc,
            )
            dt_utc = base + timedelta(seconds=(ff / fps))

            tz_name = (self.display_tz_name or "System").strip()
            if tz_name.lower() == "utc":
                tz = timezone.utc
            elif tz_name.lower() == "system":
                tz = datetime.now().astimezone().tzinfo
            else:
                if ZoneInfo is None:
                    tz = datetime.now().astimezone().tzinfo
                else:
                    tz = ZoneInfo(tz_name)

            dt_local = dt_utc.astimezone(tz)
            frac = dt_local.microsecond / 1_000_000.0
            ff_local = int(frac * fps)
            if ff_local < 0:
                ff_local = 0
            elif ff_local >= int(round(fps)):
                ff_local = int(round(fps)) - 1

            return f"{dt_local.hour:02}:{dt_local.minute:02}:{dt_local.second:02}:{ff_local:02}"
        except Exception:
            return self.decoded_tc

class TimecodeMonitorWidget(QWidget):
    def __init__(self, module: TimecodeMonitor):
        super().__init__()
        self.module = module
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50) # 20Hz UI update
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # --- Display Panel ---
        display_frame = QFrame()
        display_frame.setStyleSheet("background-color: #111; border: 2px solid #555; border-radius: 8px;")
        display_layout = QVBoxLayout(display_frame)
        
        self.tc_label = QLabel("--:--:--:--")
        self.tc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tc_label.setFont(QFont("Monospace", 64, QFont.Weight.Bold))
        self.tc_label.setStyleSheet("color: #ff3333;") # Red LED style
        display_layout.addWidget(self.tc_label)
        
        status_line = QHBoxLayout()
        self.sync_led = QLabel("SYNC")
        self.sync_led.setStyleSheet("color: #333; font-weight: bold; border: 1px solid #333; padding: 2px 5px; border-radius:4px;")
        status_line.addWidget(self.sync_led)
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #888;")
        status_line.addWidget(self.fps_label)
        
        status_line.addStretch()
        self.level_label = QLabel("-- dB")
        self.level_label.setStyleSheet("color: #888;")
        status_line.addWidget(self.level_label)
        
        display_layout.addLayout(status_line)
        layout.addWidget(display_frame)
        
        # --- Controls ---
        controls_group = QGroupBox(tr("Generator & Settings"))
        c_layout = QGridLayout()
        
        # FPS Selection
        c_layout.addWidget(QLabel(tr("Frame Rate:")), 0, 0)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["23.976", "24", "25", "29.97", "30"])
        self.fps_combo.setCurrentText("30")
        self.fps_combo.currentTextChanged.connect(self.on_fps_changed)
        c_layout.addWidget(self.fps_combo, 0, 1)
        
        # Generator Mode
        c_layout.addWidget(QLabel(tr("Gen Mode:")), 1, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Time of Day", "tod")
        self.mode_combo.addItem("Free Run", "free")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        c_layout.addWidget(self.mode_combo, 1, 1)
        
        # Generator Toggle
        self.gen_btn = QPushButton(tr("Enable Generator"))
        self.gen_btn.setCheckable(True)
        self.gen_btn.clicked.connect(self.on_gen_toggle)
        c_layout.addWidget(self.gen_btn, 0, 2, 2, 1)

        # Display timezone (display-only)
        self.display_tz_check = QCheckBox(tr("Display Local Time"))
        self.display_tz_check.setChecked(bool(self.module.display_tz_enabled))
        self.display_tz_check.toggled.connect(self.on_display_tz_toggled)
        c_layout.addWidget(self.display_tz_check, 0, 5)

        c_layout.addWidget(QLabel(tr("Display TZ:")), 1, 5)
        self.tz_combo = QComboBox()
        self.tz_combo.setEditable(True)
        self.tz_combo.addItems([
            "System",
            "UTC",
            "Asia/Tokyo",
            "Europe/London",
            "America/New_York",
        ])
        self.tz_combo.setCurrentText(self.module.display_tz_name or "System")
        self.tz_combo.currentTextChanged.connect(self.on_display_tz_changed)
        c_layout.addWidget(self.tz_combo, 1, 6)
        
        # Offsets
        c_layout.addWidget(QLabel(tr("In Delay (ms):")), 0, 3)
        self.in_offset_spin = QDoubleSpinBox()
        self.in_offset_spin.setRange(-1000, 1000)
        self.in_offset_spin.setValue(0.0)
        self.in_offset_spin.valueChanged.connect(lambda v: setattr(self.module, 'input_offset_ms', v))
        c_layout.addWidget(self.in_offset_spin, 0, 4)
        
        c_layout.addWidget(QLabel(tr("Out Delay (ms):")), 1, 3)
        self.out_offset_spin = QDoubleSpinBox()
        self.out_offset_spin.setRange(-1000, 1000)
        self.out_offset_spin.setValue(0.0)
        self.out_offset_spin.valueChanged.connect(lambda v: setattr(self.module, 'gen_offset_ms', v))
        c_layout.addWidget(self.out_offset_spin, 1, 4)
        
        # Channels
        c_layout.addWidget(QLabel(tr("In Channel:")), 2, 0)
        self.in_ch_combo = QComboBox()
        self.in_ch_combo.addItems([tr("Left (Ch 1)"), tr("Right (Ch 2)")])
        self.in_ch_combo.setCurrentIndex(0 if self.module.selected_input_channel == 0 else 1)
        self.in_ch_combo.currentIndexChanged.connect(lambda idx: setattr(self.module, 'selected_input_channel', idx))
        c_layout.addWidget(self.in_ch_combo, 2, 1)
        
        c_layout.addWidget(QLabel(tr("Out Channel:")), 2, 2)
        self.out_ch_combo = QComboBox()
        self.out_ch_combo.addItems([tr("Left (Ch 1)"), tr("Right (Ch 2)")])
        self.out_ch_combo.setCurrentIndex(0 if self.module.selected_output_channel == 0 else 1)
        self.out_ch_combo.currentIndexChanged.connect(lambda idx: setattr(self.module, 'selected_output_channel', idx))
        c_layout.addWidget(self.out_ch_combo, 2, 3)
        
        controls_group.setLayout(c_layout)
        layout.addWidget(controls_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Start module
        QTimer.singleShot(100, self.module.start_analysis)
        
    def on_fps_changed(self, text):
        try:
            fps = float(text)
            self.module.set_fps(fps)
        except:
            pass
            
    def on_mode_changed(self):
        self.module.generator_mode = self.mode_combo.currentData()
        self.module.frames_generated = 0 # Reset free run counter
        self.module.gen_buffer.clear()
        self.module._gen_current = None
        self.module._gen_pos = 0
        self.module.free_run_start_time = 0.0
        
    def on_gen_toggle(self, checked):
        self.module.generator_enabled = checked
        self.gen_btn.setText(tr("Stop Generator") if checked else tr("Enable Generator"))

    def on_display_tz_toggled(self, checked: bool):
        self.module.display_tz_enabled = bool(checked)

    def on_display_tz_changed(self, text: str):
        self.module.display_tz_name = str(text).strip() if text else "System"
        
    def update_ui(self):
        data = self.module.process()
        self.tc_label.setText(data['tc'])

        self.fps_label.setText(f"FPS: {data['fps']:.3g}")
        
        if data['locked']:
            self.sync_led.setStyleSheet("color: #0f0; font-weight: bold; border: 1px solid #0f0; background-color: #003300; padding: 2px 5px; border-radius:4px;")
            self.module.locked = False # Reset for next check (naive "locked signal" keepalive)
        else:
             self.sync_led.setStyleSheet("color: #555; font-weight: normal; border: 1px solid #555; padding: 2px 5px; border-radius:4px;")
             
        self.level_label.setText(f"{data['level']:.1f} dB")
