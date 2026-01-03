import logging
import threading

import numpy as np
import sounddevice as sd

from src.core.calibration import CalibrationManager


class AudioEngine:
    """
    Handles audio I/O operations using sounddevice.
    Implements a mixer to support multiple simultaneous clients.
    """
    def __init__(self):
        self.input_device = None
        self.output_device = None
        self.sample_rate = 48000
        self.block_size = 1024
        self.stream = None
        self.logger = logging.getLogger(__name__)

        # PipeWire/JACK resident mode: keep PortAudio stream open for the app lifetime.
        self.pipewire_jack_resident = False
        self.jack_client_name = "MeasureLab"

        # Calibration
        self.calibration = CalibrationManager()

        # Channel Configuration
        # 'stereo', 'left', 'right'
        self.input_channel_mode = 'stereo'
        self.output_channel_mode = 'stereo'

        # Mixer State
        self.callbacks = {} # id -> callback
        self.next_callback_id = 0
        self.lock = threading.Lock()

        # Status Monitoring
        # Loopback State
        self.loopback = False
        self.mute_output = False
        self.last_output_buffer = None

        # Accumulate callback status flags between UI polls.
        self.accumulated_status = sd.CallbackFlags()

    def set_pipewire_jack_resident(self, enabled: bool):
        """Enable/disable resident stream mode (useful for PipeWire/JACK routing persistence)."""
        enabled = bool(enabled)
        self.pipewire_jack_resident = enabled
        self.logger.info(f"Set PipeWire/JACK resident mode: {enabled}")

        if enabled:
            # Ensure master stream is open even with zero clients.
            self._start_master_stream()
            return

        # Disabled: revert to legacy behavior (only keep stream open while clients exist).
        with self.lock:
            has_clients = bool(self.callbacks)
        if not has_clients:
            self.stop_stream()

    def set_loopback(self, enabled):
        self.loopback = enabled
        self.logger.info(f"Set software loopback: {enabled}")

    def set_mute_output(self, enabled):
        self.mute_output = enabled
        self.logger.info(f"Set mute output: {enabled}")

    def list_devices(self):
        """Returns a list of available audio devices.

        We enrich PortAudio device info with a human-readable host API name
        (e.g. ASIO/WASAPI/DirectSound on Windows) to make UI selection clearer.
        """
        devices = sd.query_devices()

        # Try to attach host API names; fall back to raw device dicts on error.
        try:
            hostapis = sd.query_hostapis()
        except Exception:
            hostapis = None

        enriched = []
        for dev in devices:
            d = dict(dev)
            hostapi_name = None
            if hostapis is not None:
                try:
                    hostapi_idx = d.get('hostapi')
                    if hostapi_idx is not None and 0 <= int(hostapi_idx) < len(hostapis):
                        hostapi_name = hostapis[int(hostapi_idx)].get('name')
                except Exception:
                    hostapi_name = None

            if hostapi_name:
                d['hostapi_name'] = str(hostapi_name)

            enriched.append(d)

        return enriched

    def set_devices(self, input_device_id, output_device_id):
        """Sets the input and output devices."""
        self.input_device = input_device_id
        self.output_device = output_device_id
        self.logger.info(f"Set devices: Input={input_device_id}, Output={output_device_id}")
        # Restart stream if running to apply changes
        if self.is_active():
            self._restart_stream()

    def set_sample_rate(self, rate):
        self.sample_rate = rate
        self.logger.info(f"Set sample rate: {rate}")
        if self.is_active():
            self._restart_stream()

    def set_block_size(self, size):
        self.block_size = size
        self.logger.info(f"Set block size: {size}")
        if self.is_active():
            self._restart_stream()

    def set_channel_mode(self, input_mode, output_mode):
        self.input_channel_mode = input_mode
        self.output_channel_mode = output_mode
        self.logger.info(f"Set channel modes: Input={input_mode}, Output={output_mode}")
        # Note: Changing channel mode might affect active callbacks if they expect specific mapping.
        # For now, we assume global mode applies to the master stream.

    def register_callback(self, callback):
        """
        Registers a callback for audio processing.
        Returns a callback_id.
        Callback signature: callback(indata, outdata, frames, time, status)
        """
        with self.lock:
            cid = self.next_callback_id
            self.next_callback_id += 1
            self.callbacks[cid] = callback
            self.logger.info(f"Registered callback {cid}")

            # Start stream if not running
            if self.stream is None:
                self._start_master_stream()

            return cid

    def unregister_callback(self, callback_id):
        """Unregisters a callback by ID."""
        should_stop = False
        with self.lock:
            if callback_id in self.callbacks:
                del self.callbacks[callback_id]
                self.logger.info(f"Unregistered callback {callback_id}")

            # Check if we should stop the stream
            if (not self.callbacks) and (self.stream is not None) and (not self.pipewire_jack_resident):
                should_stop = True

        # Stop stream outside the lock to avoid deadlock with callback
        if should_stop:
            self.stop_stream()

    def _start_master_stream(self):
        """Starts the underlying sounddevice stream."""
        if self.stream is not None:
            return

        # Determine hardware channels needed based on mode
        in_mode = self.input_channel_mode
        out_mode = self.output_channel_mode

        hw_in_ch = 2 if in_mode in ['right', 'stereo'] else 1
        hw_out_ch = 2 if out_mode in ['right', 'stereo'] else 1

        # Reset loopback buffer
        self.last_output_buffer = None

        def master_callback(indata, outdata, frames, time, status):
            if status:
                self.accumulated_status |= status

            # Zero out master output buffer first
            outdata.fill(0)

            # Prepare logical input for clients
            # Map Hardware Input -> Logical Input (Stereo usually, or as requested)

            # If Loopback is enabled, use the last output buffer as input
            if self.loopback and self.last_output_buffer is not None and len(self.last_output_buffer) == frames:
                # We use the mixed output from the previous block
                # last_output_buffer is (frames, logical_out_ch)
                # We need to map it to logical_in (frames, 2)

                # Assuming logical_out_ch is 2 (stereo) or 1 (mono)
                # logical_in is usually stereo (2)

                lb_src = self.last_output_buffer
                logical_in = np.zeros((frames, 2), dtype='float32')

                if lb_src.shape[1] >= 2:
                    logical_in[:, :2] = lb_src[:, :2]
                elif lb_src.shape[1] == 1:
                    logical_in[:, 0] = lb_src[:, 0]
                    logical_in[:, 1] = lb_src[:, 0]
            else:
                # Standard Hardware Input Mapping
                if in_mode == 'left':
                    logical_in = indata[:, 0:1]
                elif in_mode == 'right':
                    if indata.shape[1] >= 2:
                        logical_in = indata[:, 1:2]
                    else:
                        logical_in = np.zeros((frames, 1))
                else: # stereo
                    logical_in = indata[:, 0:2]

            # Create a temp output buffer for clients
            logical_out_ch = 2 if out_mode == 'stereo' else 1

            # Snapshot of callbacks
            with self.lock:
                active_callbacks = list(self.callbacks.values())

            if not active_callbacks:
                # Even if no callbacks, we might need to update last_output_buffer (silence)
                if self.loopback:
                    if self.last_output_buffer is None or len(self.last_output_buffer) != frames:
                         self.last_output_buffer = np.zeros((frames, logical_out_ch), dtype='float32')
                    else:
                         self.last_output_buffer.fill(0)
                return

            # Mix buffer
            mix_buffer = np.zeros((frames, logical_out_ch), dtype='float32')

            for cb in active_callbacks:
                # Temp buffer for this client
                client_out = np.zeros_like(mix_buffer)

                try:
                    cb(logical_in, client_out, frames, time, status)
                except Exception as e:
                    print(f"Error in audio callback: {e}")
                    continue

                # Sum to mix
                mix_buffer += client_out

            # Store for next loopback cycle
            if self.loopback:
                self.last_output_buffer = mix_buffer.copy()

            # Map Logical Output -> Hardware Output
            if not self.mute_output:
                if out_mode == 'stereo':
                    outdata[:, 0:2] = mix_buffer
                elif out_mode == 'left':
                    outdata[:, 0:1] = mix_buffer
                    if outdata.shape[1] > 1:
                        outdata[:, 1:] = 0
                elif out_mode == 'right':
                    if outdata.shape[1] >= 2:
                        outdata[:, 1:2] = mix_buffer
                        outdata[:, 0] = 0
            # If muted, outdata is already 0 filled at start of callback

        try:
            extra_settings = None
            # If running on JACK (including PipeWire-JACK), attempt to fix the client/node name.
            try:
                hostapi_name = None
                dev_id = self.output_device
                if dev_id is None:
                    # Fallback to default output device.
                    dev_id = sd.default.device[1]
                if dev_id is not None and dev_id != -1:
                    hostapi_idx = sd.query_devices(dev_id).get('hostapi')
                    if hostapi_idx is not None:
                        hostapi_name = sd.query_hostapis(hostapi_idx).get('name')
                if hostapi_name and 'jack' in str(hostapi_name).lower():
                    extra_settings = sd.JackSettings(client_name=self.jack_client_name)
            except Exception:
                extra_settings = None

            self.stream = sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=master_callback,
                channels=(hw_in_ch, hw_out_ch),
                dtype='float32',
                latency='high',
                extra_settings=extra_settings
            )
            self.stream.start()
            self.logger.info(f"Master audio stream started. SR={self.sample_rate}, HW_Ch=({hw_in_ch}, {hw_out_ch})")
        except Exception as e:
            self.logger.error(f"Failed to start master stream: {e}")
            # Don't raise, just log. Clients will just not run.
            self.stream = None

    def stop_stream(self):
        """Stops the master audio stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.logger.info("Master audio stream stopped")

    def _restart_stream(self):
        self.stop_stream()
        if self.callbacks:
            self._start_master_stream()

    def is_active(self):
        """Returns True if the stream is active."""
        return self.stream is not None and self.stream.active

    def get_status(self):
        """Returns a dictionary containing current engine status."""
        active = self.is_active()
        cpu_load = 0.0
        if active and self.stream:
            cpu_load = self.stream.cpu_load

        with self.lock:
            client_count = len(self.callbacks)

        # Get and reset accumulated status
        current_status_flags = self.accumulated_status
        self.accumulated_status = sd.CallbackFlags()

        return {
            "active": active,
            "input_channels": self.input_channel_mode,
            "output_channels": self.output_channel_mode,
            "sample_rate": self.sample_rate,
            "cpu_load": cpu_load,
            "active_clients": client_count,
            "input_device": self.input_device,
            "output_device": self.output_device,
            "status_flags": current_status_flags
        }

    # Legacy method support (deprecated but kept for compatibility during transition if needed)
    def start_stream(self, callback, channels=2):
        """
        Deprecated: Use register_callback instead.
        This acts as a wrapper for single-client usage.
        """
        self.logger.warning("start_stream is deprecated. Use register_callback.")
        # Stop any existing stream/callbacks to mimic exclusive behavior
        self.stop_stream()
        with self.lock:
            self.callbacks.clear()

        self.register_callback(callback)
