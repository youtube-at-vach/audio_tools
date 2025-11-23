import sounddevice as sd
import numpy as np
import logging
import threading
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

    def list_devices(self):
        """Returns a list of available audio devices."""
        return sd.query_devices()

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
            if not self.callbacks and self.stream is not None:
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
        
        def master_callback(indata, outdata, frames, time, status):
            # Zero out master output buffer first
            outdata.fill(0)
            
            # Prepare logical input for clients
            # Map Hardware Input -> Logical Input (Stereo usually, or as requested)
            # For simplicity, we'll provide stereo (or 1ch) to clients based on what we have.
            # But clients currently expect (frames, channels).
            # Let's standardize on passing what we have.
            
            # Mapping logic similar to previous implementation
            if in_mode == 'left':
                logical_in = indata[:, 0:1]
            elif in_mode == 'right':
                if indata.shape[1] >= 2:
                    logical_in = indata[:, 1:2]
                else:
                    logical_in = np.zeros((frames, 1))
            else: # stereo
                logical_in = indata[:, 0:2]

            # We need a temporary buffer for clients to write to, so we can sum them.
            # Clients expect to write to 'outdata'.
            # If we pass 'outdata' directly, the first client writes, second overwrites?
            # No, we must sum.
            
            # Create a temp output buffer for clients
            # We'll assume clients want to write stereo or mono based on out_mode.
            # If out_mode is stereo, logical_out_ch = 2
            logical_out_ch = 2 if out_mode == 'stereo' else 1
            
            # Iterate over a copy of items to avoid issues if modified during iteration
            # (though we have a lock for add/remove, iteration inside callback should be safe 
            # if we copy keys or use a thread-safe structure. Python dict iteration is not thread-safe 
            # if modified. But we modify under lock. Callback runs in a separate thread.)
            
            # To be safe, acquire lock briefly or copy. 
            # Acquiring lock in audio callback is risky (priority inversion).
            # Better to copy callbacks dict when modifying, or use a flag.
            # For now, we'll try to be quick.
            
            # Snapshot of callbacks
            with self.lock:
                active_callbacks = list(self.callbacks.values())
            
            if not active_callbacks:
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
            
            # Map Logical Output -> Hardware Output
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
        
        try:
            self.stream = sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=master_callback,
                channels=(hw_in_ch, hw_out_ch),
                dtype='float32',
                latency='high'
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
