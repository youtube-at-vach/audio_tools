import sounddevice as sd
import numpy as np
import logging
from src.core.calibration import CalibrationManager

class AudioEngine:
    """
    Handles audio I/O operations using sounddevice.
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

    def list_devices(self):
        """Returns a list of available audio devices."""
        return sd.query_devices()

    def set_devices(self, input_device_id, output_device_id):
        """Sets the input and output devices."""
        self.input_device = input_device_id
        self.output_device = output_device_id
        self.logger.info(f"Set devices: Input={input_device_id}, Output={output_device_id}")

    def set_sample_rate(self, rate):
        self.sample_rate = rate
        self.logger.info(f"Set sample rate: {rate}")

    def set_block_size(self, size):
        self.block_size = size
        self.logger.info(f"Set block size: {size}")

    def set_channel_mode(self, input_mode, output_mode):
        self.input_channel_mode = input_mode
        self.output_channel_mode = output_mode
        self.logger.info(f"Set channel modes: Input={input_mode}, Output={output_mode}")

    def start_stream(self, callback, channels=2):
        """
        Starts the audio stream with the given callback.
        """
        if self.stream is not None:
            self.stop_stream()

        # Determine hardware channels needed based on mode
        # Left (idx 0) -> need 1 ch
        # Right (idx 1) -> need 2 chs (to access index 1)
        # Stereo -> need 2 chs
        
        in_mode = self.input_channel_mode
        out_mode = self.output_channel_mode
        
        hw_in_ch = 2 if in_mode in ['right', 'stereo'] else 1
        hw_out_ch = 2 if out_mode in ['right', 'stereo'] else 1
        
        # Define wrapper callback for software mapping
        def wrapped_callback(indata, outdata, frames, time, status):
            # Map Input
            if in_mode == 'left':
                logical_in = indata[:, 0:1]
            elif in_mode == 'right':
                # If we opened 2 channels, right is at index 1
                if indata.shape[1] >= 2:
                    logical_in = indata[:, 1:2]
                else:
                    # Fallback if device didn't give us 2 channels
                    logical_in = np.zeros((frames, 1))
            else: # stereo
                logical_in = indata[:, 0:2]
                
            # Prepare logical output buffer
            # We need to pass a buffer to the user callback to fill.
            # We can use a slice of outdata if the mapping allows, or a temp buffer.
            # For 'Left' mode, user sees 1 channel. We map it to outdata[:, 0].
            # For 'Right' mode, user sees 1 channel. We map it to outdata[:, 1].
            
            # To avoid copies and complexity, let's create a temp buffer for output if needed,
            # or try to use views.
            # User callback expects (frames, logical_channels)
            
            if out_mode == 'stereo':
                logical_out = outdata[:, 0:2]
                callback(logical_in, logical_out, frames, time, status)
            elif out_mode == 'left':
                # We can pass the view of the first channel
                logical_out = outdata[:, 0:1]
                callback(logical_in, logical_out, frames, time, status)
                # Zero out other channels if any?
                if outdata.shape[1] > 1:
                    outdata[:, 1:] = 0
            elif out_mode == 'right':
                # We need a temp buffer because user writes to ch 0 of logical output,
                # but we want it in ch 1 of hardware output.
                # OR we pass a view of ch 1, but user might expect 1 channel.
                # If we pass outdata[:, 1:2], it has shape (N, 1). This works!
                if outdata.shape[1] >= 2:
                    logical_out = outdata[:, 1:2]
                    callback(logical_in, logical_out, frames, time, status)
                    outdata[:, 0] = 0 # Silence left
                else:
                    # Hardware doesn't have ch 1?
                    pass
                    
        try:
            self.stream = sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=wrapped_callback,
                channels=(hw_in_ch, hw_out_ch),
                dtype='float32',
                latency='high'
            )
            self.stream.start()
            self.logger.info(f"Audio stream started. SR={self.sample_rate}, HW_Ch=({hw_in_ch}, {hw_out_ch}), Mode=({in_mode}, {out_mode})")
        except Exception as e:
            self.logger.error(f"Failed to start stream: {e}")
            raise

    def stop_stream(self):
        """Stops the audio stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.logger.info("Audio stream stopped")

    def is_active(self):
        """Returns True if the stream is active."""
        return self.stream is not None and self.stream.active
