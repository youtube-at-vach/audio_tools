import json
import os
import logging

class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self.load_config()

    def load_config(self):
        """Loads configuration from JSON file."""
        if not os.path.exists(self.config_path):
            self.logger.info("No config file found, creating default.")
            return self._default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._default_config()

    def save_config(self):
        """Saves current configuration to JSON file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info("Config saved.")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def _default_config(self):
        return {
            "audio": {
                "input_device": None,
                "output_device": None,
                "sample_rate": 48000,
                "block_size": 1024,
                "input_channels": "stereo",
                "output_channels": "stereo"
            }
        }

    def get_audio_config(self):
        """Returns a dictionary of audio configuration."""
        return self.config.get("audio", self._default_config()["audio"])

    def set_audio_config(self, input_name, output_name, sample_rate, block_size, in_ch, out_ch):
        """Updates the audio configuration."""
        if "audio" not in self.config:
            self.config["audio"] = {}
        
        self.config["audio"]["input_device"] = input_name
        self.config["audio"]["output_device"] = output_name
        self.config["audio"]["sample_rate"] = sample_rate
        self.config["audio"]["block_size"] = block_size
        self.config["audio"]["input_channels"] = in_ch
        self.config["audio"]["output_channels"] = out_ch
        self.save_config()
