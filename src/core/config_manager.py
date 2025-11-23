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
                "block_size": 1024
            }
        }

    def get_last_devices(self):
        """Returns (input_device_name, output_device_name)."""
        audio_cfg = self.config.get("audio", {})
        return audio_cfg.get("input_device"), audio_cfg.get("output_device")

    def set_last_devices(self, input_name, output_name):
        """Updates the last used devices."""
        if "audio" not in self.config:
            self.config["audio"] = {}
        
        self.config["audio"]["input_device"] = input_name
        self.config["audio"]["output_device"] = output_name
        self.save_config()
