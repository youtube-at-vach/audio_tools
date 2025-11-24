import json
import os
import numpy as np

class CalibrationManager:
    """
    Manages audio calibration data (sensitivity, gain) and conversions.
    Stores data in a JSON file.
    """
    def __init__(self, config_path="calibration.json"):
        self.config_path = config_path
        self.input_sensitivity = 1.0 # Volts per Full Scale (V/FS) (Peak)
        self.output_gain = 1.0 # Volts per Full Scale (V/FS) (Peak)
        self.frequency_calibration = 1.0 # Multiplier for frequency correction
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.input_sensitivity = data.get('input_sensitivity', 1.0)
                    self.output_gain = data.get('output_gain', 1.0)
                    self.frequency_calibration = data.get('frequency_calibration', 1.0)
            except Exception as e:
                print(f"Failed to load calibration: {e}")

    def save(self):
        data = {
            'input_sensitivity': self.input_sensitivity,
            'output_gain': self.output_gain,
            'frequency_calibration': self.frequency_calibration
        }
        try:
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save calibration: {e}")

    def set_input_sensitivity(self, v_per_fs):
        """Sets input sensitivity in Volts (Peak) corresponding to 1.0 FS."""
        self.input_sensitivity = v_per_fs
        self.save()

    def set_output_gain(self, v_per_fs):
        """Sets output gain in Volts (Peak) corresponding to 1.0 FS."""
        self.output_gain = v_per_fs
        self.save()

    def set_frequency_calibration(self, factor):
        """Sets the frequency calibration factor (multiplier)."""
        self.frequency_calibration = factor
        self.save()

    def dbfs_to_dbv(self, dbfs):
        """Converts dBFS to dBV."""
        # 0 dBFS = 20 * log10(1.0)
        # Voltage at 0 dBFS = input_sensitivity
        # dBV = 20 * log10(Voltage)
        # Voltage = 10^(dBFS/20) * input_sensitivity
        # dBV = 20 * log10(10^(dBFS/20) * input_sensitivity)
        #     = dBFS + 20 * log10(input_sensitivity)
        return dbfs + 20 * np.log10(self.input_sensitivity)

    def dbfs_to_volts(self, dbfs):
        """Converts dBFS to Volts (Peak)."""
        return (10**(dbfs/20)) * self.input_sensitivity

    def get_input_offset_db(self):
        """Returns the dB offset to add to dBFS to get dBV."""
        return 20 * np.log10(self.input_sensitivity)
