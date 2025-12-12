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
        # Whether the output gain was explicitly calibrated by the user.
        # Used to decide when to offer voltage-based UI controls.
        self.output_gain_is_calibrated = False
        self.frequency_calibration = 1.0 # Multiplier for frequency correction
        self.lockin_gain_offset = 0.0 # dB offset for Lock-in Amplifier
        # SPL calibration: maps measured (C-weighted) dBFS to SPL.
        # Stored as an offset: SPL[dB] = dBFS_C + spl_offset_db.
        self.spl_offset_db = None
        self.spl_meta = {}
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.input_sensitivity = data.get('input_sensitivity', 1.0)
                    self.output_gain = data.get('output_gain', 1.0)
                    # New flag (backward compatible)
                    if 'output_gain_is_calibrated' in data:
                        self.output_gain_is_calibrated = bool(data.get('output_gain_is_calibrated'))
                    else:
                        # Heuristic for older files: treat non-default values as calibrated.
                        try:
                            self.output_gain_is_calibrated = abs(float(self.output_gain) - 1.0) > 1e-12
                        except Exception:
                            self.output_gain_is_calibrated = False
                    self.frequency_calibration = data.get('frequency_calibration', 1.0)
                    self.lockin_gain_offset = data.get('lockin_gain_offset', 0.0)

                    # New format
                    if 'spl_offset_db' in data:
                        try:
                            self.spl_offset_db = float(data.get('spl_offset_db'))
                        except Exception:
                            self.spl_offset_db = None
                    self.spl_meta = data.get('spl_meta', {}) or {}

                    # Backward compatibility (older dict-based format)
                    if self.spl_offset_db is None:
                        legacy = data.get('spl_calibration', None)
                        if isinstance(legacy, dict) and legacy:
                            entry = legacy.get('speaker') or legacy.get('subwoofer')
                            if entry is None:
                                try:
                                    entry = next(iter(legacy.values()))
                                except Exception:
                                    entry = None
                            if isinstance(entry, dict) and 'offset_db' in entry:
                                try:
                                    self.spl_offset_db = float(entry.get('offset_db'))
                                except Exception:
                                    self.spl_offset_db = None
            except Exception as e:
                print(f"Failed to load calibration: {e}")

    def save(self):
        data = {
            'input_sensitivity': self.input_sensitivity,
            'output_gain': self.output_gain,
            'output_gain_is_calibrated': bool(self.output_gain_is_calibrated),
            'frequency_calibration': self.frequency_calibration,
            'lockin_gain_offset': self.lockin_gain_offset,
            # Keep a single SPL calibration value.
            'spl_offset_db': self.spl_offset_db,
            'spl_meta': self.spl_meta,
        }
        try:
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save calibration: {e}")

    # --- SPL Calibration ---

    def set_spl_calibration(self, measured_dbfs_c, measured_spl_db, *,
                            band_hz=None, weighting='C', notes=None):
        """Stores SPL calibration as an offset (SPL = dBFS_C + spl_offset_db)."""
        try:
            measured_dbfs_c = float(measured_dbfs_c)
            measured_spl_db = float(measured_spl_db)
        except Exception:
            raise ValueError("Invalid SPL calibration values")

        offset_db = measured_spl_db - measured_dbfs_c
        self.spl_offset_db = float(offset_db)
        meta = {
            'measured_dbfs_c': float(measured_dbfs_c),
            'measured_spl_db': float(measured_spl_db),
            'weighting': str(weighting),
        }
        if band_hz is not None:
            meta['band_hz'] = list(band_hz)
        if notes:
            meta['notes'] = str(notes)

        if not isinstance(self.spl_meta, dict):
            self.spl_meta = {}
        self.spl_meta = meta
        self.save()

    def get_spl_offset_db(self):
        try:
            return None if self.spl_offset_db is None else float(self.spl_offset_db)
        except Exception:
            return None

    def dbfs_to_spl(self, dbfs_c, profile=None):
        """Converts (C-weighted) dBFS to SPL using the stored offset."""
        off = self.get_spl_offset_db()
        if off is None:
            return None
        return float(dbfs_c) + off

    def set_input_sensitivity(self, v_per_fs):
        """Sets input sensitivity in Volts (Peak) corresponding to 1.0 FS."""
        self.input_sensitivity = v_per_fs
        self.save()

    def set_output_gain(self, v_per_fs):
        """Sets output gain in Volts (Peak) corresponding to 1.0 FS."""
        try:
            v_per_fs = float(v_per_fs)
        except Exception:
            raise ValueError("Invalid output gain")
        if not np.isfinite(v_per_fs) or v_per_fs <= 0:
            raise ValueError("Invalid output gain")

        self.output_gain = v_per_fs
        self.output_gain_is_calibrated = True
        self.save()

    def set_frequency_calibration(self, factor):
        """Sets the frequency calibration factor (multiplier)."""
        self.frequency_calibration = factor
        self.save()
        
    def set_lockin_gain_offset(self, offset_db):
        """Sets the absolute gain offset for the Lock-in Amplifier in dB."""
        self.lockin_gain_offset = offset_db
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

    # --- Frequency Correction Map ---
    
    def load_frequency_map(self, path):
        """
        Loads a frequency correction map from a JSON file.
        Format: [[freq, mag_db, phase_deg], ...]
        """
        if not os.path.exists(path):
            print(f"Calibration map not found: {path}")
            return False
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Sort by frequency just in case
                self.frequency_map = sorted(data, key=lambda x: x[0])
                print(f"Loaded calibration map with {len(self.frequency_map)} points.")
                return True
        except Exception as e:
            print(f"Failed to load calibration map: {e}")
            return False

    def save_frequency_map(self, path, data):
        """
        Saves the frequency correction map to a JSON file.
        data: list of [freq, mag_db, phase_deg]
        """
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            self.frequency_map = sorted(data, key=lambda x: x[0])
            print(f"Saved calibration map to {path}")
            return True
        except Exception as e:
            print(f"Failed to save calibration map: {e}")
            return False

    def get_frequency_correction(self, freq):
        """
        Returns (mag_correction_db, phase_correction_deg) for the given frequency.
        Uses linear interpolation.
        Returns (0.0, 0.0) if no map is loaded.
        """
        if not hasattr(self, 'frequency_map') or not self.frequency_map:
            return 0.0, 0.0
            
        # If out of range, clamp to nearest
        if freq <= self.frequency_map[0][0]:
            return self.frequency_map[0][1], self.frequency_map[0][2]
        if freq >= self.frequency_map[-1][0]:
            return self.frequency_map[-1][1], self.frequency_map[-1][2]
            
        # Binary search or simple search (map size usually < 1000)
        # np.interp is convenient if we separate arrays, but here we have list of lists.
        # Let's do a simple search or convert to numpy arrays on load? 
        # For now, simple search is fine for < 1000 points.
        
        # Optimization: Convert to numpy arrays on load if performance is critical.
        # But for now, let's just do it simply.
        
        # Find index i such that map[i][0] <= freq < map[i+1][0]
        # Using bisect would be faster but let's stick to basic python for clarity unless needed.
        
        # Let's use numpy for interpolation, it's robust.
        # We can cache the numpy arrays if this is called often (it is).
        
        if not hasattr(self, '_map_freqs'):
            self._update_map_cache()
            
        mag_corr = np.interp(freq, self._map_freqs, self._map_mags)
        phase_corr = np.interp(freq, self._map_freqs, self._map_phases)
        
        return mag_corr, phase_corr

    def _update_map_cache(self):
        if not hasattr(self, 'frequency_map') or not self.frequency_map:
            self._map_freqs = np.array([])
            self._map_mags = np.array([])
            self._map_phases = np.array([])
            return

        data = np.array(self.frequency_map)
        self._map_freqs = data[:, 0]
        self._map_mags = data[:, 1]
        self._map_phases = data[:, 2]

