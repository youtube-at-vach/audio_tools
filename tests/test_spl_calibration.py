import os
import tempfile

from src.core.calibration import CalibrationManager


def test_spl_calibration_roundtrip_and_convert():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "calibration.json")
        cal = CalibrationManager(config_path=path)

        cal.set_spl_calibration(
            measured_dbfs_c=-40.0,
            measured_spl_db=70.0,
        )

        cal2 = CalibrationManager(config_path=path)
        off = cal2.get_spl_offset_db()
        assert off is not None
        assert abs(off - 110.0) < 1e-6

        spl = cal2.dbfs_to_spl(-40.0)
        assert spl is not None
        assert abs(spl - 70.0) < 1e-6
