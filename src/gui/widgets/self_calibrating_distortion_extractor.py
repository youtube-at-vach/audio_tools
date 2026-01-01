import argparse
import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QDoubleSpinBox,
    QCheckBox,
)
from PyQt6.QtCore import QTimer

from src.measurement_modules.base import MeasurementModule
from src.core.audio_engine import AudioEngine
from src.core.localization import tr
from src.core.phase_sync import estimate_ref_phase_offset_rad, phase_to_sample_shift, fold_one_period_latest


class SelfCalibratingDistortionExtractor(MeasurementModule):
    """Phase-locked, self-synchronizing 1-cycle capture (Phase 1).

    Output: 1-period memory map (sine) repeated to stereo.
    Input: REF (R) and DUTIN (L) captured simultaneously.
    REF is used to estimate phase offset; DUTIN is folded into 1 period and aligned
    to the memory map for overlay visualization.
    """

    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine

        self.is_running = False
        self.callback_id = None

        self.target_frequency_hz = 1000.0
        self.amplitude = 0.5

        self.period_samples = 0
        self.actual_frequency_hz = 0.0
        self.memory_map = np.zeros((0,), dtype=np.float64)

        self.global_sample_counter = 0

        # Input ring buffer (stores last samples + absolute indices)
        self._ring = None
        self._ring_abs = None
        self._ring_pos = 0
        self._ring_valid = 0

        # Latest aligned one-period view (for GUI)
        self.aligned_dutin = np.zeros((0,), dtype=np.float64)
        self.last_phase_rad = 0.0
        self.last_shift_samples = 0

        # Phase 2: normalization + error/metrics (do NOT update memory map)
        self.normalized_dutin = np.zeros((0,), dtype=np.float64)
        self.error_signal = np.zeros((0,), dtype=np.float64)
        self.last_dc_offset = 0.0
        self.last_gain = 1.0
        self.last_rms_error = np.nan
        self.last_rms_error_ratio = np.nan
        self.last_thd_db = np.nan
        self.last_thd_percent = np.nan
        self.last_filled_ratio = 0.0

    @staticmethod
    def _nan_rms(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return float('nan')
        m = np.isfinite(x)
        if not np.any(m):
            return float('nan')
        v = x[m]
        return float(np.sqrt(np.mean(v * v)))

    @staticmethod
    def _compute_thd_from_one_period(x: np.ndarray, *, max_harmonic: int = 10) -> tuple[float, float]:
        """Compute THD from a one-period, DC-removed, level-scaled waveform.

        Uses coherent bin magnitudes: fundamental at bin 1, harmonics at bins k.
        Returns (thd_percent, thd_db). If invalid, returns (nan, nan).
        """
        x = np.asarray(x, dtype=np.float64)
        n = int(x.size)
        if n < 8:
            return float('nan'), float('nan')

        if np.any(~np.isfinite(x)):
            return float('nan'), float('nan')

        # rFFT bins correspond to k*sr/N. Since generator uses sr/N, fundamental is bin 1.
        X = np.fft.rfft(x)
        if X.size < 2:
            return float('nan'), float('nan')

        fund = float(np.abs(X[1]))
        if not np.isfinite(fund) or fund <= 0.0:
            return float('nan'), float('nan')

        h_max = int(min(max_harmonic, X.size - 1))
        if h_max < 2:
            return 0.0, -140.0

        harms = np.abs(X[2 : h_max + 1]).astype(np.float64, copy=False)
        thd = float(np.sqrt(np.sum(harms * harms)) / fund)
        thd_percent = float(100.0 * thd)
        thd_db = float(20.0 * np.log10(thd + 1e-15))
        return thd_percent, thd_db

    @property
    def name(self) -> str:
        return "Self-Calibrating Distortion Extractor"

    @property
    def description(self) -> str:
        return "Phase-locked 1-cycle overlay (Phase 1)."

    def run(self, args: argparse.Namespace):
        print("Self-Calibrating Distortion Extractor running from CLI (not implemented)")

    def get_widget(self):
        return SelfCalibratingDistortionExtractorWidget(self)

    def _rebuild_memory_map(self):
        sr = float(self.audio_engine.sample_rate)
        f_req = float(self.target_frequency_hz)
        if not np.isfinite(sr) or sr <= 0:
            sr = 48000.0
        if not np.isfinite(f_req) or f_req <= 0:
            f_req = 1000.0

        # Choose integer period length to avoid drift: actual f = sr / N.
        n = int(np.round(sr / f_req))
        n = max(n, 16)
        # Keep plot/CPU sane.
        n = min(n, int(sr * 2))  # <= 2 seconds per period

        self.period_samples = n
        self.actual_frequency_hz = sr / float(n)

        k = np.arange(n, dtype=np.float64)
        self.memory_map = float(self.amplitude) * np.cos(2.0 * np.pi * k / float(n))

        # Allocate ring: at least 2 seconds or 2 periods.
        ring_len = int(max(2 * sr, 2 * n))
        self._ring = np.zeros((ring_len, 2), dtype=np.float64)
        self._ring_abs = np.zeros((ring_len,), dtype=np.int64)
        self._ring_pos = 0
        self._ring_valid = 0

        self.aligned_dutin = np.zeros((n,), dtype=np.float64)

        self.normalized_dutin = np.zeros((n,), dtype=np.float64)
        self.error_signal = np.zeros((n,), dtype=np.float64)

    def start_analysis(self):
        if self.is_running:
            return

        self.is_running = True
        self.global_sample_counter = 0
        self._rebuild_memory_map()

        def callback(indata, outdata, frames, time, status):
            if status:
                # keep consistent with other modules
                print(status)

            abs_start = int(self.global_sample_counter)

            # --- Input capture (REF=R, DUTIN=L) ---
            if indata is not None and indata.size:
                if indata.shape[1] >= 2:
                    in2 = indata[:, :2]
                else:
                    in2 = np.column_stack((indata[:, 0], indata[:, 0]))

                ring_len = int(self._ring.shape[0])
                w = int(self._ring_pos)

                if w + frames <= ring_len:
                    self._ring[w : w + frames] = in2
                    self._ring_abs[w : w + frames] = np.arange(abs_start, abs_start + frames, dtype=np.int64)
                else:
                    first = ring_len - w
                    second = frames - first
                    self._ring[w:] = in2[:first]
                    self._ring_abs[w:] = np.arange(abs_start, abs_start + first, dtype=np.int64)
                    self._ring[:second] = in2[first:]
                    self._ring_abs[:second] = np.arange(abs_start + first, abs_start + frames, dtype=np.int64)

                self._ring_pos = (w + frames) % ring_len
                self._ring_valid = min(ring_len, int(self._ring_valid) + int(frames))

            # --- Output generation from memory map (stereo) ---
            outdata.fill(0)
            n = int(self.period_samples)
            if n > 0 and self.memory_map.size == n:
                idx = (np.arange(frames, dtype=np.int64) + abs_start) % n
                sig = self.memory_map[idx.astype(np.intp, copy=False)]
                if outdata.shape[1] >= 1:
                    outdata[:, 0] = sig
                if outdata.shape[1] >= 2:
                    outdata[:, 1] = sig

            self.global_sample_counter += int(frames)

        self.callback_id = self.audio_engine.register_callback(callback)

    def stop_analysis(self):
        if not self.is_running:
            return
        if self.callback_id is not None:
            self.audio_engine.unregister_callback(self.callback_id)
            self.callback_id = None
        self.is_running = False

    def _get_recent_ring(self, n_samples: int):
        if self._ring is None or self._ring_abs is None:
            return None

        ring_len = int(self._ring.shape[0])
        valid = int(self._ring_valid)
        if valid <= 0:
            return None

        n = int(min(n_samples, valid, ring_len))
        end = int(self._ring_pos)
        start = (end - n) % ring_len

        if start < end:
            x = self._ring[start:end].copy()
            sidx = self._ring_abs[start:end].copy()
        else:
            x = np.vstack((self._ring[start:], self._ring[:end])).copy()
            sidx = np.concatenate((self._ring_abs[start:], self._ring_abs[:end])).copy()

        return x, sidx

    def update_alignment(self):
        if not self.is_running:
            return
        if self.period_samples <= 0:
            return

        # Need at least one full period (prefer more for stability).
        n = int(self.period_samples)
        req = int(min(max(2 * n, 8192), self._ring.shape[0]))
        got = self._get_recent_ring(req)
        if got is None:
            return

        x, abs_idx = got
        dut = x[:, 0]
        ref = x[:, 1]

        sr = float(self.audio_engine.sample_rate)
        f = float(self.actual_frequency_hz)

        abs_start = int(abs_idx[0]) if abs_idx.size else 0
        phi = estimate_ref_phase_offset_rad(ref, abs_start_sample=abs_start, sample_rate=sr, ref_frequency_hz=f)
        shift = phase_to_sample_shift(phi, n)
        aligned = fold_one_period_latest(dut, abs_idx, period_samples=n, shift_samples=shift)

        self.last_phase_rad = float(phi)
        self.last_shift_samples = int(shift)
        self.aligned_dutin = aligned

        # --- Phase 2: normalize DUTIN + compute error/metrics (no memory_map update) ---
        mem = np.asarray(self.memory_map, dtype=np.float64)
        if mem.size != n or aligned.size != n:
            return

        m = np.isfinite(aligned)
        filled = int(np.count_nonzero(m))
        self.last_filled_ratio = float(filled / float(n)) if n > 0 else 0.0
        if filled < int(0.9 * n):
            self.normalized_dutin = np.full((n,), np.nan, dtype=np.float64)
            self.error_signal = np.full((n,), np.nan, dtype=np.float64)
            self.last_dc_offset = 0.0
            self.last_gain = 1.0
            self.last_rms_error = float('nan')
            self.last_rms_error_ratio = float('nan')
            self.last_thd_db = float('nan')
            self.last_thd_percent = float('nan')
            return

        dut_f = aligned[m]
        dc = float(np.mean(dut_f))
        dut0 = aligned - dc

        # Memory map is already cosine-like; keep symmetry by removing any numerical DC.
        mem_dc = float(np.mean(mem)) if mem.size else 0.0
        mem0 = mem - mem_dc

        # Level match: find gain minimizing ||gain*dut0 - mem0|| over finite bins.
        denom = float(np.dot(dut0[m], dut0[m]))
        if not np.isfinite(denom) or denom <= 1e-24:
            gain = 0.0
        else:
            gain = float(np.dot(dut0[m], mem0[m]) / denom)
            if not np.isfinite(gain):
                gain = 0.0

        dut_norm = dut0 * gain
        err = dut_norm - mem0

        self.normalized_dutin = dut_norm
        self.error_signal = err
        self.last_dc_offset = dc
        self.last_gain = gain

        rms_e = self._nan_rms(err)
        rms_ref = self._nan_rms(mem0)
        self.last_rms_error = float(rms_e)
        self.last_rms_error_ratio = float(rms_e / (rms_ref + 1e-15)) if np.isfinite(rms_e) and np.isfinite(rms_ref) else float('nan')

        # THD from one period (coherent bins). Use DC-removed & level-matched waveform.
        thd_percent, thd_db = self._compute_thd_from_one_period(dut_norm)
        self.last_thd_percent = float(thd_percent)
        self.last_thd_db = float(thd_db)


class SelfCalibratingDistortionExtractorWidget(QWidget):
    def __init__(self, module: SelfCalibratingDistortionExtractor):
        super().__init__()
        self.module = module

        layout = QVBoxLayout(self)

        # Controls
        control_box = QGroupBox(tr("Generator / Sync"))
        form = QFormLayout(control_box)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(1.0, 20000.0)
        self.freq_spin.setDecimals(3)
        self.freq_spin.setValue(float(self.module.target_frequency_hz))
        self.freq_spin.setSuffix(" Hz")

        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0.0, 1.0)
        self.amp_spin.setDecimals(3)
        self.amp_spin.setSingleStep(0.01)
        self.amp_spin.setValue(float(self.module.amplitude))

        self.info_label = QLabel(tr("Idle"))

        self.error_only_check = QCheckBox(tr("Error only"))
        self.error_only_check.setChecked(False)

        form.addRow(tr("Frequency"), self.freq_spin)
        form.addRow(tr("Amplitude"), self.amp_spin)
        form.addRow(tr("Plot"), self.error_only_check)
        form.addRow(tr("Info"), self.info_label)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton(tr("Start"))
        self.stop_btn = QPushButton(tr("Stop"))
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        form.addRow(btn_row)

        layout.addWidget(control_box)

        # Plot
        self.plot = pg.PlotWidget(title=tr("Memory Map vs DUTIN (Normalized)"))
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel('bottom', tr('Sample'), units='')
        self.plot.setLabel('left', tr('Amplitude'), units='')

        self.curve_mem = self.plot.plot(pen=pg.mkPen('y', width=2), name=tr('Memory'))
        self.curve_dut = self.plot.plot(pen=pg.mkPen('c', width=1), name=tr('DUTIN (Norm)'))
        self.curve_err = self.plot.plot(pen=pg.mkPen('m', width=1), name=tr('Error e[n]'))

        layout.addWidget(self.plot, stretch=1)

        # Timer for UI updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(50)  # 20 Hz

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.error_only_check.toggled.connect(self._apply_plot_mode)

        self._apply_plot_mode()

    def _apply_plot_mode(self):
        err_only = bool(self.error_only_check.isChecked())
        self.curve_mem.setVisible(not err_only)
        self.curve_dut.setVisible(not err_only)
        self.curve_err.setVisible(True)

    def _on_start(self):
        self.module.target_frequency_hz = float(self.freq_spin.value())
        self.module.amplitude = float(self.amp_spin.value())
        self.module.start_analysis()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start()

    def _on_stop(self):
        self.timer.stop()
        self.module.stop_analysis()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.info_label.setText(tr("Idle"))

    def _tick(self):
        try:
            self.module.update_alignment()
        except Exception as e:
            self.info_label.setText(tr("Error: {0}").format(str(e)))
            return

        n = int(self.module.period_samples)
        if n <= 0 or self.module.memory_map.size != n or self.module.aligned_dutin.size != n:
            return

        x = np.arange(n, dtype=np.float64)
        self.curve_mem.setData(x, self.module.memory_map)

        dut = self.module.normalized_dutin
        err = self.module.error_signal

        dut_plot = np.nan_to_num(dut, nan=0.0) if np.any(~np.isfinite(dut)) else dut
        err_plot = np.nan_to_num(err, nan=0.0) if np.any(~np.isfinite(err)) else err

        self.curve_dut.setData(x, dut_plot)
        self.curve_err.setData(x, err_plot)

        rms_ratio = float(self.module.last_rms_error_ratio)
        rms_pct = 100.0 * rms_ratio if np.isfinite(rms_ratio) else float('nan')
        thd_db = float(self.module.last_thd_db)
        thd_pct = float(self.module.last_thd_percent)

        self.info_label.setText(
            tr("N={0}, f={1:.3f} Hz, shift={2} samp, fill={3:.0f}% | gain={4:.4f}, DC={5:+.4f} | RMSerr={6:.4e} ({7:.3f}%), THD={8:.2f}% ({9:.1f} dB)").format(
                int(self.module.period_samples),
                float(self.module.actual_frequency_hz),
                int(self.module.last_shift_samples),
                float(100.0 * float(self.module.last_filled_ratio)),
                float(self.module.last_gain),
                float(self.module.last_dc_offset),
                float(self.module.last_rms_error) if np.isfinite(self.module.last_rms_error) else float('nan'),
                float(rms_pct) if np.isfinite(rms_pct) else float('nan'),
                float(thd_pct) if np.isfinite(thd_pct) else float('nan'),
                float(thd_db) if np.isfinite(thd_db) else float('nan'),
            )
        )

    def closeEvent(self, event):
        try:
            self._on_stop()
        except Exception:
            pass
        super().closeEvent(event)
