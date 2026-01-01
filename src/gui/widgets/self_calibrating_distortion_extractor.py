import argparse
import numpy as np
import pyqtgraph as pg
import time

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
    QTabWidget,
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

        # Conservative memory-map correction (partial, sliding updates)
        self.map_update_enabled = False
        self.map_update_mu = 1e-4
        self.map_update_window_pct = 10.0
        # Noise-robust guard: larger => less sensitive to noise
        self.map_update_guard_z = 3.0
        self._map_update_pos = 0
        self._map_update_best_rms_error = np.inf
        self._map_update_bad_count = 0
        self.map_update_stop_reason = ""

        # Update diagnostics (for debugging)
        self.map_update_attempts = 0
        self.map_update_accepts = 0
        self.map_update_reverts = 0
        self.last_map_update_delta_rms = float('nan')
        self.last_map_update_win_start = 0
        self.last_map_update_win_len = 0
        self.last_map_update_nw = 0
        self.last_map_update_sigma2 = float('nan')
        self.last_map_update_dsse = float('nan')
        self.last_map_update_sse_std = float('nan')

    @staticmethod
    def _wrap_indices(start: int, length: int, n: int) -> np.ndarray:
        if n <= 0 or length <= 0:
            return np.zeros((0,), dtype=np.int64)
        return (np.arange(length, dtype=np.int64) + int(start)) % int(n)

    def _maybe_update_memory_map(self):
        if not bool(self.map_update_enabled):
            return

        n = int(self.period_samples)
        if n <= 0 or self.memory_map.size != n:
            return

        if self.normalized_dutin.size != n or self.error_signal.size != n:
            return

        if not np.isfinite(self.last_filled_ratio) or float(self.last_filled_ratio) < 0.95:
            return

        mu = float(self.map_update_mu)
        if not np.isfinite(mu) or mu <= 0.0:
            return

        # Keep mu conservative.
        mu = float(min(mu, 1e-2))

        window_pct = float(self.map_update_window_pct)
        if not np.isfinite(window_pct) or window_pct <= 0.0:
            return
        window_pct = float(min(window_pct, 100.0))

        win = int(max(16, min(n, int(np.round(n * window_pct / 100.0)))))
        stride = int(max(1, win // 2))

        idx = self._wrap_indices(self._map_update_pos, win, n).astype(np.intp, copy=False)
        self.last_map_update_win_start = int(self._map_update_pos)
        self.last_map_update_win_len = int(win)
        self._map_update_pos = int((int(self._map_update_pos) + stride) % n)

        # Build error relative to current memory map with DC removed (matching Phase 2).
        mem = np.asarray(self.memory_map, dtype=np.float64)
        mem_dc = float(np.mean(mem)) if mem.size else 0.0
        mem0 = mem - mem_dc

        dut = np.asarray(self.normalized_dutin, dtype=np.float64)
        m = np.isfinite(dut) & np.isfinite(mem0)
        if not np.any(m[idx]):
            return

        err = dut - mem0
        rms_before = self._nan_rms(err)

        # Variance-aware (noise-robust) acceptance criterion uses SSE change in the updated window.
        seg_m = m[idx]
        n_w = int(np.count_nonzero(seg_m))
        if n_w < 8:
            return
        err_w = err[idx][seg_m]
        sse_before = float(np.dot(err_w, err_w))

        # Estimate residual noise variance from the full-period residual.
        sigma2 = float(np.nanvar(err[m])) if np.any(m) else float('nan')
        if not np.isfinite(sigma2) or sigma2 < 0.0:
            sigma2 = float(rms_before * rms_before) if np.isfinite(rms_before) else 0.0

        self.last_map_update_nw = int(n_w)
        self.last_map_update_sigma2 = float(sigma2)

        z = float(self.map_update_guard_z)
        if not np.isfinite(z) or z <= 0.0:
            z = 3.0
        z = float(min(max(z, 0.5), 10.0))
        # For Gaussian noise, SSE stddev ~ sqrt(2N)*sigma^2
        sse_std = float(np.sqrt(2.0 * float(n_w)) * sigma2)
        self.last_map_update_sse_std = float(sse_std)

        # Count this as an update attempt (all preconditions satisfied)
        self.map_update_attempts += 1

        # Save old segment for rollback.
        old_seg = mem[idx].copy()

        delta = np.zeros((win,), dtype=np.float64)
        delta[seg_m] = mu * err[idx][seg_m]

        # Keep DC from drifting by forcing zero-mean delta within the updated window.
        if np.any(seg_m):
            delta_mean = float(np.mean(delta[seg_m]))
            if np.isfinite(delta_mean):
                delta[seg_m] -= delta_mean

        mem[idx] = mem[idx] + delta

        # Safety: hard clip to audio-safe range.
        np.clip(mem, -1.0, 1.0, out=mem)

        # Commit tentative update.
        self.memory_map = mem

        # Re-evaluate error on the same captured period; reject if it got worse.
        mem_dc2 = float(np.mean(mem)) if mem.size else 0.0
        mem02 = mem - mem_dc2
        err2 = dut - mem02
        rms_after = self._nan_rms(err2)

        err2_w = err2[idx][seg_m]
        sse_after = float(np.dot(err2_w, err2_w))
        d_sse = float(sse_after - sse_before)
        self.last_map_update_dsse = float(d_sse)

        # Update stored residual to match the current memory map.
        self.error_signal = err2
        self.last_rms_error = float(rms_after)
        rms_ref2 = self._nan_rms(mem02)
        self.last_rms_error_ratio = float(rms_after / (rms_ref2 + 1e-15)) if np.isfinite(rms_after) and np.isfinite(rms_ref2) else float('nan')

        # Reject only if the worsening is statistically significant relative to residual variance.
        # This avoids stopping on small noise-induced fluctuations.
        if np.isfinite(d_sse) and d_sse > (z * sse_std):
            # Roll back; stop only after repeated significant worsenings.
            mem[idx] = old_seg
            self.memory_map = mem

            self.map_update_reverts += 1
            self.last_map_update_delta_rms = float(self._nan_rms(delta[seg_m])) if np.any(seg_m) else float('nan')

            # Restore residual metrics for the reverted map.
            mem_dc3 = float(np.mean(mem)) if mem.size else 0.0
            mem03 = mem - mem_dc3
            err3 = dut - mem03
            rms3 = self._nan_rms(err3)
            self.error_signal = err3
            self.last_rms_error = float(rms3)
            rms_ref3 = self._nan_rms(mem03)
            self.last_rms_error_ratio = float(rms3 / (rms_ref3 + 1e-15)) if np.isfinite(rms3) and np.isfinite(rms_ref3) else float('nan')

            self._map_update_bad_count += 1
            if self._map_update_bad_count >= 5:
                self.map_update_enabled = False
                self.map_update_stop_reason = "Significant residual increase (auto-stopped)"
                self._map_update_bad_count = 0
            return

        # Not significantly worse: gradually forgive previous bad counts.
        if self._map_update_bad_count > 0:
            self._map_update_bad_count -= 1

        self.map_update_accepts += 1
        self.last_map_update_delta_rms = float(self._nan_rms(delta[seg_m])) if np.any(seg_m) else float('nan')

        # Track best observed RMS error (informational; no stop here to avoid noise sensitivity)
        if np.isfinite(rms_after):
            self._map_update_best_rms_error = float(min(self._map_update_best_rms_error, float(rms_after)))

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
        self._map_update_pos = 0
        self._map_update_best_rms_error = np.inf
        self._map_update_bad_count = 0
        self.map_update_stop_reason = ""

        self.map_update_attempts = 0
        self.map_update_accepts = 0
        self.map_update_reverts = 0
        self.last_map_update_delta_rms = float('nan')
        self.last_map_update_win_start = 0
        self.last_map_update_win_len = 0
        self.last_map_update_nw = 0
        self.last_map_update_sigma2 = float('nan')
        self.last_map_update_dsse = float('nan')
        self.last_map_update_sse_std = float('nan')
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

        # Conservative partial update of memory map (optional)
        self._maybe_update_memory_map()


class SelfCalibratingDistortionExtractorWidget(QWidget):
    def __init__(self, module: SelfCalibratingDistortionExtractor):
        super().__init__()
        self.module = module

        # Residual plot averaging state (kept in UI to avoid touching analysis)
        self._ema_err = None

        # Residual time-series (RMS) history for debugging auto-stop
        self._ts_t0 = None
        self._ts_t = []
        self._ts_rms = []
        self._ts_enabled = []
        self._ts_max_len = 2000
        self._prev_update_enabled = False
        self._ts_discard_initial_s = 1.0

        layout = QVBoxLayout(self)

        # Controls (tabbed to keep UI compact)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        main_tab = QWidget()
        avg_tab = QWidget()
        update_tab = QWidget()

        tabs.addTab(main_tab, tr("Main"))
        tabs.addTab(avg_tab, tr("Averaging"))
        tabs.addTab(update_tab, tr("Update"))

        main_form = QFormLayout(main_tab)
        avg_form = QFormLayout(avg_tab)
        update_form = QFormLayout(update_tab)

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

        self.avg_check = QCheckBox(tr("Avg residual"))
        self.avg_check.setChecked(False)

        self.avg_tau_spin = QDoubleSpinBox()
        self.avg_tau_spin.setRange(0.05, 10.0)
        self.avg_tau_spin.setDecimals(3)
        self.avg_tau_spin.setSingleStep(0.05)
        self.avg_tau_spin.setValue(0.5)
        self.avg_tau_spin.setSuffix(" s")

        # Main tab rows
        main_form.addRow(tr("Frequency"), self.freq_spin)
        main_form.addRow(tr("Amplitude"), self.amp_spin)
        main_form.addRow(tr("Plot"), self.error_only_check)
        main_form.addRow(tr("Info"), self.info_label)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton(tr("Start"))
        self.stop_btn = QPushButton(tr("Stop"))
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        main_form.addRow(btn_row)

        # Averaging tab rows
        avg_form.addRow(tr("Avg residual"), self.avg_check)
        avg_form.addRow(tr("Avg tau"), self.avg_tau_spin)

        # Update tab rows (no phase naming)
        self.map_update_enable_check = QCheckBox(tr("Enable memory-map update"))
        self.map_update_enable_check.setChecked(bool(self.module.map_update_enabled))

        self.map_update_mu_spin = QDoubleSpinBox()
        self.map_update_mu_spin.setRange(1e-8, 1e-2)
        self.map_update_mu_spin.setDecimals(8)
        self.map_update_mu_spin.setSingleStep(1e-5)
        self.map_update_mu_spin.setValue(float(self.module.map_update_mu))

        self.map_update_window_spin = QDoubleSpinBox()
        self.map_update_window_spin.setRange(1.0, 100.0)
        self.map_update_window_spin.setDecimals(1)
        self.map_update_window_spin.setSingleStep(1.0)
        self.map_update_window_spin.setValue(float(self.module.map_update_window_pct))
        self.map_update_window_spin.setSuffix(" %")

        self.map_update_status_label = QLabel(tr("Disabled"))

        self.map_update_debug_label = QLabel(tr(""))
        self.map_update_debug_label.setWordWrap(True)

        update_form.addRow(tr("Enable"), self.map_update_enable_check)
        update_form.addRow(tr("μ"), self.map_update_mu_spin)
        update_form.addRow(tr("Window"), self.map_update_window_spin)
        update_form.addRow(tr("Status"), self.map_update_status_label)
        update_form.addRow(tr("Stats"), self.map_update_debug_label)

        # Plots (tabbed)
        plot_tabs = QTabWidget()
        layout.addWidget(plot_tabs, stretch=1)

        waveform_tab = QWidget()
        residual_tab = QWidget()
        plot_tabs.addTab(waveform_tab, tr("Waveform"))
        plot_tabs.addTab(residual_tab, tr("Residual"))

        waveform_layout = QVBoxLayout(waveform_tab)
        residual_layout = QVBoxLayout(residual_tab)

        # Plot
        self.plot = pg.PlotWidget(title=tr("Memory Map vs DUTIN (Normalized)"))
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel('bottom', tr('Sample'), units='')
        self.plot.setLabel('left', tr('Amplitude'), units='')

        self.curve_mem = self.plot.plot(pen=pg.mkPen('y', width=2), name=tr('Memory'))
        self.curve_dut = self.plot.plot(pen=pg.mkPen('c', width=1), name=tr('DUTIN (Norm)'))
        self.curve_err = self.plot.plot(pen=pg.mkPen('m', width=1), name=tr('Error e[n]'))

        waveform_layout.addWidget(self.plot)

        # Residual time-series plot (RMS over time)
        self.res_plot = pg.PlotWidget(title=tr("Residual (RMS) vs Time"))
        self.res_plot.showGrid(x=True, y=True, alpha=0.25)
        self.res_plot.setLabel('bottom', tr('Time'), units='s')
        self.res_plot.setLabel('left', tr('RMS Error'), units='')

        self.res_curve = self.res_plot.plot(pen=pg.mkPen('m', width=1), name=tr('RMSerr'))
        self.res_enable_curve = self.res_plot.plot(pen=pg.mkPen('w', width=1), name=tr('Update enabled (scaled)'))
        self.res_stop_scatter = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush('r'), size=7)
        self.res_plot.addItem(self.res_stop_scatter)

        residual_layout.addWidget(self.res_plot)

        # Timer for UI updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(50)  # 20 Hz (may be lowered when averaging)

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.error_only_check.toggled.connect(self._apply_plot_mode)

        self.avg_check.toggled.connect(self._apply_avg_settings)
        self.avg_tau_spin.valueChanged.connect(self._apply_avg_settings)

        self.map_update_enable_check.toggled.connect(self._apply_update_settings)
        self.map_update_mu_spin.valueChanged.connect(self._apply_update_settings)
        self.map_update_window_spin.valueChanged.connect(self._apply_update_settings)

        self._apply_plot_mode()
        self._apply_avg_settings()
        self._apply_update_settings()

    def _apply_update_settings(self):
        self.module.map_update_mu = float(self.map_update_mu_spin.value())
        self.module.map_update_window_pct = float(self.map_update_window_spin.value())

        want_enable = bool(self.map_update_enable_check.isChecked())
        if want_enable and not bool(self.module.map_update_enabled):
            self.module.map_update_stop_reason = ""
            self.module._map_update_bad_count = 0
            self.module._map_update_best_rms_error = np.inf
        self.module.map_update_enabled = want_enable

    def _apply_plot_mode(self):
        err_only = bool(self.error_only_check.isChecked())
        self.curve_mem.setVisible(not err_only)
        self.curve_dut.setVisible(not err_only)
        self.curve_err.setVisible(True)

    def _apply_avg_settings(self):
        # Keep UI responsive: when averaging is enabled, reduce update rate.
        if bool(self.avg_check.isChecked()):
            self.timer.setInterval(150)  # ~6.7 Hz
        else:
            self.timer.setInterval(50)  # 20 Hz

        # If settings change, reset EMA so the user sees immediate effect.
        self._ema_err = None

    def _on_start(self):
        self.module.target_frequency_hz = float(self.freq_spin.value())
        self.module.amplitude = float(self.amp_spin.value())
        self._apply_update_settings()
        self.module.start_analysis()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._ema_err = None

        self._ts_t0 = time.perf_counter()
        self._ts_t = []
        self._ts_rms = []
        self._ts_enabled = []
        self._prev_update_enabled = bool(self.module.map_update_enabled)
        self.res_stop_scatter.setData([])
        self.timer.start()

    def _on_stop(self):
        self.timer.stop()
        self.module.stop_analysis()
        self.module.map_update_enabled = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.info_label.setText(tr("Idle"))
        self._ema_err = None

    def _tick(self):
        try:
            self.module.update_alignment()
        except Exception as e:
            self.info_label.setText(tr("Error: {0}").format(str(e)))
            return

        if bool(self.module.map_update_enabled):
            self.map_update_status_label.setText(tr("Enabled"))
        else:
            r = str(self.module.map_update_stop_reason) if self.module.map_update_stop_reason else ""
            self.map_update_status_label.setText(tr("Stopped: {0}").format(r) if r else tr("Disabled"))

        # Update diagnostics text (helps confirm whether updates are actually happening)
        self.map_update_debug_label.setText(
            tr("att={0}, ok={1}, rev={2} | Δrms={3:.2e} | win={4}+{5} (nw={6}) | σ²={7:.2e} | dSSE={8:.2e}, thr={9:.2e}").format(
                int(self.module.map_update_attempts),
                int(self.module.map_update_accepts),
                int(self.module.map_update_reverts),
                float(self.module.last_map_update_delta_rms) if np.isfinite(self.module.last_map_update_delta_rms) else float('nan'),
                int(self.module.last_map_update_win_start),
                int(self.module.last_map_update_win_len),
                int(self.module.last_map_update_nw),
                float(self.module.last_map_update_sigma2) if np.isfinite(self.module.last_map_update_sigma2) else float('nan'),
                float(self.module.last_map_update_dsse) if np.isfinite(self.module.last_map_update_dsse) else float('nan'),
                float(self.module.map_update_guard_z) * float(self.module.last_map_update_sse_std)
                if np.isfinite(self.module.last_map_update_sse_std)
                else float('nan'),
            )
        )

        # --- Residual time-series logging (RMS error) ---
        if self._ts_t0 is None:
            self._ts_t0 = time.perf_counter()

        raw_t = float(time.perf_counter() - float(self._ts_t0))
        rms = float(self.module.last_rms_error)
        enabled = bool(self.module.map_update_enabled)

        # Always update previous state so stop detection doesn't glitch.
        prev_enabled = bool(self._prev_update_enabled)
        self._prev_update_enabled = enabled

        # Discard initial transient window (do not log/plot)
        if np.isfinite(raw_t) and raw_t < float(self._ts_discard_initial_s):
            return

        t = float(raw_t - float(self._ts_discard_initial_s))

        self._ts_t.append(t)
        self._ts_rms.append(rms)
        self._ts_enabled.append(1.0 if enabled else 0.0)

        if len(self._ts_t) > int(self._ts_max_len):
            drop = len(self._ts_t) - int(self._ts_max_len)
            self._ts_t = self._ts_t[drop:]
            self._ts_rms = self._ts_rms[drop:]
            self._ts_enabled = self._ts_enabled[drop:]

        # Mark the moment auto-stop happens (enabled -> disabled with a reason)
        if prev_enabled and (not enabled) and bool(self.module.map_update_stop_reason):
            spots = self.res_stop_scatter.data
            # Append a new stop marker point
            new_pt = {'pos': (t, rms)}
            if spots is None:
                self.res_stop_scatter.setData([new_pt])
            else:
                # ScatterPlotItem stores spots internally; simplest is to rebuild from current displayed points.
                existing = []
                try:
                    for s in self.res_stop_scatter.points():
                        p = s.pos()
                        existing.append({'pos': (float(p.x()), float(p.y()))})
                except Exception:
                    existing = []
                existing.append(new_pt)
                self.res_stop_scatter.setData(existing)

        # Plot time-series
        if self._ts_t:
            tt = np.asarray(self._ts_t, dtype=np.float64)
            rr = np.asarray(self._ts_rms, dtype=np.float64)
            en = np.asarray(self._ts_enabled, dtype=np.float64)

            self.res_curve.setData(tt, rr)

            # Scale enable flag to ~10% of current RMS range for visibility
            rmax = float(np.nanmax(rr)) if rr.size else 0.0
            scale = 0.1 * rmax if np.isfinite(rmax) and rmax > 0 else 1.0
            self.res_enable_curve.setData(tt, en * scale)

        n = int(self.module.period_samples)
        if n <= 0 or self.module.memory_map.size != n or self.module.aligned_dutin.size != n:
            return

        x = np.arange(n, dtype=np.float64)
        self.curve_mem.setData(x, self.module.memory_map)

        dut = self.module.normalized_dutin
        err = self.module.error_signal

        # Optional lightweight time averaging for residual plot (EMA).
        if bool(self.avg_check.isChecked()):
            tau = float(self.avg_tau_spin.value())
            dt = float(self.timer.interval()) / 1000.0
            if np.isfinite(tau) and tau > 0.0 and np.isfinite(dt) and dt > 0.0:
                alpha = float(1.0 - np.exp(-dt / tau))
            else:
                alpha = 1.0

            if self._ema_err is None or (isinstance(self._ema_err, np.ndarray) and self._ema_err.size != err.size):
                self._ema_err = np.array(err, dtype=np.float64, copy=True)
            else:
                cur = np.asarray(err, dtype=np.float64)
                ema = np.asarray(self._ema_err, dtype=np.float64)
                m = np.isfinite(cur)
                if np.any(m):
                    ema[m] = (1.0 - alpha) * ema[m] + alpha * cur[m]
                # Preserve NaNs for bins that are not yet valid.
                ema[~m] = np.nan
                self._ema_err = ema

            err = self._ema_err

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
