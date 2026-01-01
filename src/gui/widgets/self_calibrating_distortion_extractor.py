import argparse
from collections import deque
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
        self._phi_filt = None
        self._phi_lpf_alpha = 0.25
        self._shift_hist = deque(maxlen=7)
        self._align_is_stable = True

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

        # Memory-map update: simplest behavior.
        # When enabled, the memory map is rewritten back to the ideal cosine waveform,
        # starting from index 0 and advancing a write pointer (sliding update).
        self.map_update_enabled = False
        self.map_update_stop_reason = ""
        self._ideal_unit_cos = np.zeros((0,), dtype=np.float64)
        self._ideal_update_pos = 0
        self._ideal_update_step = 256
        # Blend factor for "learn" updates (smaller = slower). Kept internal.
        self._map_learn_alpha = 0.1
        # Additional smoothing on the learned waveform itself (damps occasional bad folds)
        self._learned_ema = None
        self._learned_ema_alpha = 0.25

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

        self.map_update_last_action = ""
        self.map_update_last_skip_reason = ""
        self._map_update_initial_map = None
        self.map_update_map_diff_rms = float('nan')
        self.map_update_map_diff_max = float('nan')

        # Simple deterministic update counter ("100% counter" style)
        self.map_update_counter = 0

    def _refresh_map_update_diff_metrics(self):
        n = int(self.period_samples)
        if n <= 0 or self.memory_map.size != n:
            self.map_update_map_diff_rms = float('nan')
            self.map_update_map_diff_max = float('nan')
            return
        if self._map_update_initial_map is None or getattr(self._map_update_initial_map, 'size', 0) != n:
            self.map_update_map_diff_rms = float('nan')
            self.map_update_map_diff_max = float('nan')
            return
        d = np.asarray(self.memory_map, dtype=np.float64) - np.asarray(self._map_update_initial_map, dtype=np.float64)
        self.map_update_map_diff_rms = float(np.sqrt(np.mean(d * d)))
        self.map_update_map_diff_max = float(np.max(np.abs(d)))

    def _update_memory_map_simple(self):
        """Aggressive memory-map update: learn the latest one-period waveform.

        When enabled, this replaces the memory map with the most recent folded
        1-period input waveform (DUTIN), after:
        - DC removal
        - RMS normalization to the requested output amplitude

        This is intentionally "bold" so residual/output changes are obvious.
        """
        self.map_update_last_action = ""
        self.map_update_last_skip_reason = ""

        if not bool(self.map_update_enabled):
            return

        if not bool(self._align_is_stable):
            self.map_update_last_action = "skip"
            self.map_update_last_skip_reason = "unstable align"
            return

        n = int(self.period_samples)
        if n <= 0 or self.memory_map.size != n:
            self.map_update_last_action = "skip"
            self.map_update_last_skip_reason = "bad N/memory"
            return

        if self.aligned_dutin.size != n:
            self.map_update_last_action = "skip"
            self.map_update_last_skip_reason = "no aligned period"
            return

        # Check fill ratio
        if not np.isfinite(self.last_filled_ratio) or float(self.last_filled_ratio) < 0.95:
            self.map_update_last_action = "skip"
            self.map_update_last_skip_reason = "insufficient fill"
            return

        self.map_update_attempts += 1

        aligned = np.asarray(self.aligned_dutin, dtype=np.float64)
        m = np.isfinite(aligned)
        if not np.any(m):
            self.map_update_last_action = "skip"
            self.map_update_last_skip_reason = "aligned all NaN"
            return

        # DC removal on available samples
        dc = float(np.mean(aligned[m]))
        x0 = aligned - dc

        # RMS normalize to requested output amplitude (cosine peak = amplitude)
        rms = float(self._nan_rms(x0))
        if not np.isfinite(rms) or rms <= 1e-12:
            self.map_update_last_action = "skip"
            self.map_update_last_skip_reason = "rms invalid"
            return

        target_amp = float(self.amplitude)
        if not np.isfinite(target_amp) or target_amp <= 0.0:
            target_amp = 0.5
        target_rms = target_amp / float(np.sqrt(2.0))
        scale = float(target_rms / rms)

        alpha = float(self._map_learn_alpha)
        if not np.isfinite(alpha) or alpha <= 0.0:
            alpha = 0.05
        if alpha > 1.0:
            alpha = 1.0

        # Copy-and-swap: build a new map and replace reference.
        old = np.asarray(self.memory_map, dtype=np.float64)
        new_map = np.array(old, dtype=np.float64, copy=True)
        learned = x0 * scale

        # Smooth the learned waveform over time (EMA) to suppress occasional bad folds.
        eta = float(self._learned_ema_alpha)
        if not np.isfinite(eta) or eta <= 0.0:
            eta = 0.25
        if eta > 1.0:
            eta = 1.0
        if self._learned_ema is None or (isinstance(self._learned_ema, np.ndarray) and self._learned_ema.size != learned.size):
            self._learned_ema = np.array(learned, dtype=np.float64, copy=True)
        else:
            ema = np.asarray(self._learned_ema, dtype=np.float64)
            mm = np.isfinite(learned)
            if np.any(mm):
                ema[mm] = (1.0 - eta) * ema[mm] + eta * learned[mm]
            ema[~mm] = np.nan
            self._learned_ema = ema

        learned_use = np.asarray(self._learned_ema, dtype=np.float64)
        # Slow down by blending toward the learned waveform.
        new_map[m] = (1.0 - alpha) * new_map[m] + alpha * learned_use[m]

        # Hard clip for safety
        np.clip(new_map, -2.0, 2.0, out=new_map)

        self.memory_map = new_map

        self.map_update_accepts += 1
        self.map_update_counter += 1
        self.map_update_last_action = f"learn#{int(self.map_update_counter)}"

        d = new_map - old
        self.last_map_update_delta_rms = float(self._nan_rms(d))
        self.last_map_update_win_start = 0
        self.last_map_update_win_len = int(n)
        self.last_map_update_nw = int(np.count_nonzero(m))
        self.last_map_update_sigma2 = float('nan')
        self.last_map_update_dsse = float('nan')
        self.last_map_update_sse_std = float('nan')

    def _maybe_update_memory_map(self):
        # Single entry point for memory-map update (kept minimal on purpose)
        self._update_memory_map_simple()

    @staticmethod
    def _wrap_indices(start: int, length: int, n: int) -> np.ndarray:
        if n <= 0 or length <= 0:
            return np.zeros((0,), dtype=np.int64)
        return (np.arange(length, dtype=np.int64) + int(start)) % int(n)

    def _update_memory_map_robust(self):
        # Backward-compatible alias: keep callers working, but use the minimal updater.
        self._update_memory_map_simple()

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
        self._ideal_unit_cos = np.cos(2.0 * np.pi * k / float(n))
        self.memory_map = float(self.amplitude) * np.asarray(self._ideal_unit_cos, dtype=np.float64)
        self._ideal_update_pos = 0
        self._learned_ema = None
        self._shift_hist.clear()
        self._phi_filt = None
        self._align_is_stable = True

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

        self.map_update_last_action = ""
        self.map_update_last_skip_reason = ""
        self.map_update_counter = 0
        self._learned_ema = None
        self._shift_hist.clear()
        self._phi_filt = None
        self._align_is_stable = True
        self._map_update_initial_map = None
        self.map_update_map_diff_rms = float('nan')
        self.map_update_map_diff_max = float('nan')
        self._rebuild_memory_map()

        # Baseline snapshot for confirming map changes
        if self.memory_map.size == int(self.period_samples):
            self._map_update_initial_map = np.array(self.memory_map, dtype=np.float64, copy=True)
        self._refresh_map_update_diff_metrics()

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
            mem = self.memory_map
            if n > 0 and getattr(mem, 'size', 0) == n:
                idx = (np.arange(frames, dtype=np.int64) + abs_start) % n
                sig = mem[idx.astype(np.intp, copy=False)]
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
        phi_raw = estimate_ref_phase_offset_rad(ref, abs_start_sample=abs_start, sample_rate=sr, ref_frequency_hz=f)
        if not np.isfinite(phi_raw):
            return

        # Low-pass filter phase (wrap-aware) to reduce occasional estimator jumps.
        beta = float(self._phi_lpf_alpha)
        if not np.isfinite(beta) or beta <= 0.0:
            beta = 0.25
        if beta > 1.0:
            beta = 1.0
        if self._phi_filt is None or (not np.isfinite(float(self._phi_filt))):
            phi = float(phi_raw)
        else:
            prev = float(self._phi_filt)
            dphi = float(((phi_raw - prev + np.pi) % (2.0 * np.pi)) - np.pi)
            phi = float(prev + beta * dphi)

        self._phi_filt = float(phi)

        shift_raw = int(phase_to_sample_shift(phi_raw, n))
        self._shift_hist.append(shift_raw)
        # Use median shift for folding; robust against single-tick outliers.
        shift = int(np.median(np.asarray(list(self._shift_hist), dtype=np.int64))) if self._shift_hist else int(phase_to_sample_shift(phi, n))

        # Mark alignment stability (used to gate learning updates)
        if len(self._shift_hist) >= 5:
            sh = np.asarray(list(self._shift_hist), dtype=np.int64)
            spread = int(np.max(sh) - np.min(sh))
            thresh = int(max(2, n // 32))
            self._align_is_stable = bool(spread <= thresh)
        else:
            self._align_is_stable = True

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

        # Memory-map update (optional) - minimal full-period update
        self._maybe_update_memory_map()
        self._refresh_map_update_diff_metrics()


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
        self._ts_mapdiff = []
        self._ts_upd_deltarms = []
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

        self.map_update_status_label = QLabel(tr("Disabled"))

        self.map_update_debug_label = QLabel(tr(""))
        self.map_update_debug_label.setWordWrap(True)

        update_form.addRow(tr("Enable"), self.map_update_enable_check)
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
        self.res_mapdiff_curve = self.res_plot.plot(pen=pg.mkPen('g', width=1), name=tr('Δmap (rms, scaled)'))
        self.res_upd_deltarms_curve = self.res_plot.plot(pen=pg.mkPen('y', width=1), name=tr('Δupdate (rms, scaled)'))
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

        self._apply_plot_mode()
        self._apply_avg_settings()
        self._apply_update_settings()

    def _apply_update_settings(self):
        want_enable = bool(self.map_update_enable_check.isChecked())
        if want_enable and not bool(self.module.map_update_enabled):
            self.module.map_update_stop_reason = ""
            self.module._ideal_update_pos = 0
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
        self._ts_mapdiff = []
        self._ts_upd_deltarms = []
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
            tr("{0}{1} | att={2}, ok={3}, rev={4} | Δmap(rms,max)={5:.2e},{6:.2e} | write={7}+{8} (nw={9})").format(
                str(self.module.map_update_last_action),
                ("/" + str(self.module.map_update_last_skip_reason)) if self.module.map_update_last_skip_reason else "",
                int(self.module.map_update_attempts),
                int(self.module.map_update_accepts),
                int(self.module.map_update_reverts),
                float(self.module.map_update_map_diff_rms) if np.isfinite(self.module.map_update_map_diff_rms) else float('nan'),
                float(self.module.map_update_map_diff_max) if np.isfinite(self.module.map_update_map_diff_max) else float('nan'),
                int(self.module.last_map_update_win_start),
                int(self.module.last_map_update_win_len),
                int(self.module.last_map_update_nw),
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
        self._ts_mapdiff.append(float(self.module.map_update_map_diff_rms))
        self._ts_upd_deltarms.append(float(self.module.last_map_update_delta_rms))

        if len(self._ts_t) > int(self._ts_max_len):
            drop = len(self._ts_t) - int(self._ts_max_len)
            self._ts_t = self._ts_t[drop:]
            self._ts_rms = self._ts_rms[drop:]
            self._ts_enabled = self._ts_enabled[drop:]
            self._ts_mapdiff = self._ts_mapdiff[drop:]
            self._ts_upd_deltarms = self._ts_upd_deltarms[drop:]

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
            md = np.asarray(self._ts_mapdiff, dtype=np.float64)
            du = np.asarray(self._ts_upd_deltarms, dtype=np.float64)

            self.res_curve.setData(tt, rr)

            # Scale enable flag to ~10% of current RMS range for visibility
            if rr.size and np.any(np.isfinite(rr)):
                rmax = float(np.nanmax(rr[np.isfinite(rr)]))
            else:
                rmax = 0.0
            scale = 0.1 * rmax if np.isfinite(rmax) and rmax > 0 else 1.0
            self.res_enable_curve.setData(tt, en * scale)

            # Scale map diff to be visible on the same axis (roughly the same scale as RMS)
            if md.size and np.any(np.isfinite(md)):
                mdv = float(np.nanmax(md[np.isfinite(md)]))
            else:
                mdv = 0.0
            md_scale = (0.1 * rmax / (mdv + 1e-30)) if np.isfinite(rmax) and rmax > 0 and np.isfinite(mdv) and mdv > 0 else 0.0
            self.res_mapdiff_curve.setData(tt, md * md_scale)

            # Scale per-step update magnitude similarly
            if du.size and np.any(np.isfinite(du)):
                duv = float(np.nanmax(du[np.isfinite(du)]))
            else:
                duv = 0.0
            du_scale = (0.1 * rmax / (duv + 1e-30)) if np.isfinite(rmax) and rmax > 0 and np.isfinite(duv) and duv > 0 else 0.0
            self.res_upd_deltarms_curve.setData(tt, du * du_scale)

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
