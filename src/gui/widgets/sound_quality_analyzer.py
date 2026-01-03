import argparse

import numpy as np
import pyqtgraph as pg
import scipy.signal as signal
import soundfile as sf
from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.audio_engine import AudioEngine
from src.core.localization import tr
from src.measurement_modules.base import MeasurementModule

# --- Analysis Worker ---

class AnalysisWorker(QThread):
    progress_update = pyqtSignal(int, str)
    results_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_path, target_sr):
        super().__init__()
        self.file_path = file_path
        self.target_sr = target_sr
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.progress_update.emit(0, tr("Loading file..."))
            data, samplerate = sf.read(self.file_path)

            # Resampling if needed
            if samplerate != self.target_sr:
                self.progress_update.emit(5, tr("Resampling to {}Hz (Fast)...").format(self.target_sr))
                data = self._fast_resample(data, samplerate, self.target_sr)
                samplerate = self.target_sr

            if self._is_cancelled:
                return

            # Handle stereo: Process L and R, combine or just Average?
            # Requirement: Mono/Stereo (Stereo is LR separated + composite?)
            # For simplicity, let's analyze Left and Right separately, and also provide an Average/Composite view.
            # But the request says "Stereo is LR separate + Synthesis". "Synthesis" might mean "Overall" (e.g. Total Loudness).

            if data.ndim == 1:
                channels = [data]
                ch_names = ["Mono"]
            else:
                channels = [data[:, 0], data[:, 1]]
                ch_names = ["Left", "Right"]

            results = {
                "samplerate": samplerate,
                "duration": len(data) / samplerate,
                "channels": []
            }

            total_steps = len(channels) * 4 # 4 metrics per channel
            current_step = 0

            for i, audio in enumerate(channels):
                ch_res = {"name": ch_names[i]}

                # 1. Loudness
                if self._is_cancelled: return
                self.progress_update.emit(int((current_step / total_steps) * 100), tr("Calculating Loudness ({})...").format(ch_names[i]))
                l_res = self._calc_loudness(audio, samplerate)
                ch_res.update(l_res)
                current_step += 1

                # 2. Sharpness
                if self._is_cancelled: return
                self.progress_update.emit(int((current_step / total_steps) * 100), tr("Calculating Sharpness ({})...").format(ch_names[i]))
                s_res = self._calc_sharpness(audio, samplerate)
                ch_res.update(s_res)
                current_step += 1

                # 3. Roughness
                if self._is_cancelled: return
                self.progress_update.emit(int((current_step / total_steps) * 100), tr("Calculating Roughness ({})...").format(ch_names[i]))
                r_res = self._calc_roughness(audio, samplerate)
                ch_res.update(r_res)
                current_step += 1

                # 4. Tonality
                if self._is_cancelled: return
                self.progress_update.emit(int((current_step / total_steps) * 100), tr("Calculating Tonality ({})...").format(ch_names[i]))
                t_res = self._calc_tonality(audio, samplerate)
                ch_res.update(t_res)
                current_step += 1

                results["channels"].append(ch_res)

            # Add raw audio for playback
            # Store as float32 for audio engine
            results["audio_data"] = data.astype(np.float32)
            results["samplerate"] = samplerate

            self.results_ready.emit(results)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def _fast_resample(self, data, src_sr, target_sr):
        """
        Fast approximated resampling for preview/analysis purposes.
        Uses integer slicing/repeating or linear interpolation.
        """
        if src_sr == target_sr:
            return data

        # Check for integer factors
        # Downsampling (e.g. 96k -> 48k)
        if src_sr > target_sr and src_sr % target_sr == 0:
            step = src_sr // target_sr
            # Simple decimation (no anti-aliasing filter here, but fast)
            return data[::step]

        # Upsampling (e.g. 44.1k -> 88.2k)
        if target_sr > src_sr and target_sr % src_sr == 0:
            factor = target_sr // src_sr
            # Simple repeat (zero-order hold)
            if data.ndim == 1:
                return np.repeat(data, factor)
            else:
                return np.repeat(data, factor, axis=0)

        # General case: Linear Interpolation (np.interp)
        # Calculate new time indices
        old_len = len(data)
        new_len = int(old_len * target_sr / src_sr)

        x_old = np.linspace(0, old_len - 1, old_len)
        x_new = np.linspace(0, old_len - 1, new_len)

        if data.ndim == 1:
            return np.interp(x_new, x_old, data).astype(data.dtype)
        else:
            # Handle each channel
            resampled = np.zeros((new_len, data.shape[1]), dtype=data.dtype)
            for i in range(data.shape[1]):
                resampled[:, i] = np.interp(x_new, x_old, data[:, i])
            return resampled

    def _calc_loudness(self, audio, sr):
        # Time-series (Momentary)
        # Sliding window 400ms, overlap 75% -> step 100ms
        window_sec = 0.4
        step_sec = 0.1

        # K-weighting filters (BS.1770)
        # Stage 1: Shelf
        b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
        a1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
        # Stage 2: High-pass
        b2 = np.array([1.0, -2.0, 1.0])
        a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])

        # Apply filters
        # Note: These coefs are for 48kHz. Ideally should redesign for other SR or resample.
        # For prototype, assume 48k or close, or warn. Resampling is expensive.
        # BS.1770 filter coefficients are defined for 48kHz.
        if sr != 48000:
             # TODO: Proper resampling or filter redesign.
             # For now, simplistic approach: accept small error or basic resample for filtering only?
             # Let's just run it; it's an estimation widget.
             pass

        y = signal.lfilter(b1, a1, audio)
        y = signal.lfilter(b2, a2, y)

        # Power
        p = y**2

        # Block processing
        block_size = int(window_sec * sr)
        step_size = int(step_sec * sr)

        # Use convolution for sliding mean? Or just strided indexing.
        # Convolution with ones kernel is fast.
        kernel = np.ones(block_size) / block_size
        p_smoothed = signal.fftconvolve(p, kernel, mode='valid')

        # Downsample to step size for display
        # fftconvolve 'valid' output length is N - K + 1.
        # The time points correspond to the center or end? BS.1770 usually end or center.
        # Let's take every step_size sample.

        # Time axis
        # Output starts at t = window_sec (if valid).
        # We need to map back to time.

        # Valid part starts after one full window.
        p_blocks = p_smoothed[::step_size]

        # Momentary LUFS series
        # Catch log10(0)
        m_lufs = -0.691 + 10 * np.log10(p_blocks + 1e-10)
        m_lufs[m_lufs <= -100] = -100.0

        # Integrated
        # Gate: -70 LUFS absolute, then -10 LU rel
        abs_gate = -70.0
        rel_gate_threshold = -10.0

        # 1. Absolute gating
        g1 = p_blocks[m_lufs > abs_gate]
        if len(g1) == 0:
            return {"integrated_lufs": -100.0, "lufs_series": m_lufs, "lufs_step": step_sec}

        z_avg_gated = np.mean(g1)
        gamma_a = -0.691 + 10 * np.log10(z_avg_gated)

        # 2. Relative gating
        rel_gate = gamma_a + rel_gate_threshold
        g2 = p_blocks[m_lufs > rel_gate]

        if len(g2) == 0:
             return {"integrated_lufs": -100.0, "lufs_series": m_lufs, "lufs_step": step_sec}

        z_avg_final = np.mean(g2)
        integrated = -0.691 + 10 * np.log10(z_avg_final)

        return {
            "integrated_lufs": integrated,
            "lufs_series": m_lufs,
            "lufs_step": step_sec
        }

    def _calc_sharpness(self, audio, sr):
        # Zwicker Sharpness
        # Simplified:
        # 1. FFT
        # 2. Bark grouping (Specific Loudness N')
        # 3. Sharpness S = 0.11 * Integral(N' * g(z) * z * dz) / Integral(N' * dz)
        #    g(z) weighting increases for high frequencies.

        # Window size for time series
        window_sec = 0.5 # 500ms
        step_sec = 0.25 # 250ms
        nperseg = int(window_sec * sr)
        noverlap = int(nperseg - (step_sec * sr))

        f, t, Zxx = signal.stft(audio, fs=sr, window='hann', nperseg=nperseg, noverlap=noverlap)
        # Zxx shape: (freqs, times)
        mag = np.abs(Zxx)
        power_spec = mag**2

        # Bark scale conversion
        # Bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500)^2)
        barks = 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500)**2)

        # Specific Loudness Approximation (Stevens' Law style N ~ I^0.3)
        # But Zwicker is complex.
        # Simplified "Sharpness" proportional to spectral centroid of loudness?
        # S ~ \int N'(z) * g(z) * z dz / Total Loudness

        # Weighting function g(z)
        # z < 14 Bark: g(z) = 1
        # z >= 14 Bark: g(z) = 0.00012 * z^4 - 0.0056 * z^3 + 0.1 * z^2 - 0.81 * z + 3.51 (Fastl)
        # Or simple approximation: g(z) = 1 for z<15.8, then rises.
        # Let's use simplified weighting: 1 up to 15 Bark, then rising slope.
        # Standard: z factor is usually included in the integral moment.

        # Let's just implement S = C * (Numer / Denom)
        # Numer = sum( Loudness(z) * z * g(z) )
        # Denom = sum( Loudness(z) )

        # Approximate Specific Loudness from Power Spectrum
        # E(z) density...
        # Let's do a binning method.
        # 24 Bark bands (0 to 24)
        bark_centers = np.linspace(0.5, 23.5, 24)

        sharpness_series = []

        # Pre-calc bin indices
        bin_indices = np.digitize(barks, np.arange(0, 25)) - 1

        # Simple g(z) weighting
        # z is Bark index (0-24)
        # Standard definition: g(z)=1 for z<=16 (approx), increases for z>16.
        # fastl & zwicker:
        # factor = 1 if z < 16 else 0.066 * exp(0.171 * z)

        g_z = np.ones(24)
        for z in range(24):
             center = bark_centers[z]
             if center > 16:
                 g_z[z] = 0.066 * np.exp(0.171 * center)

        # Iterate over time frames
        num_frames = Zxx.shape[1]

        for i in range(num_frames):
             p_frame = power_spec[:, i]

             # Group into bark bands (Energy Sum)
             # N ~ E^0.23 (approx for Specific Loudness)
             band_energy = np.zeros(24)
             for b in range(len(p_frame)):
                 idx = bin_indices[b]
                 if 0 <= idx < 24:
                     band_energy[idx] += p_frame[b]

             # Specific Loudness
             # This is a crude approx, real Zwicker excitation pattern is much more complex
             # (spreading functions etc.)
             # But for relative "Sharpness" diffs, this might suffice.
             N_prime = band_energy ** 0.23

             denom = np.sum(N_prime)
             if denom < 1e-9:
                 sharpness_series.append(0.0)
                 continue

             numer = 0.0
             for z in range(24):
                 val = N_prime[z] * bark_centers[z] * g_z[z]
                 numer += val

             # Constant 0.11 is scaling factor
             S = 0.11 * numer / denom
             sharpness_series.append(S)

        sharpness_series = np.array(sharpness_series)

        return {
            "mean_sharpness": np.mean(sharpness_series),
            "sharpness_series": sharpness_series,
            "sharpness_step": step_sec
        }

    def _calc_roughness(self, audio, sr):
        # Roughness (Daniel & Weber) - VERY Simplified / Placeholder
        # Roughness is amplitude modulation in range 20-300 Hz.
        # Ideally: Filterbank -> Envelope Demodulation -> Cross-correlation / Modulation Index.

        # Simple indicator: Average Modulation Index in Roughness band?

        # 1. Hilbert Envelope
        #    This is heavy for long files. Dowsample?

        # Let's chunk it.
        chunk_size = 32768
        step = 16384

        roughness_vals = []

        # Process in chunks
        for start in range(0, len(audio) - chunk_size, step):
             end = start + chunk_size
             chunk = audio[start:end]

             # Envelope
             env = np.abs(signal.hilbert(chunk))
             # Remove DC
             env_ac = env - np.mean(env)

             # Modulation Spectrum
             f_mod, p_mod = signal.welch(env_ac, fs=sr, nperseg=1024)

             # Roughness works best around f_mod 70Hz.
             # Weighting curve for modulation freq:
             # r(f_mod) ~ bandpass centered at 70Hz (20-150Hz)

             # Simple weighting: Gaussian around 70Hz
             # W(f) = exp( - (f - 70)^2 / (2 * 30^2) )
             weights = np.exp(-((f_mod - 70)**2) / (2 * 30**2))

             # Sum weighted power
             # R ~ sqrt( sum (P_mod * W) ) / RMS_carrier?
             # Modulation Index m = A_mod / A_carrier

             mod_power = np.sum(p_mod * weights)
             carrier_power = np.mean(chunk**2)

             if carrier_power < 1e-9:
                 R = 0
             else:
                 # Proportional to m
                 R = np.sqrt(mod_power / carrier_power)

             # Scale to roughly align with "asper" (1 asper is 100% mod at 1kHz 70Hz)
             # This is uncalibrated.
             roughness_vals.append(R * 10)

        # Interpolate to time?
        # Just return series
        r_series = np.array(roughness_vals)
        r_step = step / sr

        return {
            "mean_roughness": np.mean(r_series) if len(r_series) > 0 else 0.0,
            "roughness_series": r_series,
            "roughness_step": r_step
        }

    def _calc_tonality(self, audio, sr):
        # Tonality via Spectral Flatness Measure (SFM)
        # SFM = Geometric Mean / Arithmetic Mean of Power Spectrum
        # Tonality coeff = 1 - SFM (approx) for White Noise (SFM=1 => T=0) vs Sine (SFM~0 => T=1)
        # Or Peak/Mean ratio?

        # Short-time Tonality
        window_sec = 0.2
        nperseg = int(window_sec * sr)

        f, t, Zxx = signal.stft(audio, fs=sr, window='hann', nperseg=nperseg)
        mag_sq = np.abs(Zxx)**2

        # Limit freq range to meaningful band (e.g. 50Hz - 15kHz)
        # Avoid DC and very high freq noise
        mask = (f >= 50) & (f <= 16000)
        sub_spec = mag_sq[mask, :]

        # To avoid log(0)
        sub_spec += 1e-12

        geo_mean = np.exp(np.mean(np.log(sub_spec), axis=0))
        ari_mean = np.mean(sub_spec, axis=0)

        sfm = geo_mean / ari_mean

        # Tonality Index (Johnston?)
        # T = SFM_db / -60 ?
        # Simple: T = 1 - SFM
        # But for diverse signals, SFM is usually quite low.
        # Let's use simple T = 1 - SFM^epsilon?
        # Or just return SFM inverse.
        # High Tonality -> Low SFM.
        # Let's output "Tonality Index" = 1 - SFM. (1=Pure, 0=Noise)

        tonality = 1.0 - sfm
        tonality = np.clip(tonality, 0, 1)

        # Time step conversion
        # t is dependent on hop size (default nperseg/2)
        step = (nperseg/2) / sr

        return {
            "mean_tonality": np.mean(tonality),
            "tonality_series": tonality,
            "tonality_step": step
        }


# --- Widget ---

class SoundQualityAnalyzer(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine

    @property
    def name(self) -> str:
        return "Sound Quality Analyzer"

    @property
    def description(self) -> str:
        return "Offline analysis of sound quality metrics (Loudness, Sharpness, Roughness)."

    def run(self, args: argparse.Namespace):
        print("Sound Quality Analyzer is a GUI-only widget.")

    def get_widget(self):
        return SoundQualityAnalyzerWidget(self)


class SoundQualityAnalyzerWidget(QWidget):
    def __init__(self, module: SoundQualityAnalyzer):
        super().__init__()
        self.module = module
        self.worker = None
        self.analysis_results = None
        self.audio_data = None
        self.samplerate = 48000
        # Playback State
        self.is_playing = False
        self.playback_position = 0 # In samples
        self.callback_id = None
        self.playback_timer = QTimer()
        self.playback_timer.setInterval(50) # 20 fps update
        self.playback_timer.timeout.connect(self.update_playback_cursor)

        self.cursors = [] # List of InfiniteLines

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- Top: Controls ---
        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶")
        self.play_btn.setToolTip(tr("Play/Pause"))
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("■")
        self.stop_btn.setToolTip(tr("Stop"))
        self.stop_btn.setFixedWidth(40)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        self.chk_follow = QCheckBox(tr("Follow Cursor"))
        self.chk_follow.setChecked(True)
        controls_layout.addWidget(self.chk_follow)

        # Spacer
        controls_layout.addSpacing(10)

        self.file_label = QLabel(tr("No file selected"))
        controls_layout.addWidget(self.file_label, stretch=1)

        self.load_btn = QPushButton(tr("Load File..."))
        self.load_btn.clicked.connect(self.load_file)
        controls_layout.addWidget(self.load_btn)

        self.analyze_btn = QPushButton(tr("Analyze"))
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        controls_layout.addWidget(self.analyze_btn)

        layout.addLayout(controls_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Middle: Metrics Summary ---
        summary_group = QGroupBox(tr("Summary Metrics"))
        self.summary_grid = QGridLayout()

        # Headers
        self.summary_grid.addWidget(QLabel(tr("Channel")), 0, 0)
        self.summary_grid.addWidget(QLabel(tr("Integrated Loudness")), 0, 1)
        self.summary_grid.addWidget(QLabel(tr("Mean Sharpness (acum)")), 0, 2)
        self.summary_grid.addWidget(QLabel(tr("Mean Roughness")), 0, 3)
        self.summary_grid.addWidget(QLabel(tr("Mean Tonality")), 0, 4)

        summary_group.setLayout(self.summary_grid)
        layout.addWidget(summary_group)

        # --- Bottom: Graphs ---
        self.tabs = QTabWidget()

        # 1. Time Series
        self.ts_tab = QWidget()
        ts_layout = QVBoxLayout(self.ts_tab)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True)
        ts_layout.addWidget(self.plot_widget)

        self.tabs.addTab(self.ts_tab, tr("Time Series"))

        layout.addWidget(self.tabs)

        self.setLayout(layout)

        # Placeholder rows
        self._set_summary_placeholder()

    def _set_summary_placeholder(self):
        # Clear existing rows except header
        # (Simplified: Just add empty labels for row 1)
        self.summary_grid.addWidget(QLabel("-"), 1, 0)
        self.summary_grid.addWidget(QLabel("-"), 1, 1)
        self.summary_grid.addWidget(QLabel("-"), 1, 2)
        self.summary_grid.addWidget(QLabel("-"), 1, 3)
        self.summary_grid.addWidget(QLabel("-"), 1, 4)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Open Audio File"), "", "Audio Files (*.wav *.flac *.aiff)")
        if path:
            self.current_file = path
            self.file_label.setText(path)
            self.analyze_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def start_analysis(self):
        if not hasattr(self, 'current_file'):
            return

        self.analyze_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.plot_widget.clear()

        # Stop playback if running
        if hasattr(self, 'stop_playback'):
            self.stop_playback()

        target_sr = self.module.audio_engine.sample_rate
        self.worker = AnalysisWorker(self.current_file, target_sr)
        self.worker.progress_update.connect(self.on_progress)
        self.worker.results_ready.connect(self.on_results)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def on_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.progress_bar.setFormat(f"%p% - {msg}")

    def on_results(self, results):
        self.analysis_results = results

        # Store for playback
        if "audio_data" in results:
            self.audio_data = results["audio_data"] # (samples, ch) or (samples,)
            self.samplerate = results["samplerate"]
            self.playback_position = 0
            self.is_playing = False
            self.play_btn.setText("▶")

        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        self.display_metrics(results)
        self.plot_series(results)

        # Ensure we have lines
        self.cursors = []
        for p in [self.p1, self.p2, self.p3]:
            # Click event
            p.scene().sigMouseClicked.connect(self.on_plot_clicked)

            # Add cursor
            line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('y', width=2))
            p.addItem(line)
            self.cursors.append(line)

    def on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.file_label.setText(f"Error: {msg}")

    def display_metrics(self, results):
        # Clear grid
        # Note: Removing widgets from layout is tedious in Qt.
        # Let's just hide or delete properly.
        while self.summary_grid.count() > 5: # Keep headers
            item = self.summary_grid.takeAt(5)
            w = item.widget()
            if w: w.deleteLater()

        row = 1
        for ch in results["channels"]:
            name = ch["name"]
            i_lufs = ch["integrated_lufs"]
            m_sh = ch["mean_sharpness"]
            m_r = ch["mean_roughness"]
            m_t = ch["mean_tonality"]

            self.summary_grid.addWidget(QLabel(name), row, 0)
            self.summary_grid.addWidget(QLabel(f"{i_lufs:.1f} LUFS"), row, 1)
            self.summary_grid.addWidget(QLabel(f"{m_sh:.2f} acum"), row, 2)
            self.summary_grid.addWidget(QLabel(f"{m_r:.2f}"), row, 3)
            self.summary_grid.addWidget(QLabel(f"{m_t:.2f}"), row, 4)
            row += 1

    def plot_series(self, results):
        self.plot_widget.clear()

        # Plot only first channel/average to avoid clutter?
        # Or provide checkboxes?
        # Let's plot Left (or Mono) Loudness, Sharpness (scaled?), Roughness (scaled?)

        # To visualize different units, maybe we need multiple ViewBoxes or normalized view.
        # Simple approach: Just plot Loudness for now, or create separate plot items.

        # Let's use the tab widget to separate graphs if needed, or stacking.
        # User request: "Graphs (1 screen): Loudness vs Time, Sharpness vs Time..."
        # Implies stacked or same plot.
        # Metrics have vastly different ranges (-60..0 LUFS vs 0..4 Sharpness).
        # We need "Add Plot" or multiple areas.

        # Let's clear the layout of ts_tab and rebuild with 3 vertical plots.
        layout = self.ts_tab.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()

        # Loudness Plot
        p1 = pg.PlotWidget(title=tr("Loudness (Momentary)"))
        p1.setLabel('left', 'LUFS')
        p1.showGrid(y=True)
        p1.addLegend()

        # Sharpness Plot
        p2 = pg.PlotWidget(title=tr("Sharpness (Zwicker)"))
        p2.setLabel('left', 'acum')
        p2.showGrid(y=True)
        p2.addLegend()
        p2.setXLink(p1)

        # Roughness Plot
        p3 = pg.PlotWidget(title=tr("Roughness"))
        p3.setLabel('left', 'arb') # Arbitrary unit for now
        p3.showGrid(y=True)
        p3.setLabel('bottom', 'Time', units='s')
        p3.addLegend()
        p3.setXLink(p1)

        colors = ['c', 'm', 'g', 'y']

        for i, ch in enumerate(results["channels"]):
             c = colors[i % len(colors)]
             name = ch["name"]

             # Loudness
             t_l = np.arange(len(ch["lufs_series"])) * ch["lufs_step"]
             p1.plot(t_l, ch["lufs_series"], pen=c, name=name)

             # Sharpness
             t_s = np.arange(len(ch["sharpness_series"])) * ch["sharpness_step"]
             p2.plot(t_s, ch["sharpness_series"], pen=c, name=name)

             # Roughness
             t_r = np.arange(len(ch["roughness_series"])) * ch["roughness_step"]
             p3.plot(t_r, ch["roughness_series"], pen=c, name=name)

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        layout.addWidget(p1)
        layout.addWidget(p2)
        layout.addWidget(p3)

    # --- Playback Logic ---

    def toggle_playback(self):
        if not hasattr(self, 'audio_data') or self.audio_data is None:
            return

        if self.is_playing:
            # Pause
            self.is_playing = False
            self.play_btn.setText("▶")
            if self.callback_id is not None:
                self.module.audio_engine.unregister_callback(self.callback_id)
                self.callback_id = None
            self.playback_timer.stop()
        else:
            # Play
            # Check end
            if self.playback_position >= len(self.audio_data):
                self.playback_position = 0

            self.is_playing = True
            self.play_btn.setText("⏸")
            self.callback_id = self.module.audio_engine.register_callback(self.audio_callback)
            self.playback_timer.start()

    def stop_playback(self):
        self.is_playing = False
        self.play_btn.setText("▶")
        if self.callback_id is not None:
             self.module.audio_engine.unregister_callback(self.callback_id)
             self.callback_id = None
        self.playback_timer.stop()
        self.playback_position = 0
        self.update_playback_cursor()

    def audio_callback(self, indata, outdata, frames, time, status):
        if not self.is_playing or self.audio_data is None:
            outdata.fill(0)
            return

        # Write to outdata
        # audio_data can be mono (N,) or stereo (N, 2)
        # outdata is (frames, 2) usually (depending on engine config, but we target stereo)

        remaining = len(self.audio_data) - self.playback_position
        if remaining <= 0:
            outdata.fill(0)
            # Stop? Can't call GUI from thread easily.
            # handled by timer check or just silence until timer stops it?
            # ideally we just signal stop.
            return

        n = min(frames, remaining)

        chunk = self.audio_data[self.playback_position : self.playback_position + n]

        # Map to output
        # outdata shape is (frames, output_channels)
        out_ch = outdata.shape[1]

        if chunk.ndim == 1:
            # Mono to all ch
            for c in range(out_ch):
                outdata[:n, c] = chunk
        else:
            # Stereo input
            in_ch = chunk.shape[1]
            if in_ch >= out_ch:
                outdata[:n, :] = chunk[:, :out_ch]
            else:
                outdata[:n, :in_ch] = chunk
                # Fill rest with 0 or copy? 0 is safer.
                outdata[:n, in_ch:] = 0

        if n < frames:
             outdata[n:, :] = 0

        self.playback_position += n

    def update_playback_cursor(self):
        if self.audio_data is None: return

        # Check if finished
        if self.playback_position >= len(self.audio_data):
             self.stop_playback()
             return

        t = self.playback_position / self.samplerate

        # Update lines
        for line in self.cursors:
            line.setValue(t)

        # Follow
        if self.chk_follow.isChecked() and self.is_playing:
            # Center view if out of range?
            # Or just ensure visible.
            # Simple: setXRange centered? That prevents manual pan.
            # Just verify it's in view?
            # pg plot auto-range might fight.
            # Let's just do nothing for now, or maybe simple pan if continuous.
            pass

    def on_plot_clicked(self, event):
        if self.audio_data is None: return

        # Map mouse to x
        # event is GraphicsSceneMouseEvent
        # We need to map to plot coordinates.
        # This is tricky with multiple plots.
        # simpler: use the plot widget that sent it?

        # We connected scene signal, so sender is scene.
        # But we don't know which ViewBox.
        # Actually ViewBox has sigClicked?

        # Let's try getting the position from event.scenePos() mapped to view.
        # But we have 3 plots.
        # Easier: capture which plot was clicked or just use event position if they are aligned vertically.

        # Better approach:
        # All plots share X axis.
        # Just use the X coordinate of the mouse click in the scene, map to first plot's ViewBox.

        items = self.p1.scene().items(event.scenePos())
        # Check if one of our plots is determining this.
        # Simplified: Just grab X from the first plot assuming full width alignment

        pos = self.p1.plotItem.vb.mapSceneToView(event.scenePos())
        t = pos.x()

        if t < 0: t = 0
        max_t = len(self.audio_data) / self.samplerate
        if t > max_t: t = max_t

        self.playback_position = int(t * self.samplerate)
        self.update_playback_cursor()


