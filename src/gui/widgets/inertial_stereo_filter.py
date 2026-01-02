import argparse
import os

import numpy as np
import soundfile as sf
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QProgressBar,
    QFileDialog,
    QMessageBox,
)

from src.measurement_modules.base import MeasurementModule
from src.core.localization import tr
from src.core.stereo_angle_dynamics import (
    MODEL_KEYS,
    InertialAttractorState,
    process_stereo_inertial_attractor_block,
    PhaseSpaceInertiaState,
    process_stereo_phase_space_inertia_block,
    process_stereo_phase_space_inertia_corr_prefilter_block,
    DiffuseSide3DState,
    process_stereo_diffuse_side_3d_block,
)


class InertialStereoFilter(MeasurementModule):
    def __init__(self, audio_engine):
        self.audio_engine = audio_engine
        self.is_running = False

    @property
    def name(self) -> str:
        return "Inertial Stereo Filter"

    @property
    def description(self) -> str:
        return "Offline stereo filter that adds inertia to panning/phase-space angle."

    def run(self, args: argparse.Namespace):
        pass

    def get_widget(self):
        return InertialStereoFilterWidget(self)

    def start_analysis(self):
        self.is_running = True

    def stop_analysis(self):
        self.is_running = False


class ProcessingWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(
        self,
        *,
        input_path: str,
        output_path: str,
        model_key: str,
        alpha: float,
        beta: float,
        tau_ms: float,
        corr_threshold: float = 0.3,
        corr_window_frames: int = 1024,
        block_frames: int = 65536,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.model_key = model_key
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau_ms = float(tau_ms)
        self.corr_threshold = float(corr_threshold)
        self.corr_window_frames = int(corr_window_frames)
        self.block_frames = int(block_frames)

    def run(self):
        try:
            if not os.path.exists(self.input_path):
                self.finished.emit(False, tr("Input file not found."))
                return

            try:
                info = sf.info(self.input_path)
            except Exception as e:
                self.finished.emit(False, tr("Failed to read input file: {0}").format(e))
                return

            if info.channels not in (1, 2):
                self.finished.emit(False, tr("Only mono or stereo files are supported."))
                return

            sr = float(info.samplerate)
            total_frames = int(info.frames) if info.frames else 0

            tau_seconds = float(self.tau_ms) / 1000.0

            # Model dispatch (extensible)
            if self.model_key == "inertial_attractor":
                state = InertialAttractorState()
                process_block = lambda block, st: process_stereo_inertial_attractor_block(
                    block,
                    sample_rate=sr,
                    alpha=self.alpha,
                    beta=self.beta,
                    tau_seconds=tau_seconds,
                    state=st,
                )
            elif self.model_key == "phase_space_inertia":
                state = PhaseSpaceInertiaState()
                process_block = lambda block, st: process_stereo_phase_space_inertia_block(
                    block,
                    sample_rate=sr,
                    alpha=self.alpha,
                    beta=self.beta,
                    tau_seconds=tau_seconds,
                    state=st,
                )
            elif self.model_key == "phase_space_inertia_corr_prefilter":
                state = PhaseSpaceInertiaState()
                process_block = lambda block, st: process_stereo_phase_space_inertia_corr_prefilter_block(
                    block,
                    sample_rate=sr,
                    alpha=self.alpha,
                    beta=self.beta,
                    tau_seconds=tau_seconds,
                    corr_threshold=self.corr_threshold,
                    corr_window_frames=self.corr_window_frames,
                    state=st,
                )
            elif self.model_key == "diffuse_side_3d":
                state = DiffuseSide3DState()
                process_block = lambda block, st: process_stereo_diffuse_side_3d_block(
                    block,
                    sample_rate=sr,
                    alpha=self.alpha,
                    beta=self.beta,
                    tau_seconds=tau_seconds,
                    state=st,
                )
            else:
                self.finished.emit(False, tr("Unknown model: {0}").format(self.model_key))
                return

            processed_frames = 0

            with sf.SoundFile(self.input_path) as infile:
                # Preserve subtype/format when possible; fallback to WAV defaults if unknown.
                out_format = infile.format
                out_subtype = infile.subtype

                with sf.SoundFile(
                    self.output_path,
                    "w",
                    samplerate=int(sr),
                    channels=2,
                    format=out_format,
                    subtype=out_subtype,
                ) as outfile:
                    while True:
                        block = infile.read(self.block_frames, dtype="float32", always_2d=True)
                        if block.size == 0:
                            break

                        out_block, state = process_block(block, state)

                        outfile.write(out_block)

                        processed_frames += int(block.shape[0])
                        if total_frames > 0:
                            pct = int(min(100, max(0, 100.0 * processed_frames / total_frames)))
                            self.progress.emit(pct)

            self.progress.emit(100)
            self.finished.emit(True, tr("Processing complete."))

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.finished.emit(False, str(e))


class InertialStereoFilterWidget(QWidget):
    def __init__(self, module: InertialStereoFilter):
        super().__init__()
        self.module = module
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- Section 1: Model / Params ---
        params_group = QGroupBox(tr("1. Dynamics Model"))
        params_layout = QFormLayout()

        self.model_combo = QComboBox()
        for key, label in MODEL_KEYS.items():
            self.model_combo.addItem(tr(label), key)
        params_layout.addRow(tr("Model:"), self.model_combo)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.001)
        self.alpha_spin.setDecimals(4)
        self.alpha_spin.setValue(0.0500)
        self.alpha_spin.setToolTip(tr("Follow factor toward instantaneous angle (alpha)."))
        params_layout.addRow(tr("Alpha:"), self.alpha_spin)

        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 1.0)
        self.beta_spin.setSingleStep(0.001)
        self.beta_spin.setDecimals(4)
        self.beta_spin.setValue(0.0100)
        self.beta_spin.setToolTip(tr("Attraction toward mass center angle (beta)."))
        params_layout.addRow(tr("Beta:"), self.beta_spin)

        self.tau_spin = QDoubleSpinBox()
        self.tau_spin.setRange(0.0, 5000.0)
        self.tau_spin.setSingleStep(10.0)
        self.tau_spin.setDecimals(1)
        self.tau_spin.setSuffix(" ms")
        self.tau_spin.setValue(200.0)
        self.tau_spin.setToolTip(tr("EMA time constant for mass center (tau). 0 disables smoothing."))
        params_layout.addRow(tr("Tau:"), self.tau_spin)

        self.corr_thr_spin = QDoubleSpinBox()
        self.corr_thr_spin.setRange(-1.0, 1.0)
        self.corr_thr_spin.setSingleStep(0.05)
        self.corr_thr_spin.setDecimals(2)
        self.corr_thr_spin.setValue(0.30)
        self.corr_thr_spin.setToolTip(
            tr(
                "Correlation threshold for prefilter. If corr < threshold, window is collapsed to mono (mid)."
            )
        )
        params_layout.addRow(tr("Corr Thr:"), self.corr_thr_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # --- Section 2: File Processing ---
        proc_group = QGroupBox(tr("2. Offline Processing"))
        proc_layout = QFormLayout()

        in_layout = QHBoxLayout()
        self.in_path_label = QLabel(tr("No file selected"))
        in_btn = QPushButton(tr("Select Input Wav..."))
        in_btn.clicked.connect(self.select_input_file)
        in_layout.addWidget(self.in_path_label, stretch=1)
        in_layout.addWidget(in_btn)
        proc_layout.addRow(tr("Input:"), in_layout)

        out_layout = QHBoxLayout()
        self.out_path_label = QLabel(tr("No output file"))
        out_btn = QPushButton(tr("Select Output Wav..."))
        out_btn.clicked.connect(self.select_output_file)
        out_layout.addWidget(self.out_path_label, stretch=1)
        out_layout.addWidget(out_btn)
        proc_layout.addRow(tr("Output:"), out_layout)

        self.process_btn = QPushButton(tr("Process & Save"))
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        proc_layout.addRow(self.process_btn)

        self.progress = QProgressBar()
        proc_layout.addRow(self.progress)

        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)

        self.setLayout(layout)

    def select_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Open Wav File"), "", tr("Wav Files (*.wav)"))
        if not path:
            return

        self.in_path_label.setText(path)

        # Auto output path
        base, ext = os.path.splitext(path)
        self.out_path_label.setText(base + "_inertial" + (ext if ext else ".wav"))

        self._update_process_btn()

    def select_output_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("Save Wav File"),
            self.out_path_label.text(),
            tr("Wav Files (*.wav)"),
        )
        if not path:
            return

        self.out_path_label.setText(path)
        self._update_process_btn()

    def _update_process_btn(self):
        self.process_btn.setEnabled(os.path.exists(self.in_path_label.text()) and bool(self.out_path_label.text()))

    def start_processing(self):
        if self.worker and self.worker.isRunning():
            return

        input_path = self.in_path_label.text()
        output_path = self.out_path_label.text()

        model_key = self.model_combo.currentData()
        alpha = float(self.alpha_spin.value())
        beta = float(self.beta_spin.value())
        tau_ms = float(self.tau_spin.value())
        corr_thr = float(self.corr_thr_spin.value())

        self.process_btn.setEnabled(False)
        self.process_btn.setText(tr("Processing..."))
        self.progress.setValue(0)

        self.worker = ProcessingWorker(
            input_path=input_path,
            output_path=output_path,
            model_key=str(model_key),
            alpha=alpha,
            beta=beta,
            tau_ms=tau_ms,
            corr_threshold=corr_thr,
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

    def on_processing_finished(self, success: bool, msg: str):
        self.process_btn.setEnabled(True)
        self.process_btn.setText(tr("Process & Save"))

        if success:
            QMessageBox.information(self, tr("Success"), msg)
        else:
            QMessageBox.critical(self, tr("Error"), msg)

        self._update_process_btn()
