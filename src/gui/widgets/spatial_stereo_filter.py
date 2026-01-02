import os

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
from src.core.spatial_stereo_dynamics import (
    DiffuseSide3DState,
    NeuralPrecedence3DState,
    process_stereo_diffuse_side_3d_block,
    process_stereo_neural_precedence_3d_block,
)


class SpatialStereoFilter(MeasurementModule):
    def __init__(self, audio_engine):
        self.audio_engine = audio_engine
        self.is_running = False

    @property
    def name(self) -> str:
        return "Spatial Stereo Filter"

    @property
    def description(self) -> str:
        return "Offline stereo filter experimenting with neuro-inspired spatial perception."

    def run(self, args):
        pass

    def get_widget(self):
        return SpatialStereoFilterWidget(self)

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
        depth: float,
        width: float,
        size_ms: float,
        block_frames: int = 65536,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.model_key = str(model_key)
        self.depth = float(depth)
        self.width = float(width)
        self.size_ms = float(size_ms)
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

            size_seconds = float(self.size_ms) / 1000.0

            if self.model_key == "diffuse_side_3d":
                state = DiffuseSide3DState()
                process_block = lambda block, st: process_stereo_diffuse_side_3d_block(
                    block,
                    sample_rate=sr,
                    alpha=self.depth,
                    beta=self.width,
                    tau_seconds=size_seconds,
                    state=st,
                )
            elif self.model_key == "neural_precedence_3d":
                state = NeuralPrecedence3DState()
                process_block = lambda block, st: process_stereo_neural_precedence_3d_block(
                    block,
                    sample_rate=sr,
                    alpha=self.depth,
                    beta=self.width,
                    tau_seconds=size_seconds,
                    state=st,
                )
            else:
                self.finished.emit(False, tr("Unknown model: {0}").format(self.model_key))
                return

            processed_frames = 0

            with sf.SoundFile(self.input_path) as infile:
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


class SpatialStereoFilterWidget(QWidget):
    def __init__(self, module: SpatialStereoFilter):
        super().__init__()
        self.module = module
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        params_group = QGroupBox(tr("1. Spatial Model"))
        params_layout = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItem(tr("Diffuse Side 3D"), "diffuse_side_3d")
        self.model_combo.addItem(tr("Neural Precedence 3D"), "neural_precedence_3d")
        params_layout.addRow(tr("Model:"), self.model_combo)

        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(0.0, 1.0)
        self.depth_spin.setSingleStep(0.001)
        self.depth_spin.setDecimals(4)
        self.depth_spin.setValue(0.6000)
        self.depth_spin.setToolTip(tr("Effect strength/depth. Higher = more spacious, but can sound effecty."))
        params_layout.addRow(tr("Depth:"), self.depth_spin)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.0, 1.0)
        self.width_spin.setSingleStep(0.001)
        self.width_spin.setDecimals(4)
        self.width_spin.setValue(0.3000)
        self.width_spin.setToolTip(tr("Width amount (scales side component)."))
        params_layout.addRow(tr("Width:"), self.width_spin)

        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.0, 5000.0)
        self.size_spin.setSingleStep(10.0)
        self.size_spin.setDecimals(1)
        self.size_spin.setSuffix(" ms")
        self.size_spin.setValue(12.0)
        self.size_spin.setToolTip(tr("Time scale in ms. Larger = bigger space/longer precedence; too large can sound like reverb."))
        params_layout.addRow(tr("Size:"), self.size_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

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

        base, ext = os.path.splitext(path)
        self.out_path_label.setText(base + "_spatial" + (ext if ext else ".wav"))

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

        model_key = str(self.model_combo.currentData())
        depth = float(self.depth_spin.value())
        width = float(self.width_spin.value())
        size_ms = float(self.size_spin.value())

        self.process_btn.setEnabled(False)
        self.process_btn.setText(tr("Processing..."))
        self.progress.setValue(0)

        self.worker = ProcessingWorker(
            input_path=input_path,
            output_path=output_path,
            model_key=model_key,
            depth=depth,
            width=width,
            size_ms=size_ms,
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
