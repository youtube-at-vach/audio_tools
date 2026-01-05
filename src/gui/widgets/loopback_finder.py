import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.audio_engine import AudioEngine
from src.core.localization import tr
from src.measurement_modules.base import MeasurementModule


class LoopbackWorker(QThread):
    progress = pyqtSignal(int, str)
    result = pyqtSignal(list)
    error = pyqtSignal(str)
    finished_testing = pyqtSignal()

    def __init__(self, module, device_id, sample_rate=48000):
        super().__init__()
        self.module = module
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.is_running = True

    def run(self):
        try:
            self.module.perform_scan(
                self.device_id,
                self.sample_rate,
                progress_callback=self.report_progress,
                check_stop=self.check_stop
            )
            self.finished_testing.emit()
        except Exception as e:
            self.error.emit(str(e))

    def report_progress(self, value, message):
        self.progress.emit(value, message)

    def check_stop(self):
        return not self.is_running

    def stop(self):
        self.is_running = False

class LoopbackFinder(MeasurementModule):
    def __init__(self, audio_engine: AudioEngine):
        self.audio_engine = audio_engine
        self.worker = None

    @property
    def name(self) -> str:
        return "Loopback Finder"

    @property
    def description(self) -> str:
        return "Detects active loopback paths between output and input channels."

    def perform_scan(self, device_id, sample_rate, progress_callback=None, check_stop=None):
        device_info = sd.query_devices(device_id)
        max_out = device_info['max_output_channels']
        max_in = device_info['max_input_channels']

        if max_out == 0 or max_in == 0:
            raise Exception(f"Device {device_id} does not support both input and output.")

        found_paths = []
        test_freq = 440
        duration = 0.1
        threshold = 0.01 # -40dBFS approx

        t = np.linspace(0, duration, int(sample_rate * duration), False, dtype=np.float32)
        test_signal = 0.5 * np.sin(2 * np.pi * test_freq * t) # -6dBFS

        for out_ch in range(max_out):
            if check_stop and check_stop():
                break

            if progress_callback:
                progress_callback(int((out_ch / max_out) * 100), tr("Testing Output Channel {}").format(out_ch + 1))

            # Prepare output buffer for all channels
            output_signal = np.zeros((len(test_signal), max_out), dtype=np.float32)
            output_signal[:, out_ch] = test_signal

            # Play and Record
            # We record all input channels
            try:
                recorded_signal = sd.playrec(output_signal, samplerate=sample_rate,
                                           channels=max_in, device=device_id, blocking=True)
            except Exception as e:
                raise Exception(f"Error during playback/recording: {str(e)}")

            # Analyze inputs
            for in_ch in range(max_in):
                # Simple RMS check or FFT? FFT is more robust against noise.
                # Using FFT as in legacy code
                input_fft = np.fft.rfft(recorded_signal[:, in_ch])
                freqs = np.fft.rfftfreq(len(recorded_signal), 1/sample_rate)

                target_bin = np.argmin(np.abs(freqs - test_freq))
                magnitude = np.abs(input_fft[target_bin]) / len(recorded_signal) * 2

                if magnitude > threshold:
                    found_paths.append((out_ch + 1, in_ch + 1, magnitude))

        # If called from worker, we might want to emit result here or return it.
        # The worker expects result signal.
        if self.worker:
            self.worker.result.emit(found_paths)
        return found_paths

    def run(self, args):
        # CLI implementation
        print("Running Loopback Finder...")
        # Need to parse args or use defaults
        # For now just use default device from engine if available or query
        # But run(args) implies we might not have audio_engine initialized fully or we use it.
        # Let's assume audio_engine is available.

        # TODO: Parse device ID from args if provided
        device_id = self.audio_engine.output_device # Use current
        sample_rate = self.audio_engine.sample_rate

        results = self.perform_scan(device_id, sample_rate,
                                    progress_callback=lambda p, m: print(f"{p}%: {m}"))

        print("Found Paths:")
        for p in results:
            print(f"Out: {p[0]}, In: {p[1]}, Mag: {20*np.log10(p[2]):.1f} dB")

    def get_widget(self):
        return LoopbackFinderWidget(self)

class LoopbackFinderWidget(QWidget):
    def __init__(self, module: LoopbackFinder):
        super().__init__()
        self.module = module
        self._scan_available = True
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Instructions
        layout.addWidget(QLabel(tr("This tool plays a test tone on each output channel and checks all input channels for the signal.")))
        layout.addWidget(QLabel(f"<b>{tr('Note:')}</b> {tr('This will stop the main audio engine temporarily.')}"))

        # Controls
        controls_layout = QHBoxLayout()
        self.start_btn = QPushButton(tr("Start Scan"))
        self.start_btn.clicked.connect(self.start_scan)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton(tr("Stop"))
        self.stop_btn.clicked.connect(self.stop_scan)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        layout.addLayout(controls_layout)

        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel(tr("Ready"))
        layout.addWidget(self.status_label)

        # Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels([tr("Output Channel"), tr("Input Channel"), tr("Signal Level")])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.results_table)

        self.setLayout(layout)

        self._update_availability()

    def _update_availability(self):
        # Loopback scan relies on PortAudio playrec behavior that is unreliable when
        # the app runs in PipeWire/JACK resident mode (routing persistence).
        self._scan_available = not bool(getattr(self.module.audio_engine, "pipewire_jack_resident", False))

        if not self._scan_available:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.status_label.setText(
                tr("Loopback Finder is not available in PipeWire/JACK mode. Disable 'PipeWire / JACK Mode (Resident)' in Settings to use this tool.")
            )
        else:
            self.start_btn.setEnabled(True)
            self.status_label.setText(tr("Ready"))

    def start_scan(self):
        if not self._scan_available:
            QMessageBox.warning(
                self,
                tr("Unavailable"),
                tr("Loopback Finder is not available in PipeWire/JACK mode. Please disable 'PipeWire / JACK Mode (Resident)' in Settings."),
            )
            return

        # Stop main engine if running
        if self.module.audio_engine.stream and self.module.audio_engine.stream.active:
            self.module.audio_engine.stop_stream()

        # Get current device from engine
        # Note: AudioEngine stores device IDs.
        # We need to make sure we use the configured device.
        # Assuming input and output are on the same device for loopback test usually,
        # or we test the output device loopbacked to input device.
        # The legacy tool took one device ID.
        # Let's use the Output device ID from settings, and assume we record from Input device ID.
        # Wait, sd.playrec takes 'device'. If it's a tuple (in, out), that works.

        input_device = self.module.audio_engine.input_device
        output_device = self.module.audio_engine.output_device

        # If they are different, we pass (input, output) tuple to playrec
        device_arg = (input_device, output_device)

        self.module.worker = LoopbackWorker(self.module, device_arg, self.module.audio_engine.sample_rate)
        self.module.worker.progress.connect(self.update_progress)
        self.module.worker.result.connect(self.show_results)
        self.module.worker.error.connect(self.show_error)
        self.module.worker.finished_testing.connect(self.scan_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.results_table.setRowCount(0)
        self.module.worker.start()

    def stop_scan(self):
        if self.module.worker:
            self.module.worker.stop()
            self.module.worker.wait()
        self.scan_finished()

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def show_results(self, paths):
        self.results_table.setRowCount(len(paths))
        for i, (out_ch, in_ch, mag) in enumerate(paths):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(out_ch)))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(in_ch)))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{20*np.log10(mag):.1f} dB"))

        if not paths:
            self.status_label.setText(tr("No loopback paths found."))
        else:
            self.status_label.setText(tr("Found {} loopback paths.").format(len(paths)))

    def show_error(self, message):
        QMessageBox.critical(self, tr("Error"), message)
        self.scan_finished()

    def scan_finished(self):
        self.start_btn.setEnabled(self._scan_available)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
