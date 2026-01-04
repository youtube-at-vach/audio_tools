from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.core.audio_engine import AudioEngine
from src.core.config_manager import ConfigManager
from src.core.localization import get_manager, tr
from src.gui.widgets.detachable_wrapper import DetachableWidgetWrapper


def _load_module_class(module_key: str):
    """Return MeasurementModule class by key.

    Imports are intentionally inside this function to avoid importing all
    heavy GUI modules (pyqtgraph/scipy, etc.) at application startup.
    These imports remain explicit so PyInstaller can still discover them.
    """

    if module_key == "Signal Generator":
        from src.gui.widgets.signal_generator import SignalGenerator

        return SignalGenerator
    if module_key == "Spectrum Analyzer":
        from src.gui.widgets.spectrum_analyzer import SpectrumAnalyzer

        return SpectrumAnalyzer
    if module_key == "Sound Level Meter":
        from src.gui.widgets.sound_level_meter import SoundLevelMeter

        return SoundLevelMeter
    if module_key == "LUFS Meter":
        from src.gui.widgets.lufs_meter import LufsMeter

        return LufsMeter
    if module_key == "Loopback Finder":
        from src.gui.widgets.loopback_finder import LoopbackFinder

        return LoopbackFinder
    if module_key == "Distortion Analyzer":
        from src.gui.widgets.distortion_analyzer import DistortionAnalyzer

        return DistortionAnalyzer
    if module_key == "Advanced Distortion Meter":
        from src.gui.widgets.advanced_distortion_meter import AdvancedDistortionMeter

        return AdvancedDistortionMeter
    if module_key == "Network Analyzer":
        from src.gui.widgets.network_analyzer import NetworkAnalyzer

        return NetworkAnalyzer
    if module_key == "Oscilloscope":
        from src.gui.widgets.oscilloscope import Oscilloscope

        return Oscilloscope
    if module_key == "Raw Time Series":
        from src.gui.widgets.raw_time_series import RawTimeSeries

        return RawTimeSeries
    if module_key == "Lock-in Amplifier":
        from src.gui.widgets.lock_in_amplifier import LockInAmplifier

        return LockInAmplifier
    if module_key == "Lock-in THD Analyzer":
        from src.gui.widgets.lockin_thd_analyzer import LockInTHDAnalyzer

        return LockInTHDAnalyzer
    if module_key == "Frequency Counter":
        from src.gui.widgets.frequency_counter import FrequencyCounter

        return FrequencyCounter
    if module_key == "Lock-in Frequency Counter":
        from src.gui.widgets.lock_in_frequency_counter import LockInFrequencyCounter

        return LockInFrequencyCounter
    if module_key == "Spectrogram":
        from src.gui.widgets.spectrogram import Spectrogram

        return Spectrogram
    if module_key == "Boxcar Averager":
        from src.gui.widgets.boxcar_averager import BoxcarAverager

        return BoxcarAverager
    if module_key == "Goniometer":
        from src.gui.widgets.goniometer import Goniometer

        return Goniometer
    if module_key == "Impedance Analyzer":
        from src.gui.widgets.impedance_analyzer import ImpedanceAnalyzer

        return ImpedanceAnalyzer
    if module_key == "Noise Profiler":
        from src.gui.widgets.noise_profiler import NoiseProfiler

        return NoiseProfiler
    if module_key == "Recorder / Player":
        from src.gui.widgets.recorder_player import RecorderPlayer

        return RecorderPlayer
    if module_key == "Inverse Filter":
        from src.gui.widgets.inverse_filter import InverseFilter

        return InverseFilter
    if module_key == "Transient Analyzer":
        from src.gui.widgets.transient_analyzer import TransientAnalyzer

        return TransientAnalyzer
    if module_key == "Sound Quality Analyzer":
        from src.gui.widgets.sound_quality_analyzer import SoundQualityAnalyzer

        return SoundQualityAnalyzer
    if module_key == "Timecode Monitor & Generator":
        from src.gui.widgets.timecode_monitor import TimecodeMonitor

        return TimecodeMonitor
    if module_key == "BNIM Meter":
        from src.gui.widgets.bnim_meter import BNIMMeter

        return BNIMMeter

    raise KeyError(f"Unknown module key: {module_key}")


def _load_settings_widget_class():
    # Same reasoning as _load_module_class: delay heavy imports (scipy, etc.).
    from src.gui.widgets.settings import SettingsWidget

    return SettingsWidget


def _load_welcome_widget_class():
    from src.gui.widgets.welcome import WelcomeWidget

    return WelcomeWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeasureLab")
        self.resize(1000, 700)

        # Initialize Core Components
        self.config_manager = ConfigManager()

        # Initialize Localization
        lang = self.config_manager.get_language()
        get_manager().load_language(lang)

        self.audio_engine = AudioEngine()

        # Initialize Theme Manager
        from src.core.theme_manager import ThemeManager
        self.theme_manager = ThemeManager(QApplication.instance())
        # Make it accessible from app instance for SettingsWidget
        QApplication.instance().theme_manager = self.theme_manager

        # Load and apply saved theme
        saved_theme = self.config_manager.get_theme()
        self.theme_manager.set_theme(saved_theme)


        # Load saved config
        audio_cfg = self.config_manager.get_audio_config()
        last_in = audio_cfg.get('input_device')
        last_out = audio_cfg.get('output_device')

        # Default IDs
        in_id, out_id = 3, 3 # Fallback

        if last_in or last_out:
            # Find IDs by name
            devices = self.audio_engine.list_devices()
            found_in = False
            found_out = False

            for i, dev in enumerate(devices):
                if last_in and dev['name'] == last_in and dev['max_input_channels'] > 0:
                    in_id = i
                    found_in = True
                if last_out and dev['name'] == last_out and dev['max_output_channels'] > 0:
                    out_id = i
                    found_out = True

            if not found_in and last_in:
                print(f"Saved input device '{last_in}' not found, using default.")
            if not found_out and last_out:
                print(f"Saved output device '{last_out}' not found, using default.")

        try:
            self.audio_engine.set_devices(in_id, out_id)

            # Apply other settings
            sr = audio_cfg.get('sample_rate', 48000)
            self.audio_engine.set_sample_rate(sr)

            bs = audio_cfg.get('block_size', 1024)
            self.audio_engine.set_block_size(bs)

            in_ch = audio_cfg.get('input_channels', 'stereo')
            out_ch = audio_cfg.get('output_channels', 'stereo')
            self.audio_engine.set_channel_mode(in_ch, out_ch)

            # Apply PipeWire/JACK resident mode after devices + format are configured.
            self.audio_engine.set_pipewire_jack_resident(self.config_manager.get_pipewire_jack_resident())

        except Exception as e:
            print(f"Failed to set devices/settings: {e}")
            # Try default if specific failed
            try:
                self.audio_engine.set_devices(None, None)
            except Exception:
                pass

            # Even if device selection failed, honor resident setting best-effort.
            try:
                self.audio_engine.set_pipewire_jack_resident(self.config_manager.get_pipewire_jack_resident())
            except Exception:
                pass

        # Module registry (keep keys identical to module.name strings)
        self._module_keys = [
            "Signal Generator",
            "Spectrum Analyzer",
            "Sound Level Meter",
            "LUFS Meter",
            "Loopback Finder",
            "Distortion Analyzer",
            "Advanced Distortion Meter",
            "Network Analyzer",
            "Oscilloscope",
            "Raw Time Series",
            "Lock-in Amplifier",
            "Lock-in THD Analyzer",
            "Frequency Counter",
            "Lock-in Frequency Counter",
            "Spectrogram",
            "Boxcar Averager",
            "Goniometer",
            "Impedance Analyzer",
            "Noise Profiler",
            "Recorder / Player",
            "Inverse Filter",
            "Transient Analyzer",
            "Sound Quality Analyzer",
            "Timecode Monitor & Generator",
            "BNIM Meter",
        ]
        self.modules = [None] * len(self._module_keys)
        self.module_widgets = [None] * len(self._module_keys)

        # Main layout container
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Sidebar for tool selection
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.addItem(tr("Welcome"))
        self.sidebar.addItem(tr("Settings")) # Add Settings item

        for key in self._module_keys:
            self.sidebar.addItem(tr(key))

        self.sidebar.currentRowChanged.connect(self.on_tool_selected)
        layout.addWidget(self.sidebar)

        # Main content area
        self.content_area = QStackedWidget()
        layout.addWidget(self.content_area)

        # Add initial welcome page (Index 0)
        WelcomeWidget = _load_welcome_widget_class()
        self.welcome_widget = WelcomeWidget()
        self.content_area.addWidget(self.welcome_widget)

        # Add Settings Page (Index 1) - lazy loaded to avoid importing scipy at startup
        self._settings_loaded = False
        self._settings_container = QWidget()
        settings_layout = QVBoxLayout(self._settings_container)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.addWidget(QLabel(tr("Select Settings to load.")))
        self.content_area.addWidget(self._settings_container)

        # Add module pages (Index 2+) - lazy loaded per selection
        self._module_containers = []
        for key in self._module_keys:
            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(12, 12, 12, 12)
            v.addWidget(QLabel(tr("Select a module from the sidebar.")))
            self._module_containers.append(container)
            self.content_area.addWidget(container)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Status Labels
        self.status_label = QLabel(tr("Idle"))
        self.io_label = QLabel(tr("In: - | Out: -"))
        self.sr_label = QLabel(tr("SR: -"))
        self.cpu_label = QLabel(tr("CPU: 0%"))
        self.clients_label = QLabel(tr("Clients: 0"))
        self.output_dest_label = QLabel(tr("Output:"))
        self.output_dest_combo = QComboBox()
        self.output_dest_combo.addItem(tr("Physical Output"), "physical")
        self.output_dest_combo.addItem(tr("Internal Loopback (Silent)"), "loopback_silent")
        self.output_dest_combo.addItem(tr("Loopback + Physical"), "loopback_mix")
        self.output_dest_combo.setToolTip(tr("Global output destination for all modules."))
        self.output_dest_combo.currentIndexChanged.connect(self.on_output_destination_changed)

        # Add labels to status bar
        self.status_bar.addPermanentWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.io_label)
        self.status_bar.addPermanentWidget(self.sr_label)
        self.status_bar.addPermanentWidget(self.cpu_label)
        self.status_bar.addPermanentWidget(self.clients_label)
        self.status_bar.addPermanentWidget(self.output_dest_label)
        self.status_bar.addPermanentWidget(self.output_dest_combo)

        # Timer for status update
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(500) # 500ms update rate

        # Sync output destination control with engine state on startup
        self._sync_output_destination_ui(self._get_engine_output_destination(), propagate=True)

    def _replace_container_contents(self, container: QWidget, widget: QWidget):
        layout = container.layout()
        if layout is None:
            layout = QVBoxLayout(container)
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        layout.addWidget(widget)

    def _ensure_settings_loaded(self):
        if self._settings_loaded:
            return
        try:
            SettingsWidget = _load_settings_widget_class()
            self.settings_widget = SettingsWidget(self.audio_engine, self.config_manager)
            self._replace_container_contents(self._settings_container, self.settings_widget)
            self._settings_loaded = True
        except Exception as e:
            self._replace_container_contents(
                self._settings_container,
                QLabel(tr("Failed to load Settings: {0}").format(str(e))),
            )

    def _ensure_module_loaded(self, module_index: int):
        if module_index < 0 or module_index >= len(self._module_keys):
            return
        if self.modules[module_index] is not None and self.module_widgets[module_index] is not None:
            return

        key = self._module_keys[module_index]
        container = self._module_containers[module_index]
        try:
            cls = _load_module_class(key)
            module = cls(self.audio_engine)
            self.modules[module_index] = module

            widget = module.get_widget()
            if widget:
                wrapper = DetachableWidgetWrapper(widget, tr(key), self.config_manager)
                self.module_widgets[module_index] = wrapper
                self._replace_container_contents(container, wrapper)

                # Sync global output destination into newly loaded widget
                self._propagate_output_destination(self._get_engine_output_destination())
            else:
                self._replace_container_contents(container, QLabel(tr("No GUI for {0}").format(key)))
        except Exception as e:
            self._replace_container_contents(
                container,
                QLabel(tr("Failed to load module {0}: {1}").format(tr(key), str(e))),
            )

    def preload_all_modules(self, progress_callback=None):
        """Preload Settings and all modules.

        Intended to be called while a splash screen is visible so the user sees
        progress while heavy imports/widgets are created.

        progress_callback: callable(str) -> None
        """

        def report(msg: str):
            if progress_callback is None:
                return
            try:
                progress_callback(msg)
            except Exception:
                pass

        report(tr("Loading Settings..."))
        QApplication.processEvents()
        self._ensure_settings_loaded()
        QApplication.processEvents()

        total = len(self._module_keys)
        for i, key in enumerate(self._module_keys, start=1):
            report(tr("Loading {0} ({1}/{2})...").format(tr(key), i, total))
            QApplication.processEvents()
            self._ensure_module_loaded(i - 1)
            QApplication.processEvents()

    def closeEvent(self, event):
        # Ensure PortAudio stream is closed (important in resident mode).
        try:
            self.audio_engine.stop_stream()
        except Exception:
            pass
        super().closeEvent(event)

    def update_status(self):
        status = self.audio_engine.get_status()

        # Keep global output selector in sync if a widget changed it
        current_mode = self._get_engine_output_destination()
        self._sync_output_destination_ui(current_mode, propagate=True)

        # Active State
        if status['active']:
            self.status_label.setText(tr("ACTIVE"))
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText(tr("IDLE"))
            self.status_label.setStyleSheet("color: gray;")

        # I/O Mode
        in_mode = status['input_channels'].capitalize()
        out_mode = status['output_channels'].capitalize()
        self.io_label.setText(tr("In: {0} | Out: {1}").format(in_mode, out_mode))

        # Sample Rate
        self.sr_label.setText(tr("SR: {0}").format(status['sample_rate']))

        # CPU Load
        cpu = status['cpu_load'] * 100
        flags = status.get('status_flags')

        if flags:
            self.cpu_label.setText(tr("CPU: {0:.1f}% [{1}]").format(cpu, flags))
            self.cpu_label.setStyleSheet("color: red; font-weight: bold;")
            self.cpu_label.setToolTip(tr("Audio Buffer Error: {0}").format(flags))
        else:
            self.cpu_label.setText(tr("CPU: {0:.1f}%").format(cpu))
            self.cpu_label.setStyleSheet("")
            self.cpu_label.setToolTip(tr("CPU Load of Audio Thread"))

        # Clients
        self.clients_label.setText(tr("Clients: {0}").format(status['active_clients']))

    def _get_engine_output_destination(self):
        if self.audio_engine.loopback:
            return "loopback_silent" if self.audio_engine.mute_output else "loopback_mix"
        return "physical"

    def _sync_output_destination_ui(self, mode: str, propagate: bool = False):
        idx = self.output_dest_combo.findData(mode)
        if idx == -1:
            return
        if idx != self.output_dest_combo.currentIndex():
            self.output_dest_combo.blockSignals(True)
            self.output_dest_combo.setCurrentIndex(idx)
            self.output_dest_combo.blockSignals(False)
            if propagate:
                self._propagate_output_destination(mode)

    def _propagate_output_destination(self, mode: str):
        for widget in self.module_widgets:
            if widget:
                # If wrapped, get the inner content
                target = widget.content_widget if isinstance(widget, DetachableWidgetWrapper) else widget

                if hasattr(target, 'set_output_destination'):
                    try:
                        target.set_output_destination(mode)
                    except Exception as e:
                        print(f"Failed to sync output destination: {e}")

    def on_output_destination_changed(self, index):
        data = self.output_dest_combo.currentData()
        if data == "physical":
            self.audio_engine.set_loopback(False)
            self.audio_engine.set_mute_output(False)
        elif data == "loopback_silent":
            self.audio_engine.set_loopback(True)
            self.audio_engine.set_mute_output(True)
        elif data == "loopback_mix":
            self.audio_engine.set_loopback(True)
            self.audio_engine.set_mute_output(False)

        # Mirror selection to widgets that expose destination controls
        self._propagate_output_destination(data)

    def on_tool_selected(self, index):
        if index == 1:
            self._ensure_settings_loaded()
        elif index >= 2:
            self._ensure_module_loaded(index - 2)
        self.content_area.setCurrentIndex(index)
