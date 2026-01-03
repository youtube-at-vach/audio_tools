import os
import re
from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.localization import tr

try:
    from src.core.config_manager import ConfigManager
except Exception:  # pragma: no cover
    ConfigManager = None

class IndependentWindow(QMainWindow):
    """
    A separate window to hold the detached widget.
    Emits a signal when closed so the widget can be reclaimed.
    """
    closed = pyqtSignal()

    def __init__(self, title, widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)
        self.setCentralWidget(widget)

        # When this window is closed, we need to make sure the widget isn't deleted
        # if we want to reattach it. But setCentralWidget ownership rules are tricky.
        # Ideally, the wrapper will reclaim the widget before this window is fully destroyed.

    def closeEvent(self, event):
        self.closed.emit()
        # We accept the close event, effectively hiding/destroying the window.
        # The wrapper should have already removed the widget by handling the signal
        # or we rely on the wrapper to reparent it immediately.
        event.accept()

class DetachableWidgetWrapper(QWidget):
    """
    Wraps a widget to allow it to be detached into a separate window.
    """
    def __init__(self, widget: QWidget, title: str, config_manager=None):
        super().__init__()
        self.content_widget = widget
        self.title = title
        self.config_manager = config_manager
        self.is_detached = False
        self.independent_window = None

        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # --- Header ---
        self.header = QWidget()
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(5, 5, 5, 5)

        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.detach_btn = QPushButton(tr("Detach Window"))
        self.detach_btn.clicked.connect(self.toggle_detach)
        self.detach_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        self.screenshot_btn = QPushButton(tr("Screenshot"))
        self.screenshot_btn.clicked.connect(self.save_screenshot)
        self.screenshot_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.screenshot_btn)
        header_layout.addWidget(self.detach_btn)

        self.layout.addWidget(self.header)

        # --- Content Container ---
        self.content_container = QWidget()
        self.content_container_layout = QVBoxLayout(self.content_container)
        self.content_container_layout.setContentsMargins(0, 0, 0, 0)

        # Perform initial attachment
        self.content_container_layout.addWidget(self.content_widget)
        self.layout.addWidget(self.content_container)

        # --- Placeholder (shown when detached) ---
        self.placeholder_widget = QWidget()
        placeholder_layout = QVBoxLayout(self.placeholder_widget)

        info_label = QLabel(tr("Widget is detached in a separate window."))
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        reattach_btn = QPushButton(tr("Reattach"))
        reattach_btn.clicked.connect(self.reattach)
        reattach_btn.setFixedSize(150, 40)

        placeholder_layout.addStretch()
        placeholder_layout.addWidget(info_label, alignment=Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(reattach_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addStretch()

        self.placeholder_widget.hide()
        self.layout.addWidget(self.placeholder_widget)

    def _get_screenshot_output_dir(self) -> str:
        if self.config_manager is None:
            return "screenshots"
        getter = getattr(self.config_manager, "get_screenshot_output_dir", None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                return "screenshots"
        return "screenshots"

    def _safe_base_filename(self, text: str) -> str:
        text = (text or "widget").strip()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^A-Za-z0-9_\-\.\(\)\[\]]+", "_", text)
        text = text.strip("._ ")
        return text or "widget"

    def _next_available_filepath(self, directory: str, base_name: str, ext: str) -> str:
        base = os.path.join(directory, f"{base_name}.{ext}")
        if not os.path.exists(base):
            return base
        for i in range(1, 1000):
            candidate = os.path.join(directory, f"{base_name}_{i:03d}.{ext}")
            if not os.path.exists(candidate):
                return candidate
        return base

    def save_screenshot(self):
        out_dir = self._get_screenshot_output_dir()
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, tr("Error"), tr("Failed to create output folder: {0}").format(str(e)))
            return

        try:
            pixmap = self.content_widget.grab()
        except Exception as e:
            QMessageBox.warning(self, tr("Error"), tr("Failed to capture screenshot: {0}").format(str(e)))
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = self._safe_base_filename(f"{self.title}_{timestamp}")
        path = self._next_available_filepath(out_dir, base_name, "png")

        ok = False
        try:
            ok = bool(pixmap.save(path, "PNG"))
        except Exception:
            ok = False

        if not ok:
            QMessageBox.warning(self, tr("Error"), tr("Failed to save screenshot."))
            return

        QMessageBox.information(self, tr("Success"), tr("Screenshot saved to: {0}").format(path))

    def toggle_detach(self):
        if self.is_detached:
            self.reattach()
        else:
            self.detach()

    def detach(self):
        if self.is_detached:
            return

        # 1. Remove widget from local layout
        self.content_container_layout.removeWidget(self.content_widget)
        # Ensure it's not hidden (removeWidget sometimes hides it?)
        self.content_widget.setParent(None)
        self.content_widget.show()

        # 2. Create independent window
        self.independent_window = IndependentWindow(self.title, self.content_widget, self)
        self.independent_window.closed.connect(self.reattach)
        self.independent_window.show()

        # 3. Update UI state
        self.content_container.hide()
        self.placeholder_widget.show()
        self.detach_btn.setText(tr("Reattach"))
        self.detach_btn.setEnabled(False) # Use the big reattach button in placeholder or window close
        self.is_detached = True

    def reattach(self):
        if not self.is_detached:
            return

        # 1. Close external window if open
        if self.independent_window:
            # Disconnect signal to avoid recursion if we called close() manually
            try:
                self.independent_window.closed.disconnect(self.reattach)
            except TypeError:
                pass # Already disconnected

            # If the window is still visible, close it
            if self.independent_window.isVisible():
                self.independent_window.close()

            # Reparent widget back to us
            # Note: IndependentWindow.setCentralWidget gave ownership to the window.
            # When we reparent here, we take it back.
            self.content_widget.setParent(self.content_container)
            self.content_container_layout.addWidget(self.content_widget)

            self.independent_window = None

        # 2. Update UI state
        self.placeholder_widget.hide()
        self.content_container.show()

        self.detach_btn.setText(tr("Detach Window"))
        self.detach_btn.setEnabled(True)
        self.is_detached = False
