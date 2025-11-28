"""
Theme Manager for Qt6 Application

Provides theme detection and switching functionality with support for:
- System theme detection (Qt 6.5+)
- Light/Dark/System theme modes
- Dynamic theme switching with QPalette
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt, QObject, pyqtSignal
import logging


class ThemeManager(QObject):
    """Manages application theme and color scheme."""
    
    # Signal emitted when theme changes
    theme_changed = pyqtSignal(str)  # theme_name: 'light', 'dark', 'system'
    
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.logger = logging.getLogger(self.__class__.__name__)
        self.current_theme = "system"
        
        # Check if Qt supports colorScheme (Qt 6.5+)
        self.supports_color_scheme = hasattr(self.app.styleHints(), 'colorScheme')
        
        if self.supports_color_scheme:
            # Connect to system theme changes
            try:
                self.app.styleHints().colorSchemeChanged.connect(self._on_system_theme_changed)
                self.logger.info("System theme change detection enabled (Qt 6.5+)")
            except AttributeError:
                self.logger.warning("colorSchemeChanged signal not available")
    
    def set_theme(self, theme_name: str):
        """
        Set application theme.
        
        Args:
            theme_name: One of 'system', 'light', 'dark'
        """
        if theme_name not in ['system', 'light', 'dark']:
            self.logger.error(f"Invalid theme name: {theme_name}")
            return
        
        self.current_theme = theme_name
        self.logger.info(f"Setting theme to: {theme_name}")
        
        if theme_name == 'system':
            self._apply_system_theme()
        elif theme_name == 'light':
            self._apply_light_theme()
        elif theme_name == 'dark':
            self._apply_dark_theme()
        
        self.theme_changed.emit(theme_name)
    
    def get_current_theme(self) -> str:
        """Returns the current theme setting ('system', 'light', or 'dark')."""
        return self.current_theme
        
    def get_effective_theme(self) -> str:
        """
        Returns the effective theme ('light' or 'dark').
        If current_theme is 'system', detects the system theme.
        """
        if self.current_theme == 'system':
            return self._detect_system_theme()
        return self.current_theme
    
    def _on_system_theme_changed(self, scheme):
        """Handle system theme change (Qt 6.5+ only)."""
        if self.current_theme == 'system':
            self.logger.info(f"System theme changed to: {scheme}")
            self._apply_system_theme()
    
    def _detect_system_theme(self) -> str:
        """
        Detect system theme.
        
        Returns:
            'light' or 'dark'
        """
        if self.supports_color_scheme:
            try:
                from PyQt6.QtCore import Qt
                scheme = self.app.styleHints().colorScheme()
                
                # Qt.ColorScheme.Dark = 2, Qt.ColorScheme.Light = 1
                if hasattr(Qt, 'ColorScheme'):
                    if scheme == Qt.ColorScheme.Dark:
                        return 'dark'
                    elif scheme == Qt.ColorScheme.Light:
                        return 'light'
                else:
                    # Fallback for different Qt 6.5 versions
                    if int(scheme) == 2:
                        return 'dark'
                    elif int(scheme) == 1:
                        return 'light'
            except Exception as e:
                self.logger.warning(f"Failed to detect system theme: {e}")
        
        # Fallback: detect from current palette
        palette = self.app.palette()
        bg_color = palette.color(QPalette.ColorRole.Window)
        # If background is dark (low lightness), assume dark theme
        return 'dark' if bg_color.lightness() < 128 else 'light'
    
    def _apply_system_theme(self):
        """Apply system theme."""
        detected = self._detect_system_theme()
        self.logger.info(f"Applying system theme (detected: {detected})")
        
        if detected == 'dark':
            self._apply_dark_theme()
        else:
            self._apply_light_theme()
    
    def _apply_light_theme(self):
        """Apply light theme palette."""
        palette = QPalette()
        
        # Base colors
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        
        # Highlight colors
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        
        # Links
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.LinkVisited, QColor(127, 0, 127))
        
        # Tooltips
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
        
        # Disabled colors
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
        
        self.app.setPalette(palette)
        self.logger.info("Light theme applied")
    
    def _apply_dark_theme(self):
        """Apply dark theme palette."""
        palette = QPalette()
        
        # Base colors
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        
        # Highlight colors
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        
        # Links
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.LinkVisited, QColor(200, 100, 200))
        
        # Tooltips
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
        
        # Disabled colors
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))
        
        self.app.setPalette(palette)
        self.logger.info("Dark theme applied")
