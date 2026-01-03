import argparse
from abc import ABC, abstractmethod


class MeasurementModule(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def run(self, args: argparse.Namespace):
        pass

    def get_widget(self):
        """
        Returns a QWidget instance for the GUI.
        Override this method to provide a custom GUI for the module.
        """
        return None

