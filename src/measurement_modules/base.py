from abc import ABC, abstractmethod
import argparse

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
