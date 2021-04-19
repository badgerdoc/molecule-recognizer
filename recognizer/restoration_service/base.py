from abc import ABC, abstractmethod
from pathlib import Path


class BaseRestorationService(ABC):
    def __init__(self, model_path: Path):
        self.model = self._load_model(model_path)

    @abstractmethod
    def _load_model(self, model_path: Path):
        pass

    @abstractmethod
    def restore(self, img_path: Path, out_path: Path):
        pass
