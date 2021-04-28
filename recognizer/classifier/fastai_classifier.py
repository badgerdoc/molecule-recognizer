import numpy as np
from pathlib import Path

from fastai.basic_train import load_learner, Learner
from fastai.vision import pil2tensor, Image


class FastaiAtomClassifier:

    def __init__(self, model_path: Path):
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: Path) -> Learner:
        return load_learner(model_path.parent, model_path.name)

    def predict(self, img_cv2):
        img = Image(pil2tensor(img_cv2, dtype=np.float32).div_(255))
        return self.model.data.classes[int(self.model.predict(img)[0])]
