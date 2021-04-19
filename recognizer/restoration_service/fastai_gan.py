from pathlib import Path

from fastai.basic_train import load_learner, Learner
from fastai.vision import open_image

from recognizer.restoration_service.base import BaseRestorationService


class FastaiGANService(BaseRestorationService):
    def _load_model(self, model_path: Path) -> Learner:
        return load_learner(model_path.parent, model_path.name)

    def restore(self, img_path: Path, out_path: Path):
        img = open_image(img_path)
        pred = self.model.predict(img)
        pred[0].save(out_path)
