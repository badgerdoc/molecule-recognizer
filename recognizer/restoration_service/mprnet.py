from pathlib import Path
from recognizer.models.MPRNet import MPRNet

import torch

import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from skimage import img_as_ubyte

from recognizer.image_processing.utils import save_img
from recognizer.restoration_service.base import BaseRestorationService


class MPRNETService(BaseRestorationService):
    def _load_model(self, model_path: Path):
        checkpoint = torch.load(model_path, map_location='gpu')
        model = MPRNet()
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def restore(self, img_path: Path, out_path: Path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(img).unsqueeze(0)
        img_tensor = self.normalize_tensor(img_tensor)

        with torch.no_grad():
            restored = self.model(img_tensor)
        restored = restored[0]
        restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        save_img(out_path, restored)

    @staticmethod
    def normalize_tensor(img_tensor, dim_base=8):
        """Ensure that image dimensions are product of `dim_base`."""
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        _h, _w = ((h + dim_base) // dim_base) * dim_base, (
                    (w + dim_base) // dim_base) * dim_base
        padh = _h - h if h % dim_base != 0 else 0
        padw = _w - w if w % dim_base != 0 else 0
        return F.pad(img_tensor, (0, padw, 0, padh), 'reflect')
