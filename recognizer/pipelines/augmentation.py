import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from recognizer.common.molecule_generator import MoleculeImageGenerator
from recognizer.dataset import MoleculesDataset, MoleculesImageItem
from recognizer.image_processing.augmentation import add_noise
from recognizer.image_processing.utils import to_binary_img_naive, norm_dims_base, \
    to_bgr

TARGET_DIR = 'target'
INPUT_DIR = 'input'

TRAIN = 'train'
VAL = 'val'
TEST = 'test'


class GanAugmentationConfig:
    def __init__(
        self,
        dim_base: int = 8,
        train: float = 0.7,
        val: float = 0.2,
        test: float = 0.1,
        img_size: Tuple[int, int] = (400, 400),
        bond_length: int = 27,
    ):
        self.dim_base = dim_base
        self.train = train
        self.val = val
        self.test = test
        self.img_size = img_size
        self.bond_length = bond_length


# TODO: add random seed
class GanAugmentationPipeline:
    def __init__(
        self,
        dataset: MoleculesDataset,
        out_path: Path,
        config: Optional[GanAugmentationConfig] = None
    ):
        self.dataset = dataset
        self.out_path = out_path
        self.config = GanAugmentationConfig() if config is None else config
        self.distortion = Distortion()
        self.molecule_generator = MoleculeImageGenerator()

    def process_item(self, item: MoleculesImageItem):
        target_img = self.generate_target(item)
        distorted = self.distortion(target_img)
        self.save_augmented_img(distorted, target_img, item.path)

    def generate_target(self, item: MoleculesImageItem):
        target_img = self.molecule_generator.inchi_to_image(
            item.ground_truth,
            img_size=self.config.img_size,
            bond_length=self.config.bond_length,
        )
        target_img = self.normalize(target_img)
        return target_img

    def save_augmented_img(self, distorted, target_img, img_path: Path):
        subset = self._select_subset()
        for input_img, params in distorted:
            new_name = self._get_param_img_name(img_path, params)
            target_dir = _make_dir(self.out_path / subset / TARGET_DIR)
            input_dir = _make_dir(self.out_path / subset / INPUT_DIR)
            new_target_path = str(target_dir / new_name)
            new_input_path = str(input_dir / new_name)
            cv2.imwrite(new_target_path, to_bgr(target_img))
            cv2.imwrite(new_input_path, to_bgr(input_img))

    @staticmethod
    def _get_param_img_name(img_path: Path, params):
        return (
            f'{img_path.stem}_{"".join([str(int(p)) for p in params])}{img_path.suffix}'
        )

    def normalize(self, img):
        return norm_dims_base(img, self.config.dim_base)

    def process_batch(self, _slice: slice = slice(None)):
        items = list(self.dataset.items.values())[_slice]
        for item in items:
            self.process_item(item)

    def _select_subset(self):
        rand_val = np.random.uniform()
        if 0 <= rand_val < self.config.train:
            return TRAIN
        elif self.config.train <= rand_val < self.config.train + self.config.val:
            return VAL
        else:
            return TEST


class Distortion:
    def __init__(self):
        n_params = 2
        self.distort_params = [
            (i % 2 != 0, j % 2 != 0) for i in range(n_params) for j in range(n_params)
        ]

    @staticmethod
    def distort_image(img, binary: bool, dilated: bool):
        img = img.copy()
        if binary:
            img = to_binary_img_naive(img)
        if dilated:
            img = cv2.dilate(img, (2, 2), iterations=1)
        img = add_noise(img, 255, 0.7)
        img = add_noise(img, 0, 0.9995)
        # binary_img = apply_binary_threshold(img, 140)
        return img

    def __call__(self, img):
        """Apply distortion with different parameters."""
        res = []
        for params in self.distort_params:
            output_img = img.copy()
            output_img = self.distort_image(output_img, *params)
            res.append((output_img, params))
        return res


def _make_dir(dir_path: Path):
    os.makedirs(dir_path, exist_ok=True)
    return dir_path
