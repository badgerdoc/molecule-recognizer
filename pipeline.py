import os
import subprocess
from typing import Optional

import Levenshtein
from pathlib import Path

import cv2
from fastai.basic_train import Learner
from fastai.vision import open_image

from dataset import MoleculesDataset, MoleculesImageItem
from rdkit import Chem

from detector.inference import CascadeRCNNInferenceService
from molecules import inchi_to_mol, mol_to_png


RESIZED_DIR = 'resized'
RESTORED_DIR = 'restored'
DETECTION_DIR = 'detected'
MOL_DIR = 'mol'
INCHI_DIR = 'inchi'
IMG_FROM_INCHI_DIR = 'img_from_inchi'


class Pipeline:
    def __init__(self, dataset: MoleculesDataset, out_path: Path, gan_model: Learner, detector: CascadeRCNNInferenceService):
        self.dataset = dataset
        self.out_path = out_path
        self._dirs_init = False
        self.gan_model = gan_model
        self.detector = detector

    def _create_dirs(self):
        for dirname in (RESIZED_DIR, RESTORED_DIR, DETECTION_DIR, MOL_DIR, INCHI_DIR, IMG_FROM_INCHI_DIR):
            os.makedirs(self.out_path / dirname, exist_ok=True)
        self._dirs_init = True

    def resize(self, item: MoleculesImageItem) -> Path:
        img = cv2.imread(str(item.path))
        h, w = img.shape[:2]
        if h % 2 == 1: h += 1
        if w % 2 == 1: w += 1
        img = cv2.resize(img, (w, h))
        img_path = (self.out_path / RESIZED_DIR) / item.path.name
        cv2.imwrite(str(img_path), img)
        return img_path

    def detect_structure(self, img_path: Path, item: MoleculesImageItem):
        detection_path = (self.out_path / DETECTION_DIR) / item.path.name
        self.detector.inference_image(img_path, detection_path)

    def restore_image(self, resized_img_path: Path, item: MoleculesImageItem) -> Path:
        img = open_image(resized_img_path)
        pred = self.gan_model.predict(img)
        restored_path = (self.out_path / RESTORED_DIR) / item.path.name
        pred[0].save(restored_path)
        return restored_path

    def get_mol_file(self, item: MoleculesImageItem, img_path: Path) -> Optional[Path]:
        mol_path = (self.out_path / MOL_DIR) / f'{item.path.stem}.mol'
        try:
            subprocess.run(['bin/imago_console', '-o', str(mol_path), str(img_path)])
        except Exception as e:
            print(e)
            return None
        return mol_path

    def mol_to_inchi(self, mol_path: Path) -> str:
        mol = Chem.MolFromMolFile(str(mol_path))
        try:
            return Chem.MolToInchi(mol)
        except Exception as e:
            raise ValueError(e.args)

    def check_inchi(self, inchi: str, item: MoleculesImageItem) -> int:
        dist = Levenshtein.distance(inchi, item.ground_truth)
        inchi_path = (self.out_path / INCHI_DIR) / f'{item.path.stem}.txt'
        with open(inchi_path, 'w') as f:
            f.write(
                f'Predicted: {inchi}\nGround truth: {item.ground_truth}\nDistance: {dist}'
            )
        return dist

    def ground_truth_inchi_to_image(self, item: MoleculesImageItem):
        out_img_path = self.out_path / IMG_FROM_INCHI_DIR
        mol = inchi_to_mol(item.ground_truth)
        mol_to_png(mol, item.path.stem, out_img_path)

    def process_item(self, item: MoleculesImageItem) -> int:
        if not self._dirs_init:
            self._create_dirs()
        resized_img_path = self.resize(item)
        self.detect_structure(resized_img_path, item)
        restored_img_path = self.restore_image(resized_img_path, item)
        mol_path = self.get_mol_file(item, restored_img_path)
        if not mol_path:
            raise ValueError(f'Imago failed to parse image {item.path}')
        self.ground_truth_inchi_to_image(item)
        inchi = self.mol_to_inchi(mol_path)
        return self.check_inchi(inchi, item)

    def process_batch(self, _slice: slice = slice(None)):
        items = list(self.dataset.items.values())[_slice]
        total_dist = 0
        failed = 0
        for item in items:
            try:
                total_dist += self.process_item(item)
                # FIXME: handle exceptions carefully
            except ValueError as e:
                print(e)
                failed += 1
        succeeded = (len(items) - failed)
        if succeeded:
            print(f'Mean distance: {total_dist / succeeded}')
            print(f'Failed: {failed}')
        else:
            print('All failed')
