import os
import subprocess
import Levenshtein
from pathlib import Path

import cv2
from fastai.basic_train import Learner
from fastai.vision import open_image

from dataset import MoleculesDataset, MoleculesImageItem
from rdkit import Chem

RESIZED_DIR = 'resized'
RESTORED_DIR = 'restored'
MOL_DIR = 'mol'
INCHI_DIR = 'inchi'


class Pipeline:
    def __init__(self, dataset: MoleculesDataset, out_path: Path, model: Learner):
        self.dataset = dataset
        self.out_path = out_path
        self._dirs_init = False
        self.model = model

    def _create_dirs(self):
        for dirname in (RESIZED_DIR, RESTORED_DIR, MOL_DIR, INCHI_DIR):
            os.makedirs(self.out_path / dirname, exist_ok=True)
        self._dirs_init = True

    def resize(self, item: MoleculesImageItem) -> Path:
        img = cv2.imread(str(item.path))
        h, w = img.shape[:2]
        if h % 2 == 1: h += 1
        if w % 2 == 1: w += 1
        img = cv2.resize(img, (w, h))
        img_path = (self.out_path / RESIZED_DIR) / f'{item.path.name}'
        cv2.imwrite(str(img_path), img)
        return img_path

    def restore_image(self, item: MoleculesImageItem) -> Path:
        resized_img_path = self.resize(item)
        img = open_image(resized_img_path)
        pred = self.model.predict(img)
        restored_path = (self.out_path / RESTORED_DIR) / f'{item.path.name}'
        pred[0].save(restored_path)
        return restored_path

    def get_mol_file(self, item: MoleculesImageItem, img_path: Path):
        mol_path = (self.out_path / MOL_DIR) / f'{item.path.stem}.mol'
        subprocess.run(['bin/imago_console', '-o', str(mol_path), str(img_path)])
        return mol_path

    def mol_to_inchi(self, mol_path: Path):
        mol = Chem.MolFromMolFile(str(mol_path))
        return Chem.MolToInchi(mol)

    def check_inchi(self, inchi: str, item: MoleculesImageItem) -> int:
        dist = Levenshtein.distance(inchi, item.ground_truth)
        inchi_path = (self.out_path / INCHI_DIR) / f'{item.path.stem}.txt'
        with open(inchi_path, 'w') as f:
            f.write(
                f'Predicted: {inchi}\nGround truth: {item.ground_truth}\nDistance: {dist}'
            )
        return dist

    def process_item(self, item: MoleculesImageItem) -> int:
        if not self._dirs_init:
            self._create_dirs()
        restored_img_path = self.restore_image(item)
        mol_path = self.get_mol_file(item, restored_img_path)
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
            except Exception:
                failed += 1
        print(f'Mean distance: {total_dist / len(items)}')
        print(f'Failed: {failed}')
