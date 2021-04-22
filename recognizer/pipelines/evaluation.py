import logging
import os
from typing import Optional

import Levenshtein
from pathlib import Path

import cv2

from recognizer.dataset import MoleculesDataset, MoleculesImageItem
from rdkit import Chem

from recognizer.detector.inference import CascadeRCNNInferenceService
from recognizer.image_processing.normalization import make_even_dimensions
from recognizer.imago_service.imago import ImagoService
from recognizer.pipelines.molecules import inchi_to_mol, mol_to_png
from recognizer.restoration_service.base import BaseRestorationService

# logging.basicConfig(filename='run.log', level=logging.INFO)
logger = logging.getLogger(__name__)

RESIZED_DIR = 'resized'
RESTORED_DIR = 'restored'
DETECTION_DIR = 'detected'
MOL_DIR = 'mol'
INCHI_DIR = 'inchi'

GROUND_TRUTH_DIR = Path('ground_truth')
IMG_FROM_INCHI_DIR = GROUND_TRUTH_DIR / 'img_from_inchi'
MOL_FROM_INCHI_DIR = GROUND_TRUTH_DIR / 'mol_from_inchi'


class EvaluationPipeline:
    def __init__(
        self,
        dataset: MoleculesDataset,
        out_path: Path,
        restoration_service: BaseRestorationService,
        detector: CascadeRCNNInferenceService,
        imago: ImagoService
    ):
        self.dataset = dataset
        self.out_path = out_path
        self.restoration_service = restoration_service
        self.detector = detector
        self.imago = imago
        self._dirs_init = False

    def _create_dirs(self):
        for dirname in (
            RESIZED_DIR,
            RESTORED_DIR,
            DETECTION_DIR,
            MOL_DIR,
            INCHI_DIR,
            IMG_FROM_INCHI_DIR,
            MOL_FROM_INCHI_DIR,
        ):
            os.makedirs(self.out_path / dirname, exist_ok=True)
        self._dirs_init = True

    def resize(self, img_path: Path) -> Path:
        img = cv2.imread(str(img_path))
        img = make_even_dimensions(img)
        res_img_path = (self.out_path / RESIZED_DIR) / img_path.name
        cv2.imwrite(str(res_img_path), img)
        return res_img_path

    def detect_structure(self, img_path: Path, item: MoleculesImageItem):
        detection_path = (self.out_path / DETECTION_DIR) / item.path.name
        self.detector.inference_image(img_path, detection_path)

    def restore_image(self, resized_img_path: Path, item: MoleculesImageItem) -> Path:
        restored_path = (self.out_path / RESTORED_DIR) / item.path.name
        self.restoration_service.restore(resized_img_path, restored_path)
        return restored_path

    def get_mol_file(self, item: MoleculesImageItem, img_path: Path) -> Optional[Path]:
        mol_path = (self.out_path / MOL_DIR) / f'{item.path.stem}.mol'
        self.imago.image_to_mol(img_path, mol_path)
        return mol_path

    @staticmethod
    def mol_to_inchi(mol_path: Path) -> str:
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

    def ground_truth_inchi_to_mol(self, item: MoleculesImageItem) -> Path:
        mol_path = self.out_path / MOL_FROM_INCHI_DIR / self._get_mol_name(item)
        with open(mol_path, 'w') as f:
            mol = inchi_to_mol(item.ground_truth)
            mol_block = Chem.MolToMolBlock(mol)
            f.write(str(mol_block))
        return mol_path

    def ground_truth_inchi_to_image(self, item: MoleculesImageItem) -> Path:
        out_img_path = self.out_path / IMG_FROM_INCHI_DIR
        mol = inchi_to_mol(item.ground_truth)
        mol_to_png(mol, item.path.stem, out_img_path)
        return out_img_path

    @staticmethod
    def _get_mol_name(item: MoleculesImageItem) -> str:
        return f'{item.path.stem}.mol'

    def process_item(self, item: MoleculesImageItem) -> int:
        if not self._dirs_init:
            self._create_dirs()
        self.ground_truth_inchi_to_image(item)
        self.ground_truth_inchi_to_mol(item)
        resized_img_path = self.resize(item.path)
        self.detect_structure(resized_img_path, item)
        restored_img_path = self.restore_image(resized_img_path, item)
        mol_path = self.get_mol_file(item, restored_img_path)
        if not mol_path:
            raise ValueError(f'Imago failed to parse image {item.path}')
        inchi = self.mol_to_inchi(mol_path)
        return self.check_inchi(inchi, item)

    def process_batch(self, _slice: slice = slice(None)):
        items = list(self.dataset.items.values())[_slice]
        total_dist = 0
        failed = 0
        for item in items:
            try:
                dist = self.process_item(item)
                logger.info(f'Image {item.path.name}, distance {dist}')
                total_dist += dist
            except ValueError as e:
                logger.error(e)
                failed += 1
        succeeded = (len(items) - failed)
        if succeeded:
            logger.info(f'Mean distance: {total_dist / succeeded}')
            logger.info(f'Failed: {failed}')
        else:
            logger.info('All failed')
