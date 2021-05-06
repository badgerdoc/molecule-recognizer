import logging
import os
from typing import Optional

import Levenshtein
from pathlib import Path

import cv2

from recognizer.common.molecule_generator import MoleculeImageGenerator
from recognizer.dataset import MoleculesDataset, MoleculesImageItem
from rdkit import Chem

from recognizer.detector.inference import CascadeRCNNInferenceService
from recognizer.image_processing.normalization import make_even_dimensions
from recognizer.image_processing.utils import save_img
from recognizer.imago_service.imago import ImagoService
from recognizer.pipelines.molecules import inchi_to_mol
from recognizer.restoration_service.base import BaseRestorationService

# logging.basicConfig(filename='run.log', level=logging.INFO)
from recognizer.restoration_service.text import TextRestorationService

GT_IMG_SIZE = (400, 400)
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
        detector: CascadeRCNNInferenceService,  # TODO: Going to be deprecated
        text_restoration_service: TextRestorationService,
        imago: ImagoService,
        restore_text: bool = False
    ):
        self.dataset = dataset
        self.out_path = out_path
        self.restoration_service = restoration_service
        self.detector = detector
        self.text_restoration_service = text_restoration_service
        self.imago = imago
        self.molecule_generator = MoleculeImageGenerator(add_padding=False)
        self.restore_text = restore_text
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
        return res_img_path, img

    def detect_structure(self, img, item: MoleculesImageItem):
        detection_path = (self.out_path / DETECTION_DIR) / item.path.name
        structure = self.detector.inference_image(img)
        self.detector.visualize_boxes(structure, img, detection_path)

    def restore_image(self, resized_img_path: Path, item: MoleculesImageItem) -> Path:
        restored_path = (self.out_path / RESTORED_DIR) / item.path.name
        self.restoration_service.restore(resized_img_path, restored_path)
        if self.restore_text:
            ref_img = cv2.imread(str(restored_path))
            img = cv2.imread(str(restored_path))
            restored_img = self.text_restoration_service.restore(ref_img, img)
            cv2.imwrite(str(restored_path), restored_img)
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

    def ground_truth_inchi_to_image(self, item: MoleculesImageItem):
        img_path = self.out_path / IMG_FROM_INCHI_DIR / item.path.name
        img = self.molecule_generator.inchi_to_image(
            item.ground_truth,
            GT_IMG_SIZE
        )
        save_img(img_path, img)

    @staticmethod
    def _get_mol_name(item: MoleculesImageItem) -> str:
        return f'{item.path.stem}.mol'

    def process_item(self, item: MoleculesImageItem) -> int:
        if not self._dirs_init:
            self._create_dirs()
#       self.ground_truth_inchi_to_image(item)
        self.ground_truth_inchi_to_mol(item)
        resized_img_path, resized_img = self.resize(item.path)
#       self.detect_structure(resized_img, item)
        restored_img_path = self.restore_image(resized_img_path, item)
        mol_path = self.get_mol_file(item, restored_img_path)
        if not mol_path:
            raise ValueError(f'Imago failed to parse image {item.path}')
        return 0

    def process_batch(self, _slice: slice = slice(None)):
        items = list(self.dataset.items.values())[_slice]
        total_dist = 0
        failed = 0
        succeeded = 0
        for item in items:
            try:
                dist = self.process_item(item)
                logger.info(f'Image {item.path.name}, distance {dist}, succeeded {succeeded}')
                succeeded += 1
                total_dist += dist
            except ValueError as e:
                logger.error(e)
                failed += 1
        if succeeded:
            logger.info(f'Mean distance: {total_dist / succeeded}')
            logger.info(f'Failed: {failed}')
        else:
            logger.info('All failed')
