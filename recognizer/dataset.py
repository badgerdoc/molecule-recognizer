from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path
import pandas as pd


@dataclass
class MoleculesImageItem:
    path: Path
    ground_truth: Optional[str] = None

    @property
    def svg_name(self):
        return f'{self.path.stem}.svg'


class MoleculesDataset:
    def __init__(self, img_dir: Path, csv_path: Path):
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.items: Dict[str, MoleculesImageItem] = {}
        self._load()

    def _load(self):
        self._load_images()
        self._load_ground_truth()

    def _load_images(self):
        for l1_dir in self.img_dir.iterdir():
            for l2_dir in l1_dir.iterdir():
                for l3_dir in l2_dir.iterdir():
                    for img_path in l3_dir.iterdir():
                        self.items[img_path.stem] = MoleculesImageItem(img_path)

    def _load_ground_truth(self):
        gt = pd.read_csv(self.csv_path)
        for idx, rec in gt.iterrows():
            self.items[rec['image_id']].ground_truth = rec['InChI']
