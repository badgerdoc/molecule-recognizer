import json
import os
from pathlib import Path
from dataset import MoleculesDataset, MoleculesImageItem
from molecules import process_single


ANNOTATIONS_PATH = 'annotations.json'


class InchiPipeline:
    def __init__(self, dataset: MoleculesDataset,
                 out_path: Path,
                 size: tuple,
                 annotations_path: Path = ANNOTATIONS_PATH
    ):
        self.dataset = dataset
        self.out_path = out_path
        self.annotations_path = annotations_path
        self.size = size

    def _create_dirs(self):
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
            print(f"{self.out_path} dir was created")

    def process_item(self, item: MoleculesImageItem) -> int:
        return process_single(item.ground_truth, self.out_path, name=item.path.stem, img_id=1)

    def process_batch(self, _slice: slice = slice(None)):
        self._create_dirs()

        items = list(self.dataset.items.values())[_slice]
        annotations = []
        for item in items:
            _, annotation = self.process_item(item)
            annotations.append(annotation)
        print(f"{len(items)} images saved to {self.out_path}")

        with open(Path(self.out_path, self.annotations_path), 'w') as f:
            json.dump(annotations, f)
        print(f"Annotations were saved to {self.annotations_path}")

        if self.size:
            print(f"size: {self.size}")
