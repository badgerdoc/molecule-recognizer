from pathlib import Path
from dataset import MoleculesDataset
from inchi_pipeline import InchiPipeline

IN_PATH: Path = Path('datasets/sample_train_dataset/')
OUT_PATH: Path = Path('inchi_images')
ANNOTATIONS_PATH: Path = Path('annotations.json')
IMAGE_SIZE: tuple = (400, 400)
SLICE = slice(0, 10)

if __name__ == '__main__':
    img_dir = Path(IN_PATH, 'train')
    csv_path = Path(IN_PATH, 'train_sample_dataset.csv')
    molecule_dataset = MoleculesDataset(img_dir, csv_path)
    pipeline = InchiPipeline(dataset=molecule_dataset,
                             out_path=OUT_PATH,
                             size=IMAGE_SIZE,
                             annotations_path=ANNOTATIONS_PATH)
    pipeline.process_batch(SLICE)
