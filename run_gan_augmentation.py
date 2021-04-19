from pathlib import Path

from recognizer.dataset import MoleculesDataset
from recognizer.pipelines.augmentation import GanAugmentationPipeline

if __name__ == '__main__':
    img_dir = Path('datasets/sample_train_dataset/train')
    csv_path = Path('datasets/sample_train_dataset/train_sample_dataset.csv')
    molecule_dataset = MoleculesDataset(img_dir, csv_path)

    pipeline = GanAugmentationPipeline(
        molecule_dataset,
        Path('augmented_dataset')
    )
    pipeline.process_batch(slice(0, 25))
