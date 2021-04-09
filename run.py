from pathlib import Path

from fastai.basic_train import load_learner

from dataset import MoleculesDataset
from pipeline import Pipeline

if __name__ == '__main__':
    img_dir = Path('datasets/sample_train_dataset/train')
    csv_path = Path('datasets/sample_train_dataset/train_sample_dataset.csv')
    molecule_dataset = MoleculesDataset(img_dir, csv_path)
    model = load_learner('models', file='gen.pkl')
    pipeline = Pipeline(molecule_dataset, Path('output'), model)
    pipeline.process_batch(slice(0, 5))
