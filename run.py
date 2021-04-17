from pathlib import Path

from fastai.basic_train import load_learner

from dataset import MoleculesDataset
from detector.inference import CascadeRCNNInferenceService
from imago_service.imago import ImagoService
from pipeline import Pipeline

if __name__ == '__main__':
    img_dir = Path('datasets/sample_train_dataset/train')
    csv_path = Path('datasets/sample_train_dataset/train_sample_dataset.csv')
    molecule_dataset = MoleculesDataset(img_dir, csv_path)

    gan_model = load_learner('models', file='gen_8.pkl')

    det_cfg_path = Path('configs/detector_config.py')
    det_model_path = Path('models/epoch_15.pth')
    detector_service = CascadeRCNNInferenceService(det_cfg_path, det_model_path, True)

    imago = ImagoService(Path('bin/imago_feature_console'))

    pipeline = Pipeline(molecule_dataset, Path('output'), gan_model, detector_service, imago)
    pipeline.process_batch(slice(0, 5))
