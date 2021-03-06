from pathlib import Path

from fastai.basic_train import load_learner

from recognizer.classifier.fastai_classifier import FastaiAtomClassifier
from recognizer.dataset import MoleculesDataset
from recognizer.detector.inference import CascadeRCNNInferenceService
from recognizer.drawing.text import TextRenderer, TextRendererConfig
from recognizer.imago_service.imago import ImagoService
from recognizer.pipelines.evaluation import EvaluationPipeline
from recognizer.restoration_service.fastai_gan import FastaiGANService
from recognizer.restoration_service.mprnet import MPRNETService
from recognizer.restoration_service.text import TextRestorationService

if __name__ == '__main__':
    img_dir = Path('datasets/sample_train_dataset/train')
    csv_path = Path('datasets/sample_train_dataset/train_sample_dataset.csv')
    molecule_dataset = MoleculesDataset(img_dir, csv_path)

    # restoration_service = FastaiGANService(model_path=Path('models/gen_8.pkl'))
    restoration_service = MPRNETService(model_path=Path('models/mprnet.pth'))

    det_cfg_path = Path('configs/detector_config.py')
    det_model_path = Path('models/epoch_15.pth')
    detector_service = CascadeRCNNInferenceService(det_cfg_path, det_model_path, True)

    text_renderer_cfg = TextRendererConfig(
        font_path=Path('fonts/Inconsolata-Regular.ttf')
    )
    text_renderer = TextRenderer(text_renderer_cfg)
    atom_classifier = FastaiAtomClassifier(Path('models/classifier.pth'))
    text_restoration = TextRestorationService(
        atom_classifier, detector_service, text_renderer
    )

    imago = ImagoService(Path('bin/imago_console'))

    pipeline = EvaluationPipeline(
        molecule_dataset, Path('output'),
        restoration_service,
        detector_service,
        text_restoration,
        imago
    )
    pipeline.process_batch(slice(10, 11))
