from pathlib import Path

INCHI_PREFIX = 'InChI=1S/'
PICKLED_TRAIN_DF = 'content/df.pkl'
PREPROCESSED_TRAIN_DF = 'content/prep_train.pkl'
TOKENIZER_PATH = 'tokenizer.pth'

PROJECT_PATH = Path('content/drive/MyDrive/EPAM/Lectures/Image Captioning')
DRIVE_DATASETS_PATH = PROJECT_PATH / 'datasets'
GPU_DATASET_PATH = DRIVE_DATASETS_PATH / 'bms_fold_0.zip'
CHECKPOINT_PATH = PROJECT_PATH / 'checkpoints'

LATEST = 'latest'
EXPERIMENT_TITLE = 'EfficientNetV2'

OUTPUT_DIR = Path('content/output')
