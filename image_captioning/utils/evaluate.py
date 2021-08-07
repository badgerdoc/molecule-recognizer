import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from image_captioning.constants import ENCODER_CONFIG_YML, PIPELINE_CONFIG_YML
from image_captioning.datasets import TestDataset
from image_captioning.tokenizer import Tokenizer
from image_captioning.train import split_df_into_folds
from image_captioning.utils.helpers import load_config, load_checkpoint, seed_torch
from image_captioning.utils.training import get_transforms, valid_fn


def evaluation(checkpoint: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipeline_cfg = load_config(checkpoint / PIPELINE_CONFIG_YML)

    project_dir = pipeline_cfg.workdir.parent
    tokenizer_path = project_dir / 'tokenizer.pth'
    tokenizer: Tokenizer = torch.load(tokenizer_path)
    prep_train_df = pd.read_pickle(pipeline_cfg.preprocessed_train_df_path)
    encoder, decoder = load_checkpoint(checkpoint, device)
    encoder_config = load_config(checkpoint / ENCODER_CONFIG_YML)
    image_transforms = get_transforms(encoder_config)

    seed_torch(seed=pipeline_cfg.seed)
    folds = split_df_into_folds(prep_train_df, pipeline_cfg)

    val_idx = folds[folds['fold'] == pipeline_cfg.dataset.validation_fold].index

    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['InChI'].values

    images_path = pipeline_cfg.dataset.images_path

    valid_dataset = TestDataset(
        valid_folds, images_path, tokenizer, transform=image_transforms
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=pipeline_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    valid_fn(valid_loader=valid_loader,
             encoder=encoder,
             decoder=decoder,
             tokenizer=tokenizer,
             valid_labels=valid_labels,
             device=device)


if __name__ == '__main__':
    from image_captioning.cli.cli_template import evaluate
    pipeline_config_path = Path(r"D:\EPAM\EpamLab\MolecularRecognition\molecule-recognizer\workdir\checkpoints\effnetv2_l_300x400_transformer-encoder-decoder\latest\pipeline_config.yml")
    checkpoint_path = Path(r"D:\EPAM\EpamLab\MolecularRecognition\molecule-recognizer\workdir\checkpoints\effnetv2_l_300x400_transformer-encoder-decoder\latest")
    evaluate(pipeline_config_path, checkpoint_path)
