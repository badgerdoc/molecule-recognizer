import Levenshtein
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import DataLoader

from image_captioning.constants import ENCODER_CONFIG_YML
from image_captioning.datasets import TestDataset
from image_captioning.tokenizer import Tokenizer
from image_captioning.train import split_df_into_folds
from image_captioning.utils.helpers import load_config, load_checkpoint, seed_torch
from image_captioning.utils.training import get_transforms


def get_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score


def valid_fn(
        encoder,
        decoder,
        folds,
        validation_fold,
        tokenizer,
        device,
        pipeline_config,
        encoder_config,
        val_slice=100
):
    val_idx = folds[folds['fold'] == validation_fold].index

    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['InChI'].values

    images_path = pipeline_config.dataset.images_path

    image_transforms = get_transforms(encoder_config)

    valid_dataset = TestDataset(
        valid_folds, images_path, tokenizer, transform=image_transforms
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=pipeline_config.batch_size,
        shuffle=False,
        num_workers=pipeline_config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    encoder.eval()
    decoder.eval()

    text_preds = []
    max_length = 300
    for i, images in enumerate(valid_loader):
        batch_size = images.size(0)
        ys = torch.full((batch_size, 1), tokenizer.sos_idx, dtype=torch.long).to(device)
        images = images.to(device)

        with torch.no_grad():
            features = encoder(images)
            features.to(device)

            memory = decoder.encode(features)

            for j in range(max_length - 1):
                out = decoder.decode(ys, memory)
                pred = decoder.generator(out)
                _, next_token = torch.max(pred, dim=2)
                ys = torch.cat((ys, next_token[-1].unsqueeze(1)), dim=1)
                if next_token[-1].item() == tokenizer.eos_idx:
                    break

            text_preds.append(ys.squeeze().tolist())

        if i == val_slice:
            break

    text_preds = tokenizer.predict_captions(text_preds)
    text_preds = [f"InChI=1S/{text[5:]}" for text in text_preds]
    score = get_score(valid_labels[:val_slice + 1], text_preds)
    print(score)
    print()


def evaluate(pipeline: Path, checkpoint: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipeline_cfg = load_config(Path(pipeline))

    project_dir = Path(r'D:\EPAM\EpamLab\MolecularRecognition')
    tokenizer_path = project_dir / 'tokenizer.pth'
    tokenizer: Tokenizer = torch.load(tokenizer_path)
    prep_train_df = pd.read_pickle(pipeline_cfg.preprocessed_train_df_path)
    encoder, decoder = load_checkpoint(checkpoint, device)
    encoder_config = load_config(checkpoint / ENCODER_CONFIG_YML)
    image_transforms = get_transforms(encoder_config)

    seed_torch(seed=pipeline_cfg.seed)
    folds = split_df_into_folds(prep_train_df, pipeline_cfg)

    valid_fn(encoder=encoder,
             decoder=decoder,
             folds=folds,
             validation_fold=pipeline_cfg.dataset.validation_fold,
             tokenizer=tokenizer,
             device=device,
             pipeline_config=pipeline_cfg,
             encoder_config=encoder_config)
