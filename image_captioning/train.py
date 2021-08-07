from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from image_captioning import get_model, EfficientNetV2Config, CaptionTransformerConfig
from image_captioning.pipeline.config import (
    CheckpointConfig,
    DatasetConfig,
    OptimizerConfig,
    PipelineConfig,
    SchedulerConfig,
)

from image_captioning.models.encoders.efficient_net_v2.encoder import EFFNET_V2_L

from image_captioning.tokenizer import Tokenizer
from image_captioning.utils.helpers import seed_torch, load_models
from image_captioning.utils.training import train_loop


def main():
    # TODO: replace prints with logging
    project_dir = Path(r'D:\EPAM\EpamLab\MolecularRecognition')
    tokenizer_path = Path(project_dir / r'molecule-recognizer\tokenizer.pth')
    preprocessed_train_df = Path(project_dir / r'molecule-recognizer\content\prep_train.pkl')
    tokenizer: Tokenizer = torch.load(tokenizer_path)
    prep_train_df = pd.read_pickle(preprocessed_train_df)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pipeline_config = PipelineConfig(
        workdir=Path(project_dir / r'molecule-recognizer\workdir'),
        dataset=DatasetConfig(
            labels_path=Path(project_dir / r'molecule-recognizer\bms_fold_0\train.csv'),
            images_path=Path(project_dir / r'molecule-recognizer\bms_fold_0\train'),
            n_fold=5,
            validation_fold=2
        ),
        checkpoint=CheckpointConfig(
            load_from_checkpoint=True,
            frequency=500,
            number_to_keep=5,
            resume_from='latest',  # TODO: implement resume functionality
            skip_steps=True,
            samples_trained=1376000
        ),
        encoder_train=OptimizerConfig(lr=1e-4, wd=1e-6),
        decoder_train=OptimizerConfig(lr=4e-4, wd=1e-6),
        scheduler=SchedulerConfig(
            name='CosineAnnealingLR', params={'T_max': 4, 'eta_min': 1e-6}
        ),
        seed=1,
        epochs=1,
        batch_size=4,
        num_workers=2,
        print_frequency=1,
        gradient_accumulation_steps=1,
        max_grad_norm=5,
    )

    seed_torch(seed=pipeline_config.seed)

    encoder_config = EfficientNetV2Config(name='effnetv2', model=EFFNET_V2_L, size=(300, 400))
    decoder_config = CaptionTransformerConfig(
        name='transformer-encoder-decoder',
        num_encoder_layers=3,
        num_decoder_layers=3,
        features_size=1792,
        emb_size=512,
        nhead=8,
        tgt_vocab_size=len(tokenizer.itos),
        dim_feedforward=512,
        dropout=0.1,
    )

    if not pipeline_config.checkpoint.load_from_checkpoint:
        encoder = get_model(encoder_config)
        decoder = get_model(decoder_config)
        print("WARNING: NO LOADING!")
    else:
        encoder, decoder = load_models("latest", encoder_config, decoder_config, pipeline_config)

    folds = split_df_into_folds(prep_train_df, pipeline_config)

    train_loop(
        encoder=encoder,
        decoder=decoder,
        folds=folds,
        validation_fold=2,
        tokenizer=tokenizer,
        device=device,
        pipeline_config=pipeline_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


def split_df_into_folds(prep_train_df, pipeline_config) -> pd.DataFrame:
    folds = prep_train_df.copy()
    k_fold = StratifiedKFold(
        n_splits=pipeline_config.dataset.n_fold,
        shuffle=True,
        random_state=pipeline_config.seed,
    )
    for n, (train_index, val_index) in enumerate(
        k_fold.split(folds, folds['InChI_length'])
    ):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    print(folds.groupby(['fold']).size())
    return folds


if __name__ == '__main__':
    main()
