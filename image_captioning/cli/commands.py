from pathlib import Path

import pandas as pd
import torch

from image_captioning import (CaptionTransformerConfig, EfficientNetV2Config,
                              get_model)
from image_captioning.base import LibRegistry
from image_captioning.pipeline.config import PipelineConfig
from image_captioning.tokenizer import Tokenizer
from image_captioning.train import split_df_into_folds
from image_captioning.utils.helpers import seed_torch, load_config, load_models, PIPELINE_CONFIG_YML, \
    ENCODER_CONFIG_YML, DECODER_CONFIG_YML
from image_captioning.utils.training import train_loop


def create_new_model(pipeline: Path, encoder: Path, decoder: Path):
    # FIXME: temporary fix to register configs, should be removed, when config registration is fixed
    LibRegistry.configs["PipelineConfig"] = PipelineConfig
    LibRegistry.configs["CaptionTransformerConfig"] = CaptionTransformerConfig
    LibRegistry.configs["EfficientNetV2Config"] = EfficientNetV2Config

    pipeline_cfg = load_config(Path(pipeline))
    encoder_cfg = load_config(Path(encoder))
    decoder_cfg = load_config(Path(decoder))

    tokenizer: Tokenizer = torch.load(pipeline_cfg.tokenizer_path)
    prep_train_df = pd.read_pickle(pipeline_cfg.preprocessed_train_df_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = get_model(encoder_cfg)
    decoder = get_model(decoder_cfg)

    seed_torch(seed=pipeline_cfg.seed)
    folds = split_df_into_folds(prep_train_df, pipeline_cfg)

    train_loop(
        encoder=encoder,
        decoder=decoder,
        folds=folds,
        validation_fold=2,
        tokenizer=tokenizer,
        device=device,
        pipeline_config=pipeline_cfg,
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
    )


def resume_from(checkpoint: Path):
    # FIXME: temporary fix to register configs, should be removed, when config registration is fixed
    LibRegistry.configs["PipelineConfig"] = PipelineConfig
    LibRegistry.configs["CaptionTransformerConfig"] = CaptionTransformerConfig
    LibRegistry.configs["EfficientNetV2Config"] = EfficientNetV2Config

    pipeline_cfg = load_config(Path(checkpoint / PIPELINE_CONFIG_YML))
    encoder_cfg = load_config(Path(checkpoint / ENCODER_CONFIG_YML))
    decoder_cfg = load_config(Path(checkpoint / DECODER_CONFIG_YML))

    tokenizer: Tokenizer = torch.load(pipeline_cfg.tokenizer_path)
    prep_train_df = pd.read_pickle(pipeline_cfg.preprocessed_train_df_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder = load_models('latest', encoder_cfg, decoder_cfg, pipeline_cfg)

    seed_torch(seed=pipeline_cfg.seed)
    folds = split_df_into_folds(prep_train_df, pipeline_cfg)

    train_loop(
            encoder=encoder,
            decoder=decoder,
            folds=folds,
            validation_fold=2,
            tokenizer=tokenizer,
            device=device,
            pipeline_config=pipeline_cfg,
            encoder_config=encoder_cfg,
            decoder_config=decoder_cfg,
        )