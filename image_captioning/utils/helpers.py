import json
import math
import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from pydantic import BaseModel
from image_captioning.base import LibRegistry
from image_captioning.constants import PIPELINE_CONFIG_YML, DECODER_CONFIG_YML, ENCODER_CONFIG_YML, DECODER_FILENAME, \
    ENCODER_FILENAME
from image_captioning.exceptions import FolderDoesNotExist

CFG_CLS = '_cfg_cls'

if TYPE_CHECKING:
    from image_captioning.base import (
        DecoderBaseConfig,
        EncoderBaseConfig
)
    from image_captioning.pipeline.config import PipelineConfig


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seconds2minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_remaining_time(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (seconds2minutes(s), seconds2minutes(rs))


def get_model_id(
    encoder_config: 'EncoderBaseConfig', decoder_config: 'DecoderBaseConfig'
) -> str:
    return f'{encoder_config.id}_{decoder_config.id}'


def _add_cfg_class(data: dict, config: BaseModel) -> dict:
    data[CFG_CLS] = config.__class__.__name__
    return data


def _get_cfg_class(data: dict):
    cls = LibRegistry.configs.get(data[CFG_CLS])
    if not cls:
        raise KeyError(f'Config class "{data[CFG_CLS]}" not found in `LibRegistry`.')
    return cls


def save_config(config: BaseModel, file_path: Path):
    with open(file_path, 'w') as f:
        f.write(
            yaml.dump(
                data=_add_cfg_class(
                    json.loads(
                        config.json()  # Converting to json first to simulate yaml.safe_dump, which does not work here
                    ),
                    config
                )
            )
        )


def load_config(file_path: Path):
    with open(file_path, 'r') as f:
        data = yaml.load(f)  # warning. could be replaced by safe_load?
        cls = _get_cfg_class(data)
        data.pop(CFG_CLS)
        return cls(**data)


def save_checkpoint(
    name,
    encoder,
    decoder,
    encoder_config: 'EncoderBaseConfig',
    decoder_config: 'DecoderBaseConfig',
    pipeline_config: 'PipelineConfig',
):
    model_id = get_model_id(encoder_config, decoder_config)
    checkpoint = pipeline_config.checkpoint_path / model_id / name
    os.makedirs(checkpoint, exist_ok=True)

    torch.save(encoder, checkpoint / ENCODER_FILENAME)
    torch.save(decoder, checkpoint / DECODER_FILENAME)

    save_config(encoder_config, checkpoint / ENCODER_CONFIG_YML)
    save_config(decoder_config, checkpoint / DECODER_CONFIG_YML)
    save_config(pipeline_config, checkpoint / PIPELINE_CONFIG_YML)
    print(f'\nSaved checkpoint: {checkpoint}\n')


def load_checkpoint(checkpoint: Path, device: str):
    if not os.path.exists(checkpoint):
        raise FolderDoesNotExist(f"Checkpoint folder at {checkpoint} doesn't exist.")

    encoder = torch.load(checkpoint / ENCODER_FILENAME, map_location=device)
    decoder = torch.load(checkpoint / DECODER_FILENAME, map_location=device)

    print(f'\nLoaded checkpoint: {checkpoint}\n')
    return encoder, decoder


def load_models(name, encoder_config: 'EncoderBaseConfig', decoder_config: 'DecoderBaseConfig',
                pipeline_config: 'PipelineConfig'):
    # FIXME Reuse load_checkpoint
    model_id = get_model_id(encoder_config, decoder_config)
    checkpoint = pipeline_config.checkpoint_path / model_id / name
    enc_pth = checkpoint / ENCODER_FILENAME
    dec_pth = checkpoint / DECODER_FILENAME
    encoder = torch.load(enc_pth)
    decoder = torch.load(dec_pth)
    print(f'\nLoaded models:\n{enc_pth}\n{dec_pth}\n')
    return encoder, decoder


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
