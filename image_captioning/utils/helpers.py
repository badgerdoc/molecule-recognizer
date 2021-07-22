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

if TYPE_CHECKING:
    from image_captioning.models.base import (
        DecoderBaseConfig,
        EncoderBaseConfig,
    )
    from image_captioning.configs.pipeline import PipelineConfig

PIPELINE_CONFIG_YML = 'pipeline_config.yml'
DECODER_CONFIG_YML = 'decoder_config.yml'
ENCODER_CONFIG_YML = 'encoder_config.yml'
DECODER_FILENAME = 'decoder.pth'
ENCODER_FILENAME = 'encoder.pth'


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


def save_config(config: BaseModel, file_path: Path):
    with open(file_path, 'w') as f:
        f.write(
            yaml.dump(
                data=json.loads(
                    config.json()  # Converting to json first to simulate yaml.safe_dump, which does not work here
                )
            )
        )


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


# def load_models(postfix):
#     os.makedirs(CFG.latest_checkpoint, exist_ok=True)
#     enc_pth = CFG.latest_checkpoint / f'encoder_{postfix}.pth'
#     dec_pth = CFG.latest_checkpoint / f'decoder_{postfix}.pth'
#     encoder = torch.load(enc_pth)
#     decoder = torch.load(dec_pth)
#     print(f'\nLoaded models:\n{enc_pth}\n{dec_pth}\n')
#     return encoder, decoder


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
