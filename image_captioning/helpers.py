import math
import os
import time

import torch

from image_captioning.config import CFG
from image_captioning.models import Encoder, DecoderWithAttention


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def save_models(encoder: Encoder, decoder: DecoderWithAttention, postfix):
    print('NOT SAVING')
    # os.makedirs(CFG.latest_checkpoint, exist_ok=True)
    # enc_pth = CFG.latest_checkpoint / f'encoder_{postfix}.pth'
    # dec_pth = CFG.latest_checkpoint / f'decoder_{postfix}.pth'
    # torch.save(encoder, enc_pth)
    # torch.save(decoder, dec_pth)
    # LOGGER.info(f'\nSaved models:\n{enc_pth}\n{dec_pth}\n')


def load_models(postfix):
    os.makedirs(CFG.latest_checkpoint, exist_ok=True)
    enc_pth = CFG.latest_checkpoint / f'encoder_{postfix}.pth'
    dec_pth = CFG.latest_checkpoint / f'decoder_{postfix}.pth'
    encoder = torch.load(enc_pth)
    decoder = torch.load(dec_pth)
    print(f'\nLoaded models:\n{enc_pth}\n{dec_pth}\n')
    return encoder, decoder
