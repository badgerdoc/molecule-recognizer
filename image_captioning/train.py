import json
import math
import os
import random
import re
import time
from pathlib import Path
from pprint import pprint

import cv2
import Levenshtein
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from albumentations import (
    Blur,
    Compose,
    Cutout,
    HorizontalFlip,
    IAAAdditiveGaussianNoise,
    ImageOnlyTransform,
    Normalize,
    OneOf,
    RandomBrightness,
    RandomContrast,
    RandomCrop,
    Resize,
    Rotate,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
)
from albumentations.augmentations.transforms import *
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertGenerationDecoder, DistilBertTokenizer

from config import CFG
from constants import *
from effnetv2 import *
from effnetv2 import MBConv, _make_divisible, conv_1x1_bn, conv_3x3_bn
from image_captioning.datasets import TrainDataset, TestDataset, bms_collate
from image_captioning.helpers import save_models, AverageMeter, timeSince
from image_captioning.models import Encoder, DecoderWithAttention
from tokenization import Tokenizer, apply_tokenizer_to_df  # required


def get_transforms():
    return Compose([
        Resize(CFG.size[0], CFG.size[1]),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score


def init_logger(log_file=OUTPUT_DIR / 'train.log'):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_fn(train_loader, encoder, decoder, criterion,
             encoder_optimizer, decoder_optimizer, epoch,
             encoder_scheduler, decoder_scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    encoder.train()
    decoder.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels, label_lengths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        batch_size = images.size(0)
        features = encoder(images)
        predictions, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, labels, label_lengths)
        # predictions = decoder(inputs_embeds=features)
        targets = caps_sorted[:, 1:]
        predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss = criterion(predictions, targets)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        loss.backward()
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), CFG.max_grad_norm)
        decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step !=0 and (step % CFG.print_freq == 0 or step == (len(train_loader)-1)):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Encoder Grad: {encoder_grad_norm:.4f}  '
                  'Decoder Grad: {decoder_grad_norm:.4f}  '
                  #'Encoder LR: {encoder_lr:.6f}  '
                  #'Decoder LR: {decoder_lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   encoder_grad_norm=encoder_grad_norm,
                   decoder_grad_norm=decoder_grad_norm,
                   #encoder_lr=encoder_scheduler.get_lr()[0],
                   #decoder_lr=decoder_scheduler.get_lr()[0],
                   ))
        if step % CFG.checkpoint_freq == 0 or step == (len(train_loader)-1):
            save_models(encoder, decoder, postfix=LATEST)
    return losses.avg


def train_loop(folds, fold, tokenizer, device):
    print(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['InChI'].values

    # train_dataset = TrainDataset(train_folds, tokenizer, transform=get_transforms())
    train_dataset = TrainDataset(train_folds, CFG.train_path, tokenizer, transform=get_transforms())
    # valid_dataset = TestDataset(valid_folds, transform=get_transforms())
    valid_dataset = TestDataset(valid_folds, CFG.train_path, tokenizer, transform=get_transforms())
    from functools import partial
    collate = partial(bms_collate, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=False)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True,
                                          eps=CFG.eps)
        elif CFG.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    # if CFG.continue_from:
    #     encoder, decoder = load_models(CFG.continue_from)
    # else:
    print('WARNING: NO LOADING!')
    encoder = Encoder(CFG.encoder_model, pretrained=True)
    decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                                embed_dim=CFG.embed_dim,
                                decoder_dim=CFG.decoder_dim,
                                vocab_size=len(tokenizer),
                                dropout=CFG.dropout,
                                device=device,
                                encoder_dim=CFG.encoder_dim)

    encoder.to(device)
    decoder.to(device)
    encoder_optimizer = Adam(encoder.parameters(), lr=CFG.encoder_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    encoder_scheduler = get_scheduler(encoder_optimizer)
    decoder_optimizer = Adam(decoder.parameters(), lr=CFG.decoder_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    decoder_scheduler = get_scheduler(decoder_optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])

    best_score = np.inf
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, encoder, decoder, criterion,
                            encoder_optimizer, decoder_optimizer, epoch,
                            encoder_scheduler, decoder_scheduler, device)

        # eval
        # text_preds = valid_fn(valid_loader, encoder, decoder, tokenizer, criterion, device)
        # text_preds = [f"InChI=1S/{text}" for text in text_preds]
        # print(f"labels: {valid_labels[:5]}")
        # print(f"preds: {text_preds[:5]}")
        #
        # # scoring
        # score = get_score(valid_labels, text_preds)
        #
        # if isinstance(encoder_scheduler, ReduceLROnPlateau):
        #     encoder_scheduler.step(score)
        # elif isinstance(encoder_scheduler, CosineAnnealingLR):
        #     encoder_scheduler.step()
        # elif isinstance(encoder_scheduler, CosineAnnealingWarmRestarts):
        #     encoder_scheduler.step()
        #
        # if isinstance(decoder_scheduler, ReduceLROnPlateau):
        #     decoder_scheduler.step(score)
        # elif isinstance(decoder_scheduler, CosineAnnealingLR):
        #     decoder_scheduler.step()
        # elif isinstance(decoder_scheduler, CosineAnnealingWarmRestarts):
        #     decoder_scheduler.step()
        #
        # elapsed = time.time() - start_time
        #
        # print(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        # print(f'Epoch {epoch + 1} - Score: {score:.4f}')
        #
        # if score < best_score:
        #     best_score = score
        #     print(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
        #     torch.save({'encoder': encoder.state_dict(),
        #                 'encoder_optimizer': encoder_optimizer.state_dict(),
        #                 'encoder_scheduler': encoder_scheduler.state_dict(),
        #                 'decoder': decoder.state_dict(),
        #                 'decoder_optimizer': decoder_optimizer.state_dict(),
        #                 'decoder_scheduler': decoder_scheduler.state_dict(),
        #                 'text_preds': text_preds,
        #                 },
        #                OUTPUT_DIR / f'{CFG.effnet_cfg.name}_fold{fold}_best.pth')


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER = init_logger()
    seed_torch(seed=CFG.seed)
    tokenizer = torch.load(TOKENIZER_PATH)
    prep_train_df = pd.read_pickle(PREPROCESSED_TRAIN_DF)
    train_dataset = TrainDataset(prep_train_df, CFG.train_path, tokenizer, transform=get_transforms())

    folds = prep_train_df.copy()
    Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['InChI_length'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    print(folds.groupby(['fold']).size())

    if CFG.train:
        oof_df = pd.DataFrame()

        # train on all folds
        # for fold in range(CFG.n_fold):
        #     if fold in CFG.trn_fold:
        #         train_loop(folds, fold)

        # select specific fold
        train_loop(folds, 2, tokenizer, device)


if __name__ == '__main__':
    main()
