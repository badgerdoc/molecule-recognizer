import time
from typing import TYPE_CHECKING

import Levenshtein
import numpy as np
import torch
import torch.nn as nn
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from image_captioning.datasets import TestDataset, TrainDataset, bms_collate
from image_captioning.tokenizer import Tokenizer
from image_captioning.utils.helpers import (
    AverageMeter,
    get_remaining_time,
    save_checkpoint,
)

if TYPE_CHECKING:
    from pandas import DataFrame

    from image_captioning.base import EncoderBaseConfig, DecoderBaseConfig
    from image_captioning.pipeline.config import PipelineConfig


def get_transforms(encoder_config: 'EncoderBaseConfig'):
    return Compose(
        [
            Resize(encoder_config.size[0], encoder_config.size[1]),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
            .masked_fill(mask == 0, float('-inf'))
            .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(tgt, tokenizer, device):
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

    tgt_padding_mask = (tgt == tokenizer.pad_idx).transpose(0, 1)
    return tgt_mask, tgt_padding_mask


def get_scheduler(optimizer, pipeline_config: 'PipelineConfig'):
    sched_conf = pipeline_config.scheduler
    # FIXME: Support other schedulers
    # if CFG.scheduler == 'ReduceLROnPlateau':
    #     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True,
    #                                   eps=CFG.eps)
    # elif CFG.scheduler == 'CosineAnnealingLR':
    #     scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
    # elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
    #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
    if sched_conf.name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer, last_epoch=-1, **pipeline_config.scheduler.params
        )
    else:
        raise ValueError('Unsupported scheduler')
    return scheduler


def train_loop(
        encoder,
        decoder,
        folds: 'DataFrame',
        validation_fold: int,
        tokenizer: Tokenizer,
        device: str,
        pipeline_config: 'PipelineConfig',
        encoder_config: 'EncoderBaseConfig',
        decoder_config: 'DecoderBaseConfig',
):
    print(f'========== fold: {validation_fold} training ==========')

    trn_idx = folds[folds['fold'] != validation_fold].index
    val_idx = folds[folds['fold'] == validation_fold].index

    train_folds: 'DataFrame' = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds: 'DataFrame' = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds['InChI'].values

    images_path = pipeline_config.dataset.images_path
    encoder_train = pipeline_config.encoder_train
    decoder_train = pipeline_config.decoder_train

    image_transforms = get_transforms(encoder_config)
    train_dataset = TrainDataset(
        train_folds, images_path, tokenizer, transform=image_transforms
    )
    valid_dataset = TestDataset(
        valid_folds, images_path, tokenizer, transform=image_transforms
    )

    from functools import partial

    collate = partial(
        bms_collate, tokenizer=tokenizer
    )  # TODO: rewrite as callable class

    train_loader = DataLoader(
        train_dataset,
        batch_size=pipeline_config.batch_size,
        shuffle=True,
        num_workers=pipeline_config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=pipeline_config.batch_size,
        shuffle=False,
        num_workers=pipeline_config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    encoder.to(device)
    decoder.to(device)
    encoder_optimizer = Adam(
        encoder.parameters(),
        lr=encoder_train.lr,
        weight_decay=encoder_train.wd,
        amsgrad=False,
    )
    encoder_scheduler = get_scheduler(encoder_optimizer, pipeline_config)
    decoder_optimizer = Adam(
        decoder.parameters(),
        lr=decoder_train.lr,
        weight_decay=decoder_train.wd,
        amsgrad=False,
    )
    decoder_scheduler = get_scheduler(decoder_optimizer, pipeline_config)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    best_score = np.inf
    best_loss = np.inf

    for epoch in range(pipeline_config.epochs):
        start_time = time.time()

        avg_loss = train_fn(
           train_loader,
           encoder,
           decoder,
           criterion,
           encoder_optimizer,
           decoder_optimizer,
           epoch,
           encoder_scheduler,
           decoder_scheduler,
           device,
           tokenizer,
           pipeline_config,
           encoder_config,
           decoder_config,
        )

        valid_fn(valid_dataset, encoder, decoder, tokenizer, valid_labels, device)


def train_fn(
        train_loader,
        encoder,
        decoder,
        criterion,
        encoder_optimizer,
        decoder_optimizer,
        epoch,
        encoder_scheduler,
        decoder_scheduler,
        device,
        tokenizer,
        pipeline_config: 'PipelineConfig',
        encoder_config: 'EncoderBaseConfig',
        decoder_config: 'DecoderBaseConfig',
) -> float:
    train_info = TrainingInfo(len(train_loader), pipeline_config)

    encoder.train()
    decoder.train()

    dataset_processing_start = prev_batch_end = time.time()
    global_step = 0

    trained_steps = pipeline_config.checkpoint.samples_trained // pipeline_config.batch_size

    for step, (images, labels, label_lengths) in enumerate(train_loader):
        if pipeline_config.checkpoint.skip_steps and trained_steps > step:
            if pipeline_config.checkpoint.skip_steps and step % 1000 == 0 and trained_steps >= step:
                print(f"{step} Skipped")
            continue

        # Measure data loading time
        train_info.data_time.update(time.time() - prev_batch_end)

        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        features = encoder(images)

        # Sort all tensors by the length of target label
        label_lengths, sort_ind = label_lengths.squeeze(1).sort(dim=0, descending=True)
        label_lengths = (label_lengths - 1)

        features = features[sort_ind]
        labels = labels[sort_ind]

        targets_input = labels[:, :-1]
        targets_output = labels[:, 1:]

        tgt_mask, tgt_padding_mask = create_mask(targets_input, tokenizer, device)
        output = decoder(
            src=features,
            tgt=targets_input,
            tgt_mask=tgt_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=None,
        )

        predictions = output.permute(1, 0, 2)

        # FIXME: (not tested on GPU yet) if crushes during training on GPU, send `label_lengths` to CPU.
        predictions = _get_padded_sequence(predictions, label_lengths)
        targets_output = _get_padded_sequence(targets_output, label_lengths)
        loss = criterion(predictions, targets_output)

        # Record loss
        train_info.losses.update(loss.item(), batch_size)
        gradient_accum_steps = pipeline_config.gradient_accumulation_steps
        if gradient_accum_steps > 1:
            loss = loss / gradient_accum_steps
        loss.backward()
        encoder_grad_norm = nn.utils.clip_grad_norm_(
            encoder.parameters(), pipeline_config.max_grad_norm
        )
        decoder_grad_norm = nn.utils.clip_grad_norm_(
            decoder.parameters(), pipeline_config.max_grad_norm
        )
        if (step + 1) % gradient_accum_steps == 0:
            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            global_step += 1
            if pipeline_config.checkpoint.skip_steps:
                samples_trained = (trained_steps + global_step) * batch_size
                pipeline_config.checkpoint.samples_trained = samples_trained
            elif trained_steps < global_step:
                pipeline_config.checkpoint.samples_trained = global_step * batch_size

        # Measure batch processing time
        batch_end = time.time()
        train_info.batch_time.update(batch_end - prev_batch_end)
        prev_batch_end = batch_end

        if step != 0 and (
                step % pipeline_config.print_frequency == 0
                or step == (len(train_loader) - 1)
        ):
            train_info.print_stats(
                encoder_scheduler,
                decoder_scheduler,
                decoder_grad_norm,
                encoder_grad_norm,
                epoch,
                dataset_processing_start,
                step,
            )
        if (
                step != 0
                and step % pipeline_config.checkpoint.frequency == 0
                or step == (len(train_loader) - 1)
        ):
            save_checkpoint(
                'latest',  # TODO: use timestamp, when "keep n checkpoints" logic is implemented.
                encoder,
                decoder,
                encoder_config,
                decoder_config,
                pipeline_config,
            )
    return train_info.losses.avg


def valid_fn(valid_loader, encoder, decoder, tokenizer, valid_labels, device, batch_size=1, num_workers=2):

    encoder.eval()
    decoder.eval()

    text_preds = []
    max_length = 300
    val_slice = 100
    for i, images in enumerate(valid_loader):
        batch_size = images.size(0)
        ys = torch.full((batch_size, 1), tokenizer.sos_idx, dtype=torch.long).to(device)
        images = images.to(device)

        # plt.imshow(images.cpu().squeeze().permute(1, 2, 0))
        # plt.title(valid_labels[i])
        # plt.savefig(f'image_{i}.png')

        with torch.no_grad():
            features = encoder(images)
            features.to(device)
            # plt.imshow(features.cpu().squeeze().reshape(130, 1792))
            # plt.title(valid_labels[i])
            # plt.savefig(f'feature_{i}.png')

            memory = decoder.encode(features)
            # plt.imshow(memory.cpu().squeeze())
            # plt.title(valid_labels[i])
            # plt.savefig(f'memory_{i}.png')

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
    # print(text_preds)
    # print(valid_labels[i:i+batch_size])
    score = get_score(valid_labels[:val_slice+1], text_preds)
    print(score)
    print()


def get_score(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score


def _get_padded_sequence(predictions, label_lengths):
    return pack_padded_sequence(predictions, label_lengths.view(-1), batch_first=True).data


class TrainingInfo:
    def __init__(self, loader_size: int, config: 'PipelineConfig'):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.loader_size = loader_size
        self.config = config

    def print_stats(
            self,
            encoder_scheduler,
            decoder_scheduler,
            decoder_grad_norm,
            encoder_grad_norm,
            epoch,
            start,
            step,
    ):
        print(
            (
                'Epoch: [{0}][{1}/{2}] '
                'Data loading: {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Elapsed {remain:s} '
                'Loss: {loss.val:.6f}({loss.avg:.6f}) '
                'Encoder Grad: {encoder_grad_norm:.4f}  '
                'Decoder Grad: {decoder_grad_norm:.4f}  '
                'Encoder LR: {encoder_lr:.6f}  '
                'Decoder LR: {decoder_lr:.6f}  '
            ).format(
                epoch + 1,
                step,
                self.loader_size,
                batch_time=self.batch_time,
                data_time=self.data_time,
                loss=self.losses,
                remain=get_remaining_time(start, float(step + 1) / self.loader_size),
                encoder_grad_norm=encoder_grad_norm,
                decoder_grad_norm=decoder_grad_norm,
                encoder_lr=encoder_scheduler.get_lr()[0],
                decoder_lr=decoder_scheduler.get_lr()[0],
            )
        )
