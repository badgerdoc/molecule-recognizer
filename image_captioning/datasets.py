from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


if TYPE_CHECKING:
    from image_captioning.tokenizer import Tokenizer
    from image_captioning.pipeline.config import PipelineConfig
    from image_captioning.base import EncoderBaseConfig

INCHI_PREFIX = 'InChI=1S/'


def get_train_file_path(dataset_path: Path, image_id: str) -> str:
    return str(
        dataset_path
        / '{}/{}/{}/{}.png'.format(image_id[0], image_id[1], image_id[2], image_id)
    )


class ImageCaptioningDataset(Dataset, ABC):
    def __init__(self, df: pd.DataFrame, path: Path, tokenizer: 'Tokenizer', transform):
        super().__init__()
        self.df = df
        self.path = path
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)


class TrainDataset(ImageCaptioningDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path: Path,
        tokenizer: 'Tokenizer',
        transform,
        pipeline_config: 'PipelineConfig',
        encoder_config: 'EncoderBaseConfig',
    ):
        super(TrainDataset, self).__init__(df, path, tokenizer, transform)
        self.pipeline_config = pipeline_config
        self.encoder_config = encoder_config
        self.skip_to_idx = (
            pipeline_config.checkpoint.samples_trained
            if pipeline_config.checkpoint.skip_steps
            else 0
        )

    def __getitem__(self, idx):
        bs = self.pipeline_config.batch_size
        h, w = self.encoder_config.size
        label = self.df['InChI_text'][idx].replace(INCHI_PREFIX, '', 1)
        label = self.tokenizer.text_to_sequence(label)
        skip = torch.Tensor([0])
        if self.skip_to_idx != 0:
            self.skip_to_idx -= 1
            skip[0] = 1
            empty_tensor = torch.zeros(bs, h, w)
            return empty_tensor, torch.zeros(len(label)), torch.zeros(1), skip

        image_id = self.df['image_id'][idx]
        file_path = get_train_file_path(self.path, image_id)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        transformed = self.transform(image=image)
        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        return transformed['image'], torch.LongTensor(label), label_length, skip


class TestDataset(ImageCaptioningDataset):
    def __getitem__(self, idx):
        image_id = self.df['image_id'][idx]
        file_path = get_train_file_path(self.path, image_id)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        transformed = self.transform(image=image)
        return transformed['image']


def bms_collate(batch, tokenizer: 'Tokenizer'):
    imgs, labels, label_lengths, skip = [], [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
        skip.append(data_point[3])
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_idx)
    return (
        torch.stack(imgs),
        labels,
        torch.stack(label_lengths).reshape(-1, 1),
        torch.stack(skip),
    )
