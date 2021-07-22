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
    def __getitem__(self, idx):
        image_id = self.df['image_id'][idx]
        file_path = get_train_file_path(self.path, image_id)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        transformed = self.transform(image=image)

        label = self.df['InChI_text'][idx].replace(INCHI_PREFIX, '', 1)
        label = self.tokenizer.text_to_sequence(label)
        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        return transformed['image'], torch.LongTensor(label), label_length


class TestDataset(ImageCaptioningDataset):
    def __getitem__(self, idx):
        image_id = self.df['image_id'][idx]
        file_path = get_train_file_path(self.path, image_id)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        transformed = self.transform(image=image)
        return transformed['image']


def bms_collate(batch, tokenizer: 'Tokenizer'):
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_idx)
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)
