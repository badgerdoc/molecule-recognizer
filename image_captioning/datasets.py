from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import (
    Compose,
    Resize,
)
from albumentations.augmentations.transforms import *
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from image_captioning.config import CFG
from image_captioning.constants import INCHI_PREFIX
from image_captioning.tokenization import Tokenizer


def get_train_file_path(dataset_path: Path, image_id: str) -> str:
    return str(
        dataset_path
        / "{}/{}/{}/{}.png".format(image_id[0], image_id[1], image_id[2], image_id)
    )


class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, path: Path, tokenizer: Tokenizer, transform):
        super().__init__()
        self.df = df
        self.path = path
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

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


class TestDataset(TrainDataset):

    def __getitem__(self, idx):
        image_id = self.df['image_id'][idx]
        file_path = get_train_file_path(self.path, image_id)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        transformed = self.transform(image=image)
        return transformed['image']


def get_transforms():
    return Compose([
        Resize(CFG.size[0], CFG.size[1]),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def bms_collate(batch, tokenizer):
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)
