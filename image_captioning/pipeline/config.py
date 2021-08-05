from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel

from image_captioning.base import RegisteredConfigMixin


class DatasetConfig(BaseModel):
    labels_path: Path
    images_path: Path
    n_fold: int
    validation_fold: int


class CheckpointConfig(BaseModel):
    # TODO: implement best checkpoint storage functionality
    #  disallow to remove best checkpoint even if it is old according to `number_to_keep`, keep it until better
    #  candidate emerges.

    # TODO: store configuration and logs with checkpoints, or have one log file to append records to
    load_from_checkpoint: bool
    frequency: int
    number_to_keep: int
    resume_from: str = 'latest'  # | 'best'
    skip_steps: bool
    samples_trained: int


class OptimizerConfig(BaseModel):
    lr: float
    wd: float


class SchedulerConfig(BaseModel):
    name: str
    params: Dict[str, Any]


class PipelineConfig(BaseModel, RegisteredConfigMixin):
    workdir: Path
    dataset: DatasetConfig
    checkpoint: CheckpointConfig

    encoder_train: OptimizerConfig
    decoder_train: OptimizerConfig
    scheduler: SchedulerConfig

    seed: int
    epochs: int
    batch_size: int
    num_workers: int

    print_frequency: int
    gradient_accumulation_steps: int
    max_grad_norm: int

    @property
    def checkpoint_path(self) -> Path:
        return self.workdir / 'checkpoints'

    @property
    def tokenizer_path(self) -> Path:
        return self.workdir.parent / 'tokenizer.pth'

    @property
    def preprocessed_train_df_path(self) -> Path:
        return self.workdir.parent / 'content/prep_train.pkl'
