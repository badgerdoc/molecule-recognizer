from efficientnet_pytorch import EfficientNet
import torch.nn as nn

from image_captioning.base import register_model, ConfigurableModel
from image_captioning.models.encoders.efficient_net_v1.config import (
    EfficientNetV1Config,
)


@register_model('effnetv1')
class EfficientNetV1Encoder(nn.Module, ConfigurableModel):
    def __init__(self, model):
        super().__init__()
        # TODO: disable or enable pretraining
        self.efnet = EfficientNet.from_pretrained(model)

    def forward(self, x):
        return self.efnet.extract_features(x)

    @classmethod
    def from_config(cls, config: EfficientNetV1Config) -> 'EfficientNetV1Encoder':
        return cls(model=config.model)
