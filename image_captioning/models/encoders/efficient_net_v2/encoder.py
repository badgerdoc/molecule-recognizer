import torch.nn as nn

from image_captioning.base import register_model, ConfigurableModel
from image_captioning.models.encoders.efficient_net_v2.config import EfficientNetV2Config
from image_captioning.models.encoders.efficient_net_v2.model import custom_effnetv2_xl, custom_effnetv2_l

EFFNET_V2_XL = 'effnetv2_xl'
EFFNET_V2_L = 'effnetv2_l'

_EFFICIENT_NET_V2_MODELS = {EFFNET_V2_L, EFFNET_V2_XL}


@register_model('effnetv2')
class EfficientNetV2Encoder(nn.Module, ConfigurableModel):

    def __init__(self, model: str):
        super().__init__()
        self.efficient_net = self._select_model(model)

    def forward(self, x):
        features = self.efficient_net.extract_features(x)
        features = features.permute(0, 2, 3, 1)
        return features

    @staticmethod
    def _select_model(model: str):
        if model == EFFNET_V2_XL:
            model = custom_effnetv2_xl()
        elif model == EFFNET_V2_L:
            model = custom_effnetv2_l()
        else:
            raise ValueError(
                f'Unknown model "{model}" select one of the following models: {" ".join(_EFFICIENT_NET_V2_MODELS)}'
            )
        return model

    @classmethod
    def from_config(cls, config: EfficientNetV2Config) -> 'EfficientNetV2Encoder':
        # TODO: What about other parameters like context tensor dimensions?
        return cls(model=config.model)
