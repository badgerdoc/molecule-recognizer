from image_captioning.base import EncoderBaseConfig


class EfficientNetV1Config(EncoderBaseConfig):
    """
    Configuration for `EfficientNetV1` model.
    Attributes:
        model (str): name of the model (e.g. efficientnet-b2, efficientnet-b4, efficientnet-b7)
    """

    model: str

    @property
    def id(self) -> str:
        return f'{self.model}_{self.size[0]}x{self.size[1]}'
