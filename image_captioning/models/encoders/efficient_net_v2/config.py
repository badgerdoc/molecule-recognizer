from image_captioning.models.base import EncoderBaseConfig


class EfficientNetV2Config(EncoderBaseConfig):
    model: str

    @property
    def id(self) -> str:
        return f'{self.model}_{self.size[0]}x{self.size[1]}'
